import os
import torch
import torch.optim as optim
import numpy as np
import random
import configparser
import time
from datetime import datetime

# Import our modules
from models.conv_vae_model import ConvVae, vae_loss_function as conv_vae_loss_fn
from models.transformer_vae_model import TransformerVae, vae_loss_function as transformer_vae_loss_fn
from models.conv_vqvae_model import ConvVQVAE, vqvae_loss_function as conv_vqvae_loss_fn
from models.transformer_vqvae_model import TransformerVQVAE, vqvae_loss_function as transformer_vqvae_loss_fn
from trainer.trainer_vae import VAETrainer
from trainer.trainer_vqvae import VQVAETrainer
from dataloader.dataloader import load_fraud_data, load_config
from utils.model_saver import save_model, get_save_directory
from utils.wandb_logger_lr import WandBLogger
from utils.evaluation_utils import extract_recon_loss
from utils.lr_scheduler import create_scheduler

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_model(config_path, model_class, config_section, loss_fn):
    """
    Train a model with the given configuration.
    
    Args:
        config_path: Path to the config file
        model_class: Model class to instantiate
        config_section: Section in config file with model parameters
        loss_fn: Loss function to use for training
    """
    print(f"\n{'='*80}")
    print(f"Training with config: {config_path}")
    print(f"Model class: {model_class.__name__}")
    print(f"Config section: {config_section}")
    print(f"{'='*80}\n")
    

    # Determine the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load configuration
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    model_config = config_parser[config_section]
    train_config = config_parser["Trainer"]
    
    # Create config dictionary for WandB
    config_dict = {}
    for section in config_parser.sections():
        config_dict[section] = dict(config_parser[section])
    
    # Initialize WandB logger
    wandb_logger = WandBLogger(config_dict)
    
    # Load data
    data = load_fraud_data(config_path=config_path)
    dataloaders = data['dataloaders']
    input_dim = data['input_dim']
    
    # Create model
    model = model_class(model_config)
    model= model.to(device)
    print("Model parameters:")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_params}")
    
    # Training parameters
    lr = train_config.getfloat("lr")
    num_epochs = train_config.getint("num_epochs")
    patience = train_config.getint("patience", 15)  # Get early stopping patience from config
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = create_scheduler(train_config, optimizer)
    
    # Create trainer with scheduler
    if 'vqvae' in model_class.__name__.lower():
        trainer = VQVAETrainer(model, dataloaders, loss_fn, optimizer, scheduler,device=device)
    else:
        trainer = VAETrainer(model, dataloaders, loss_fn, optimizer, scheduler,device=device)
    
    # Create save directory
    save_dir = get_save_directory(config_path)
    print(f"Models will be saved to: {save_dir}")
    
    # Track best validation loss
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_recon_losses, val_recon_losses = [], []
    no_improve_count = 0  # Counter for early stopping
    
    try:
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Training epoch - handle different return values between VAE and VQVAE
            if 'vqvae' in model_class.__name__.lower():
                train_loss, train_recon_loss, train_vq_loss = trainer.train_epoch()
                val_loss, val_recon_loss, val_vq_loss = trainer.validate_epoch()
            else:
                train_loss, train_recon_loss = trainer.train_epoch()
                val_loss, val_recon_loss = trainer.validate_epoch()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_recon_losses.append(train_recon_loss)
            val_recon_losses.append(val_recon_loss)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to WandB (if enabled)
            try:
                wandb_logger.log_epoch(
                    epoch, train_loss, val_loss, 
                    train_recon_loss, val_recon_loss,
                    learning_rate=current_lr
                )
            except Exception as e:
                print(f"WandB logging error: {e}")
                wandb_logger.log_epoch(epoch, train_loss, val_loss)
            
            # Print progress with learning rate and reconstruction loss
            if 'vqvae' in model_class.__name__.lower():
                print(f"Epoch {epoch}/{num_epochs}: "
                      f"Train Loss = {train_loss:.6f} (Recon: {train_recon_loss:.6f}, VQ: {train_vq_loss:.6f}), "
                      f"Val Loss = {val_loss:.6f} (Recon: {val_recon_loss:.6f}, VQ: {val_vq_loss:.6f}), "
                      f"LR = {current_lr:.1e}")
            else:
                print(f"Epoch {epoch}/{num_epochs}: "
                      f"Train Loss = {train_loss:.6f} (Recon: {train_recon_loss:.6f}), "
                      f"Val Loss = {val_loss:.6f} (Recon: {val_recon_loss:.6f}), "
                      f"LR = {current_lr:.1e}")
            
            # Update scheduler based on validation loss
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Save best model and handle early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0  # Reset counter when we see improvement
                
                # Save the best model
                metadata = {
                    'epoch': epoch, 
                    'train_loss': train_loss, 
                    'val_loss': val_loss,
                    'train_recon_loss': train_recon_loss,
                    'val_recon_loss': val_recon_loss,
                    'learning_rate': current_lr,
                    'model_class': model_class.__name__,
                    'config_path': config_path
                }
                model_path = save_model(model, save_dir, 'best_model.pt', metadata)
                print(f"New best validation loss: {val_loss:.6f}")
                wandb_logger.log_model(model_path, metadata)
            else:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} epochs. Best val loss: {best_val_loss:.6f}")
                
                # Check if we should stop early
                if no_improve_count >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
    
        # Save final model if different from best
        if val_losses[-1] > best_val_loss:
            metadata = {
                'epoch': epoch, 
                'train_loss': train_losses[-1], 
                'val_loss': val_losses[-1],
                'train_recon_loss': train_recon_losses[-1],
                'val_recon_loss': val_recon_losses[-1],
                'learning_rate': optimizer.param_groups[0]['lr'],
                'model_class': model_class.__name__,
                'config_path': config_path
            }
            save_model(model, save_dir, 'final_model.pt', metadata)
    
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        return {
            'config_path': config_path,
            'model_class': model_class.__name__,
            'best_val_loss': best_val_loss,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_recon_loss': val_recon_losses[val_losses.index(best_val_loss)],
            'status': 'completed'
        }
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        metadata = {
            'epoch': epoch,
            'train_loss': train_losses[-1] if train_losses else None,
            'val_loss': val_losses[-1] if val_losses else None,
            'train_recon_loss': train_recon_losses[-1] if train_recon_losses else None,
            'val_recon_loss': val_recon_losses[-1] if val_recon_losses else None,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'model_class': model_class.__name__,
            'config_path': config_path,
            'interrupted': True
        }
        save_model(model, save_dir, 'interrupted_model.pt', metadata)
        print(f"Saved interrupted model at epoch {epoch}")
        return {
            'config_path': config_path,
            'model_class': model_class.__name__,
            'best_val_loss': best_val_loss,
            'status': 'interrupted'
        }
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_path': config_path,
            'model_class': model_class.__name__,
            'error': str(e),
            'status': 'failed'
        }
    
    finally:
        # Clean up WandB
        try:
            wandb_logger.finish()
        except:
            pass
        print("Training session complete.")


if __name__ == "__main__":
    # Define model configurations to loop through
    SEED=42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configs = {
        # VAE models
        "conv_vae": {
            "fraud": "configs/conv_vae/fraud_conv_vae.config",
            "normal": "configs/conv_vae/normal_conv_vae.config",
            "model_class": ConvVae,
            "config_section": "Conv_VAE",
            "loss_fn": conv_vae_loss_fn
        },
        "transformer_vae": {
            "fraud": "configs/transformer_vae/fraud_transformer_vae.config",
            "normal": "configs/transformer_vae/normal_transformer_vae.config",
            "model_class": TransformerVae,
            "config_section": "Transformer_VAE",
            "loss_fn": transformer_vae_loss_fn
        },
        # VQVAE models
        "conv_vqvae": {
            "fraud": "configs/conv_vqvae/fraud_conv_vqvae.config",
            "normal": "configs/conv_vqvae/normal_conv_vqvae.config",
            "model_class": ConvVQVAE,
            "config_section": "Conv_VQVAE",
            "loss_fn": conv_vqvae_loss_fn
        },
        "transformer_vqvae": {
            "fraud": "configs/transformer_vqvae/fraud_transformer_vqvae.config",
            "normal": "configs/transformer_vqvae/normal_transformer_vqvae.config",
            "model_class": TransformerVQVAE,
            "config_section": "Transformer_VQVAE",
            "loss_fn": transformer_vqvae_loss_fn
        }
    }
    
    # Track results
    results = []
    
    # Start timer
    start_time = time.time()
    
    # Loop through VAE vs VQVAE architectures
    arch_types = ["vae", "vqvae"]
    for arch in arch_types:
        # Loop through conv vs transformer
        model_types = ["conv", "transformer"]
        for model_type in model_types:
            # Combine to get full model type
            full_model_type = f"{model_type}_{arch}"
            if full_model_type not in configs:
                print(f"Skipping unknown model type: {full_model_type}")
                continue
                
            # Get model config
            model_config = configs[full_model_type]
            
            # Loop through fraud vs normal
            class_types = ["fraud", "normal"]
            for class_type in class_types:
                if class_type not in model_config:
                    print(f"Skipping missing class type {class_type} for {full_model_type}")
                    continue
                    
                config_path = model_config[class_type]
                if not os.path.exists(config_path):
                    print(f"Config file not found: {config_path}")
                    continue
                
                print(f"\n\n{'#'*100}")
                print(f"TRAINING {full_model_type.upper()} ON {class_type.upper()} DATA")
                print(f"{'#'*100}\n")
                
                # Train the model
                result = train_model(
                    config_path=config_path,
                    model_class=model_config["model_class"],
                    config_section=model_config["config_section"],
                    loss_fn=model_config["loss_fn"]
                )
                
                # Add more info to result
                result["arch_type"] = arch
                result["model_type"] = model_type
                result["full_model_type"] = full_model_type
                result["class_type"] = class_type
                
                # Store result
                results.append(result)
                
                # Just a short pause between trainings
                time.sleep(2)
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary
    print("\n\n" + "="*100)
    print(f"TRAINING COMPLETE - Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("="*100)
    
    print(f"\n{'Model Type':<20} {'Class':<10} {'Best Val Loss':<15} {'Best Recon Loss':<15} {'Status':<10}")
    print("-"*75)
    
    for result in results:
            model_type = result['full_model_type']
            class_type = result['class_type']
            best_val_loss = result.get('best_val_loss', float('inf'))
            best_recon_loss = result.get('best_recon_loss', float('inf'))
            status = result.get('status', 'Unknown')
            
            # Format the loss values
            if best_val_loss == float('inf'):
                val_loss_str = 'N/A'
            else:
                val_loss_str = f"{best_val_loss:.6f}"
                
            if best_recon_loss == float('inf'):
                recon_loss_str = 'N/A'
            else:
                recon_loss_str = f"{best_recon_loss:.6f}"
            
            print(f"{model_type:<20} {class_type:<10} {val_loss_str:<15} {recon_loss_str:<15} {status:<10}")
        
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"training_results_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'total_time_seconds': total_time
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
import os
import torch
import torch.optim as optim
import numpy as np
import random
import configparser

# Import our modules
from models.conv_vae_model import ConvVae, vae_loss_function, print_num_params
from trainer.trainer_vae import VAETrainer  # Updated trainer
# Add this import for VQVAETrainer
from trainer.trainer_vqvae import VQVAETrainer
#from dataloader.dataloader import load_fraud_data, load_config
from dataloader.dataloader import load_fraud_data, load_config
from utils.model_saver import save_model, get_save_directory
from utils.wandb_logger_lr import WandBLogger
from utils.lr_scheduler import create_scheduler  # New scheduler utility
from models.transformer_vae_model import TransformerVae
from models.transformer_vqvae_model import TransformerVQVAE, vqvae_loss_function
from models.conv_vqvae_model import ConvVQVAE, vqvae_loss_function,VectorQuantizer
# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Build the config path
config_path = "configs/conv_vae/fraud_conv_vae.config"
#config_path = "configs/conv_vae/normal_conv_vae.config"

#config_path="configs/conv_vqvae/fraud_conv_vqvae.config"
#config_path="configs/conv_vqvae/normal_conv_vqvae.config"

#config_path = "configs/transformer_vqvae/fraud_transformer_vqvae.config"
#config_path = "configs/transformer_vqvae/normal_transformer_vqvae.config"
#config_path = "configs/transformer_vae/fraud_transformer_vae.config"
#config_path = "configs/transformer_vae/normal_transformer_vae.config"


#config_path = "configs/conv_vae/fraud_conv_vae_test.config"

# Load configuration
config_parser = configparser.ConfigParser()
config_parser.read(config_path)
conv_vae_config = config_parser["Conv_VAE"]
#conv_vae_config = config_parser["Transformer_VAE"]
#conv_vae_config = config_parser["Transformer_VQVAE"]
#conv_vae_config = config_parser["Conv_VQVAE"]

train_config = config_parser["Trainer"]

# Create config dictionary for WandB
config_dict = {}
for section in config_parser.sections():
    config_dict[section] = dict(config_parser[section])

# Initialize WandB logger
wandb_logger = WandBLogger(config_dict)

# Load data
config_dict = load_config(config_path)
data = load_fraud_data(config_path=config_path)
dataloaders = data['dataloaders']
input_dim = data['input_dim']

# Create model
model = ConvVae(conv_vae_config)
#model=ConvVQVAE(conv_vae_config)
#model = TransformerVae(conv_vae_config)
#model = TransformerVQVAE(conv_vae_config)

print("Model parameters:")
print_num_params(model)

# Training parameters
lr = train_config.getfloat("lr")
num_epochs = train_config.getint("num_epochs")
patience = train_config.getint("patience", 15)  # Get early stopping patience from config

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr)

# Create learning rate scheduler
scheduler = create_scheduler(train_config, optimizer)

# Define loss function
loss_fn = vae_loss_function
#loss_fn = vqvae_loss_function

# Create trainer with scheduler
trainer = VAETrainer(model, dataloaders, loss_fn, optimizer, scheduler)
#trainer = VQVAETrainer(model, dataloaders, loss_fn, optimizer, scheduler)

# Create save directory
save_dir = get_save_directory(config_path)
print(f"Models will be saved to: {save_dir}")

# Track best validation loss
best_val_loss = float('inf')
train_losses, val_losses = [], []
train_recon_losses, val_recon_losses = [], []
#train_vq_losses, val_vq_losses = [], []  # Added for VQ losses
no_improve_count = 0  # Counter for early stopping

try:
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Get both total loss and reconstruction loss
        train_loss, train_recon_loss = trainer.train_epoch()
        val_loss, val_recon_loss = trainer.validate_epoch()
        #train_loss, train_recon_loss, train_vq_loss = trainer.train_epoch()
        #val_loss, val_recon_loss, val_vq_loss = trainer.validate_epoch()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_recon_losses.append(train_recon_loss)
        val_recon_losses.append(val_recon_loss)
        #train_vq_losses.append(train_vq_loss)  # Added for VQ losses
        #val_vq_losses.append(val_vq_loss)      # Added for VQ losses
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to WandB (if enabled)
        try:
            # Now we have direct access to recon losses - no need for extract_recon_loss function
            wandb_logger.log_epoch(
                epoch, train_loss, val_loss, 
                train_recon_loss, val_recon_loss,
                learning_rate=current_lr
            )
        except Exception as e:
            print(f"WandB logging error: {e}")
            wandb_logger.log_epoch(epoch, train_loss, val_loss)
        
        # Print progress with learning rate and reconstruction losses
        # print(f"Epoch {epoch}/{num_epochs}: "
        #       f"Train Loss = {train_loss:.6f} (Recon: {train_recon_loss:.6f}), "
        #       f"Val Loss = {val_loss:.6f} (Recon: {val_recon_loss:.6f}), "
        #       f"LR = {current_lr:.1e}")
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train Loss = {train_loss:.6f} (Recon: {train_recon_loss:.6f}"#, VQ: {train_vq_loss:.6f}), "
              f"Val Loss = {val_loss:.6f} (Recon: {val_recon_loss:.6f}"#, VQ: {val_vq_loss:.6f}), "
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
                #'train_vq_loss': train_vq_loss,  # Added for VQ losses
                #'val_vq_loss': val_vq_loss,      # Added for VQ losses
                'learning_rate': current_lr
            }
            model_path = save_model(model, save_dir, 'best_model.pt', metadata)
            # print(f"New best validation loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f})")
            print(f"New best validation loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}")#, VQ: {val_vq_loss:.6f})")
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
            #'train_vq_loss': train_vq_losses[-1],  # Added for VQ losses
            #'val_vq_loss': val_vq_losses[-1],      # Added for VQ losses 
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        save_model(model, save_dir, 'final_model.pt', metadata)

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current state...")
    # Save interrupted model logic...
    metadata = {
        'epoch': epoch,
        'train_loss': train_losses[-1] if train_losses else None,
        'val_loss': val_losses[-1] if val_losses else None,
        'train_recon_loss': train_recon_losses[-1] if train_recon_losses else None,
        'val_recon_loss': val_recon_losses[-1] if val_recon_losses else None,
        #'train_vq_loss': train_vq_losses[-1] if train_vq_losses else None,  # Added for VQ losses
        #'val_vq_loss': val_vq_losses[-1] if val_vq_losses else None,        # Added for VQ losses
        'learning_rate': optimizer.param_groups[0]['lr'],
        'interrupted': True
    }
    save_model(model, save_dir, 'interrupted_model.pt', metadata)
    print(f"Saved interrupted model at epoch {epoch}")

except Exception as e:
    print(f"\nTraining failed with error: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clean up WandB
    try:
        wandb_logger.finish()
    except:
        pass
    print("Training session complete.")
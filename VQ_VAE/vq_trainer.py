import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import wandb
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import custom modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the VQVAE model and data loading utilities
from vq_vae_model import VQVAE, load_config
from vq_dataloader import load_fraud_data


def train_epoch(model: VQVAE, 
                train_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The VQVAE model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_perplexity = 0.0
    batch_count = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(train_loader, desc=f"Training", leave=False)
    
    for batch in progress_bar:
        # Move batch to device
        batch = batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        
        # Get losses
        loss = outputs['loss']
        recon_loss = outputs['reconstruction_loss']
        vq_loss = outputs['vq_loss']
        perplexity = outputs['perplexity']
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            'perplexity': perplexity.item()
        })
    
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_recon_loss = total_recon_loss / batch_count
    avg_vq_loss = total_vq_loss / batch_count
    avg_perplexity = total_perplexity / batch_count
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'vq_loss': avg_vq_loss,
        'perplexity': avg_perplexity
    }


def validate(model: VQVAE, 
             val_loader: torch.utils.data.DataLoader, 
             device: torch.device) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: The VQVAE model
        val_loader: DataLoader for validation data
        device: Device to validate on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_perplexity = 0.0
    batch_count = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(val_loader, desc=f"Validating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Get losses
            loss = outputs['loss']
            recon_loss = outputs['reconstruction_loss']
            vq_loss = outputs['vq_loss']
            perplexity = outputs['perplexity']
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'val_loss': loss.item(),
                'val_recon_loss': recon_loss.item(),
                'val_vq_loss': vq_loss.item(),
                'val_perplexity': perplexity.item()
            })
    
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_recon_loss = total_recon_loss / batch_count
    avg_vq_loss = total_vq_loss / batch_count
    avg_perplexity = total_perplexity / batch_count
    
    return {
        'val_loss': avg_loss,
        'val_reconstruction_loss': avg_recon_loss,
        'val_vq_loss': avg_vq_loss,
        'val_perplexity': avg_perplexity
    }


def calculate_anomaly_scores(model: VQVAE, 
                              data_loader: torch.utils.data.DataLoader, 
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate anomaly scores for all samples in the dataset.
    
    Args:
        model: The VQVAE model
        data_loader: DataLoader for the dataset
        device: Device to use
        
    Returns:
        Tuple of (anomaly_scores, labels)
    """
    model.eval()
    all_scores = []
    all_encodings = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating anomaly scores"):
            # Move batch to device
            batch = batch.to(device)
            
            # Calculate anomaly scores
            scores = model.calculate_anomaly_scores(batch)
            encodings = model.encode_to_indices(batch)
            
            # Store scores and encodings
            all_scores.append(scores.cpu())
            all_encodings.append(encodings.cpu())
    
    # Concatenate all scores and encodings
    all_scores = torch.cat(all_scores)
    all_encodings = torch.cat(all_encodings)
    
    return all_scores, all_encodings


def train_model(config_path: str, model_path: Optional[str] = None):
    """
    Train the VQ-VAE model using the provided configuration.
    
    Args:
        config_path: Path to the configuration file
        model_path: Path to load pretrained model (optional)
    """
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")
    
    # Extract configuration parameters
    vqvae_config = config.get('VQVAE', {})
    dataloader_config = config.get('DataLoader', {})
    trainer_config = config.get('Trainer', {})
    wandb_config = config.get('WandB', {})
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data = load_fraud_data(config_path=config_path)
    train_loader = data['dataloaders']['train']
    val_loader = data['dataloaders']['val']
    test_loader = data['dataloaders']['test']
    input_dim = data['input_dim']
    
    print(f"Data loaded successfully. Input dimension: {input_dim}")
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize or load the model
    if model_path and os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        model = VQVAE.load(model_path, device=device)
    else:
        print("Initializing new model")
        model = VQVAE(input_dim=input_dim, config_path=config_path)
    
    model = model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print model architecture details
    print(f"\nModel Architecture:")
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Hidden dimensions: {vqvae_config.get('hidden_dims', [128, 64])}")
    print(f"  Latent dimension: {model.latent_dim}")
    print(f"  Codebook size: {model.num_embeddings}")
    print(f"  Embedding dimension: {model.embedding_dim}")
    print(f"  Commitment cost: {vqvae_config.get('commitment_cost', 0.25)}")
    print(f"  Decay: {vqvae_config.get('decay', 0.99)}")
    print(f"  Dropout rate: {vqvae_config.get('dropout_rate', 0.1)}")
    print("")
    
    # Initialize optimizer
    learning_rate = trainer_config.get('learning_rate', 1e-3)
    weight_decay = trainer_config.get('weight_decay', 1e-5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Training parameters
    num_epochs = trainer_config.get('num_epochs', 50)
    base_save_dir = trainer_config.get('save_dir', 'saved_models/vq_vae')
    validate_every = trainer_config.get('validate_every', 1)
    save_best_only = trainer_config.get('save_best_only', True)
    early_stopping = trainer_config.get('early_stopping', True)
    patience = trainer_config.get('patience', 15)
    
    # Create timestamped directory for this run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = wandb_config.get('name', 'vqvae-training')
    save_dir = os.path.join(base_save_dir, f"{timestamp}_{run_name}")
    print(f"Models will be saved to: {save_dir}")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize WandB if configured
    use_wandb = wandb_config.get('use_wandb', False)
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project', 'fraud-vqvae'),
            name=wandb_config.get('name', 'vqvae-fraud-training'),
            entity=wandb_config.get('entity', None),
            tags=wandb_config.get('tags', ['fraud', 'vqvae']),
            config={
                **vqvae_config,
                **{f"dataloader_{k}": v for k, v in dataloader_config.items()},
                **{f"trainer_{k}": v for k, v in trainer_config.items()},
                'input_dim': input_dim
            }
        )
        # Watch the model to track gradients and parameters
        wandb.watch(model)
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Validate if needed
        if (epoch + 1) % validate_every == 0:
            val_metrics = validate(model, val_loader, device)
            
            # Update learning rate scheduler
            scheduler.step(val_metrics['val_loss'])
            
            # Check if this is the best model so far
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                epochs_without_improvement = 0
                
                # Save best model with validation loss in filename
                model_save_path = os.path.join(save_dir, f'best_model_val_loss_{val_metrics["val_loss"]:.4f}.pt')
                model.save(model_save_path)
                
                # Also save as best_model.pt for easy reference
                best_model_path = os.path.join(save_dir, 'best_model.pt')
                model.save(best_model_path)
                
                print(f"Saved best model (val_loss: {val_metrics['val_loss']:.4f}) to {model_save_path}")
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if early_stopping and epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs without improvement")
                break
        else:
            val_metrics = {}
        
        # Save checkpoint with metrics in filename
        if not save_best_only or (epoch + 1) % 10 == 0:
            checkpoint_name = f'model_epoch_{epoch+1}_loss_{train_metrics["loss"]:.4f}'
            if val_metrics:
                checkpoint_name += f'_val_loss_{val_metrics["val_loss"]:.4f}'
            model_save_path = os.path.join(save_dir, f'{checkpoint_name}.pt')
            model.save(model_save_path)
            print(f"Saved checkpoint to {model_save_path}")
        
        # Log metrics
        epoch_time = time.time() - epoch_start_time
        metrics = {**train_metrics, **val_metrics, 'epoch_time': epoch_time}
        
        # Print metrics
        metrics_str = f"Epoch {epoch + 1}/{num_epochs} - Time: {epoch_time:.2f}s - "
        metrics_str += " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'epoch_time'])
        print(metrics_str)
        
        # Log to WandB
        if use_wandb:
            wandb.log(metrics, step=epoch + 1)
    
    print("Training completed")
    
    # Final evaluation on test set
    print("Evaluating on test set")
    test_metrics = validate(model, test_loader, device)
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Calculate and analyze anomaly scores
    print("Calculating anomaly scores")
    anomaly_scores, encodings = calculate_anomaly_scores(model, test_loader, device)
    
    # Save final model with test metrics
    final_model_name = f'final_model_test_loss_{test_metrics["val_loss"]:.4f}'
    final_model_path = os.path.join(save_dir, f'{final_model_name}.pt')
    model.save(final_model_path)
    
    # Also save as final_model.pt for easy reference
    simple_final_path = os.path.join(save_dir, 'final_model.pt')
    model.save(simple_final_path)
    
    print(f"Saved final model to {final_model_path}")
    
    # Save a summary of the training run
    summary_path = os.path.join(save_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Configuration file: {config_path}\n")
        f.write(f"Input dimension: {input_dim}\n")
        f.write(f"Epochs trained: {epoch + 1}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n\n")
        
        f.write("Final test metrics:\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v:.6f}\n")
            
        f.write("\nModel architecture:\n")
        f.write(f"  Input dimension: {model.input_dim}\n")
        f.write(f"  Latent dimension: {model.latent_dim}\n")
        f.write(f"  Codebook size: {model.num_embeddings}\n")
        f.write(f"  Total parameters: {total_params:,}\n")
        f.write(f"  Trainable parameters: {trainable_params:,}\n")
        
    print(f"Saved training summary to {summary_path}")
    
    # Close WandB
    if use_wandb:
        wandb.log(test_metrics)
        wandb.finish()
    
    return model, test_metrics, anomaly_scores, encodings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VQ-VAE model')
    parser.add_argument('--config', type=str, default='configs/vq_vae/vq_vae_test.config',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model (optional)')
    args = parser.parse_args()
    
    # Change to parent directory if needed (when running from subdirectory)
    if not os.path.exists(args.config) and os.path.exists(os.path.join('..', args.config)):
        os.chdir('..')
        print(f"Changed working directory to {os.getcwd()}")
    
    # Train the model
    model, test_metrics, anomaly_scores, encodings = train_model(args.config, args.model)
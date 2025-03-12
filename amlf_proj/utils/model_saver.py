import os
import torch
from datetime import datetime
import re

def save_model(model, config_path, epoch=None, is_best=False, val_loss=None, train_loss=None):
    """
    Save model checkpoints based on config path.
    
    Args:
        model (torch.nn.Module): The model to save.
        config_path (str): Path to the config file (e.g. "configs/conv_vqvae/fraud_conv_vqvae.config").
        epoch (int, optional): Current epoch number.
        is_best (bool): Whether this is the best model so far.
        val_loss (float, optional): Validation loss to record.
        train_loss (float, optional): Training loss to record.
    
    Returns:
        str: Path where the model was saved.
    """
    # Remove 'configs/' from the beginning and '.config' from the end
    stripped_path = re.sub(r'^configs/', '', config_path)
    stripped_path = re.sub(r'\.config$', '', stripped_path)
    
    # Create a timestamp folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the base directory if it doesn't exist
    base_dir = os.path.join('saved_models', stripped_path)
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a timestamped subdirectory for this run if it doesn't exist yet
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Define filename
    if is_best:
        filename = 'best_model.pt'
    elif epoch is not None:
        filename = f'model_epoch_{epoch}.pt'
    else:
        filename = 'final_model.pt'
    
    # Full path to save
    save_path = os.path.join(save_dir, filename)
    
    # Prepare metadata
    metadata = {}
    if epoch is not None:
        metadata['epoch'] = epoch
    if val_loss is not None:
        metadata['val_loss'] = val_loss
    if train_loss is not None:
        metadata['train_loss'] = train_loss
    
    # Save the model along with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, save_path)
    
    print(f"Model saved to {save_path}")
    return save_path
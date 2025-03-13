import os
import torch
from datetime import datetime
import re

def get_save_directory(config_path):
    """
    Create a directory structure based on the config path with a timestamp.
    
    Args:
        config_path (str): Path to the config file.
        
    Returns:
        str: Path to the timestamped directory where models should be saved.
    """
    # Remove 'configs/' from the beginning and '.config' from the end
    stripped_path = re.sub(r'^configs/', '', config_path)
    stripped_path = re.sub(r'\.config$', '', stripped_path)
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the full path
    save_dir = os.path.join('saved_models', stripped_path, timestamp)
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    return save_dir

def save_model(model, save_dir, filename, metadata=None):
    """
    Save a model checkpoint to the specified directory.
    
    Args:
        model (torch.nn.Module): The model to save.
        save_dir (str): Directory to save the model in.
        filename (str): Filename for the saved model.
        metadata (dict, optional): Additional metadata to save with the model.
    
    Returns:
        str: Path where the model was saved.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Full path to save
    save_path = os.path.join(save_dir, filename)
    
    # Prepare metadata if not provided
    if metadata is None:
        metadata = {}
    
    # Save the model along with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, save_path)
    
    print(f"Model saved to {save_path}")
    return save_path
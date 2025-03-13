import os
import torch
import wandb
from dotenv import load_dotenv
from pathlib import Path

class WandBLogger:
    """
    A simple WandB logger for tracking VAE/VQ-VAE training.
    Handles configuration, initialization, and logging metrics to WandB.
    """
    def __init__(self, config_dict):
        """
        Initialize the WandB logger if wandb is enabled in config.
        
        Args:
            config_dict (dict): The full configuration dictionary
        """
        # Check if WandB should be used
        wandb_config = config_dict.get('WandB', {})
        self.use_wandb = wandb_config.get('use_wandb', False)
        
        if not self.use_wandb:
            print("WandB logging is disabled in config.")
            return
        
        # Load API key from .env file
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        
        api_key = os.environ.get('WANDB_API_KEY')
        if not api_key:
            print("Warning: WANDB_API_KEY not found in .env file. WandB logging disabled.")
            self.use_wandb = False
            return
        
        # Get project settings from config
        self.project = wandb_config.get('project', 'fraud-vae')
        self.entity = wandb_config.get('entity', None)
        self.name = wandb_config.get('name', None)
        self.tags = wandb_config.get('tags', [])
        if isinstance(self.tags, str):
            self.tags = [tag.strip() for tag in self.tags.split(',')]
        
        # Initialize WandB session
        print(f"Initializing WandB run: {self.name} (Project: {self.project}, Entity: {self.entity})")
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            tags=self.tags,
            config=config_dict
        )
    
    def log_epoch(self, epoch, train_loss, val_loss, recon_loss=None, val_recon_loss=None):
        """
        Log epoch results to WandB.
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss
            val_loss (float): Validation loss
            recon_loss (float, optional): Training reconstruction loss component
            val_recon_loss (float, optional): Validation reconstruction loss component
        """
        if not self.use_wandb:
            return
        
        log_dict = {
            'epoch': epoch,
            'train/total_loss': train_loss,
            'val/total_loss': val_loss
        }
        
        # Add reconstruction losses if provided
        if recon_loss is not None:
            log_dict['train/recon_loss'] = recon_loss
        if val_recon_loss is not None:
            log_dict['val/recon_loss'] = val_recon_loss
        
        wandb.log(log_dict)
    
    def log_model(self, model_path, metadata=None):
        """
        Log model artifact to WandB.
        
        Args:
            model_path (str): Path to saved model file
            metadata (dict, optional): Additional metadata about the model
        """
        if not self.use_wandb:
            return
        
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}", 
            type="model",
            metadata=metadata
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    def finish(self):
        """End the WandB run."""
        if self.use_wandb:
            wandb.finish()
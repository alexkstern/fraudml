import os
import torch
import wandb
import time
from datetime import datetime
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
        self.initialized = False
        
        # Check if WandB should be used - handle string values like "False" correctly
        wandb_config = config_dict.get('WandB', {})
        use_wandb_value = wandb_config.get('use_wandb', False)
        
        # Handle string values like "False" or "false"
        if isinstance(use_wandb_value, str):
            self.use_wandb = use_wandb_value.lower() == 'true'
        else:
            self.use_wandb = bool(use_wandb_value)
        
        # Exit early if WandB is disabled - don't initialize anything else
        if not self.use_wandb:
            print("WandB disabled: use_wandb is False in config")
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
        
        # Add timestamp to the run name for uniqueness
        base_name = wandb_config.get('name', None)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = f"{base_name}_{timestamp}" if base_name else timestamp
        
        self.tags = wandb_config.get('tags', [])
        if isinstance(self.tags, str):
            self.tags = [tag.strip() for tag in self.tags.split(',')]
        
        # Initialize WandB session with better error handling
        print(f"Initializing WandB run: {self.name} (Project: {self.project}, Entity: {self.entity})")
        try:
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                tags=self.tags,
                config=config_dict
            )
            self.initialized = True
        except Exception as e:
            print(f"Error initializing WandB: {e}")
            self.use_wandb = False
    
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
        if not self.use_wandb or not self.initialized:
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
        
        try:
            wandb.log(log_dict)
        except Exception as e:
            print(f"Error logging to WandB: {e}")
            # If we hit an error, disable subsequent logging
            self.initialized = False
    
    def log_model(self, model_path, metadata=None):
        """
        Log model artifact to WandB.
        
        Args:
            model_path (str): Path to saved model file
            metadata (dict, optional): Additional metadata about the model
        """
        if not self.use_wandb or not self.initialized:
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}", 
                type="model",
                metadata=metadata
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Error logging model artifact to WandB: {e}")
            # If we hit an error, disable subsequent logging
            self.initialized = False
    
    def finish(self):
        """
        End the WandB run safely.
        This method is safe to call multiple times and in any state.
        Handles broken pipe errors that can happen during keyboard interrupts.
        """
        if not self.use_wandb or not self.initialized:
            return
            
        try:
            # Check if wandb.run exists and is not None
            if wandb.run is not None:
                # First try a clean shutdown
                try:
                    wandb.finish()
                except BrokenPipeError:
                    print("WandB connection already closed. Skipping clean shutdown.")
                except Exception as e:
                    print(f"Warning: Error during WandB finish: {e}")
                    
                # If that fails, try to force terminate after waiting a moment
                time.sleep(0.5)  # Give a short delay for any operations to complete
                try:
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)  # Force exit
                except Exception:
                    # At this point, we've done our best
                    pass
        except Exception as e:
            print(f"Final error in WandB cleanup: {e}")
            # Don't re-raise, as this is cleanup code
        finally:
            # Mark as no longer initialized
            self.initialized = False
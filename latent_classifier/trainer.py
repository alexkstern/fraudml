import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import logging
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("trainer")

class Trainer:
    """
    A flexible trainer class for PyTorch models with optional WandB integration.
    Uses a config file with sections for VAE, DataLoader, Trainer, and WandB.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        config: Dict[str, Dict[str, Any]],
        optimizer_class: Optional[Any] = None,
        loss_fn: Optional[callable] = None,
        device: Optional[str] = None,
        load_checkpoint: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            dataloaders: Dictionary with keys 'train', 'val', and optionally 'test'
            config: Configuration dictionary with sections as keys
            optimizer_class: Optimizer class (default: torch.optim.Adam)
            loss_fn: Loss function (default: None, will use model's loss function)
            device: Device to train on (default: automatically determined)
            load_checkpoint: Path to checkpoint to load (default: None)
        """
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set optimizer
        if optimizer_class is None:
            optimizer_class = optim.Adam
        
        # Get optimizer params (default to 1e-3 if not specified)
        optimizer_config = config.get('Optimizer', {})
        lr = optimizer_config.get('lr', 1e-3)
        self.optimizer = optimizer_class(model.parameters(), lr=lr)
        
        # Set loss function
        self.loss_fn = loss_fn
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.last_save_path = None
        
        # Extract training settings from config
        trainer_config = config.get('Trainer', {})
        self.num_epochs = trainer_config.get('num_epochs', 100)
        self.save_dir = trainer_config.get('save_dir', 'saved_models')
        self.validate_every = trainer_config.get('validate_every', 1)
        self.save_best_only = trainer_config.get('save_best_only', True)
        self.early_stopping = trainer_config.get('early_stopping', False)
        self.patience = trainer_config.get('patience', 10)
        self.patience_counter = 0
        
        # WandB settings
        wandb_config = config.get('WandB', {})
        self.use_wandb = wandb_config.get('use_wandb', False)
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize model name/id
        model_name = model.__class__.__name__
        run_id = wandb_config.get('run_id', time.strftime("%Y%m%d-%H%M%S"))
        self.model_id = f"{model_name}_{run_id}"
        
        # Get class info for model name
        dataloader_config = config.get('DataLoader', {})
        class_filter = dataloader_config.get('class_filter')
        if class_filter is not None:
            class_name = "fraud" if class_filter == 1 else "non-fraud" if class_filter == 0 else "all"
            self.model_id = f"{model_name}_{class_name}_{run_id}"
        
        # Load checkpoint if provided
        if load_checkpoint:
            self._load_checkpoint(load_checkpoint)
        
        # Initialize WandB if enabled
        self.wandb_initialized = False
        if self.use_wandb:
            self._initialize_wandb()
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases for experiment tracking with robust error handling."""
        if not self.use_wandb:
            return False
            
        try:
            import wandb
            
            load_dotenv()  # Load environment variables from .env file
            
            # Get WandB API key from environment variable
            api_key = os.getenv('WANDB_API_KEY')
            if not api_key:
                logger.warning("WANDB_API_KEY not found in .env file. WandB logging disabled.")
                self.use_wandb = False
                return False
            
            # Get WandB configuration from config
            wandb_config = self.config.get('WandB', {})
            project = wandb_config.get('project', 'vae-fraud-detection')
            entity = wandb_config.get('entity', None)
            name = wandb_config.get('name', self.model_id)
            
            # Handle tags as a list or comma-separated string
            tags = wandb_config.get('tags', [])
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',')]
            
            # Check if wandb is already initialized
            if wandb.run is not None:
                logger.info("WandB already initialized")
                self.wandb_initialized = True
                return True
                
            # Initialize WandB
            try:
                wandb.login(key=api_key)
                wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    config=self.config,
                    tags=tags
                )
                
                # Verify initialization was successful
                if wandb.run is None:
                    logger.warning("WandB initialization failed. Disabling WandB logging.")
                    self.use_wandb = False
                    return False
                    
                logger.info(f"WandB initialized: project={project}, run={name}")
                
                # Watch model
                wandb.watch(self.model, log="all", log_freq=100)
                self.wandb_initialized = True
                return True
                
            except Exception as e:
                logger.warning(f"Error during wandb.init(): {e}")
                self.use_wandb = False
                return False
                
        except ImportError:
            logger.warning("wandb package not installed. WandB logging disabled.")
            self.use_wandb = False
            return False
        except Exception as e:
            logger.error(f"Error initializing WandB: {e}")
            self.use_wandb = False
            return False
        
    def _log_to_wandb(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to WandB if enabled with robust error handling.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step (epoch)
            prefix: Prefix for metric names
        """
        if not self.use_wandb:
            return
        
        try:
            import wandb
            
            # First check if wandb is properly initialized
            if wandb.run is None:
                logger.warning("WandB run is not initialized. Skipping logging.")
                self.wandb_initialized = False  # Mark as not initialized to prevent future attempts
                return
            
            # Filter metrics to include only what's essential
            essential_metrics = {}
            
            # For batch metrics, only include overall loss and skip detailed reporting
            if prefix == "batch":
                # Only log the overall loss for batches if needed
                essential_metrics['batch/loss'] = metrics.get('batch/loss', 0)
            else:
                # For epoch metrics, include only train and val losses with their components
                for key, value in metrics.items():
                    # Include only train and val loss metrics
                    if key.startswith('train/') or key.startswith('val/') or key.startswith('test/'):
                        if 'loss' in key:  # Only include loss-related metrics
                            essential_metrics[key] = value
            
            # Log the filtered metrics
            if essential_metrics:
                wandb.log(essential_metrics, step=step)
                
        except ImportError:
            logger.warning("WandB not installed. Disabling WandB logging.")
            self.wandb_initialized = False
            self.use_wandb = False
        except (AttributeError, Exception) as e:
            logger.warning(f"Error logging to WandB: {e}")
            logger.warning("Disabling WandB logging for future steps.")
            self.wandb_initialized = False

    def cleanup_wandb(self):
        """
        Properly finish the WandB run if it's initialized.
        Call this method when you're done with training/evaluation or in case of an exception.
        """
        if self.use_wandb and self.wandb_initialized:
            try:
                import wandb
                if wandb.run is not None:
                    logger.info("Finishing WandB run")
                    wandb.finish()
                self.wandb_initialized = False
            except (ImportError, Exception) as e:
                logger.warning(f"Error finishing WandB run: {e}")
                self.wandb_initialized = False
    
    def _save_checkpoint(self, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        if self.save_best_only and not is_best:
            return
        
        file_name = f"{self.model_id}_epoch{self.current_epoch+1}_val{val_loss:.4f}.pt"
        save_path = os.path.join(self.save_dir, file_name)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        self.last_save_path = save_path
        
        # Log to WandB
        if self.use_wandb and self.wandb_initialized:
            try:
                import wandb
                if wandb.run is not None:
                    artifact = wandb.Artifact(
                        name=f"model-checkpoint-{self.current_epoch+1}",
                        type="model",
                        description=f"Model checkpoint at epoch {self.current_epoch+1}"
                    )
                    artifact.add_file(save_path)
                    wandb.log_artifact(artifact)
            except ImportError:
                pass
            except Exception as e:
                logger.error(f"Error logging artifact to WandB: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint {checkpoint_path} not found")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.best_val_loss = checkpoint['val_loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']+1})")
    
    def vae_loss_function(self, recon_x, x, mu, logvar):
        """
        Default VAE loss function.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            total_loss, (recon_loss, kl_loss)
        """
        # Extract weights from config
        vae_config = self.config.get('VAE', {})
        recon_weight = vae_config.get('recon_weight', 1.0)
        kl_weight = vae_config.get('kl_weight', 1.0)
        
        # Calculate losses
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_weight * recon_loss + kl_weight * kl_loss
        
        return total_loss, (recon_loss, kl_loss)
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss, metrics
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        samples_processed = 0
        
        train_loader = self.dataloaders['train']
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass (assuming VAE model)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(batch)
            
            # Compute loss
            if self.loss_fn is not None:
                loss, (recon_loss, kl_loss) = self.loss_fn(recon_batch, batch, mu, logvar)
            else:
                loss, (recon_loss, kl_loss) = self.vae_loss_function(recon_batch, batch, mu, logvar)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_size = batch.size(0)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            samples_processed += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() / batch_size,
                'recon': recon_loss.item() / batch_size,
                'kl': kl_loss.item() / batch_size
            })
            
            # REMOVED: Batch-level WandB logging to reduce reports
            # Only log every 50 batches if you really need batch-level metrics
            # if self.use_wandb and batch_idx % 50 == 0 and self.wandb_initialized:
            #    step = self.current_epoch * len(train_loader) + batch_idx
            #    self._log_to_wandb({'batch/loss': loss.item() / batch_size}, step=step)
        
        # Calculate average losses
        avg_loss = total_loss / samples_processed
        avg_recon_loss = total_recon_loss / samples_processed
        avg_kl_loss = total_kl_loss / samples_processed
        
        metrics = {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'kl_ratio': avg_kl_loss / avg_loss if avg_loss > 0 else 0
        }
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate model on validation set.
        
        Returns:
            avg_loss, metrics
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        samples_processed = 0
        
        val_loader = self.dataloaders['val']
        progress_bar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Val]")
        
        for batch in progress_bar:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass
            recon_batch, mu, logvar = self.model(batch)
            
            # Compute loss
            if self.loss_fn is not None:
                loss, (recon_loss, kl_loss) = self.loss_fn(recon_batch, batch, mu, logvar)
            else:
                loss, (recon_loss, kl_loss) = self.vae_loss_function(recon_batch, batch, mu, logvar)
            
            # Update metrics
            batch_size = batch.size(0)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            samples_processed += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'val_loss': loss.item() / batch_size
            })
        
        # Calculate average losses
        avg_loss = total_loss / samples_processed
        avg_recon_loss = total_recon_loss / samples_processed
        avg_kl_loss = total_kl_loss / samples_processed
        
        metrics = {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'kl_ratio': avg_kl_loss / avg_loss if avg_loss > 0 else 0
        }
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def evaluate_on_test(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary of metrics
        """
        if 'test' not in self.dataloaders:
            logger.warning("Test dataloader not provided. Skipping test evaluation.")
            return {}
        
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        samples_processed = 0
        
        test_loader = self.dataloaders['test']
        progress_bar = tqdm(test_loader, desc="Testing")
        
        for batch in progress_bar:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass
            recon_batch, mu, logvar = self.model(batch)
            
            # Compute loss
            if self.loss_fn is not None:
                loss, (recon_loss, kl_loss) = self.loss_fn(recon_batch, batch, mu, logvar)
            else:
                loss, (recon_loss, kl_loss) = self.vae_loss_function(recon_batch, batch, mu, logvar)
            
            # Update metrics
            batch_size = batch.size(0)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            samples_processed += batch_size
        
        # Calculate average losses
        avg_loss = total_loss / samples_processed
        avg_recon_loss = total_recon_loss / samples_processed
        avg_kl_loss = total_kl_loss / samples_processed
        
        # Only report essential metrics
        metrics = {
            'test/loss': avg_loss,
            'test/recon_loss': avg_recon_loss,
            'test/kl_loss': avg_kl_loss
        }
        
        # Log only essential metrics to wandb - with proper checks
        if self.use_wandb and self.wandb_initialized:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'test/loss': avg_loss,
                        'test/recon_loss': avg_recon_loss,
                        'test/kl_loss': avg_kl_loss
                    }, step=self.current_epoch)
            except (ImportError, AttributeError, Exception) as e:
                logger.warning(f"Error logging to WandB: {e}")
                logger.warning("Continuing without WandB logging")
        
        return metrics
    
    @torch.no_grad()
    def collect_latent_embeddings(self, dataset: str = 'train', max_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect latent space embeddings.
        
        Args:
            dataset: Dataset to use ('train', 'val', or 'test')
            max_samples: Maximum number of samples to collect
            
        Returns:
            latent_embeddings, original_data
        """
        if dataset not in self.dataloaders:
            logger.warning(f"{dataset} dataloader not provided. Skipping embedding collection.")
            return np.array([]), np.array([])
        
        self.model.eval()
        
        loader = self.dataloaders[dataset]
        all_embeddings = []
        all_data = []
        total_samples = 0
        
        for batch in tqdm(loader, desc=f"Collecting embeddings ({dataset})"):
            # Move batch to device
            batch = batch.to(self.device)
            
            # Get latent representation
            mu, _ = self.model.encode(batch)
            
            # Store embeddings and data
            embeddings = mu.cpu().numpy()
            data = batch.cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_data.append(data)
            
            total_samples += embeddings.shape[0]
            if max_samples is not None and total_samples >= max_samples:
                break
        
        # Concatenate all embeddings and data
        all_embeddings = np.vstack(all_embeddings)
        all_data = np.vstack(all_data)
        
        return all_embeddings, all_data
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary with training history
        """
        logger.info("Starting training")
        start_time = time.time()
        
        try:
            # Training loop
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Train for one epoch
                train_loss, train_metrics = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # Validate if it's time
                if (epoch + 1) % self.validate_every == 0:
                    val_loss, val_metrics = self.validate()
                    self.val_losses.append(val_loss)
                    
                    # Check if best model
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # Save checkpoint
                    self._save_checkpoint(val_loss, is_best)
                    
                    # Early stopping check
                    if self.early_stopping and self.patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                else:
                    val_loss = None
                    val_metrics = {}
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Print epoch summary
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Time: {epoch_time:.2f}s")
                logger.info(f"  Train Loss: {train_loss:.6f} (Recon: {train_metrics['recon_loss']:.6f}, KL: {train_metrics['kl_loss']:.6f})")
                if val_loss is not None:
                    logger.info(f"  Val Loss: {val_loss:.6f} (Recon: {val_metrics['recon_loss']:.6f}, KL: {val_metrics['kl_loss']:.6f})")
                
                # Log to WandB - SIMPLIFIED
                if self.use_wandb and self.wandb_initialized:
                    # Only log essential metrics to WandB
                    metrics_to_log = {
                        'train/loss': train_loss,
                        'train/recon_loss': train_metrics['recon_loss'],
                        'train/kl_loss': train_metrics['kl_loss'],
                    }
                    
                    if val_loss is not None:
                        metrics_to_log.update({
                            'val/loss': val_loss,
                            'val/recon_loss': val_metrics['recon_loss'],
                            'val/kl_loss': val_metrics['kl_loss'],
                        })
                    
                    self._log_to_wandb(metrics_to_log, step=epoch)
            
            # Training complete
            total_time = time.time() - start_time
            logger.info(f"Training complete - Total time: {total_time:.2f}s")
            
            # Collect history
            history = {
                'train_loss': self.train_losses,
                'val_loss': self.val_losses
            }
            
            # Log the best model as an artifact
            if self.use_wandb and self.wandb_initialized and self.last_save_path:
                try:
                    import wandb
                    logger.info(f"Logging final model to WandB")
                    artifact = wandb.Artifact(
                        name="best-model",
                        type="model",
                        description=f"Best model with validation loss {self.best_val_loss:.6f}"
                    )
                    artifact.add_file(self.last_save_path)
                    wandb.log_artifact(artifact)
                except Exception as e:
                    logger.error(f"Error logging artifact to WandB: {e}")
            
            return history
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise e
        
        finally:
            # Always clean up WandB at the end
            self.cleanup_wandb()
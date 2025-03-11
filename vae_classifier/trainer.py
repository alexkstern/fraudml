import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from typing import Dict, List, Optional, Union, Tuple, Callable
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm, trange  # Added tqdm

try:
    import wandb
except ImportError:
    wandb = None

class VAETrainer:
    """
    Trainer class for the VAE Classifier.
    
    This class handles the training process, validation, early stopping,
    checkpoint saving, and metrics tracking.
    """
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        loss_fn: Callable,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The VAE Classifier model to train
            dataloaders: Dict with 'train', 'val', and 'test' dataloaders
            loss_fn: Loss function to use for training
            optimizer: Optimizer to use (if None, will create Adam)
            scheduler: Learning rate scheduler (optional)
            config: Configuration dictionary (optional)
            device: Device to train on (if None, will use CUDA if available)
        """
        self.model = model
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Load config if provided
        self.config = self._load_default_config()
        if config is not None:
            # Update default config with provided values
            self._update_config(config)
            
        # Set optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.get('learning_rate', 0.001)
            )
        else:
            self.optimizer = optimizer
            
        # Set scheduler
        self.scheduler = scheduler
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {
            'train': {'loss': [], 'recon_loss': [], 'kl_loss': [], 'class_loss': [], 
                      'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
            'val': {'loss': [], 'recon_loss': [], 'kl_loss': [], 'class_loss': [], 
                    'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
            'test': {'loss': [], 'recon_loss': [], 'kl_loss': [], 'class_loss': [], 
                     'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
        }
        
        # Early stopping variables
        self.patience_counter = 0
        
        # Initialize WandB if requested
        self._init_wandb()
        
    def _load_default_config(self) -> Dict:
        """Load default configuration for training."""
        return {
            'num_epochs': 50,
            'learning_rate': 0.001,
            'save_dir': 'saved_models/vae_classifier',
            'save_best_only': True,
            'early_stopping': True,
            'patience': 10,
            'validate_every': 1,
            'wandb': {
                'use_wandb': False,
                'project': 'vae-classifier',
                'name': 'vae-classifier-run',
                'entity': None,
                'tags': ['vae', 'classifier'],
            },
            'recon_weight': 1.0,
            'kl_weight': 1.0,
            'class_weight': 1.0,
        }
    
    def _update_config(self, config: Dict):
        """Update configuration with provided values."""
        # Update top-level configs
        for key, value in config.items():
            if key.lower() == 'vae' and isinstance(value, dict):
                # Extract specific VAE configs
                self.config['recon_weight'] = value.get('recon_weight', self.config.get('recon_weight'))
                self.config['kl_weight'] = value.get('kl_weight', self.config.get('kl_weight'))
            elif key.lower() == 'trainer' and isinstance(value, dict):
                # Update trainer configs
                for k, v in value.items():
                    self.config[k] = v
            elif key.lower() == 'wandb' and isinstance(value, dict):
                # Update WandB configs
                for k, v in value.items():
                    self.config['wandb'][k] = v
            else:
                # Set other configs directly
                self.config[key] = value
    
    def _init_wandb(self):
        """Initialize WandB if requested in config."""
        self.use_wandb = self.config['wandb']['use_wandb'] and wandb is not None
        
        if self.use_wandb:
            wandb.init(
                project=self.config['wandb']['project'],
                name=self.config['wandb']['name'],
                entity=self.config['wandb']['entity'],
                tags=self.config['wandb']['tags'],
                config=self.config
            )
            wandb.watch(self.model)
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        # Create save directory if it doesn't exist
        # If path is relative, make sure it's relative to the parent directory
        save_dir = self.config['save_dir']
        if not os.path.isabs(save_dir):
            # Check if we're in a subdirectory and need to go up one level
            if os.path.basename(os.getcwd()) in ['fraudml', 'vae_classifier']:
                save_dir = os.path.join('..', save_dir)
                
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics_history,
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(save_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint to {best_path}")
        
        # Save numbered checkpoint
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{self.current_epoch}.pt')
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        # Handle relative paths that might need to go up a directory
        if not os.path.isabs(checkpoint_path):
            # Check if we're in a subdirectory and need to go up one level
            if os.path.basename(os.getcwd()) in ['fraudml', 'vae_classifier']:
                checkpoint_path = os.path.join('..', checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load other information
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load metrics if available
        if 'metrics' in checkpoint:
            self.metrics_history = checkpoint['metrics']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
        return True
    
    def _process_batch(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        phase: str = 'train'
    ) -> Dict[str, float]:
        """
        Process a single batch of data.
        
        Args:
            batch: Tuple of (features, labels)
            phase: Current phase ('train', 'val', or 'test')
            
        Returns:
            Dictionary with batch metrics
        """
        features, labels = batch
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        reconstructed, mu, sigma, class_output = self.model(features)
        
        # Calculate loss components
        # Assuming the loss function signature is compatible with our VAE classifier
        loss = self.loss_fn(reconstructed, features, mu, sigma, labels, class_output)
        
        # Calculate individual loss components for logging
        recon_loss = torch.nn.functional.mse_loss(reconstructed, features, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        
        # Apply class weights to cross entropy
        class_weights = torch.tensor([1.0, 100.0], device=self.device)  # From the original model
        class_loss = torch.nn.functional.cross_entropy(class_output, labels, weight=class_weights)
        
        # Calculate classification metrics
        with torch.no_grad():
            # Get predictions
            _, predicted = torch.max(class_output, 1)
            predicted = predicted.cpu().numpy()
            true_labels = labels.cpu().numpy()
            
            # Calculate classification metrics
            accuracy = accuracy_score(true_labels, predicted)
            
            # Handle cases with only one class present
            if len(np.unique(true_labels)) == 1:
                precision = np.nan
                recall = np.nan
                f1 = np.nan
                auc = np.nan
            else:
                # Handle class imbalance with different averaging methods
                precision = precision_score(true_labels, predicted, zero_division=0)
                recall = recall_score(true_labels, predicted, zero_division=0)
                f1 = f1_score(true_labels, predicted, zero_division=0)
                
                # AUC requires probability scores
                prob_positive = torch.softmax(class_output, dim=1)[:, 1].cpu().numpy()
                auc = roc_auc_score(true_labels, prob_positive) if len(np.unique(true_labels)) > 1 else np.nan
        
        # Create metrics dictionary
        batch_metrics = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'class_loss': class_loss.item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        # Backward pass and optimization if in training phase
        if phase == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return batch_metrics
    
    def _run_epoch(self, phase: str = 'train') -> Dict[str, float]:
        """
        Run a single epoch.
        
        Args:
            phase: Current phase ('train', 'val', or 'test')
            
        Returns:
            Dictionary with epoch metrics
        """
        # Set model to appropriate mode
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        # Initialize metrics
        epoch_metrics = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'class_loss': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0
        }
        
        # Get dataloader
        dataloader = self.dataloaders[phase]
        
        # Track number of batches
        batch_count = 0
        
        # Initialize arrays for confusion matrix
        all_labels = []
        all_preds = []
        
        # Process batches with tqdm progress bar
        progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {self.current_epoch+1}")
        
        for batch in progress_bar:
            # Train or evaluate on batch
            with torch.set_grad_enabled(phase == 'train'):
                batch_metrics = self._process_batch(batch, phase)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                if not np.isnan(value):  # Skip NaN values when accumulating
                    epoch_metrics[key] += value
            
            # For confusion matrix
            features, labels = batch
            features = features.to(self.device)
            with torch.no_grad():
                _, _, _, class_output = self.model(features)
                _, predicted = torch.max(class_output, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
            
            batch_count += 1
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=f"{batch_metrics['loss']:.4f}")
        
        # Average metrics
        for key in epoch_metrics:
            if key not in ['confusion_matrix']:  # Don't average confusion matrix
                epoch_metrics[key] /= batch_count
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        epoch_metrics['confusion_matrix'] = conf_matrix
        
        # Log metrics
        print(f"{phase.capitalize()} Epoch {self.current_epoch+1}:")
        print(f"  Loss: {epoch_metrics['loss']:.4f}")
        print(f"  Recon Loss: {epoch_metrics['recon_loss']:.4f}")
        print(f"  KL Loss: {epoch_metrics['kl_loss']:.4f}")
        print(f"  Class Loss: {epoch_metrics['class_loss']:.4f}")
        print(f"  Accuracy: {epoch_metrics['accuracy']:.4f}")
        print(f"  Precision: {epoch_metrics['precision']:.4f}")
        print(f"  Recall: {epoch_metrics['recall']:.4f}")
        print(f"  F1: {epoch_metrics['f1']:.4f}")
        print(f"  AUC: {epoch_metrics['auc']:.4f}")
        print(f"  Confusion Matrix:\n{conf_matrix}")
        
        # Record metrics history (excluding confusion matrix for now)
        for key, value in epoch_metrics.items():
            if key != 'confusion_matrix':  # Don't store confusion matrix in history
                self.metrics_history[phase][key].append(value)
        
        # Log to WandB if enabled
        if self.use_wandb:
            wandb_log = {f"{phase}/{key}": value for key, value in epoch_metrics.items() 
                         if key != 'confusion_matrix'}
            # Log confusion matrix separately
            if phase in ['val', 'test']:
                wandb_log[f"{phase}/confusion_matrix"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=["Normal", "Fraud"]
                )
            
            wandb.log(wandb_log, step=self.current_epoch)
        
        return epoch_metrics
    
    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train (override config)
            
        Returns:
            Dictionary with training history
        """
        # Set number of epochs
        if num_epochs is not None:
            self.config['num_epochs'] = num_epochs
        
        # Initialize timer
        start_time = time.time()
        
        # Main training loop with tqdm progress bar
        epochs = trange(self.current_epoch, self.config['num_epochs'], desc="Training Progress")
        for epoch in epochs:
            self.current_epoch = epoch
            epochs.set_description(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train for one epoch
            train_metrics = self._run_epoch(phase='train')
            
            # Validate if needed
            if epoch % self.config['validate_every'] == 0:
                val_metrics = self._run_epoch(phase='val')
                
                # Check for improvement
                current_val_loss = val_metrics['loss']
                
                if current_val_loss < self.best_val_loss:
                    print(f"Validation loss improved from {self.best_val_loss:.4f} to {current_val_loss:.4f}")
                    self.best_val_loss = current_val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    
                    # Save best model
                    if self.config['save_best_only']:
                        self._save_checkpoint(is_best=True)
                    
                    # Reset patience counter
                    self.patience_counter = 0
                else:
                    print(f"Validation loss did not improve. Best: {self.best_val_loss:.4f}, Current: {current_val_loss:.4f}")
                    self.patience_counter += 1
                    
                    # Check for early stopping
                    if self.config['early_stopping'] and self.patience_counter >= self.config['patience']:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Step the scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint if not save_best_only
            if not self.config['save_best_only']:
                self._save_checkpoint()
                
            # Update progress bar with current metrics
            epochs.set_postfix(train_loss=f"{train_metrics['loss']:.4f}", 
                              val_loss=f"{val_metrics['loss']:.4f}" if 'val_metrics' in locals() else "N/A")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model state")
        
        # Evaluate on test set
        if 'test' in self.dataloaders:
            print("Evaluating on test set...")
            test_metrics = self._run_epoch(phase='test')
        
        # Return training history
        return self.metrics_history
    
    def evaluate(self, dataloader_key: str = 'test') -> Dict:
        """
        Evaluate the model on a specific dataset.
        
        Args:
            dataloader_key: Which dataloader to use ('train', 'val', 'test')
            
        Returns:
            Dictionary with evaluation metrics
        """
        if dataloader_key not in self.dataloaders:
            raise ValueError(f"Dataloader '{dataloader_key}' not found")
        
        print(f"Evaluating model on {dataloader_key} dataset...")
        metrics = self._run_epoch(phase=dataloader_key)
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot losses
        axs[0, 0].plot(self.metrics_history['train']['loss'], label='Train')
        axs[0, 0].plot(self.metrics_history['val']['loss'], label='Validation')
        axs[0, 0].set_title('Total Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        
        # Plot reconstruction loss
        axs[0, 1].plot(self.metrics_history['train']['recon_loss'], label='Train')
        axs[0, 1].plot(self.metrics_history['val']['recon_loss'], label='Validation')
        axs[0, 1].set_title('Reconstruction Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        
        # Plot KL loss
        axs[0, 2].plot(self.metrics_history['train']['kl_loss'], label='Train')
        axs[0, 2].plot(self.metrics_history['val']['kl_loss'], label='Validation')
        axs[0, 2].set_title('KL Divergence Loss')
        axs[0, 2].set_xlabel('Epoch')
        axs[0, 2].set_ylabel('Loss')
        axs[0, 2].legend()
        
        # Plot classification loss
        axs[1, 0].plot(self.metrics_history['train']['class_loss'], label='Train')
        axs[1, 0].plot(self.metrics_history['val']['class_loss'], label='Validation')
        axs[1, 0].set_title('Classification Loss')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        
        # Plot accuracy
        axs[1, 1].plot(self.metrics_history['train']['accuracy'], label='Train')
        axs[1, 1].plot(self.metrics_history['val']['accuracy'], label='Validation')
        axs[1, 1].set_title('Accuracy')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].legend()
        
        # Plot F1 score
        axs[1, 2].plot(self.metrics_history['train']['f1'], label='Train')
        axs[1, 2].plot(self.metrics_history['val']['f1'], label='Validation')
        axs[1, 2].set_title('F1 Score')
        axs[1, 2].set_xlabel('Epoch')
        axs[1, 2].set_ylabel('F1')
        axs[1, 2].legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            print(f"Saved training history plot to {save_path}")
        else:
            plt.show()
    
    def generate_anomaly_scores(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate anomaly scores for a dataset.
        
        Args:
            dataloader: DataLoader with the dataset
            
        Returns:
            Tuple of (anomaly_scores, labels, predictions)
        """
        self.model.eval()
        anomaly_scores = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch
                features = features.to(self.device)
                
                # Forward pass
                reconstructed, mu, sigma, class_output = self.model(features)
                
                # Calculate reconstruction error for each sample
                recon_errors = torch.sum((reconstructed - features) ** 2, dim=1).cpu().numpy()
                
                # Get class predictions
                _, predicted = torch.max(class_output, 1)
                
                # Store results
                anomaly_scores.extend(recon_errors)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        return np.array(anomaly_scores), np.array(all_labels), np.array(all_preds)
    
    def plot_latent_space(self, dataloader: DataLoader, save_path: Optional[str] = None):
        """
        Plot the latent space for visualization.
        
        Args:
            dataloader: DataLoader with the dataset
            save_path: Path to save the plot (optional)
        """
        self.model.eval()
        
        # Get latent representations
        z_list = []
        label_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch
                features = features.to(self.device)
                
                # Get latent space encoding
                mu, sigma = self.model.encode(features)
                z = self.model.reparametrize(mu, sigma)
                
                z_list.append(z.cpu().numpy())
                label_list.append(labels.cpu().numpy())
        
        # Combine all batches
        z_all = np.vstack(z_list)
        labels_all = np.concatenate(label_list)
        
        # For visualization, use t-SNE if dimensionality > 2
        if z_all.shape[1] > 2:
            try:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42)
                z_2d = tsne.fit_transform(z_all)
            except ImportError:
                print("scikit-learn not available for t-SNE. Using first 2 dimensions.")
                z_2d = z_all[:, :2]
        else:
            z_2d = z_all
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_all, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label='Class')
        plt.title('Latent Space Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            print(f"Saved latent space plot to {save_path}")
        else:
            plt.show()


def train_vae_classifier(
    model, 
    dataloaders, 
    loss_fn, 
    config=None, 
    optimizer=None, 
    scheduler=None,
    device=None
):
    """
    Convenience function to train a VAE Classifier.
    
    Args:
        model: The VAE Classifier model
        dataloaders: Dictionary with train, val, and test dataloaders
        loss_fn: Loss function for training
        config: Configuration dictionary (optional)
        optimizer: Optimizer to use (optional, default: Adam)
        scheduler: Learning rate scheduler (optional)
        device: Device to train on (optional)
        
    Returns:
        Trained model and trainer object
    """
    # Create trainer
    trainer = VAETrainer(
        model=model,
        dataloaders=dataloaders,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    # Train the model
    history = trainer.train()
    
    # Evaluate on test set
    if 'test' in dataloaders:
        test_metrics = trainer.evaluate('test')
        print("\nTest Metrics:")
        for key, value in test_metrics.items():
            if key != 'confusion_matrix':
                print(f"  {key}: {value:.4f}")
        print(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    return model, trainer


def load_trained_vae(checkpoint_path, model, device=None):
    """
    Load a trained VAE Classifier from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Uninitialized model of the same architecture
        device: Device to load the model onto (optional)
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle relative paths that might need to go up a directory
    if not os.path.isabs(checkpoint_path):
        # Check if we're in a subdirectory and need to go up one level
        if os.path.basename(os.getcwd()) in ['fraudml', 'vae_classifier']:
            checkpoint_path = os.path.join('..', checkpoint_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    return model



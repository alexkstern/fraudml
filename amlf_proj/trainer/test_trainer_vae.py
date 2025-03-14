import torch
from tqdm import tqdm

class VAETrainer:
    def __init__(self, model, dataloaders, loss_function, optimizer, scheduler=None, device=None):
        """
        Args:
            model: The VAE model instance.
            dataloaders: Dictionary with keys 'train' and 'val' holding DataLoader objects.
            loss_function: Function that computes loss as loss = loss_fn(recon, x, mu, logvar).
            optimizer: Optimizer for model parameters.
            scheduler: Learning rate scheduler (optional).
            device: Device string ('cuda' or 'cpu'). If None, it auto-detects.
        """
        self.model = model
        self.dataloaders = dataloaders
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.dataloaders['train'], desc="Training VAE", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(batch)
            loss = self.loss_function(recon, batch, mu, logvar)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
        return epoch_loss

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.dataloaders['val'], desc="Validating VAE", leave=False)
            for batch in pbar:
                batch = batch.to(self.device)
                recon, mu, logvar = self.model(batch)
                loss = self.loss_function(recon, batch, mu, logvar)
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(self.dataloaders['val'].dataset)
        return epoch_loss

    def train(self, num_epochs, print_every=1):
        train_losses = []
        val_losses = []
        
        # Track best validation loss for early stopping and model saving
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Step the scheduler if it exists
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.1e}")
            
            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
            else:
                is_best = False
                
            # Additional logic for early stopping could be added here
                
        return train_losses, val_losses
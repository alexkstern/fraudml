import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def create_scheduler(config, optimizer):
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        config: Configuration dictionary or section with scheduler parameters
        optimizer: PyTorch optimizer to schedule
        
    Returns:
        PyTorch scheduler object
    """
    # Extract scheduler parameters from config with defaults
    use_scheduler = config.getboolean('use_scheduler', fallback=True)
    if not use_scheduler:
        return None
    
    scheduler_type = config.get('scheduler_type', fallback='plateau')
    
    if scheduler_type.lower() == 'plateau':
        patience = config.getint('scheduler_patience', fallback=5)
        factor = config.getfloat('scheduler_factor', fallback=0.5)
        min_lr = config.getfloat('scheduler_min_lr', fallback=1e-6)
        
        return ReduceLROnPlateau(
            optimizer, 
            mode='min',               # We're monitoring a loss, so lower is better
            factor=factor,            # Multiply LR by this factor on plateau
            patience=patience,        # Number of epochs with no improvement after which LR will be reduced
            verbose=True,             # Print message when LR is reduced
            min_lr=min_lr             # Lower bound on the learning rate
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
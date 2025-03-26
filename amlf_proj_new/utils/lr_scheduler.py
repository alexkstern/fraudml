import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, MultiStepLR, ExponentialLR

def create_scheduler(config, optimizer):
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        config: Configuration dictionary or section with scheduler parameters
        optimizer: PyTorch optimizer to schedule
        
    Returns:
        PyTorch scheduler object or None if scheduler is disabled
    """
    # Extract scheduler parameters from config with defaults
    use_scheduler = config.getboolean('use_scheduler', fallback=True)
    if not use_scheduler:
        return None
    
    scheduler_type = config.get('scheduler_type', fallback='plateau').lower()
    
    if scheduler_type == 'plateau':
        # Defaults: patience=5, factor=0.5, min_lr=1e-6
        patience = config.getint('scheduler_patience', fallback=10)
        factor = config.getfloat('scheduler_factor', fallback=0.8)
        min_lr = config.getfloat('scheduler_min_lr', fallback=1e-6)
        return ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=factor,
            patience=patience,
            verbose=True,
            min_lr=min_lr
        )
    elif scheduler_type == 'step':
        # Defaults: step_size=30, gamma=0.1
        step_size = config.getint('scheduler_step_size', fallback=20)
        gamma = config.getfloat('scheduler_gamma', fallback=0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == 'cosine':
        # CosineAnnealingLR: Defaults T_max=50, eta_min=0
        T_max = config.getint('scheduler_T_max', fallback=25)
        eta_min = config.getfloat('scheduler_eta_min', fallback=0.0002)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == 'cosinewarm':
        # CosineAnnealingWarmRestarts: Defaults T_0=10, T_mult=1
        T_0 = config.getint('scheduler_T_0', fallback=10)
        T_mult = config.getint('scheduler_T_mult', fallback=1)
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    elif scheduler_type == 'onecycle':
        # OneCycleLR: These defaults work for many cases; max_lr should be provided by the optimizer or config
        max_lr = config.getfloat('scheduler_max_lr', fallback=0.01)
        total_steps = config.getint('scheduler_total_steps', fallback=100)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr, 
            total_steps=total_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

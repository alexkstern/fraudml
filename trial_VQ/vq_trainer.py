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
    total_class_loss = 0.0  # Track classification loss
    batch_count = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(train_loader, desc=f"Training", leave=False)
    
    for batch_data in progress_bar:
        # Handle the case where the dataset returns features and labels
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            batch, labels = batch_data
            batch = batch.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with labels
            outputs = model(batch, labels)
        else:
            # Original case - just features, no labels
            batch = batch_data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass without labels
            outputs = model(batch)
        
        # Get losses
        loss = outputs['loss']
        recon_loss = outputs['reconstruction_loss']
        vq_loss = outputs['vq_loss']
        perplexity = outputs['perplexity']
        
        # Get classification loss if available
        class_loss = outputs.get('classification_loss', torch.tensor(0.0, device=device))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        if 'classification_loss' in outputs:
            total_class_loss += class_loss.item()
        batch_count += 1
        
        # Update progress bar
        progress_dict = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            'perplexity': perplexity.item()
        }
        if 'classification_loss' in outputs:
            progress_dict['class_loss'] = class_loss.item()
        progress_bar.set_postfix(progress_dict)
    
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_recon_loss = total_recon_loss / batch_count
    avg_vq_loss = total_vq_loss / batch_count
    avg_perplexity = total_perplexity / batch_count
    
    # Prepare metrics dictionary
    metrics = {
        'loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'vq_loss': avg_vq_loss,
        'perplexity': avg_perplexity
    }
    
    # Add classification metrics if available
    if total_class_loss > 0:
        metrics['classification_loss'] = total_class_loss / batch_count
    
    return metrics


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
    total_class_loss = 0.0  # Track classification loss
    batch_count = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(val_loader, desc=f"Validating", leave=False)
    
    with torch.no_grad():
        for batch_data in progress_bar:
            # Handle the case where the dataset returns features and labels
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                batch, labels = batch_data
                batch = batch.to(device)
                labels = labels.to(device)
                
                # Forward pass with labels
                outputs = model(batch, labels)
            else:
                # Original case - just features, no labels
                batch = batch_data.to(device)
                
                # Forward pass without labels
                outputs = model(batch)
            
            # Get losses
            loss = outputs['loss']
            recon_loss = outputs['reconstruction_loss']
            vq_loss = outputs['vq_loss']
            perplexity = outputs['perplexity']
            
            # Get classification loss if available
            class_loss = outputs.get('classification_loss', torch.tensor(0.0, device=device))
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            if 'classification_loss' in outputs:
                total_class_loss += class_loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_dict = {
                'val_loss': loss.item(),
                'val_recon_loss': recon_loss.item(),
                'val_vq_loss': vq_loss.item(),
                'val_perplexity': perplexity.item()
            }
            if 'classification_loss' in outputs:
                progress_dict['val_class_loss'] = class_loss.item()
            progress_bar.set_postfix(progress_dict)
    
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_recon_loss = total_recon_loss / batch_count
    avg_vq_loss = total_vq_loss / batch_count
    avg_perplexity = total_perplexity / batch_count
    
    # Prepare metrics dictionary
    metrics = {
        'val_loss': avg_loss,
        'val_reconstruction_loss': avg_recon_loss,
        'val_vq_loss': avg_vq_loss,
        'val_perplexity': avg_perplexity
    }
    
    # Add classification metrics if available
    if total_class_loss > 0:
        metrics['val_classification_loss'] = total_class_loss / batch_count
    
    return metrics


def calculate_anomaly_scores(model: VQVAE, 
                              data_loader: torch.utils.data.DataLoader, 
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Calculate anomaly scores for all samples in the dataset.
    
    Args:
        model: The VQVAE model
        data_loader: DataLoader for the dataset
        device: Device to use
        
    Returns:
        Tuple of (anomaly_scores, encodings, labels)
    """
    model.eval()
    all_scores = []
    all_encodings = []
    all_labels = []
    has_labels = False
    
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Calculating anomaly scores"):
            # Handle the case where the dataset returns features and labels
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                batch, labels = batch_data
                batch = batch.to(device)
                all_labels.append(labels.cpu())
                has_labels = True
            else:
                # Original case - just features
                batch = batch_data.to(device)
            
            # Calculate anomaly scores
            scores = model.calculate_anomaly_scores(batch)
            encodings = model.encode_to_indices(batch)
            
            # Store scores and encodings
            all_scores.append(scores.cpu())
            all_encodings.append(encodings.cpu())
    
    # Concatenate all scores and encodings
    all_scores = torch.cat(all_scores)
    all_encodings = torch.cat(all_encodings)
    
    # Concatenate labels if available
    if has_labels:
        all_labels = torch.cat(all_labels)
        return all_scores, all_encodings, all_labels
    else:
        return all_scores, all_encodings, None





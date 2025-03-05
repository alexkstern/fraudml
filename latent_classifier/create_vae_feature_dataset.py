import torch
import numpy as np
from dataloader import load_fraud_data, load_config
from model_VAE import VariationalAutoEncoder
from tqdm import tqdm
import os

def load_vae_model(checkpoint_path, config):
    """
    Load a VAE model from a checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config: Configuration dictionary with VAE section
        
    Returns:
        Loaded model
    """
    print(f"Loading model from: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = VariationalAutoEncoder(config['VAE'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def process_dataset(model, dataloader, device, prefix=""):
    """
    Get latent vectors and reconstruction errors from a VAE
    
    Args:
        model: VAE model
        dataloader: DataLoader with data to encode
        device: Device to run inference on
        prefix: Prefix for feature names (e.g., "fraud_" or "non_fraud_")
        
    Returns:
        Dictionary with latent vectors and reconstruction errors
    """
    all_latent_vectors = []
    all_recon_errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {prefix} features"):
            # Get original batch
            batch = batch.to(device)
            
            # Get latent vectors and reconstructions
            mu, _ = model.encode(batch)
            reconstructed = model.decode(mu)
            
            # Calculate reconstruction error per sample (element-wise MSE)
            recon_error = torch.mean((reconstructed - batch)**2, dim=1, keepdim=True)
            
            # Store data
            all_latent_vectors.append(mu.detach().cpu().numpy())
            all_recon_errors.append(recon_error.detach().cpu().numpy())
    
    # Concatenate all batches
    latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    recon_errors = np.concatenate(all_recon_errors, axis=0)
    
    # Double-check shapes
    print(f"{prefix} latent vectors shape: {latent_vectors.shape}")
    print(f"{prefix} recon errors shape: {recon_errors.shape}")
    
    # Feature names
    latent_dim = latent_vectors.shape[1]
    latent_feature_names = [f"{prefix}latent_{i}" for i in range(latent_dim)]
    recon_feature_name = [f"{prefix}recon_error"]
    
    return {
        "latent_vectors": latent_vectors,
        "recon_errors": recon_errors,
        "latent_feature_names": latent_feature_names,
        "recon_feature_name": recon_feature_name
    }

def main():
    # Configuration paths
    config_path = 'configs/classifier/classifier.config'
    fraud_model_path = 'saved_models/fraud/v1/VariationalAutoEncoder_fraud_20250305-161711_epoch36_val61.1807.pt'
    non_fraud_model_path = 'saved_models/non_fraud/v1/VariationalAutoEncoder_non-fraud_20250305-163324_epoch5_val18.5573.pt'
    output_dir = 'vae_dataset'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Load raw datasets (without class filtering)
    print("Loading raw datasets (without filtering)...")
    data = load_fraud_data(config_path=config_path, class_filter=None)
    
    # Load both models
    fraud_model = load_vae_model(fraud_model_path, config)
    non_fraud_model = load_vae_model(non_fraud_model_path, config)
    
    # Process all splits (train, val, test)
    for split, dataloader in data['dataloaders'].items():
        print(f"\nProcessing {split} split...")
        
        # Get the dataset for this split to extract labels and original data
        dataset = dataloader.dataset.dataset
        feature_cols = dataloader.dataset.feature_cols
        
        # Collect all samples individually to ensure consistency
        all_samples = []
        all_labels = []
        
        # Extract samples and labels directly from the dataset
        print(f"Extracting original data and labels from {split} dataset...")
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            features = []
            for col in feature_cols:
                features.append(sample[col])
            
            all_samples.append(features)
            
            # Get label (assuming 'Class' is the label column)
            if 'Class' in sample:
                all_labels.append(sample['Class'])
        
        # Convert to numpy arrays
        original_data = np.array(all_samples)
        if all_labels:
            labels = np.array(all_labels)
            
        print(f"Original data shape: {original_data.shape}")
        if all_labels:
            print(f"Labels shape: {labels.shape}")
        
        # Process through VAEs
        fraud_features = process_dataset(fraud_model, dataloader, device, prefix="fraud_")
        non_fraud_features = process_dataset(non_fraud_model, dataloader, device, prefix="non_fraud_")
        
        # Combine features
        combined_features = np.hstack([
            fraud_features['latent_vectors'],
            fraud_features['recon_errors'],
            non_fraud_features['latent_vectors'],
            non_fraud_features['recon_errors']
        ])
        
        print(f"Combined features shape: {combined_features.shape}")
        
        # Combined feature names
        all_feature_names = (
            fraud_features['latent_feature_names'] + 
            fraud_features['recon_feature_name'] +
            non_fraud_features['latent_feature_names'] + 
            non_fraud_features['recon_feature_name']
        )
        
        # Save the combined features, feature names, original data and labels
        np.save(os.path.join(output_dir, f'{split}_combined_features.npy'), combined_features)
        np.save(os.path.join(output_dir, f'{split}_feature_names.npy'), np.array(all_feature_names))
        np.save(os.path.join(output_dir, f'{split}_original_data.npy'), original_data)
        if all_labels:
            np.save(os.path.join(output_dir, f'{split}_labels.npy'), labels)
        
        print(f"Saved {split} features to {output_dir} directory")
        
        # Save a metadata file with descriptions
        with open(os.path.join(output_dir, f'{split}_metadata.txt'), 'w') as f:
            f.write(f"Combined features shape: {combined_features.shape}\n")
            f.write(f"Feature names: {all_feature_names}\n")
            if all_labels:
                f.write(f"Labels shape: {labels.shape}\n")
            f.write(f"Original data shape: {original_data.shape}\n")
            f.write("\nFeature descriptions:\n")
            f.write("- fraud_latent_*: Latent vectors from the fraud VAE\n")
            f.write("- fraud_recon_error: Reconstruction error from the fraud VAE\n")
            f.write("- non_fraud_latent_*: Latent vectors from the non-fraud VAE\n")
            f.write("- non_fraud_recon_error: Reconstruction error from the non-fraud VAE\n")
    
    print("\nAll features have been created and saved to the 'vae_dataset' directory!")
    print("\nNext steps: You can now use these features for downstream classification tasks.")

if __name__ == "__main__":
    main()
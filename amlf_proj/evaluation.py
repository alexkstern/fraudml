import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
from dataloader.dataloader_classifier import load_fraud_classification_data, load_config
from models.conv_vae_model import ConvVae
from models.transformer_vae_model import TransformerVae
from models.conv_vqvae_model import ConvVQVAE
from models.transformer_vqvae_model import TransformerVQVAE
import configparser

def load_model(model_path, model_class, config):
    """
    Load a saved model from a checkpoint.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        model_class: The model class (ConvVae, TransformerVae, etc.)
        config: The model configuration section
        
    Returns:
        The loaded model
    """
    model = model_class(config)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def calculate_reconstruction_error(model, data_batch, device, model_type='vae'):
    """
    Calculate the reconstruction error for a batch of data.
    
    Args:
        model: The model to evaluate
        data_batch: A batch of data (features)
        device: The device to run evaluation on
        model_type: Type of model ('vae' or 'vqvae')
        
    Returns:
        Reconstruction error for each sample in the batch
    """
    with torch.no_grad():
        data_batch = data_batch.to(device)
        
        if model_type.lower() == 'vae':
            # For VAE models
            recon, mu, logvar = model(data_batch)
            # Calculate per-sample MSE
            recon_error = ((recon - data_batch) ** 2).mean(dim=(1, 2))
        else:  # vqvae
            # For VQVAE models
            recon, _ = model(data_batch)
            # Calculate per-sample MSE
            recon_error = ((recon - data_batch) ** 2).mean(dim=(1, 2))
            
    return recon_error.cpu().numpy()

def evaluate_models(fraud_model_path, normal_model_path, model_class, config, dataloader, device=None, model_type='vae'):
    """
    Evaluate fraud and normal models on test data.
    
    Args:
        fraud_model_path: Path to the fraud model checkpoint
        normal_model_path: Path to the normal model checkpoint
        model_class: The model class
        config: The model configuration section
        dataloader: Test dataloader
        device: Device to run evaluation on
        model_type: Type of model ('vae' or 'vqvae')
        
    Returns:
        Dictionary of evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print(f"Loading fraud model from {fraud_model_path}")
    fraud_model = load_model(fraud_model_path, model_class, config)
    
    print(f"Loading normal model from {normal_model_path}")
    normal_model = load_model(normal_model_path, model_class, config)
    
    fraud_model.to(device)
    normal_model.to(device)
    
    # Initialize arrays to store results
    all_fraud_errors = []
    all_normal_errors = []
    all_true_labels = []
    
    # Process test set
    print("Evaluating models on test data...")
    for batch_features, batch_labels in dataloader:
        # Calculate reconstruction errors
        fraud_error = calculate_reconstruction_error(fraud_model, batch_features, device, model_type)
        normal_error = calculate_reconstruction_error(normal_model, batch_features, device, model_type)
        
        all_fraud_errors.extend(fraud_error)
        all_normal_errors.extend(normal_error)
        all_true_labels.extend(batch_labels.numpy())
    
    # Convert to numpy arrays
    fraud_errors = np.array(all_fraud_errors)
    normal_errors = np.array(all_normal_errors)
    true_labels = np.array(all_true_labels)
    
    # Calculate error ratio (fraud_error / normal_error)
    error_ratios = fraud_errors / normal_errors
    
    return {
        'fraud_errors': fraud_errors,
        'normal_errors': normal_errors,
        'error_ratios': error_ratios,
        'true_labels': true_labels
    }

def classify_samples(error_ratios, threshold=1.0):
    """
    Classify samples based on error ratios.
    
    Args:
        error_ratios: Array of error ratios (fraud_error / normal_error)
        threshold: Threshold for classification (default: 1.0)
        
    Returns:
        Array of predicted labels (0 for normal, 1 for fraud)
    """
    # If ratio < threshold, classify as fraud (1), else as normal (0)
    predictions = (error_ratios < threshold).astype(int)
    return predictions

def compute_metrics(true_labels, predictions):
    """
    Compute classification metrics.
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted labels
        
    Returns:
        Dictionary of metrics
    """
    conf_matrix = confusion_matrix(true_labels, predictions)
    cls_report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    return {
        'confusion_matrix': conf_matrix,
        'classification_report': cls_report,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_reconstruction_errors(fraud_errors, normal_errors, true_labels, save_path=None):
    """
    Create a scatter plot of reconstruction errors colored by true class.
    
    Args:
        fraud_errors: Array of fraud model reconstruction errors
        normal_errors: Array of normal model reconstruction errors
        true_labels: Array of true labels
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with colors based on true labels
    scatter = plt.scatter(
        normal_errors, 
        fraud_errors, 
        c=true_labels, 
        cmap='coolwarm', 
        alpha=0.7
    )
    
    # Add color bar and labels
    cbar = plt.colorbar(scatter)
    cbar.set_label('True Class (0=Normal, 1=Fraud)')
    
    plt.xlabel('Normal Model Reconstruction Error')
    plt.ylabel('Fraud Model Reconstruction Error')
    plt.title('Reconstruction Errors by Model Type')
    
    # Add decision boundary line (y = x)
    max_error = max(normal_errors.max(), fraud_errors.max())
    min_error = min(normal_errors.min(), fraud_errors.min())
    plt.plot([min_error, max_error], [min_error, max_error], 'k--', label='Decision Boundary')
    plt.legend()
    
    # Add grid and make plot look nicer
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(conf_matrix, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: The confusion matrix
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Normal', 'Fraud'],
        yticklabels=['Normal', 'Fraud']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def find_best_threshold(error_ratios, true_labels, thresholds=None):
    """
    Find the best classification threshold by maximizing F1 score.
    
    Args:
        error_ratios: Array of error ratios
        true_labels: Array of true labels
        thresholds: List of thresholds to try (default: 100 values from 0.1 to 2.0)
        
    Returns:
        best_threshold, best_f1_score
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 2.0, 100)
    
    best_threshold = 1.0
    best_f1 = 0.0
    
    for threshold in thresholds:
        predictions = classify_samples(error_ratios, threshold)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

"""def main():
    # Configuration
    config_path = "configs/classifier/classifier.config"
    
    # Model paths - REPLACE THESE WITH YOUR ACTUAL MODEL PATHS
    # For Conv VAE
    fraud_model_path = "saved_models/conv_vae/fraud_conv_vae/20250313_173415/best_model.pt"
    normal_model_path = "saved_models/conv_vae/normal_conv_vae/20250313_175713/best_model.pt"
    
    model_config="configs/conv_vae/fraud_conv_vae.config"
    model_normal_config="configs/conv_vae/normal_conv_vae.config"
    # You can add paths for other model types as needed
    # transformer_fraud_model_path = "saved_models/transformer_vae/fraud_transformer_vae/YYYYMMDD_HHMMSS/best_model.pt"
    # transformer_normal_model_path = "saved_models/transformer_vae/normal_transformer_vae/YYYYMMDD_HHMMSS/best_model.pt"
    
    # Load configuration
    config_dict = load_config(model_config)
    
    # Get the specific model config section
    vae_config = config_dict.get("Conv_VAE", {})
    
    # Load classification data that includes labels
    data = load_fraud_classification_data(config_path=config_path)
    test_dataloader = data['dataloaders']['test']
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dictionary to store results for each model type
    all_results = {}
    
    # Evaluate Conv VAE
    print("\n--- Evaluating Conv VAE models ---")
    conv_vae_results = evaluate_models(
        fraud_model_path,
        normal_model_path,
        ConvVae,
        vae_config,
        test_dataloader,
        device,
        model_type='vae'
    )"""
def main():
    # Configuration
    config_path = "configs/classifier/classifier.config"
    
    # Model paths - REPLACE THESE WITH YOUR ACTUAL MODEL PATHS
    # For Conv VAE
    fraud_model_path = "saved_models/conv_vae/fraud_conv_vae/20250313_173415/best_model.pt"
    normal_model_path = "saved_models/conv_vae/normal_conv_vae/20250313_175713/best_model.pt"
    
    model_config = "configs/conv_vae/fraud_conv_vae.config"
    
    # Load classification data that includes labels
    data = load_fraud_classification_data(config_path=config_path)
    test_dataloader = data['dataloaders']['test']
    
    # Load configuration for the model
    config_parser = configparser.ConfigParser()
    config_parser.read(model_config)
    vae_config = config_parser["Conv_VAE"]
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dictionary to store results for each model type
    all_results = {}
    
    # Evaluate Conv VAE
    print("\n--- Evaluating Conv VAE models ---")
    conv_vae_results = evaluate_models(
        fraud_model_path,
        normal_model_path,
        ConvVae,
        vae_config,
        test_dataloader,
        device,
        model_type='vae'
    )
    # ... rest of the function ...
    all_results['conv_vae'] = conv_vae_results
    
    # Find the best threshold
    best_threshold, best_f1 = find_best_threshold(
        conv_vae_results['error_ratios'],
        conv_vae_results['true_labels']
    )
    print(f"Best threshold: {best_threshold:.4f} (F1 Score: {best_f1:.4f})")
    
    # Make predictions using the best threshold
    predictions = classify_samples(conv_vae_results['error_ratios'], threshold=best_threshold)
    
    # Compute metrics
    metrics = compute_metrics(conv_vae_results['true_labels'], predictions)
    
    # Print metrics
    print("\nClassification Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])


if __name__ == "__main__":
    main()


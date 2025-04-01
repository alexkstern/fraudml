import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter errors

import os
import torch
import numpy as np
import configparser
from tqdm import tqdm
from dataloader.dataloader_classifier import load_fraud_classification_data
from models.conv_vae_model import ConvVae
from models.transformer_vae_model import TransformerVae
from models.conv_vqvae_model import ConvVQVAE
from models.transformer_vqvae_model import TransformerVQVAE

def load_model(model_type, config_path, model_path, device):
    """Load a model with its config and weights"""
    model_mapping = {
        'conv_vae': {'class': ConvVae, 'section': 'Conv_VAE', 'loss_type': 'vae'},
        'transformer_vae': {'class': TransformerVae, 'section': 'Transformer_VAE', 'loss_type': 'vae'},
        'conv_vqvae': {'class': ConvVQVAE, 'section': 'Conv_VQVAE', 'loss_type': 'vqvae'},
        'transformer_vqvae': {'class': TransformerVQVAE, 'section': 'Transformer_VQVAE', 'loss_type': 'vqvae'}
    }
    
    model_info = model_mapping[model_type]
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    model_config = config_parser[model_info['section']]
    model = model_info['class'](model_config)
    
    try:
        # Try with weights_only=True first
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        # Fall back to loading without weights_only
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model, model_info['loss_type']

def calculate_reconstruction_errors(model, dataloader, loss_type, device):
    """Calculate reconstruction errors for all samples in dataloader"""
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in tqdm(dataloader, desc="Calculating errors"):
            batch_features = batch_features.to(device)
            
            try:
                if loss_type == 'vae':
                    recon, _, _ = model(batch_features)
                else:  # vqvae
                    recon, _ = model(batch_features)
                
                # Per-sample MSE
                recon_error = ((recon - batch_features) ** 2).mean(dim=tuple(range(1, recon.ndim)))
                all_errors.extend(recon_error.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
            except Exception as e:
                print(f"Error during forward pass: {e}")
                continue
    
    return np.array(all_errors), np.array(all_labels)

def calculate_error_statistics(errors, labels):
    """Calculate statistics for the errors"""
    normal_errors = errors[labels == 0]
    fraud_errors = errors[labels == 1]
    
    # Helper function to calculate statistics safely
    def get_stats(data):
        if len(data) == 0:
            return {
                'count': 0, 'min': None, 'max': None, 'mean': None, 'median': None,
                'std': None, 'q1': None, 'q3': None, 'iqr': None
            }
        return {
            'count': len(data),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25))
        }
    
    return {
        'all': get_stats(errors),
        'normal': get_stats(normal_errors),
        'fraud': get_stats(fraud_errors)
    }

def main():
    # Create output directory
    output_dir = "recon_error_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Open a single text file for all results
    results_file_path = os.path.join(output_dir, "reconstruction_error_statistics.txt")
    results_file = open(results_file_path, 'w')
    
    # Header
    header = "RECONSTRUCTION ERROR STATISTICS ACROSS MODELS\n"
    header += "=" * 80 + "\n\n"
    results_file.write(header)
    print(header)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_line = f"Using device: {device}\n"
    results_file.write(log_line)
    print(log_line)
    
    # Load classification data (which contains both normal and fraud samples)
    classifier_config_path = "configs/classifier/classifier.config"
    log_line = f"Loading evaluation data using config: {classifier_config_path}\n\n"
    results_file.write(log_line)
    print(log_line)
    data = load_fraud_classification_data(config_path=classifier_config_path)
    train_dataloader = data['dataloaders']['train']
    
    # Define model configurations
    model_configs = [
        {
            'type': 'conv_vae',
            'class_type': 'normal',
            'model_path': "./saved_models/conv_vae/normal_conv_vae/20250327_175631/best_model.pt",
            'config': "configs/conv_vae/normal_conv_vae.config",
        },
        {
            'type': 'conv_vae',
            'class_type': 'fraud',
            'model_path': "./saved_models/conv_vae/fraud_conv_vae/20250327_175451/best_model.pt", 
            'config': "configs/conv_vae/fraud_conv_vae.config",
        },
        {
            'type': 'transformer_vae',
            'class_type': 'normal',
            'model_path': "./saved_models/transformer_vae/normal_transformer_vae/20250327_180851/best_model.pt",
            'config': "configs/transformer_vae/normal_transformer_vae.config",
        },
        {
            'type': 'transformer_vae',
            'class_type': 'fraud',
            'model_path': "./saved_models/transformer_vae/fraud_transformer_vae/20250327_180740/best_model.pt",
            'config': "configs/transformer_vae/fraud_transformer_vae.config",
        },
        {
            'type': 'conv_vqvae',
            'class_type': 'normal',
            'model_path': "./saved_models/conv_vqvae/normal_conv_vqvae/20250327_181723/best_model.pt",
            'config': "configs/conv_vqvae/normal_conv_vqvae.config",
        },
        {
            'type': 'conv_vqvae',
            'class_type': 'fraud',
            'model_path': "./saved_models/conv_vqvae/fraud_conv_vqvae/20250327_181652/best_model.pt",
            'config': "configs/conv_vqvae/fraud_conv_vqvae.config",
        },
        {
            'type': 'transformer_vqvae',
            'class_type': 'normal',
            'model_path': "./saved_models/transformer_vqvae/normal_transformer_vqvae/20250327_182930/best_model.pt",
            'config': "configs/transformer_vqvae/normal_transformer_vqvae.config",
        },
        {
            'type': 'transformer_vqvae',
            'class_type': 'fraud',
            'model_path': "./saved_models/transformer_vqvae/fraud_transformer_vqvae/20250327_182749/best_model.pt",
            'config': "configs/transformer_vqvae/fraud_transformer_vqvae.config",
        }
    ]
    
    # Process each model
    for model_config in model_configs:
        model_type = model_config['type']
        class_type = model_config['class_type']
        
        section_header = f"\n{'-'*40}\n"
        section_header += f"MODEL: {model_type.upper()} - CLASS: {class_type.upper()}\n"
        section_header += f"{'-'*40}\n"
        results_file.write(section_header)
        print(section_header)
        
        # Load model
        model, loss_type = load_model(
            model_config['type'], 
            model_config['config'], 
            model_config['model_path'], 
            device
        )
        
        # Calculate errors
        log_line = "Calculating reconstruction errors...\n"
        results_file.write(log_line)
        print(log_line)
        errors, labels = calculate_reconstruction_errors(model, train_dataloader, loss_type, device)
        
        # Calculate statistics
        stats = calculate_error_statistics(errors, labels)
        
        # Save raw errors for potential future analysis
        np.savez(
            os.path.join(output_dir, f"{model_type}_{class_type}_errors.npz"),
            errors=errors,
            labels=labels
        )
        
        # Report statistics
        for subset_name, subset_stats in stats.items():
            subset_header = f"\n{subset_name.upper()} SAMPLES (Count: {subset_stats['count']})\n"
            results_file.write(subset_header)
            print(subset_header)
            
            stats_text = ""
            for stat_name, stat_value in subset_stats.items():
                if stat_name != 'count':  # Already printed in header
                    # Format the value with appropriate precision
                    if stat_value is not None:
                        if stat_name in ['min', 'max', 'mean', 'median']:
                            formatted_value = f"{stat_value:.6f}"
                        else:
                            formatted_value = f"{stat_value:.6f}"
                    else:
                        formatted_value = "N/A"
                    
                    stats_text += f"  {stat_name.upper()}: {formatted_value}\n"
            
            results_file.write(stats_text)
            print(stats_text)
        
        # Calculate ratio of fraud to normal (if possible)
        if (stats['normal']['mean'] is not None and 
            stats['fraud']['mean'] is not None and
            stats['normal']['mean'] != 0):
            
            mean_ratio = stats['fraud']['mean'] / stats['normal']['mean']
            median_ratio = stats['fraud']['median'] / stats['normal']['median'] if stats['normal']['median'] != 0 else None
            
            ratio_text = f"\nRATIOS (Fraud / Normal):\n"
            ratio_text += f"  MEAN RATIO: {mean_ratio:.6f}\n"
            if median_ratio is not None:
                ratio_text += f"  MEDIAN RATIO: {median_ratio:.6f}\n"
            
            results_file.write(ratio_text)
            print(ratio_text)
    
    # Create a summary table at the end
    summary_header = "\n\n" + "=" * 80 + "\n"
    summary_header += "SUMMARY OF RECONSTRUCTION ERRORS ACROSS MODELS\n"
    summary_header += "=" * 80 + "\n\n"
    results_file.write(summary_header)
    print(summary_header)
    
    header_row = f"{'MODEL TYPE':<20} {'CLASS TYPE':<10} {'NORMAL MEAN':<15} {'NORMAL IQR':<15} {'FRAUD MEAN':<15} {'FRAUD IQR':<15} {'MEAN RATIO':<15}\n"
    header_row += "-" * 105 + "\n"
    results_file.write(header_row)
    print(header_row)
    
    for model_config in model_configs:
        model_type = model_config['type']
        class_type = model_config['class_type']
        
        # Find this model in our processed results
        model_file = os.path.join(output_dir, f"{model_type}_{class_type}_errors.npz")
        if not os.path.exists(model_file):
            continue
            
        data = np.load(model_file)
        errors, labels = data['errors'], data['labels']
        stats = calculate_error_statistics(errors, labels)
        
        normal_mean = stats['normal']['mean'] or 0
        normal_iqr = stats['normal']['iqr'] or 0
        fraud_mean = stats['fraud']['mean'] or 0
        fraud_iqr = stats['fraud']['iqr'] or 0
        mean_ratio = fraud_mean / normal_mean if normal_mean != 0 else "N/A"
        
        if isinstance(mean_ratio, float):
            mean_ratio = f"{mean_ratio:.4f}"
        
        summary_row = f"{model_type:<20} {class_type:<10} {normal_mean:<15.6f} {normal_iqr:<15.6f} {fraud_mean:<15.6f} {fraud_iqr:<15.6f} {mean_ratio:<15}\n"
        results_file.write(summary_row)
        print(summary_row)
    
    # Close the results file
    results_file.close()
    
    print(f"\nAnalysis complete! Results saved to {results_file_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
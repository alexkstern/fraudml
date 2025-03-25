import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import configparser
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
from dataloader.dataloader_classifier import load_fraud_classification_data
from models.conv_vae_model import ConvVae
from models.transformer_vae_model import TransformerVae
from models.conv_vqvae_model import ConvVQVAE
from models.transformer_vqvae_model import TransformerVQVAE

class ModelEvaluator:
    """
    A class to evaluate different types of autoencoders for fraud detection.
    """
    def __init__(self, model_type, fraud_model_path, normal_model_path, config_path, test_dataloader, device=None):
        """
        Initialize the evaluator with model paths and configuration.
        
        Args:
            model_type (str): Type of model ('conv_vae', 'transformer_vae', 'conv_vqvae', 'transformer_vqvae')
            fraud_model_path (str): Path to the fraud model checkpoint
            normal_model_path (str): Path to the normal model checkpoint
            config_path (str): Path to the model configuration file
            test_dataloader: Test dataloader with labeled data
            device: Device to run evaluation on (default: auto-detect)
        """
        self.model_type = model_type
        self.fraud_model_path = fraud_model_path
        self.normal_model_path = normal_model_path
        self.config_path = config_path
        self.test_dataloader = test_dataloader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Map model types to their classes and config sections
        self.model_mapping = {
            'conv_vae': {'class': ConvVae, 'section': 'Conv_VAE', 'loss_type': 'vae'},
            'transformer_vae': {'class': TransformerVae, 'section': 'Transformer_VAE', 'loss_type': 'vae'},
            'conv_vqvae': {'class': ConvVQVAE, 'section': 'Conv_VQVAE', 'loss_type': 'vqvae'},
            'transformer_vqvae': {'class': TransformerVQVAE, 'section': 'Transformer_VQVAE', 'loss_type': 'vqvae'}
        }
        
        if model_type not in self.model_mapping:
            raise ValueError(f"Unsupported model type: {model_type}. Must be one of {list(self.model_mapping.keys())}")
            
        # Load model config
        self.model_info = self.model_mapping[model_type]
        config_parser = configparser.ConfigParser()
        config_parser.read(config_path)
        self.model_config = config_parser[self.model_info['section']]
        
        print(f"Loaded {model_type} configuration from {config_path}")
        
    def load_model(self, model_path):
        """Load a model from checkpoint."""
        model_class = self.model_info['class']
        model = model_class(self.model_config)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
        
    def calculate_reconstruction_error(self, model, data_batch):
        """Calculate reconstruction error for a batch of data."""
        with torch.no_grad():
            data_batch = data_batch.to(self.device)
            
            if self.model_info['loss_type'] == 'vae':
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
    
    def evaluate(self):
        """
        Evaluate fraud and normal models on test data.
        
        Returns:
            Dictionary of evaluation results
        """
        print(f"Loading fraud model from {self.fraud_model_path}")
        fraud_model = self.load_model(self.fraud_model_path)
        
        print(f"Loading normal model from {self.normal_model_path}")
        normal_model = self.load_model(self.normal_model_path)
        
        # Initialize arrays to store results
        all_fraud_errors = []
        all_normal_errors = []
        all_true_labels = []
        
        # Process test set
        print(f"Evaluating {self.model_type} models on test data...")
        for batch_features, batch_labels in self.test_dataloader:
            # Calculate reconstruction errors
            fraud_error = self.calculate_reconstruction_error(fraud_model, batch_features)
            normal_error = self.calculate_reconstruction_error(normal_model, batch_features)
            
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
            'model_type': self.model_type,
            'fraud_errors': fraud_errors,
            'normal_errors': normal_errors,
            'error_ratios': error_ratios,
            'true_labels': true_labels
        }
    
    def find_best_threshold(self, error_ratios, true_labels, thresholds=None):
        """Find the best classification threshold by maximizing F1 score."""
        if thresholds is None:
            thresholds = np.linspace(0.1, 2.0, 100)
        
        best_threshold = 1.0
        best_f1 = 0.0
        
        for threshold in thresholds:
            predictions = (error_ratios < threshold).astype(int)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        print(f"Best threshold: {best_threshold:.4f} (F1 Score: {best_f1:.4f})")
        return best_threshold, best_f1
    
    def compute_metrics(self, error_ratios, true_labels, threshold):
        """Compute classification metrics using the specified threshold."""
        predictions = (error_ratios < threshold).astype(int)
        
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
    
    def plot_results(self, results, save_dir=None):
        """Plot evaluation results."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. Plot reconstruction errors scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            results['normal_errors'], 
            results['fraud_errors'], 
            c=results['true_labels'], 
            cmap='coolwarm', 
            alpha=0.7
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('True Class (0=Normal, 1=Fraud)')
        plt.xlabel('Normal Model Reconstruction Error')
        plt.ylabel('Fraud Model Reconstruction Error')
        plt.title(f'{self.model_type} - Reconstruction Errors')
        plt.grid(True, alpha=0.3)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{self.model_type}_recon_errors.png"))
        plt.show()
        
        # 2. Plot confusion matrix
        best_threshold, _ = self.find_best_threshold(results['error_ratios'], results['true_labels'])
        metrics = self.compute_metrics(results['error_ratios'], results['true_labels'], best_threshold)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metrics['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud']
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{self.model_type} - Confusion Matrix (threshold={best_threshold:.4f})')
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{self.model_type}_confusion_matrix.png"))
        plt.show()
        
    def run_full_evaluation(self, save_dir=None):
        """Run a complete evaluation and return the results."""
        # Evaluate models
        results = self.evaluate()
        
        # Find the best threshold
        best_threshold, best_f1 = self.find_best_threshold(results['error_ratios'], results['true_labels'])
        print(f"Best threshold: {best_threshold:.4f} (F1 Score: {best_f1:.4f})")
        
        # Compute metrics
        metrics = self.compute_metrics(results['error_ratios'], results['true_labels'], best_threshold)
        
        # Print metrics
        print("\nClassification Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Plot results
        if save_dir:
            self.plot_results(results, save_dir)
        
        return {
            'results': results,
            'best_threshold': best_threshold,
            'metrics': metrics
        }
    
def main():
    # Configuration
    classifier_config_path = "configs/classifier/classifier.config"
    
    # Load classification data that includes labels
    data = load_fraud_classification_data(config_path=classifier_config_path)
    test_dataloader = data['dataloaders']['test']
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dictionary to store all evaluation results
    all_evaluations = {}
    
    # Model configurations
    model_configs = [
        {
            'type': 'conv_vae',
            'fraud_model': "saved_models/conv_vae/fraud_conv_vae/20250313_173415/best_model.pt",
            'normal_model': "saved_models/conv_vae/normal_conv_vae/20250313_175713/best_model.pt",
            'config': "configs/conv_vae/fraud_conv_vae.config",
            'save_dir': "evaluation_results/conv_vae"
        },
        {
            'type': 'transformer_vae',
            'fraud_model': "saved_models/transformer_vae/fraud_transformer_vae/20250313_190737/best_model.pt",
            'normal_model': "saved_models/transformer_vae/normal_transformer_vae/20250313_232852/best_model.pt",
            'config': "configs/transformer_vae/fraud_transformer_vae.config",
            'save_dir': "evaluation_results/transformer_vae"
        },
        {
            'type': 'conv_vqvae',
            'fraud_model': "saved_models/conv_vqvae/fraud_conv_vqvae/20250313_235605/best_model.pt",
            'normal_model': "saved_models/conv_vqvae/normal_conv_vqvae/20250313_235821/best_model.pt",
            'config': "configs/conv_vqvae/fraud_conv_vqvae.config",
            'save_dir': "evaluation_results/conv_vqvae"
        },
        {
            'type': 'transformer_vqvae',
            'fraud_model': "saved_models/transformer_vqvae/fraud_transformer_vqvae/20250314_001329/best_model.pt",
            'normal_model': "saved_models/transformer_vqvae/normal_transformer_vqvae/20250314_001750/best_model.pt",
            'config': "configs/transformer_vqvae/fraud_transformer_vqvae.config",
            'save_dir': "evaluation_results/transformer_vqvae"
        }
    ]
    
    # Evaluate each model type
    for model_config in model_configs:
        print(f"\n----- Evaluating {model_config['type']} -----")
        
        try:
            evaluator = ModelEvaluator(
                model_type=model_config['type'],
                fraud_model_path=model_config['fraud_model'],
                normal_model_path=model_config['normal_model'],
                config_path=model_config['config'],
                test_dataloader=test_dataloader,
                device=device
            )
            
            evaluation = evaluator.run_full_evaluation(model_config['save_dir'])
            all_evaluations[model_config['type']] = evaluation
            
        except Exception as e:
            print(f"Error evaluating {model_config['type']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Optionally: Compare all model types
    if len(all_evaluations) > 1:
        print("\n----- Model Comparison -----")
        print("Model Type    | Accuracy | Precision | Recall  | F1 Score")
        print("-------------|----------|-----------|---------|--------")
        for model_type, eval_data in all_evaluations.items():
            metrics = eval_data['metrics']
            print(f"{model_type:13} | {metrics['accuracy']:.4f}   | {metrics['precision']:.4f}    | {metrics['recall']:.4f}  | {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
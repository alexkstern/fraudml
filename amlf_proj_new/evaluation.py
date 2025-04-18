import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, f1_score,
                             accuracy_score, precision_score, recall_score,
                             precision_recall_curve)
import seaborn as sns
from dataloader.dataloader_classifier import load_fraud_classification_data
from models.conv_vae_model import ConvVae
from models.transformer_vae_model import TransformerVae
from models.conv_vqvae_model import ConvVQVAE
from models.transformer_vqvae_model import TransformerVQVAE

torch.multiprocessing.set_sharing_strategy('file_system')

class IndependentModelEvaluator:
    def __init__(self, model_type, model_path, config_path, device=None, class_type=None):
        self.model_type = model_type
        self.model_path = model_path
        self.config_path = config_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Determine class_type from path if not provided
        if class_type is None:
             if "normal" in model_path.lower(): self.class_type = "normal"
             elif "fraud" in model_path.lower(): self.class_type = "fraud"
             else: raise ValueError("Cannot determine class_type (normal/fraud) from model_path")
        else:
            self.class_type = class_type

        self.model_mapping = {
            'conv_vae': {'class': ConvVae, 'section': 'Conv_VAE', 'loss_type': 'vae'},
            'transformer_vae': {'class': TransformerVae, 'section': 'Transformer_VAE', 'loss_type': 'vae'},
            'conv_vqvae': {'class': ConvVQVAE, 'section': 'Conv_VQVAE', 'loss_type': 'vqvae'},
            'transformer_vqvae': {'class': TransformerVQVAE, 'section': 'Transformer_VQVAE', 'loss_type': 'vqvae'}
        }
        if model_type not in self.model_mapping:
            raise ValueError(f"Unsupported model type: {model_type}. Must be one of {list(self.model_mapping.keys())}")

        self.model_info = self.model_mapping[model_type]
        import configparser
        config_parser = configparser.ConfigParser()
        config_parser.read(config_path)
        # Handle potential missing section more gracefully
        if self.model_info['section'] not in config_parser:
             raise ValueError(f"Config section '[{self.model_info['section']}]' not found in {config_path}")
        self.model_config = config_parser[self.model_info['section']]

        print(f"Loaded {model_type} ({self.class_type}) configuration from {config_path}")
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model_class = self.model_info['class']
        # Ensure input_dim is passed correctly if model expects it (check model __init__)
        if 'input_dim' not in self.model_config and hasattr(model_class, '__init__') and 'input_dim' in model_class.__init__.__code__.co_varnames:
             print(f"Warning: 'input_dim' might be missing in config section [{self.model_info['section']}] for {model_class.__name__}")
             # Attempt to load anyway, might fail if model needs it strictly
             model = model_class(self.model_config)
        else:
            model = model_class(self.model_config)

        try:
            # Try loading with weights_only=True first for security
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
             print(f"Warning: Failed loading with weights_only=True ({e}). Trying without...")
             try:
                # Fallback for older checkpoints saved without weights_only
                checkpoint = torch.load(model_path, map_location=self.device)
                # Check if it's the new format (dict) or potentially just the state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict): # Might be the state_dict itself
                     model.load_state_dict(checkpoint)
                else: # Assume it's just the state dict
                     model.load_state_dict(checkpoint)
             except Exception as load_err:
                  print(f"FATAL: Failed to load model checkpoint from {model_path}: {load_err}")
                  raise load_err # Re-raise the error

        model.to(self.device)
        model.eval()
        return model

    def calculate_reconstruction_errors(self, dataloader):
        print(f"Processing data with {self.model_type} {self.class_type} model...")
        all_errors = []
        all_labels = []

        with torch.no_grad():
            # Use tqdm for progress bar
            from tqdm import tqdm
            pbar = tqdm(dataloader, desc=f"Eval {self.model_type} ({self.class_type})", leave=False)
            for batch_data in pbar:
                # Handle potential None batches from collate errors
                if batch_data is None or batch_data[0] is None:
                     print("Warning: Skipping None batch in evaluation.")
                     continue
                batch_features, batch_labels = batch_data
                batch_features = batch_features.to(self.device)

                recon = None # Initialize recon
                try:
                    if self.model_info['loss_type'] == 'vae':
                        recon, mu, logvar = self.model(batch_features)
                    else: # vqvae
                        recon, vq_loss = self.model(batch_features) # Unpack both outputs

                    # Calculate per-sample MSE, ensure dimensions match
                    if recon.shape != batch_features.shape:
                         print(f"!!! Shape mismatch during error calc: Recon {recon.shape}, Input {batch_features.shape}. Skipping batch.")
                         continue # Skip batch if shapes don't match

                    # Mean over channel and feature dims (dims 1 and 2)
                    recon_error = ((recon - batch_features) ** 2).mean(dim=tuple(range(1, recon.ndim)))
                    all_errors.extend(recon_error.cpu().numpy())
                    all_labels.extend(batch_labels.numpy())
                except Exception as e:
                     print(f"!!! Error during model forward/error calc: {e}. Input shape: {batch_features.shape}. Skipping batch.")
                     import traceback
                     traceback.print_exc() # Print detailed traceback
                     continue # Skip problematic batch

        if not all_errors: # Handle case where all batches failed
            print("Warning: No errors calculated, possibly due to errors in all batches.")
            return np.array([]), np.array([])

        return np.array(all_errors), np.array(all_labels)


    def find_best_threshold(self, errors, true_labels):
        """
        Find the optimal threshold by considering candidate thresholds computed 
        only from the fraud errors, then maximize F1 on the combined set.
        """
        # Identify fraud errors
        fraud_errors = errors[true_labels == 1]
        if len(fraud_errors) == 0:
            raise ValueError("No fraud examples found for threshold selection.")
        
        # Generate candidate thresholds from fraud errors
        candidate_thresholds = np.percentile(fraud_errors, np.linspace(0.1, 99.9, 200))
        candidate_thresholds = np.unique(candidate_thresholds)
        
        best_f1 = -1.0
        best_threshold = np.median(fraud_errors)
        best_condition_is_greater = True  # default assumption

        # Try both conditions: predict fraud when error > threshold or when error <= threshold
        for condition_is_greater in [True, False]:
            for t in candidate_thresholds:
                if condition_is_greater:
                    predictions = (errors > t).astype(int)
                else:
                    predictions = (errors <= t).astype(int)
                
                current_f1 = f1_score(true_labels, predictions, pos_label=1, zero_division=0)
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = t
                    best_condition_is_greater = condition_is_greater

        condition_description = ">" if best_condition_is_greater else "<="
        print(f"Best threshold (fraud subset): {best_threshold:.6f} with condition 'errors {condition_description} threshold'")
        print(f"Max F1-score (using fraud subset candidates): {best_f1:.4f}")
        
        return best_threshold, best_f1, best_condition_is_greater



    def compute_metrics(self, errors, true_labels, threshold, condition_is_greater=True):
        """Compute classification metrics using the specified threshold and condition."""
        if condition_is_greater:
            # Predict fraud when error > threshold
            predictions = (errors > threshold).astype(int)
        else:
            # Predict fraud when error <= threshold
            predictions = (errors <= threshold).astype(int)
        
        conf_matrix = confusion_matrix(true_labels, predictions)
        # Ensure labels=[0, 1] if not all classes present in predictions
        cls_report_dict = classification_report(true_labels, predictions, output_dict=True, zero_division=0, labels=[0, 1])
        
        # Calculate metrics specifically for the positive class (fraud=1)
        precision = precision_score(true_labels, predictions, pos_label=1, zero_division=0)
        recall = recall_score(true_labels, predictions, pos_label=1, zero_division=0)
        f1 = f1_score(true_labels, predictions, pos_label=1, zero_division=0)
        accuracy = accuracy_score(true_labels, predictions)
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': cls_report_dict,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'threshold': threshold,
            'condition_is_greater': condition_is_greater
        }

    def plot_results(self, results, metrics, save_dir=None):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        errors = results['errors']
        true_labels = results['true_labels']
        threshold = metrics['threshold']
        condition_is_greater = metrics['condition_is_greater']
        condition_description = ">" if condition_is_greater else "<="
        
        if len(errors) == 0:
             print("Skipping plots as no errors were recorded.")
             return

        # Plot 1: Error distribution by class
        plt.figure(figsize=(10, 6))
        try:
             # Use log scale for y-axis if densities vary widely? Or clip errors?
             sns.kdeplot(errors[true_labels == 0], label='Normal (Class 0)', fill=True, warn_singular=False, log_scale=False)
             sns.kdeplot(errors[true_labels == 1], label='Fraud (Class 1)', fill=True, warn_singular=False, log_scale=False)
             plt.axvline(threshold, color='r', linestyle='--', 
                         label=f'Threshold: {threshold:.4f} (Fraud when error {condition_description} threshold)')
             plt.xlabel('Reconstruction Error')
             plt.ylabel('Density')
             plt.title(f'{self.model_type} ({self.class_type}) - Reconstruction Error Distribution')
             # Set xlim potentially based on percentiles to avoid extreme outliers dominating plot
             lower_lim = np.percentile(errors, 0.1)
             upper_lim = np.percentile(errors, 99.9)
             plt.xlim(max(0, lower_lim - 0.1*(upper_lim-lower_lim)), upper_lim + 0.1*(upper_lim-lower_lim))
             plt.legend()
             plt.grid(True, alpha=0.3)
             if save_dir:
                 plt.savefig(os.path.join(save_dir, f"{self.model_type}_{self.class_type}_error_dist.png"))
             plt.close()
        except Exception as e:
             print(f"Error plotting error distribution: {e}")
             plt.close() # Ensure plot is closed even on error

        # Plot 2: Confusion matrix
        plt.figure(figsize=(8, 6))
        try:
             sns.heatmap(
                 metrics['confusion_matrix'],
                 annot=True,
                 fmt='d',
                 cmap='Blues',
                 xticklabels=['Pred Normal', 'Pred Fraud'],
                 yticklabels=['True Normal', 'True Fraud']
             )
             plt.xlabel('Predicted Label')
             plt.ylabel('True Label')
             plt.title(f'{self.model_type} ({self.class_type}) - Confusion Matrix\n(Thresh={threshold:.4f}, Fraud when error {condition_description} threshold)')
             if save_dir:
                 plt.savefig(os.path.join(save_dir, f"{self.model_type}_{self.class_type}_confusion_matrix.png"))
             plt.close()
        except Exception as e:
             print(f"Error plotting confusion matrix: {e}")
             plt.close()

def evaluate_all_models(classifier_config_path, model_configs, base_save_dir="evaluation_results_independent"):
    # Load classification data (using the specific eval config)
    print(f"Loading evaluation data using config: {classifier_config_path}")
    data = load_fraud_classification_data(config_path=classifier_config_path)
    # Check if dataloaders are empty
    if not data['dataloaders']['train'] or not data['dataloaders']['val'] or not data['dataloaders']['test']:
         print("FATAL: One or more evaluation dataloaders are empty. Check data loading and config.")
         return {}, []

    train_dataloader = data['dataloaders']['train']
    val_dataloader = data['dataloaders']['val']
    test_dataloader = data['dataloaders']['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_evaluations = {}
    compare_results = []

    # Create a summary table for all models
    print("\n----- Model Threshold and F1 Score Summary -----")
    print(f"{'Model Type':<20} | {'Class':<6} | {'Max F1 (train+val)':<17} | {'Threshold':<10} | {'Condition':<10} | {'Test F1':<8}")
    print("-" * 85)

    for model_config in model_configs:
        try:
            print(f"\n----- Evaluating {model_config['type']} ({model_config['class_type']}) -----")
            model_id = f"{model_config['type']}_{model_config['class_type']}"
            save_dir = os.path.join(base_save_dir, model_id)
            os.makedirs(save_dir, exist_ok=True)

            evaluator = IndependentModelEvaluator(
                model_type=model_config['type'],
                model_path=model_config['model_path'],
                config_path=model_config['config'],
                device=device,
                class_type=model_config['class_type']
            )

            # Step 1: Collect errors from training and validation sets
            print("Collecting reconstruction errors from training set...")
            train_errors, train_labels = evaluator.calculate_reconstruction_errors(train_dataloader)
            print(f"Collected {len(train_errors)} training errors.")
            if len(train_errors) == 0:
                 print("Skipping evaluation for this model due to no training errors.")
                 continue

            print("Collecting reconstruction errors from validation set...")
            val_errors, val_labels = evaluator.calculate_reconstruction_errors(val_dataloader)
            print(f"Collected {len(val_errors)} validation errors.")
            if len(val_errors) == 0:
                 print("Skipping evaluation for this model due to no validation errors.")
                 continue

            combined_errors = np.concatenate([train_errors, val_errors])
            combined_labels = np.concatenate([train_labels, val_labels])

            # Step 2: Find optimal threshold on combined data using empirical F1 search
            best_threshold, best_f1_on_val, condition_is_greater = evaluator.find_best_threshold(
                combined_errors, combined_labels)
            condition_desc = ">" if condition_is_greater else "<="
            print(f"Optimal threshold on train+val (max F1={best_f1_on_val:.4f}): {best_threshold:.6f} "
                  f"with condition 'errors {condition_desc} threshold'")

            # Step 3: Evaluate on test set
            print("Collecting reconstruction errors from test set...")
            test_errors, test_labels = evaluator.calculate_reconstruction_errors(test_dataloader)
            print(f"Collected {len(test_errors)} test errors.")
            if len(test_errors) == 0:
                 print("Skipping test evaluation for this model due to no test errors.")
                 continue

            # Step 4: Compute metrics using the found best_threshold and condition
            test_metrics = evaluator.compute_metrics(
                test_errors, test_labels, best_threshold, condition_is_greater)

            # Print metrics
            print("\nTest Set Classification Metrics (Class 1 = Fraud):")
            print(f"Threshold: {test_metrics['threshold']:.6f} (Fraud when errors {condition_desc} threshold)")
            print(f"Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Precision (Fraud): {test_metrics['precision']:.4f}")
            print(f"Recall (Fraud): {test_metrics['recall']:.4f}")
            print(f"F1 Score (Fraud): {test_metrics['f1_score']:.4f}")
            print("\nConfusion Matrix (Rows: True, Cols: Pred):")
            print("             Pred Normal  Pred Fraud")
            print(f"True Normal  {test_metrics['confusion_matrix'][0,0]:<11d} {test_metrics['confusion_matrix'][0,1]:<10d}")
            print(f"True Fraud   {test_metrics['confusion_matrix'][1,0]:<11d} {test_metrics['confusion_matrix'][1,1]:<10d}")

            # Plot results
            test_results = {'errors': test_errors, 'true_labels': test_labels}
            evaluator.plot_results(test_results, test_metrics, save_dir)

            all_evaluations[model_id] = {
                'results': test_results,
                'metrics': test_metrics,
                'train_val_f1': best_f1_on_val,
                'threshold': best_threshold,
                'condition_is_greater': condition_is_greater
            }

            # Store simplified results for comparison table
            # Extract model base type (conv, transformer) and vae/vqvae part
            base_model_type = model_config['type'].split('_')[0] # conv or transformer
            vae_type = model_config['type'].split('_')[1] # vae or vqvae
            row = {
                'model_type': f"{base_model_type}_{vae_type}",
                'class_type': model_config['class_type'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1_score': test_metrics['f1_score'],
                'accuracy': test_metrics['accuracy'],
                'train_val_f1': best_f1_on_val,
                'threshold': best_threshold,
                'condition': '>' if condition_is_greater else '<='
            }
            compare_results.append(row)

        except Exception as e:
            print(f"Error evaluating {model_config.get('type', 'N/A')}_{model_config.get('class_type', 'N/A')}: {e}")
            import traceback
            traceback.print_exc()

    # --- Comparison Table and Plot (adjusted formatting to include condition) ---
    print("\n----- Model Comparison Summary -----")
    print(f"{'Model Type':<20} | {'Class':<6} | {'Max F1 (train+val)':<17} | {'Threshold':<10} | {'Condition':<10} | {'Test F1':<8} | {'Precision':<9} | {'Recall':<8} | {'Accuracy':<8}")
    print("-" * 110)

    # Sort for consistent table output
    compare_results.sort(key=lambda x: (x['model_type'], x['class_type']))

    for row in compare_results:
        print(f"{row['model_type']:<20} | {row['class_type']:<6} | {row['train_val_f1']:<17.4f} | {row['threshold']:<10.6f} | {row['condition']:<10} | {row['f1_score']:<8.4f} | {row['precision']:<9.4f} | {row['recall']:<8.4f} | {row['accuracy']:<8.4f}")

    # Save comparison results to CSV
    import csv
    os.makedirs(base_save_dir, exist_ok=True)
    comparison_path = os.path.join(base_save_dir, "model_comparison.csv")
    if compare_results: # Only write if there are results
        with open(comparison_path, 'w', newline='') as csvfile:
            fieldnames = ['model_type', 'class_type', 'train_val_f1', 'threshold', 'condition', 'precision', 'recall', 'f1_score', 'accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(compare_results)
        print(f"\nComparison results saved to {comparison_path}")
    else:
        print("\nNo evaluation results to save to CSV.")

    # Create comparison plot - now with two plots
    if compare_results:
        # Figure 1: Compare train+val F1 with test F1 by model
        plt.figure(figsize=(14, 7))
        
        # Prepare data for plotting
        model_names = sorted(list(set(r['model_type'] for r in compare_results)))
        x = np.arange(len(model_names))
        width = 0.2  # narrower bars to fit more data
        
        # Extract F1 scores
        normal_train_val_f1 = []
        normal_test_f1 = []
        fraud_train_val_f1 = []
        fraud_test_f1 = []
        
        for name in model_names:
            normal_row = next((r for r in compare_results if r['model_type'] == name and r['class_type'] == 'normal'), None)
            fraud_row = next((r for r in compare_results if r['model_type'] == name and r['class_type'] == 'fraud'), None)
            
            normal_train_val_f1.append(normal_row['train_val_f1'] if normal_row else 0)
            normal_test_f1.append(normal_row['f1_score'] if normal_row else 0)
            fraud_train_val_f1.append(fraud_row['train_val_f1'] if fraud_row else 0)
            fraud_test_f1.append(fraud_row['f1_score'] if fraud_row else 0)
        
        # Plot the four bar sets
        plt.bar(x - 1.5*width, normal_train_val_f1, width, label='Normal Model (Train+Val F1)')
        plt.bar(x - 0.5*width, normal_test_f1, width, label='Normal Model (Test F1)')
        plt.bar(x + 0.5*width, fraud_train_val_f1, width, label='Fraud Model (Train+Val F1)')
        plt.bar(x + 1.5*width, fraud_test_f1, width, label='Fraud Model (Test F1)')
        
        plt.ylabel('F1 Score')
        plt.title('Comparison of Training+Validation F1 vs Test F1 by Model Type')
        plt.xticks(x, model_names, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        f1_comparison_path = os.path.join(base_save_dir, "model_train_test_f1_comparison.png")
        plt.savefig(f1_comparison_path)
        plt.close()
        print(f"F1 comparison plot saved to {f1_comparison_path}")
        
        # Original F1 plot for consistency
        plt.figure(figsize=(12, 7))
        rects1 = plt.bar(x - width/2, normal_test_f1, width, label='Normal Model (Test F1)')
        rects2 = plt.bar(x + width/2, fraud_test_f1, width, label='Fraud Model (Test F1)')
        
        plt.ylabel('F1 Score (Fraud Detection)')
        plt.title('Model F1 Scores on Test Set')
        plt.xticks(x, model_names, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        original_plot_path = os.path.join(base_save_dir, "model_comparison_f1.png")
        plt.savefig(original_plot_path)
        plt.close()
        print(f"Original comparison plot saved to {original_plot_path}")

        # New plot: Conditions by model type
        plt.figure(figsize=(12, 7))
        normal_conditions = []
        fraud_conditions = []
        
        for name in model_names:
            normal_row = next((r for r in compare_results if r['model_type'] == name and r['class_type'] == 'normal'), None)
            fraud_row = next((r for r in compare_results if r['model_type'] == name and r['class_type'] == 'fraud'), None)
            
            # Convert condition to numeric for visualization (1 for '>', 0 for '<=')
            normal_cond = 1 if normal_row and normal_row['condition'] == '>' else 0
            fraud_cond = 1 if fraud_row and fraud_row['condition'] == '>' else 0
            
            normal_conditions.append(normal_cond)
            fraud_conditions.append(fraud_cond)
        
        # Plot conditions (1 for '>', 0 for '<=')
        plt.bar(x - width/2, normal_conditions, width, label='Normal Model')
        plt.bar(x + width/2, fraud_conditions, width, label='Fraud Model')
        
        plt.ylabel('Condition (1 = ">" threshold, 0 = "<=" threshold)')
        plt.title('Empirically Determined Conditions by Model Type')
        plt.xticks(x, model_names, rotation=45, ha="right")
        plt.yticks([0, 1], ['<= threshold', '> threshold'])
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        conditions_plot_path = os.path.join(base_save_dir, "model_conditions.png")
        plt.savefig(conditions_plot_path)
        plt.close()
        print(f"Conditions plot saved to {conditions_plot_path}")

    return all_evaluations, compare_results

def main():
    classifier_config_path = "configs/classifier/classifier.config"
    # Update paths if needed, e.g., if you retrained fraud models
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

    print(f"Evaluating {len(model_configs)} models:")
    for config in model_configs:
        print(f"  - {config['type']} ({config['class_type']}): {config['model_path']}")

    evaluate_all_models(classifier_config_path, model_configs)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) # Suppress some common warnings
    main()
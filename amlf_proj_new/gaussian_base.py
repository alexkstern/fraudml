import os
import torch
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
from dataloader.dataloader import load_fraud_data
from dataloader.dataloader_classifier import load_fraud_classification_data
import datetime


def fit_gaussian_distribution(dataloader, device='cpu'):
    """
    Fit a multivariate Gaussian to the data from the dataloader.
    """
    all_features = []
    
    # Collect all features from the dataloader
    for batch in dataloader:
        batch = batch.to(device)
        # Remove channel dimension if present
        if batch.dim() == 3:  # (batch, channels, features)
            batch = batch.squeeze(1)
        all_features.append(batch.cpu().numpy())
    
    # Combine all batches
    all_features = np.vstack(all_features)
    
    # Fit the Gaussian
    mean = np.mean(all_features, axis=0)
    cov = np.cov(all_features, rowvar=False)
    
    # Add small regularization to ensure covariance matrix is positive definite
    epsilon = 1e-6
    n_features = cov.shape[0]
    cov += epsilon * np.eye(n_features)
    
    return mean, cov

def calculate_likelihood(features, mean, cov):
    """
    Calculate the likelihood of features under a multivariate Gaussian distribution.
    Lower likelihood indicates more anomalous (potentially fraud) instances.
    """
    # Create multivariate normal distribution
    dist = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
    
    # Calculate log likelihood for numerical stability
    log_likelihood = dist.logpdf(features)
    
    return log_likelihood

def find_optimal_threshold(likelihoods, true_labels, class_type):
    """
    Find the threshold that maximizes F1 score.
    For normal model: samples with likelihood < threshold are classified as fraud
    For fraud model: samples with likelihood > threshold are classified as fraud
    """
    # Determine if we're using normal or fraud model to adjust comparison direction
    if class_type == 'normal':
        # For normal model, lower likelihood = more anomalous = more likely fraud
        scores = -likelihoods  # Negate so higher score = more likely fraud
    else:  # fraud model
        # For fraud model, higher likelihood = more likely fraud
        scores = likelihoods
    
    # Use sklearn's precision_recall_curve to generate good threshold candidates
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:  # Avoid division by zero
            f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            f1_scores.append((f1, i))
    
    # Find threshold with maximum F1 score
    if not f1_scores:
        return 0.0, 0.0
    
    best_score, best_idx = max(f1_scores, key=lambda x: x[0])
    
    # Get the corresponding threshold
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        # Handle edge case for the last point in precision_recall_curve
        best_threshold = 0.0
    
    return best_threshold, best_score

def extract_features_and_labels(dataloader, device='cpu'):
    """
    Extract features and labels from a classification dataloader.
    """
    all_features = []
    all_labels = []
    
    for batch_features, batch_labels in dataloader:
        # Move to device
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        # Remove channel dimension if present
        if batch_features.dim() == 3:  # (batch, channels, features)
            batch_features = batch_features.squeeze(1)
        
        all_features.append(batch_features.cpu().numpy())
        all_labels.append(batch_labels.cpu().numpy())
    
    # Combine all batches
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    return all_features, all_labels

def plot_likelihood_distribution(likelihoods, labels, threshold, class_type, save_path):
    """
    Plot the distribution of likelihoods with the threshold.
    """
    plt.figure(figsize=(10, 6))
    
    # Get likelihoods for normal and fraud samples
    normal_likelihoods = likelihoods[labels == 0]
    fraud_likelihoods = likelihoods[labels == 1]
    
    # Plot histograms
    plt.hist(normal_likelihoods, bins=50, alpha=0.5, label='Normal')
    plt.hist(fraud_likelihoods, bins=50, alpha=0.5, label='Fraud')
    
    # Plot threshold line (direction depends on class_type)
    if class_type == 'normal':
        plt.axvline(-threshold, color='red', linestyle='--', 
                    label=f'Threshold: {-threshold:.2f}')
        plt.title(f'Normal Model: Log-Likelihood Distribution (samples below threshold classified as fraud)')
    else:  # fraud model
        plt.axvline(threshold, color='red', linestyle='--', 
                    label=f'Threshold: {threshold:.2f}')
        plt.title(f'Fraud Model: Log-Likelihood Distribution (samples above threshold classified as fraud)')
    
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(features, true_labels, mean, cov, threshold, class_type):
    """
    Evaluate the model on the given features and labels.
    """
    likelihoods = calculate_likelihood(features, mean, cov)
    
    # Make predictions based on model type
    if class_type == 'normal':
        # For normal model: likelihood < threshold => fraud
        predictions = (likelihoods < -threshold).astype(int)
    else:  # fraud model
        # For fraud model: likelihood > threshold => fraud
        predictions = (likelihoods > threshold).astype(int)
    
    # Calculate metrics
    f1 = f1_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions)
    
    return likelihoods, predictions, f1, conf_matrix, class_report

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('gaussian_model_results', exist_ok=True)
    
    # Open a text file to save results
    results_file = open('gaussian_model_results/results_gaussian.txt', 'w')
    results_file.write(f"Gaussian Models Evaluation Results\n")
    results_file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_file.write(f"{'='*80}\n\n")
    
    # Define config paths
    fraud_config_path = "configs/conv_vae/fraud_conv_vae.config"
    normal_config_path = "configs/conv_vae/normal_conv_vae.config"
    classifier_config_path = "configs/classifier/classifier.config"
    
    # Load classification data (for threshold optimization and testing)
    print("Loading classification data...")
    classification_data = load_fraud_classification_data(config_path=classifier_config_path)
    
    # Extract features and labels
    print("Extracting features from classification data...")
    train_features, train_labels = extract_features_and_labels(classification_data['dataloaders']['train'], device)
    val_features, val_labels = extract_features_and_labels(classification_data['dataloaders']['val'], device)
    test_features, test_labels = extract_features_and_labels(classification_data['dataloaders']['test'], device)
    
    # Combine train and validation for threshold optimization
    combined_features = np.vstack([train_features, val_features])
    combined_labels = np.concatenate([train_labels, val_labels])
    
    # Store results for summary
    model_results = []
    
    # Process both model types (normal and fraud)
    for model_type, config_path in [('normal', normal_config_path), ('fraud', fraud_config_path)]:
        print(f"\n{'='*80}\nProcessing {model_type.upper()} model\n{'='*80}")
        results_file.write(f"\n{model_type.upper()} MODEL\n{'-'*50}\n")
        
        # 1. Load model-specific training data 
        print(f"Loading {model_type} training data...")
        model_data = load_fraud_data(config_path=config_path)
        train_loader = model_data['dataloaders']['train']
        
        # 2. Fit Gaussian distribution
        print(f"Fitting Gaussian distribution to {model_type} data...")
        mean, cov = fit_gaussian_distribution(train_loader, device)
        print(f"Distribution shape: mean {mean.shape}, cov {cov.shape}")
        
        # 3. Calculate likelihoods for combined train+val set
        print("Calculating likelihoods for threshold optimization...")
        combined_likelihoods = calculate_likelihood(combined_features, mean, cov)
        
        # 4. Find optimal threshold
        print("Finding optimal threshold...")
        threshold, best_f1 = find_optimal_threshold(combined_likelihoods, combined_labels, model_type)
        direction = "<" if model_type == 'normal' else ">"
        print(f"Optimal threshold: likelihood {direction} {threshold:.6f}")
        print(f"F1 score on train+val: {best_f1:.4f}")
        
        results_file.write(f"Optimal threshold: likelihood {direction} {threshold:.6f}\n")
        results_file.write(f"F1 score on train+val: {best_f1:.4f}\n\n")
        
        # 5. Plot likelihood distributions with threshold
        print("Plotting likelihood distributions...")
        plot_path = f"gaussian_model_results/{model_type}_likelihood_distribution.png"
        plot_likelihood_distribution(
            combined_likelihoods, combined_labels, threshold, model_type, plot_path)
        
        # 6. Evaluate on test data
        print("Evaluating on test data...")
        test_likelihoods, test_predictions, test_f1, conf_matrix, class_report = evaluate_model(
            test_features, test_labels, mean, cov, threshold, model_type)
        
        # 7. Print and save results
        print("\nTest Results:")
        print(f"F1 Score: {test_f1:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        
        # Write results to the text file
        results_file.write(f"TEST RESULTS:\n")
        results_file.write(f"F1 Score: {test_f1:.4f}\n\n")
        results_file.write(f"Confusion Matrix:\n")
        results_file.write(f"             Pred Normal  Pred Fraud\n")
        results_file.write(f"True Normal  {conf_matrix[0,0]:<11d} {conf_matrix[0,1]:<10d}\n")
        results_file.write(f"True Fraud   {conf_matrix[1,0]:<11d} {conf_matrix[1,1]:<10d}\n\n")
        results_file.write(f"Classification Report:\n{class_report}\n\n")
        
        # Store results for summary
        model_results.append({
            'model_type': model_type,
            'threshold': threshold,
            'train_val_f1': best_f1,
            'test_f1': test_f1
        })
        
        # 8. Save model parameters
        np.savez(f'gaussian_model_results/{model_type}_gaussian_model.npz', 
                 mean=mean, 
                 cov=cov,
                 threshold=threshold,
                 class_type=model_type)
        
        # 9. Save confusion matrix visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{model_type.capitalize()} Model Confusion Matrix')
        plt.colorbar()
        
        classes = ['Normal', 'Fraud']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]),
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f'gaussian_model_results/{model_type}_confusion_matrix.png')
        plt.close()
    
    # Write summary comparison
    results_file.write(f"\n{'='*80}\nMODEL COMPARISON SUMMARY\n{'='*80}\n")
    results_file.write(f"{'Model Type':<15} {'Threshold':<15} {'Train+Val F1':<15} {'Test F1':<15}\n")
    results_file.write(f"{'-'*60}\n")
    
    for result in model_results:
        results_file.write(f"{result['model_type']:<15} {result['threshold']:<15.6f} {result['train_val_f1']:<15.4f} {result['test_f1']:<15.4f}\n")
    
    # Close the results file
    results_file.close()
    
    print("\nEvaluation complete. Results saved to 'gaussian_model_results/results_gaussian.txt'")

if __name__ == "__main__":
    main()
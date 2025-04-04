# Credit Card Fraud Detection with Autoencoder Anomaly Detection

## Project Overview
This project implements and evaluates various autoencoder architectures for credit card fraud detection using reconstruction-based anomaly detection. The primary approach is to train separate models on normal and fraudulent transactions, then analyze the reconstruction errors to identify fraudulent activities.

## Dataset
- European credit card transaction dataset with highly imbalanced classes
- 0.17% fraud cases in all dataset splits
- Train set: 394 fraud out of 227,845 transactions
- Validation/Test sets: 49 fraud out of 28,481 transactions each

## Project Structure

### Core Components

#### Models
Four types of autoencoders were implemented and compared:
- `models/conv_vae_model.py`: Convolutional Variational Autoencoder
- `models/transformer_vae_model.py`: Transformer-based Variational Autoencoder
- `models/conv_vqvae_model.py`: Convolutional Vector Quantized VAE
- `models/transformer_vqvae_model.py`: Transformer-based Vector Quantized VAE

#### Data Loading
- `dataloader/dataloader.py`: Base data loader for autoencoder training 
- `dataloader/dataloader_classifier.py`: Extended data loader that preserves labels for evaluation

#### Training
- `trainer/trainer_vae.py`: VAE model trainer
- `trainer/trainer_vqvae.py`: VQ-VAE model trainer
- `main_train.py`: Main training script that cycles through all model configurations
- `train_one_model.py`: Script for training individual models

#### Evaluation
- `evaluation.py`: Comprehensive evaluation of trained models
- `error_distributions.py`: Analysis of reconstruction error distributions
- `gaussian_base.py`: Baseline Gaussian model for comparison

#### Configurations
- `configs/`: Configuration files for each model type and variant
  - `conv_vae/`: Convolutional VAE configs
  - `transformer_vae/`: Transformer VAE configs
  - `conv_vqvae/`: Convolutional VQ-VAE configs
  - `transformer_vqvae/`: Transformer VQ-VAE configs
  - `classifier/`: Evaluation configs

#### Utils
- `utils/evaluation_utils.py`: Utility functions for evaluation
- `utils/lr_scheduler.py`: Learning rate scheduler implementations
- `utils/model_saver.py`: Model checkpoint saving functionality
- `utils/wandb_logger.py` & `utils/wandb_logger_lr.py`: Weights & Biases logging

### Results and Analysis
- `results.txt`: Summary of training results
- `training_results_20250327_183655.json`: Detailed training metrics
- `gaussian_model_results/results_gaussian.txt`: Results for the Gaussian baseline model

## Methodology

### Model Training Strategy
1. For each architecture (VAE, VQ-VAE):
   - Train separate models on normal-only and fraud-only data
   - Models learn the distribution of either normal or fraudulent patterns

2. Model variants explored:
   - Convolutional encoders/decoders for capturing local feature relationships
   - Transformer-based encoders/decoders for capturing global dependencies
   - Both VAE (continuous latent space) and VQ-VAE (discrete latent space) approaches

### Anomaly Detection Approach
1. For each trained model:
   - Calculate reconstruction errors on test transactions
   - Find optimal threshold that maximizes F1 score on combined training+validation data
   - Transactions with reconstruction errors beyond the threshold are classified as anomalies

2. Each model type uses a different mechanism for detecting anomalies:
   - Normal models: High reconstruction error indicates potential fraud
   - Fraud models: Low reconstruction error indicates potential fraud

### Baseline Comparison
- Gaussian models were trained as baseline for both normal and fraud classes
- Results in `gaussian_model_results/` provide comparison points

## Key Findings
1. The best-performing models were:
   - Transformer-based VQ-VAE on normal data (best validation loss: 1.27)
   - Transformer-based VQ-VAE on fraud data (best validation loss: 24.25)

2. Most models achieved good discrimination between normal and fraudulent transactions
   - See full evaluation metrics in `evaluation_results_independent/` 

3. Reconstruction error ratios between fraud/normal samples provide a robust signal for anomaly detection

## Running the Code
To train all models:
```bash
python main_train.py
```

To train a single model (edit config path in script):
```bash
python train_one_model.py
```

To evaluate trained models:
```bash
python evaluation.py
```

To run baseline Gaussian models:
```bash
python gaussian_base.py
```

## Dependencies
- PyTorch
- NumPy
- scikit-learn
- Weights & Biases (for logging)
- Matplotlib, Seaborn (for visualization)
- Hugging Face Datasets (for data loading)
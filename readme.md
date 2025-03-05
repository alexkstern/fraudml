# Credit Card Fraud Detection with VAE

This project uses Variational Autoencoders (VAEs) to extract features for credit card fraud detection. All project files are contained in the `latent_classifier` folder, as this repository will be used for other projects in the future.

## Environment Setup

To set up the environment used for this project:

1. **Create a new conda environment**:
   ```bash
   conda create -n credit_vae python=3.8
   conda activate credit_vae
   ```

2. **Install pip within the conda environment**:
   ```bash
   conda install pip
   ```

3. **Install the requirements using pip**:
   ```bash
   pip install -r requirements.txt
   ```

This will recreate the exact environment used for the project.

## Project Structure

All files are located in the `latent_classifier` folder:

- `latent_classifier/dataloader.py`: Handles loading and preprocessing the credit card fraud dataset
- `latent_classifier/model_VAE.py`: Implements the Variational Autoencoder model
- `latent_classifier/trainer.py`: Contains the trainer class for model training and evaluation
- `latent_classifier/create_vae_dataset.py`: Creates a dataset combining features from both VAEs
- `latent_classifier/configs/`: Configuration files for VAE models and training parameters
- `latent_classifier/saved_models/`: Directory for saved model checkpoints
- `latent_classifier/vae_dataset/`: Generated dataset with VAE features for downstream tasks

## Usage

1. **Set up the environment** using the instructions above
2. **Navigate to the project folder**:
   ```bash
   cd latent_classifier
   ```
3. **Create the VAE dataset** by running:
   ```bash
   python create_vae_dataset.py
   ```
4. **Train classification models** using the generated features

## Notes

- The VAE feature dataset combines latent vectors and reconstruction errors from both fraud and non-fraud VAE models
- Both the VAE features and original features are preserved for comparison
- The dataset is separated into train, validation, and test splits
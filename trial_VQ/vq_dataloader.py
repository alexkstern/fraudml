import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from typing import Dict, Tuple, List, Union, Optional
import configparser
import os

def load_config(config_path):
    """
    Load configuration from a .config file with support for multiple sections
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary with sections as keys
    """
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Dictionary to hold all sections
        config_dict = {}
        
        # Process each section
        for section in config.sections():
            config_dict[section] = {}
            for key, value in config[section].items():
                # Strip whitespace from the value
                value = value.strip()
                
                # Handle lists first - check if value contains commas
                if ',' in value:
                    # Split by comma and process each item
                    items = [item.strip() for item in value.split(',')]
                    processed_items = []
                    
                    for item in items:
                        # Skip empty items
                        if not item:
                            continue
                            
                        # Try to convert to numbers if possible
                        try:
                            if '.' in item:
                                processed_items.append(float(item))
                            else:
                                processed_items.append(int(item))
                        except ValueError:
                            # Handle booleans
                            if item.lower() in ['true', 'yes']:
                                processed_items.append(True)
                            elif item.lower() in ['false', 'no']:
                                processed_items.append(False)
                            else:
                                processed_items.append(item)
                    
                    config_dict[section][key] = processed_items
                else:
                    # Handle non-list values
                    try:
                        if '.' in value:
                            config_dict[section][key] = float(value)
                        else:
                            config_dict[section][key] = int(value)
                    except ValueError:
                        # Handle booleans
                        if value.lower() in ['true', 'yes']:
                            config_dict[section][key] = True
                        elif value.lower() in ['false', 'no']:
                            config_dict[section][key] = False
                        else:
                            config_dict[section][key] = value
        
        return config_dict
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default configuration instead.")
        return {}


class CreditCardFraudDataset(Dataset):
    """
    Dataset class for credit card fraud detection.
    
    This class handles loading data from Hugging Face datasets and preparing it for VAE training.
    It performs normalization on numerical features based on training data statistics.
    """
    def __init__(self, 
                 dataset=None,
                 dataset_name: str = "stanpony/european_credit_card_fraud_dataset",
                 split: str = 'train',
                 exclude_cols: List[str] = ['original_index', 'Class'],
                 normalize_cols: List[str] = ['Time', 'Amount'],
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 normalize: bool = True,
                 stats_dict: Dict = None,
                 device: str = None,
                 return_labels: bool = False):
        """
        Initialize the dataset.
        """
        self.split = split
        
        # Ensure exclude_cols is a list
        if isinstance(exclude_cols, str):
            self.exclude_cols = [exclude_cols]
        else:
            self.exclude_cols = list(exclude_cols)  # Convert to list and create a copy
            
        self.normalize_cols = normalize_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalize = normalize
        self.return_labels = return_labels
        
        # If returning labels, make sure 'Class' is not in exclude_cols
        if self.return_labels and 'Class' in self.exclude_cols:
            self.exclude_cols.remove('Class')
            print(f"Removed 'Class' from exclude_cols because return_labels=True")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load the dataset if not provided
        if dataset is None:
            self.dataset = load_dataset(dataset_name, split=split)
        else:
            self.dataset = dataset
        
        print(f"Using all classes: {len(self.dataset)} samples")
            
        # Get feature columns (excluding specified columns)
        self.feature_cols = [col for col in self.dataset.column_names if col not in self.exclude_cols]
        
        # Calculate or use provided normalization statistics
        self.stats_dict = stats_dict
        if self.normalize and self.stats_dict is None:
            self.stats_dict = self._calculate_normalization_stats()
    
    def _calculate_normalization_stats(self) -> Dict:
        stats = {}
        for col in self.normalize_cols:
            if col in self.dataset.column_names:
                values = np.array(self.dataset[col])
                stats[col] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        return stats
    
    def _normalize_value(self, value: float, col: str) -> float:
        if col in self.stats_dict:
            return (value - self.stats_dict[col]['mean']) / self.stats_dict[col]['std']
        return value
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sample = self.dataset[idx]
        
        # Extract features (exclude 'Class' and other excluded columns)
        features = []
        for col in self.feature_cols:
            if col != 'Class':  # Skip 'Class' for features
                value = sample[col]
                if self.normalize and col in self.normalize_cols:
                    value = self._normalize_value(value, col)
                features.append(value)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # If returning labels, extract 'Class' separately
        if self.return_labels:
            label = sample['Class']
            label_tensor = torch.tensor(label, dtype=torch.long)
            return features_tensor, label_tensor
        else:
            return features_tensor
    
    def get_dataloader(self) -> DataLoader:
        """
        Create and return a DataLoader for this dataset.
        
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
            drop_last=False
        )
    
    def get_input_dim(self) -> int:
        """
        Get the input dimension (number of features).
        
        Returns:
            Integer representing the number of input features
        """
        return len([col for col in self.feature_cols if col != 'Class'])


class FraudDataModule:
    """
    Data module class to manage all datasets and dataloaders.
    
    Provides easy access to train, validation, and test dataloaders.
    """
    def __init__(self,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None,
                 dataset_name: str = "stanpony/european_credit_card_fraud_dataset",
                 exclude_cols: List[str] = ['original_index', 'Class'],
                 normalize_cols: List[str] = ['Time', 'Amount'],
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 normalize: bool = True,
                 device: str = None,
                 config_path: str = None,
                 return_labels: bool = False):
        """
        Initialize the data module.
        
        Args:
            train_dataset: Pre-loaded train dataset (optional)
            val_dataset: Pre-loaded validation dataset (optional)
            test_dataset: Pre-loaded test dataset (optional)
            dataset_name: Name of the Hugging Face dataset
            exclude_cols: Columns to exclude from the input features
            normalize_cols: Columns to normalize
            batch_size: Batch size for DataLoaders
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for DataLoaders
            normalize: Whether to normalize the data
            device: Device to use ('cuda' or 'cpu'). If None, will be automatically determined
            config_path: Path to configuration file (optional)
            return_labels: Whether to return labels with features
        """
        # Load config if provided
        if config_path and os.path.exists(config_path):
            config = load_config(config_path)
            
            # Extract DataLoader config if present
            dataloader_config = config.get('DataLoader', {})
            
            # Extract VQVAE config to check if classification is enabled
            vqvae_config = config.get('VQVAE', {})
            classification_enabled = vqvae_config.get('classification', False)
            
            # If classification is enabled in the model config, we need to return labels
            if classification_enabled:
                return_labels = True
                print("Classification enabled in model config - ensuring labels are returned")
            
            # Override defaults with config values if present
            if dataloader_config:
                dataset_name = dataloader_config.get('dataset_name', dataset_name)
                exclude_cols = dataloader_config.get('exclude_cols', exclude_cols)
                normalize_cols = dataloader_config.get('normalize_cols', normalize_cols)
                batch_size = dataloader_config.get('batch_size', batch_size)
                shuffle = dataloader_config.get('shuffle', shuffle)
                num_workers = dataloader_config.get('num_workers', num_workers)
                normalize = dataloader_config.get('normalize', normalize)
                return_labels = dataloader_config.get('return_labels', return_labels)
        
        self.dataset_name = dataset_name
        
        # Ensure exclude_cols is a list
        if isinstance(exclude_cols, str):
            self.exclude_cols = [exclude_cols]
        else:
            self.exclude_cols = list(exclude_cols)  # Convert to list and create a copy
            
        self.normalize_cols = normalize_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalize = normalize
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.return_labels = return_labels
        
        # If returning labels, make sure 'Class' is not in exclude_cols
        if self.return_labels and 'Class' in self.exclude_cols:
            self.exclude_cols.remove('Class')
            print(f"Removed 'Class' from exclude_cols because return_labels=True")
        
        # Load datasets if not provided
        self._load_datasets(train_dataset, val_dataset, test_dataset)
        
        # Calculate normalization statistics from training data
        if self.normalize:
            # Create a temporary dataset with all classes to calculate stats
            temp_dataset = CreditCardFraudDataset(
                dataset=self.train_dataset_raw,
                normalize=False,
                batch_size=batch_size,
                device=device,
                return_labels=False  # Don't need labels for stats calculation
            )
            self.stats_dict = temp_dataset._calculate_normalization_stats()
            print(f"Normalization statistics: {self.stats_dict}")
        else:
            self.stats_dict = None

        # Create dataset objects with proper normalization
        self.setup()
    
    def _load_datasets(self, train_dataset, val_dataset, test_dataset):
        """
        Load datasets from Hugging Face or use provided datasets.
        
        Args:
            train_dataset: Pre-loaded train dataset
            val_dataset: Pre-loaded validation dataset
            test_dataset: Pre-loaded test dataset
        """
        # Store raw datasets
        self.train_dataset_raw = train_dataset or load_dataset(self.dataset_name, split='train')
        self.val_dataset_raw = val_dataset or load_dataset(self.dataset_name, split='validation')
        try:
            self.test_dataset_raw = test_dataset or load_dataset(self.dataset_name, split='test')
        except:
            print("Test dataset not available. Using validation dataset as test.")
            self.test_dataset_raw = self.val_dataset_raw
    
    def setup(self):
        """
        Set up datasets with proper normalization.
        """
        # Create train dataset
        self.train_dataset = CreditCardFraudDataset(
            dataset=self.train_dataset_raw,
            exclude_cols=self.exclude_cols,
            normalize_cols=self.normalize_cols,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            normalize=self.normalize,
            stats_dict=self.stats_dict,
            device=self.device,
            return_labels=self.return_labels
        )
        
        # Create validation dataset
        self.val_dataset = CreditCardFraudDataset(
            dataset=self.val_dataset_raw,
            exclude_cols=self.exclude_cols,
            normalize_cols=self.normalize_cols,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.num_workers,
            normalize=self.normalize,
            stats_dict=self.stats_dict,  # Use the same stats as train
            device=self.device,
            return_labels=self.return_labels
        )
        
        # Create test dataset
        self.test_dataset = CreditCardFraudDataset(
            dataset=self.test_dataset_raw,
            exclude_cols=self.exclude_cols,
            normalize_cols=self.normalize_cols,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle test data
            num_workers=self.num_workers,
            normalize=self.normalize,
            stats_dict=self.stats_dict,  # Use the same stats as train
            device=self.device,
            return_labels=self.return_labels
        )
        
        # Create dataloaders
        self.train_dataloader = self.train_dataset.get_dataloader()
        self.val_dataloader = self.val_dataset.get_dataloader()
        self.test_dataloader = self.test_dataset.get_dataloader()
    
    def get_input_dim(self) -> int:
        """
        Get the input dimension for the model.
        
        Returns:
            Integer representing the number of input features
        """
        return self.train_dataset.get_input_dim()
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Get all dataloaders.
        
        Returns:
            Dictionary with keys 'train', 'val', 'test' and DataLoader values
        """
        return {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader
        }
    
    def get_stats_dict(self) -> Dict:
        """
        Get normalization statistics.
        
        Returns:
            Dictionary with normalization statistics
        """
        return self.stats_dict


def load_fraud_data(train_dataset=None,
                   val_dataset=None,
                   test_dataset=None,
                   dataset_name: str = "stanpony/european_credit_card_fraud_dataset",
                   exclude_cols: List[str] = ['original_index', 'Class'],
                   normalize_cols: List[str] = ['Time', 'Amount'],
                   batch_size: int = 64,
                   shuffle: bool = True,
                   num_workers: int = 4,
                   normalize: bool = True,
                   config_path: str = None,
                   return_labels: bool = False) -> Dict:
    """
    Convenience function to load all data and return loaders and metadata.
    
    Args:
        train_dataset: Pre-loaded train dataset (optional)
        val_dataset: Pre-loaded validation dataset (optional)
        test_dataset: Pre-loaded test dataset (optional)
        dataset_name: Name of the Hugging Face dataset
        exclude_cols: Columns to exclude from the input features
        normalize_cols: Columns to normalize
        batch_size: Batch size for DataLoaders
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for DataLoaders
        normalize: Whether to normalize the data
        config_path: Path to configuration file (optional)
        return_labels: Whether to return labels with features
        
    Returns:
        Dictionary with keys:
            - 'dataloaders': Dictionary of DataLoaders for each split
            - 'input_dim': Input dimension size
            - 'stats_dict': Normalization statistics
    """
    # Load config if provided
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded configuration from {config_path}")
        
        # Extract DataLoader config if present
        dataloader_config = config.get('DataLoader', {})
        
        # Extract VQVAE config to check if classification is enabled
        vqvae_config = config.get('VQVAE', {})
        classification_enabled = vqvae_config.get('classification', False)
        
        # If classification is enabled in the model config, we need to return labels
        if classification_enabled:
            return_labels = True
            print("Classification enabled in model config - ensuring labels are returned")
        
        # Override defaults with config values if present
        if dataloader_config:
            dataset_name = dataloader_config.get('dataset_name', dataset_name)
            exclude_cols = dataloader_config.get('exclude_cols', exclude_cols)
            normalize_cols = dataloader_config.get('normalize_cols', normalize_cols)
            batch_size = dataloader_config.get('batch_size', batch_size)
            shuffle = dataloader_config.get('shuffle', shuffle)
            num_workers = dataloader_config.get('num_workers', num_workers)
            normalize = dataloader_config.get('normalize', normalize)
            return_labels = dataloader_config.get('return_labels', return_labels)

    # Ensure 'Class' is not excluded if we need to return labels
    # Convert exclude_cols to a list if it's a string
    if isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
    else:
        exclude_cols = list(exclude_cols)  # Ensure it's a list and create a copy
        
    if return_labels and 'Class' in exclude_cols:
        exclude_cols.remove('Class')
        print("Removed 'Class' from exclude_cols because return_labels=True")

    data_module = FraudDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        dataset_name=dataset_name,
        exclude_cols=exclude_cols,
        normalize_cols=normalize_cols,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        normalize=normalize,
        return_labels=return_labels
    )
    
    return {
        'dataloaders': data_module.get_dataloaders(),
        'input_dim': data_module.get_input_dim(),
        'stats_dict': data_module.get_stats_dict()
    }


# Example usage
if __name__ == "__main__":
    # Path to configuration file
    config_path = 'configs/vq_vae/vq_vae_test.config'
    # Change to parent directory if running from VQ_VAE/
    if not os.path.exists(config_path) and os.path.exists(os.path.join('..', config_path)):
        os.chdir('..')
    
    # Example usage with configuration file
    if os.path.exists(config_path):
        data = load_fraud_data(config_path=config_path)
    else:
        # Example usage without configuration file
        data = load_fraud_data(batch_size=128)
    
    print(f"Input dimension: {data['input_dim']}")
    print(f"Normalization statistics: {data['stats_dict']}")
    
    # Iterate through one batch of each loader
    for split, loader in data['dataloaders'].items():
        batch_data = next(iter(loader))
        
        # Check if we're getting features and labels
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            features, labels = batch_data
            print(f"{split} features shape: {features.shape}")
            print(f"{split} labels shape: {labels.shape}")
            print(f"Total batches in {split}: {len(loader)}")
            # Print an example
            print(f"Example {split} batch features:")
            print(features[0])
            print(f"Example {split} batch label: {labels[0].item()}")
        else:
            features = batch_data
            print(f"{split} batch shape: {features.shape}")
            print(f"Total batches in {split}: {len(loader)}")
            # Print an example
            print(f"Example {split} batch:")
            print(features[0])
        
    # Print confirmation that we're using all classes
    print("DataLoader contains both fraud and non-fraud transactions")
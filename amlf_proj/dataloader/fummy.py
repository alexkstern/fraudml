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
                # Try to convert to int or float if possible
                try:
                    if '.' in value:
                        config_dict[section][key] = float(value)
                    else:
                        config_dict[section][key] = int(value)
                except ValueError:
                    # Handle lists and booleans
                    if value.lower() in ['true', 'yes']:
                        config_dict[section][key] = True
                    elif value.lower() in ['false', 'no']:
                        config_dict[section][key] = False
                    elif ',' in value:
                        # Assume it's a list
                        config_dict[section][key] = [item.strip() for item in value.split(',')]
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
                 class_filter: Optional[int] = None):  # Added class_filter parameter
        """
        Initialize the dataset.
        
        Args:
            dataset: Pre-loaded dataset (optional)
            dataset_name: Name of the Hugging Face dataset (used if dataset is None)
            split: Dataset split to use ('train', 'validation', or 'test')
            exclude_cols: Columns to exclude from the input features
            normalize_cols: Columns to normalize
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for DataLoader
            normalize: Whether to normalize the data
            stats_dict: Dictionary with normalization stats (used if provided)
            device: Device to use ('cuda' or 'cpu'). If None, will be automatically determined
            class_filter: Filter to specific class (0 for non-fraud, 1 for fraud, None for all)
        """
        self.split = split
        self.exclude_cols = exclude_cols
        self.normalize_cols = normalize_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalize = normalize
        self.class_filter = class_filter
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load the dataset if not provided
        if dataset is None:
            self.dataset = load_dataset(dataset_name, split=split)
        else:
            self.dataset = dataset
        
        # Filter by class if specified
        if self.class_filter is not None:
            # Filter dataset to only include samples of the specified class
            filtered_dataset = self.dataset.filter(lambda example: example['Class'] == self.class_filter)
            self.dataset = filtered_dataset
            print(f"Filtered dataset to class {self.class_filter}: {len(self.dataset)} samples")
            
        # Get all feature columns (excluding the ones in exclude_cols)
        self.feature_cols = [col for col in self.dataset.column_names if col not in exclude_cols]
        
        # Calculate or use provided normalization statistics
        self.stats_dict = stats_dict
        if self.normalize and self.stats_dict is None:
            self.stats_dict = self._calculate_normalization_stats()
    
    def _calculate_normalization_stats(self) -> Dict:
        """
        Calculate normalization statistics (mean and std) for specified columns.
        
        Returns:
            Dictionary with normalization statistics for each column
        """
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
        """
        Normalize a single value.
        
        Args:
            value: Value to normalize
            col: Column name
            
        Returns:
            Normalized value
        """
        if col in self.stats_dict:
            return (value - self.stats_dict[col]['mean']) / self.stats_dict[col]['std']
        return value
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Feature tensor (no label since we're only doing VAE training)
        """
        sample = self.dataset[idx]
        
        # Extract features and normalize if needed
        features = []
        for col in self.feature_cols:
            value = sample[col]
            if self.normalize and col in self.normalize_cols:
                value = self._normalize_value(value, col)
            features.append(value)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # For VAE training, we only need the features, not the labels
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
            pin_memory=(self.device == 'cuda')
        )
    
    def get_input_dim(self) -> int:
        """
        Get the input dimension (number of features).
        
        Returns:
            Integer representing the number of input features
        """
        return len(self.feature_cols)


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
                 class_filter: Optional[int] = None):  # Added class_filter parameter
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
            class_filter: Filter to specific class (0 for non-fraud, 1 for fraud, None for all)
        """
        # Load config if provided
        if config_path and os.path.exists(config_path):
            config = load_config(config_path)
            
            # Extract DataLoader config if present
            dataloader_config = config.get('DataLoader', {})
            
            # Override defaults with config values if present
            if dataloader_config:
                dataset_name = dataloader_config.get('dataset_name', dataset_name)
                exclude_cols = dataloader_config.get('exclude_cols', exclude_cols)
                normalize_cols = dataloader_config.get('normalize_cols', normalize_cols)
                batch_size = dataloader_config.get('batch_size', batch_size)
                shuffle = dataloader_config.get('shuffle', shuffle)
                num_workers = dataloader_config.get('num_workers', num_workers)
                normalize = dataloader_config.get('normalize', normalize)
                class_filter = dataloader_config.get('class_filter', class_filter)
        
        self.dataset_name = dataset_name
        self.exclude_cols = exclude_cols
        self.normalize_cols = normalize_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalize = normalize
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_filter = class_filter
        
        # Load datasets if not provided
        self._load_datasets(train_dataset, val_dataset, test_dataset)
        
        # Calculate normalization statistics from training data
        # Calculate normalization statistics from training data
        if self.normalize:
            # Create a temporary dataset with the same class filter to calculate stats
            temp_dataset = CreditCardFraudDataset(
                dataset=self.train_dataset_raw,
                normalize=False,
                batch_size=batch_size,
                device=device,
                class_filter=self.class_filter  # Use the same class filter
            )
            self.stats_dict = temp_dataset._calculate_normalization_stats()
            print(f"Normalization statistics (calculated from class {self.class_filter}): {self.stats_dict}")
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
            class_filter=self.class_filter
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
            class_filter=self.class_filter
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
            class_filter=self.class_filter
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
                   class_filter: Optional[int] = None) -> Dict:
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
        class_filter: Filter to specific class (0 for non-fraud, 1 for fraud, None for all)
        
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
        
        # Override defaults with config values if present
        if dataloader_config:
            dataset_name = dataloader_config.get('dataset_name', dataset_name)
            exclude_cols = dataloader_config.get('exclude_cols', exclude_cols)
            normalize_cols = dataloader_config.get('normalize_cols', normalize_cols)
            batch_size = dataloader_config.get('batch_size', batch_size)
            shuffle = dataloader_config.get('shuffle', shuffle)
            num_workers = dataloader_config.get('num_workers', num_workers)
            normalize = dataloader_config.get('normalize', normalize)
            class_filter = dataloader_config.get('class_filter', class_filter)

        
    
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
        class_filter=class_filter
    )
    
    return {
        'dataloaders': data_module.get_dataloaders(),
        'input_dim': data_module.get_input_dim(),
        'stats_dict': data_module.get_stats_dict()
    }


if __name__ == "__main__":
    # Build the config path relative to this file's directory
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "conv_vae", "fraud_conv_vae.config")
    
    # Load configuration from file
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    
    # Optionally, print a message to confirm loading the config
    print(f"Loaded configuration from: {config_path}")
    
    # Instantiate the data module and load data using the config file
    data = load_fraud_data(config_path=config_path)
    
    # Print key metadata
    print(f"Input dimension: {data['input_dim']}")
    print(f"Normalization statistics: {data['stats_dict']}")
    
    # Iterate through one batch of each dataloader and print its shape and number of batches
    for split, loader in data['dataloaders'].items():
        features = next(iter(loader))
        print(f"{split} batch shape: {features.shape}")
        print(f"{split} number of batches: {len(loader)}")

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
import configparser
import os

def load_config(config_path):
    """
    Load configuration from a .config file with support for multiple sections.
    
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
                value = value.strip()
                # Try to convert to int or float if possible
                try:
                    if '.' in value:
                        config_dict[section][key] = float(value)
                    else:
                        config_dict[section][key] = int(value)
                except ValueError:
                    if value.lower() in ['true', 'yes']:
                        config_dict[section][key] = True
                    elif value.lower() in ['false', 'no']:
                        config_dict[section][key] = False
                    elif ',' in value:
                        config_dict[section][key] = [item.strip() for item in value.split(',')]
                    else:
                        config_dict[section][key] = value
        
        return config_dict
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default configuration instead.")
        return {}

class CreditCardFraudClassificationDataset(Dataset):
    """
    Dataset class for credit card fraud detection classification.
    
    This class extends the basic dataset to keep track of the class labels
    while still performing the same normalization steps as the original.
    """
    def __init__(self, 
                 dataset=None,
                 dataset_name: str = "stanpony/european_credit_card_fraud_dataset",
                 split: str = 'train',
                 exclude_cols: List[str] = ['original_index'],
                 include_label_col: str = 'Class',
                 normalize_cols: List[str] = ['Time', 'Amount'],
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 normalize: bool = True,
                 stats_dict: Dict = None,
                 device: str = None,
                 conv: bool = False):
        """
        Initialize the classification dataset.
        
        Args:
            dataset: Pre-loaded dataset (optional)
            dataset_name: Name of the dataset to load from Hugging Face
            split: Dataset split ('train', 'validation', or 'test')
            exclude_cols: Columns to exclude from features (except label column)
            include_label_col: Column name for classification label
            normalize_cols: Columns to normalize
            batch_size: Batch size for dataloader
            shuffle: Whether to shuffle data
            num_workers: Number of workers for dataloader
            normalize: Whether to normalize data
            stats_dict: Precomputed statistics for normalization
            device: Device for computation
            conv: Whether to reshape for convolutional models
        """
        self.split = split
        self.exclude_cols = exclude_cols
        self.include_label_col = include_label_col
        self.normalize_cols = normalize_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalize = normalize
        self.conv = conv
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load the dataset if not provided
        if dataset is None:
            self.dataset = load_dataset(dataset_name, split=split)
        else:
            self.dataset = dataset
            
        # Store original labels
        self.labels = []
        for sample in self.dataset:
            self.labels.append(sample[self.include_label_col])
        
        # Get feature columns (excluding labels and specified columns)
        self.feature_cols = [col for col in self.dataset.column_names 
                            if col not in exclude_cols and col != self.include_label_col]
        
        # Calculate or use provided normalization statistics
        self.stats_dict = stats_dict
        if self.normalize and self.stats_dict is None:
            self.stats_dict = self._calculate_normalization_stats()
    
    def _calculate_normalization_stats(self) -> Dict:
        """
        Calculate normalization statistics (mean and std) for specified columns.
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
        """
        if col in self.stats_dict:
            return (value - self.stats_dict[col]['mean']) / self.stats_dict[col]['std']
        return value
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample and its label.
        
        Returns:
            Tuple of (features_tensor, label)
        """
        sample = self.dataset[idx]
        features = []
        
        for col in self.feature_cols:
            value = sample[col]
            if self.normalize and col in self.normalize_cols:
                value = self._normalize_value(value, col)
            # If the value is a list or array, extend the features list with its elements
            if isinstance(value, (list, np.ndarray)):
                # Ensure it's a flat list
                value = np.array(value).flatten().tolist()
                features.extend(value)
            else:
                features.append(value)
        
        # Get the label
        label = sample[self.include_label_col]
        
        # Return features and label as tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features_tensor, label_tensor
    
    def get_dataloader(self) -> DataLoader:
        """
        Create and return a DataLoader for this dataset.
        If conv mode is enabled, the features are reshaped to include a channel dimension.
        """
        def collate_fn(batch):
            # Separate features and labels
            features, labels = zip(*batch)
            
            # Stack features into a tensor of shape (batch, total_num_features)
            features_tensor = torch.stack(features)
            
            # If conv flag is True, reshape to get shape (batch, 1, total_num_features)
            if self.conv:
                features_tensor = features_tensor.view(features_tensor.size(0), 1, -1)
            
            # Stack labels into a tensor
            labels_tensor = torch.stack(labels)
            
            return features_tensor, labels_tensor
        
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
            collate_fn=collate_fn
        )
    
    def get_input_dim(self) -> int:
        """Get the input dimension for the model."""
        sample_features, _ = self.__getitem__(0)
        return sample_features.numel()
    
    def get_all_labels(self) -> List[int]:
        """Get all labels in the dataset."""
        return self.labels


class FraudClassificationDataModule:
    """
    Data module class to manage classification datasets and dataloaders.
    Provides access to train, validation, and test dataloaders with labels.
    """
    def __init__(self,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None,
                 dataset_name: str = "stanpony/european_credit_card_fraud_dataset",
                 exclude_cols: List[str] = ['original_index'],
                 include_label_col: str = 'Class',
                 normalize_cols: List[str] = ['Time', 'Amount'],
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 normalize: bool = True,
                 device: str = None,
                 config_path: str = None,
                 conv: bool = False):
        """
        Initialize the data module for classification.
        """
        # Load config if provided
        if config_path and os.path.exists(config_path):
            config = load_config(config_path)
            dataloader_config = config.get('DataLoader', {})
            if dataloader_config:
                dataset_name = dataloader_config.get('dataset_name', dataset_name)
                exclude_cols = dataloader_config.get('exclude_cols', exclude_cols)
                normalize_cols = dataloader_config.get('normalize_cols', normalize_cols)
                batch_size = dataloader_config.get('batch_size', batch_size)
                shuffle = dataloader_config.get('shuffle', shuffle)
                num_workers = dataloader_config.get('num_workers', num_workers)
                normalize = dataloader_config.get('normalize', normalize)
                conv = dataloader_config.get('conv', conv)
        
        self.dataset_name = dataset_name
        self.exclude_cols = exclude_cols
        self.include_label_col = include_label_col
        self.normalize_cols = normalize_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalize = normalize
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv = conv
        
        # Load datasets if not provided
        self._load_datasets(train_dataset, val_dataset, test_dataset)
        
        # Calculate normalization statistics from training data
        if self.normalize:
            temp_dataset = CreditCardFraudClassificationDataset(
                dataset=self.train_dataset_raw,
                normalize=False,
                batch_size=batch_size,
                device=device,
                conv=self.conv
            )
            self.stats_dict = temp_dataset._calculate_normalization_stats()
            print(f"Normalization statistics: {self.stats_dict}")
        else:
            self.stats_dict = None
        
        self.setup()
    
    def _load_datasets(self, train_dataset, val_dataset, test_dataset):
        """
        Load datasets from Hugging Face or use provided datasets.
        """
        self.train_dataset_raw = train_dataset or load_dataset(self.dataset_name, split='train')
        self.val_dataset_raw = val_dataset or load_dataset(self.dataset_name, split='validation')
        try:
            self.test_dataset_raw = test_dataset or load_dataset(self.dataset_name, split='test')
        except:
            print("Test dataset not available. Using validation dataset as test.")
            self.test_dataset_raw = self.val_dataset_raw
    
    def setup(self):
        """
        Set up classification datasets with proper normalization.
        """
        self.train_dataset = CreditCardFraudClassificationDataset(
            dataset=self.train_dataset_raw,
            exclude_cols=self.exclude_cols,
            include_label_col=self.include_label_col,
            normalize_cols=self.normalize_cols,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            normalize=self.normalize,
            stats_dict=self.stats_dict,
            device=self.device,
            conv=self.conv
        )
        
        self.val_dataset = CreditCardFraudClassificationDataset(
            dataset=self.val_dataset_raw,
            exclude_cols=self.exclude_cols,
            include_label_col=self.include_label_col,
            normalize_cols=self.normalize_cols,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            normalize=self.normalize,
            stats_dict=self.stats_dict,
            device=self.device,
            conv=self.conv
        )
        
        self.test_dataset = CreditCardFraudClassificationDataset(
            dataset=self.test_dataset_raw,
            exclude_cols=self.exclude_cols,
            include_label_col=self.include_label_col,
            normalize_cols=self.normalize_cols,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            normalize=self.normalize,
            stats_dict=self.stats_dict,
            device=self.device,
            conv=self.conv
        )
        
        self.train_dataloader = self.train_dataset.get_dataloader()
        self.val_dataloader = self.val_dataset.get_dataloader()
        self.test_dataloader = self.test_dataset.get_dataloader()
    
    def get_input_dim(self) -> int:
        """Get the input dimension for the model."""
        return self.train_dataset.get_input_dim()
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Get all dataloaders."""
        return {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader
        }
    
    def get_stats_dict(self) -> Dict:
        """Get normalization statistics."""
        return self.stats_dict
    
    def get_all_labels(self):
        """Get all labels for each dataset."""
        return {
            'train': self.train_dataset.get_all_labels(),
            'val': self.val_dataset.get_all_labels(),
            'test': self.test_dataset.get_all_labels()
        }


def load_fraud_classification_data(
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    dataset_name: str = "stanpony/european_credit_card_fraud_dataset",
    exclude_cols: List[str] = ['original_index'],
    include_label_col: str = 'Class',
    normalize_cols: List[str] = ['Time', 'Amount'],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    normalize: bool = True,
    config_path: str = None) -> Dict:
    """
    Convenience function to load all data and return loaders and metadata.
    """
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded configuration from {config_path}")
        dataloader_config = config.get('DataLoader', {})
        if dataloader_config:
            dataset_name = dataloader_config.get('dataset_name', dataset_name)
            exclude_cols = dataloader_config.get('exclude_cols', exclude_cols)
            normalize_cols = dataloader_config.get('normalize_cols', normalize_cols)
            batch_size = dataloader_config.get('batch_size', batch_size)
            shuffle = dataloader_config.get('shuffle', shuffle)
            num_workers = dataloader_config.get('num_workers', num_workers)
            normalize = dataloader_config.get('normalize', normalize)
            conv = dataloader_config.get('conv', False)
        else:
            conv = False
    else:
        conv = False
    
    data_module = FraudClassificationDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        dataset_name=dataset_name,
        exclude_cols=exclude_cols,
        include_label_col=include_label_col,
        normalize_cols=normalize_cols,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        normalize=normalize,
        conv=conv,
        config_path=config_path
    )
    
    return {
        'dataloaders': data_module.get_dataloaders(),
        'input_dim': data_module.get_input_dim(),
        'stats_dict': data_module.get_stats_dict(),
        'labels': data_module.get_all_labels()
    }


# Example usage
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "classifier", "classifier.config")
    
    data = load_fraud_classification_data(config_path=config_path)
    print(f"Input dimension: {data['input_dim']}")
    print(f"Normalization statistics: {data['stats_dict']}")
    
    # Check class distribution in each split
    for split, labels in data['labels'].items():
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"{split.capitalize()} set class distribution: {class_dist}")
    
    # Check data shapes
    for split, loader in data['dataloaders'].items():
        features, labels = next(iter(loader))
        print(f"{split} features shape: {features.shape}, labels shape: {labels.shape}")
        print(f"{split} number of batches: {len(loader)}")
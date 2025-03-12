import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Optional
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

class CreditCardFraudDataset(Dataset):
    """
    Dataset class for credit card fraud detection.
    
    This class handles loading data from Hugging Face datasets and preparing it for VAE training.
    It performs normalization on numerical features based on training data statistics.
    Optionally, if conv=True, it will later be collated to include a channel dimension.
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
                 class_filter: Optional[int] = None,
                 conv: bool = False):   # New parameter to indicate conv mode
        """
        Initialize the dataset.
        
        Args:
            conv: If True, then during batching an extra channel dimension will be added.
            (Other parameters as before)
        """
        self.split = split
        self.exclude_cols = exclude_cols
        self.normalize_cols = normalize_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalize = normalize
        self.class_filter = class_filter
        self.conv = conv  # Store the conv flag
        
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
    
    def __getitem__(self, idx: int) -> torch.Tensor:
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
        
        # Return a 1D tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return features_tensor


    
    def get_dataloader(self) -> DataLoader:
        """
        Create and return a DataLoader for this dataset.
        If conv mode is enabled, a custom collate function is used to unsqueeze the tensor.
        """
        def conv_collate_fn(batch):
            # Stack into a tensor of shape (batch, total_num_features)
            data = torch.stack(batch)
            # If conv flag is True and the tensor is 2D, unsqueeze to get shape (batch, 1, total_num_features)
            if self.conv and data.ndim == 2:
                data = data.unsqueeze(1)
            return data

        
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
            collate_fn=conv_collate_fn
        )
    
    def get_input_dim(self) -> int:
        # Compute the total number of scalars in one sample by checking __getitem__(0)
        sample = self.__getitem__(0)
        return sample.numel()


class FraudDataModule:
    """
    Data module class to manage datasets and dataloaders.
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
                 class_filter: Optional[int] = None,
                 conv: bool = False):   # New conv flag
        """
        Initialize the data module.
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
                class_filter = dataloader_config.get('class_filter', class_filter)
                conv = dataloader_config.get('conv', conv)
        
        self.dataset_name = dataset_name
        self.exclude_cols = exclude_cols
        self.normalize_cols = normalize_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalize = normalize
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_filter = class_filter
        self.conv = conv
        
        # Load datasets if not provided
        self._load_datasets(train_dataset, val_dataset, test_dataset)
        
        # Calculate normalization statistics from training data
        if self.normalize:
            temp_dataset = CreditCardFraudDataset(
                dataset=self.train_dataset_raw,
                normalize=False,
                batch_size=batch_size,
                device=device,
                class_filter=self.class_filter,
                conv=self.conv
            )
            self.stats_dict = temp_dataset._calculate_normalization_stats()
            print(f"Normalization statistics (calculated from class {self.class_filter}): {self.stats_dict}")
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
        Set up datasets with proper normalization.
        """
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
            class_filter=self.class_filter,
            conv=self.conv
        )
        
        self.val_dataset = CreditCardFraudDataset(
            dataset=self.val_dataset_raw,
            exclude_cols=self.exclude_cols,
            normalize_cols=self.normalize_cols,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            normalize=self.normalize,
            stats_dict=self.stats_dict,
            device=self.device,
            class_filter=self.class_filter,
            conv=self.conv
        )
        
        self.test_dataset = CreditCardFraudDataset(
            dataset=self.test_dataset_raw,
            exclude_cols=self.exclude_cols,
            normalize_cols=self.normalize_cols,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            normalize=self.normalize,
            stats_dict=self.stats_dict,
            device=self.device,
            class_filter=self.class_filter,
            conv=self.conv
        )
        
        self.train_dataloader = self.train_dataset.get_dataloader()
        self.val_dataloader = self.val_dataset.get_dataloader()
        self.test_dataloader = self.test_dataset.get_dataloader()
    
    def get_input_dim(self) -> int:
        """
        Get the input dimension for the model.
        """
        return self.train_dataset.get_input_dim()
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Get all dataloaders.
        """
        return {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader
        }
    
    def get_stats_dict(self) -> Dict:
        """
        Get normalization statistics.
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
            class_filter = dataloader_config.get('class_filter', class_filter)
            conv = dataloader_config.get('conv', False)
        else:
            conv = False
    else:
        conv = False
    
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
        class_filter=class_filter,
        conv=conv
    )
    
    return {
        'dataloaders': data_module.get_dataloaders(),
        'input_dim': data_module.get_input_dim(),
        'stats_dict': data_module.get_stats_dict()
    }

# Example usage of the dataloader script (this part runs when executing the script directly)
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "conv_vae", "fraud_conv_vae.config")
    data = load_fraud_data(config_path=config_path)
    print(f"Input dimension: {data['input_dim']}")
    print(f"Normalization statistics: {data['stats_dict']}")
    
    for split, loader in data['dataloaders'].items():
        features = next(iter(loader))
        print(f"{split} batch shape: {features.shape}")
        print(f"{split} number of batches: {len(loader)}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
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


class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for VQ-VAE
    
    Implementation of the vector quantization from the paper
    "Neural Discrete Representation Learning" (https://arxiv.org/abs/1711.00937)
    """
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 epsilon: float = 1e-5):
        """
        Initialize the Vector Quantizer.
        
        Args:
            num_embeddings: Number of embedding vectors (size of codebook)
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for commitment loss
            decay: Decay factor for exponential moving average (EMA) updates
            epsilon: Small constant to avoid division by zero
        """
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # EMA update parameters
        self.decay = decay
        self.epsilon = epsilon
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))
        
        # Initialize the embedding vectors
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through the Vector Quantizer.
        
        Args:
            inputs: Input tensor [B, D, ...]
            
        Returns:
            quantized: Quantized tensor [B, D, ...]
            loss: VQ loss
            info: Dictionary with metrics
        """
        # Save input shape
        input_shape = inputs.shape
        
        # Flatten inputs to [N, D] where N is batch size * flattened spatial dims
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances between inputs and embedding vectors
        # |x - e_j|^2 = |x|^2 + |e_j|^2 - 2 * x^T * e_j
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))
        
        # Get indices of nearest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize and unflatten
        quantized = self.embedding(encoding_indices).reshape(input_shape)
        
        # Calculate loss
        # Codebook loss: move codebook vectors closer to encoder output
        # Commitment loss: prevent encoder output from fluctuating too much
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        # Pass gradients from quantized to inputs
        quantized = inputs + (quantized - inputs).detach()
        
        # EMA update (only during training)
        if self.training:
            # Calculate new cluster sizes
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            cluster_size = encodings.sum(0)
            
            # Update EMA tracking of cluster sizes
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * cluster_size
            
            # Update EMA tracking of embeddings
            new_cluster_sum = torch.matmul(encodings.t(), flat_inputs)
            self.ema_w = self.ema_cluster_size[:, None] * self.embedding.weight.data.clone()
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * new_cluster_sum
            
            # Update embeddings
            n = torch.sum(self.ema_cluster_size)
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)
        
        # Calculate perplexity (measure of codebook usage)
        avg_probs = torch.mean(F.one_hot(encoding_indices, self.num_embeddings).float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Return quantized representations and metrics
        return quantized, loss, {"perplexity": perplexity, "encodings": encoding_indices}


class Encoder(nn.Module):
    """
    Encoder network for VQ-VAE
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int],
                 latent_dim: int,
                 dropout_rate: float = 0.1):
        """
        Initialize the encoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout_rate: Dropout rate
        """
        super(Encoder, self).__init__()
        
        # Build encoder layers
        layers = []
        in_features = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(in_features, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder network for VQ-VAE
    """
    def __init__(self, 
                 latent_dim: int, 
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout_rate: float = 0.1):
        """
        Initialize the decoder.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output features
            dropout_rate: Dropout rate
        """
        super(Decoder, self).__init__()
        
        # Build decoder layers
        layers = []
        in_features = latent_dim
        
        # Reverse the list of hidden dimensions for the decoder
        hidden_dims = hidden_dims[::-1]
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        
        # Final layer to output space
        layers.append(nn.Linear(in_features, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder."""
        return self.decoder(x)


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) for tabular data
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64],
                 latent_dim: int = 32,
                 num_embeddings: int = 512,
                 embedding_dim: int = 32,
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 dropout_rate: float = 0.1,
                 config_path: Optional[str] = None,
                 classification: bool = False,
                 num_classes: int = 2):
        """
        Initialize the VQ-VAE model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space (before quantization)
            num_embeddings: Number of embedding vectors (codebook size)
            embedding_dim: Dimension of each embedding vector (must match latent_dim)
            commitment_cost: Weight for commitment loss in vector quantizer
            decay: Decay factor for EMA updates
            dropout_rate: Dropout rate for encoder and decoder
            config_path: Path to configuration file (optional)
        """
        super(VQVAE, self).__init__()
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            config = load_config(config_path)
            
            # Extract VQ-VAE config if present
            vqvae_config = config.get('VQVAE', {})
            
            # Override defaults with config values if present
            if vqvae_config:
                hidden_dims = vqvae_config.get('hidden_dims', hidden_dims)
                latent_dim = vqvae_config.get('latent_dim', latent_dim)
                num_embeddings = vqvae_config.get('num_embeddings', num_embeddings)
                embedding_dim = vqvae_config.get('embedding_dim', embedding_dim)
                commitment_cost = vqvae_config.get('commitment_cost', commitment_cost)
                decay = vqvae_config.get('decay', decay)
                dropout_rate = vqvae_config.get('dropout_rate', dropout_rate)
        
        # Check that latent_dim matches embedding_dim
        assert latent_dim == embedding_dim, "Latent dimension must match embedding dimension"
        
        # Initialize model components
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout_rate)
        self.vector_quantizer = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost, decay
        )
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, dropout_rate)
        
        # Save model parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
      # Classification head (optional)
        self.classification = classification  # Store flag
        if classification:
            self.classifier = nn.Linear(embedding_dim, num_classes)  # Classifier from quantized embeddings
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Encode inputs to quantized latent representations.
        
        Args:
            x: Input tensor [B, D]
            
        Returns:
            quantized: Quantized latent tensor
            vq_loss: Vector quantization loss
            info: Dictionary with additional information
        """
        # Encode inputs to latent representations
        z = self.encoder(x)
        
        # Reshape for vector quantizer if needed (for 1D data, this is a no-op)
        z = z.reshape(-1, self.embedding_dim)
        
        # Quantize latent representations
        quantized, vq_loss, info = self.vector_quantizer(z)
        
        return quantized, vq_loss, info
    
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent representations to reconstructions.
        
        Args:
            quantized: Quantized latent tensor
            
        Returns:
            reconstructions: Reconstructed tensor
        """
        return self.decoder(quantized)
    
    #def forward(self, x: torch.Tensor) -> Dict:
        """
        Forward pass through the VQ-VAE.
        
        Args:
            x: Input tensor [B, D]
            
        Returns:
            Dictionary with:
                - reconstructions: Reconstructed outputs
                - quantized: Quantized latent representations
                - vq_loss: Vector quantization loss
                - reconstruction_loss: Reconstruction loss (MSE)
                - perplexity: Perplexity of the codebook usage
                - encodings: Indices of the selected codebook entries
        """
        # Encode inputs to quantized latent representations
    #    quantized, vq_loss, info = self.encode(x)
        
        # Decode quantized representations to reconstructions
    #    reconstructions = self.decode(quantized)
        
        # Calculate reconstruction loss
     #   reconstruction_loss = F.mse_loss(reconstructions, x)
        
        # Return all outputs and metrics
    #    return {
    #        "reconstructions": reconstructions,
    #        "quantized": quantized,
    #        "vq_loss": vq_loss,
    #        "reconstruction_loss": reconstruction_loss,
    #        "loss": reconstruction_loss + vq_loss,
    #        "perplexity": info["perplexity"],
    #        "encodings": info["encodings"]
    #    }
   
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass through the VQ-VAE.
        
        Args:
            x: Input tensor [B, D]
            labels: Ground truth labels [B] (only needed if classification=True)
            
        Returns:
            Dictionary with:
                - reconstructions: Reconstructed outputs
                - quantized: Quantized latent representations
                - vq_loss: Vector quantization loss
                - reconstruction_loss: Reconstruction loss (MSE)
                - classification_logits (if classification=True)
                - classification_loss (if classification=True and labels provided)
                - loss: Total loss (sum of all active losses)
        """
        quantized, vq_loss, info = self.encode(x)  # Encode & Quantize
        reconstructions = self.decode(quantized)  # Decode

        # Calculate reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructions, x)

        loss = reconstruction_loss + vq_loss  # Base loss

        results = {
            "reconstructions": reconstructions,
            "quantized": quantized,
            "vq_loss": vq_loss,
            "reconstruction_loss": reconstruction_loss,
            "loss": loss,
            "perplexity": info["perplexity"],
            "encodings": info["encodings"]
        }

        # If classification is enabled, classify from quantized embeddings
        if self.classification:
            classification_logits = self.classifier(quantized)  # Pass through classifier
            results["classification_logits"] = classification_logits

            # If labels are provided, compute classification loss
            if labels is not None:
                classification_loss = F.cross_entropy(classification_logits, labels)
                results["classification_loss"] = classification_loss
                results["loss"] += classification_loss  # Add to total loss

        return results

    
    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample random vectors from the codebook and decode them.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to use for tensors
            
        Returns:
            Generated samples
        """
        # Sample random indices from the codebook
        indices = torch.randint(0, self.num_embeddings, (num_samples,), device=device)
        
        # Get quantized vectors from the codebook
        quantized = self.vector_quantizer.embedding(indices)
        
        # Decode quantized vectors to samples
        samples = self.decode(quantized)
        
        return samples
    
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode inputs directly to codebook indices.
        
        Args:
            x: Input tensor [B, D]
            
        Returns:
            indices: Codebook indices for each input
        """
        # Encode inputs to latent representations
        z = self.encoder(x)
        
        # Calculate distances between inputs and embedding vectors
        distances = (torch.sum(z**2, dim=1, keepdim=True) 
                    + torch.sum(self.vector_quantizer.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z, self.vector_quantizer.embedding.weight.t()))
        
        # Get indices of nearest embedding vectors
        indices = torch.argmin(distances, dim=1)
        
        return indices
    
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from codebook indices directly.
        
        Args:
            indices: Codebook indices
            
        Returns:
            reconstructions: Reconstructed outputs
        """
        # Get quantized vectors from the codebook
        quantized = self.vector_quantizer.embedding(indices)
        
        # Decode quantized vectors to reconstructions
        reconstructions = self.decode(quantized)
        
        return reconstructions
    
    def classify_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Classify inputs based on codebook indices.
        
        Args:
            indices: Codebook indices
        """

    
    def calculate_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate anomaly scores based on reconstruction error.
        
        Args:
            x: Input tensor [B, D]
            
        Returns:
            anomaly_scores: Anomaly scores for each input
        """
        # Get reconstructions
        with torch.no_grad():
            outputs = self.forward(x)
            reconstructions = outputs["reconstructions"]
        
        # Calculate anomaly scores as MSE per sample
        anomaly_scores = torch.mean((x - reconstructions) ** 2, dim=1)
        
        return anomaly_scores
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """
        Load model from file.
        
        Args:
            path: Path to saved model
            device: Device to load model to
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"], 
            num_embeddings=checkpoint["num_embeddings"],
            embedding_dim=checkpoint["embedding_dim"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

"""
# Example usage
if __name__ == "__main__":
    # Create a sample input
    batch_size = 64
    input_dim = 30  # Typical for credit card data with ~30 features
    x = torch.randn(batch_size, input_dim)
    

    # Path to configuration file
    config_path = 'configs/vq_vae/vq_vae_test.config'
    os.chdir('..')
    # Initialize model
    if os.path.exists(config_path):
        model = VQVAE(input_dim=input_dim, classification=False,config_path=config_path)
        print(f"Created VQ-VAE model from config: {config_path}")
    else:
        model = VQVAE(input_dim=input_dim)
        print("Created VQ-VAE model with default parameters")
    
    # Forward pass
    outputs = model(x)
    
    # Print output shapes and metrics
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {outputs['reconstructions'].shape}")
    print(f"Quantized shape: {outputs['quantized'].shape}")
    print(f"VQ Loss: {outputs['vq_loss'].item():.4f}")
    print(f"Reconstruction Loss: {outputs['reconstruction_loss'].item():.4f}")
    print(f"Perplexity: {outputs['perplexity'].item():.2f} (out of {model.num_embeddings})")
    
    # Sample from model
    samples = model.sample(5)
    print(f"Generated samples shape: {samples.shape}")
    
    
    # Calculate anomaly scores
    anomaly_scores = model.calculate_anomaly_scores(x)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Average anomaly score: {anomaly_scores.mean().item():.4f}")"
"""

# Example usage
if __name__ == "__main__":
    # Create a sample input with random data
    batch_size = 64
    input_dim = 30  # Typical for credit card data with ~30 features
    x = torch.randn(batch_size, input_dim)
    
    # Path to configuration file
    config_path = 'configs/vq_vae/vq_vae_test.config'
    os.chdir('..')
    # Initialize model
    if os.path.exists(config_path):
        model = VQVAE(input_dim=input_dim, classification=False, config_path=config_path)
        print(f"Created VQ-VAE model from config: {config_path}")
    else:
        model = VQVAE(input_dim=input_dim)
        print("Created VQ-VAE model with default parameters")
    
    # Forward pass with random data
    outputs = model(x)
    
    # Print output shapes and metrics for random data
    print(f"Random Input shape: {x.shape}")
    print(f"Reconstruction shape: {outputs['reconstructions'].shape}")
    print(f"Quantized shape: {outputs['quantized'].shape}")
    print(f"VQ Loss: {outputs['vq_loss'].item():.4f}")
    print(f"Reconstruction Loss: {outputs['reconstruction_loss'].item():.4f}")
    print(f"Perplexity: {outputs['perplexity'].item():.2f} (out of {model.num_embeddings})")
    
    # Sample from model
    samples = model.sample(5)
    print(f"Generated samples shape: {samples.shape}")
    
    # Calculate anomaly scores on random data
    anomaly_scores = model.calculate_anomaly_scores(x)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Average anomaly score: {anomaly_scores.mean().item():.4f}")
    
    try_model_on_real_data = True
    if try_model_on_real_data:
        # ---------------------------------------------
        # Now import the dataloader and do one forward pass on a real batch,
        # using configuration parameters from the config file.
        # ---------------------------------------------
        from vq_dataloader import load_fraud_data  # adjust the module name/path as needed

        # Path to configuration file
        config_path = 'configs/vq_vae/vq_vae_test.config'
        
        # Load data using the configuration file (parameters such as batch_size, return_labels, etc. are overridden by config)
        data_info = load_fraud_data(config_path=config_path)
        dataloaders = data_info['dataloaders']
        train_loader = dataloaders['train']

        # Get one batch from the train DataLoader
        batch = next(iter(train_loader))
        if isinstance(batch, (tuple, list)):
            features, labels = batch  # If labels are included
        else:
            features = batch

        # Move features to the same device as the model if necessary
        features = features.to(next(model.parameters()).device)

        # Forward pass on the real data batch
        outputs_data = model(features)

        # Print output shapes and metrics for the data batch
        print("Forward pass on real data batch:")
        print(f"Data Input shape: {features.shape}")
        print(f"Reconstruction shape: {outputs_data['reconstructions'].shape}")
        print(f"VQ Loss: {outputs_data['vq_loss'].item():.4f}")
        print(f"Reconstruction Loss: {outputs_data['reconstruction_loss'].item():.4f}")
        print(f"Perplexity: {outputs_data['perplexity'].item():.2f} (out of {model.num_embeddings})")

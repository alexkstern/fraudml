import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser

class VectorQuantizer(nn.Module):
    """
    A simple vector quantizer module for VQ-VAE.
    It maps continuous latent representations to the nearest embedding vector in a codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize the codebook embeddings (shape: [num_embeddings, embedding_dim])
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)
    
    def forward(self, inputs):
        # inputs: (batch, sequence_length, embedding_dim)
        # Flatten input: (batch * sequence_length, embedding_dim)
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Compute squared L2 distance between flat_input and codebook embeddings:
        distances = (flat_input.pow(2).sum(1, keepdim=True) +
                     self.embedding.weight.pow(2).sum(1) -
                     2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find the nearest code for each latent vector:
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)
        
        # Reshape quantized vectors back to original shape:
        quantized = quantized.view_as(inputs)
        
        # Compute the losses:
        # Codebook loss: encourage the codebook embeddings to move toward encoder outputs
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        # Commitment loss: encourage the encoder outputs not to fluctuate too much
        commitment_loss = F.mse_loss(inputs, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator:
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss

class TransformerVQVAE(nn.Module):
    """
    A Transformer-based VQ-VAE for tabular data.
    The model uses:
      - A transformer encoder to embed and process the input sequence.
      - A vector quantizer to discretize the latent representations.
      - A transformer "decoder" (built from encoder layers) to reconstruct the input.
    The expected input shape is (batch, 1, input_dim) where input_dim is the number of features.
    """
    def __init__(self, config):
        super().__init__()
        # Basic parameters
        self.input_dim = config.getint("input_dim")    # e.g., 30 features
        self.hidden_dim = config.getint("hidden_dim")    # (unused here but provided in config)
        self.recon_weight = config.getfloat("recon_weight")
        
        # VQ-VAE specific parameters
        self.vq_commitment_cost = config.getfloat("vq_commitment_cost", fallback=0.25)
        self.num_embeddings = config.getint("num_embeddings", fallback=512)
        self.embedding_dim = config.getint("embedding_dim", fallback=64)  # should equal d_model
        
        # Transformer hyperparameters
        self.d_model = config.getint("d_model", fallback=64)
        self.nhead = config.getint("nhead", fallback=8)
        self.encoder_layers = config.getint("encoder_layers", fallback=2)
        self.encoder_ff_dim = config.getint("encoder_ff_dim", fallback=128)
        self.decoder_layers = config.getint("decoder_layers", fallback=2)
        self.decoder_ff_dim = config.getint("decoder_ff_dim", fallback=128)
        
        # ------------------------------
        # Encoder: Embed and process input sequence
        # ------------------------------
        # Input: (batch, 1, input_dim) → Transpose to (batch, input_dim, 1)
        # Then each scalar feature is embedded to a d_model-dimensional token.
        self.embedding = nn.Linear(1, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.input_dim, self.d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.encoder_ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.encoder_layers
        )
        
        # ------------------------------
        # Vector Quantizer: Discretize latent representations
        # ------------------------------
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=self.vq_commitment_cost
        )
        
        # ------------------------------
        # Decoder: Reconstruct from quantized latent representations
        # ------------------------------
        # The decoder processes the quantized sequence (batch, input_dim, d_model)
        # with an added positional encoding.
        self.dec_pos_embedding = nn.Parameter(torch.zeros(1, self.input_dim, self.d_model))
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.decoder_ff_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=self.decoder_layers
        )
        # Map each token back to a scalar
        self.output_layer = nn.Linear(self.d_model, 1)
    
    def encode(self, x):
        """
        Encoder forward pass.
        x: tensor of shape (batch, 1, input_dim)
        Returns latent representations of shape (batch, input_dim, d_model)
        """
        x = x.transpose(1, 2)          # -> (batch, input_dim, 1)
        x = self.embedding(x)          # -> (batch, input_dim, d_model)
        x = x + self.pos_embedding     # add positional encoding
        x = self.transformer_encoder(x)  # -> (batch, input_dim, d_model)
        return x
    
    def decode(self, quantized):
        """
        Decoder forward pass.
        quantized: (batch, input_dim, d_model)
        Returns reconstruction of shape (batch, 1, input_dim)
        """
        x = quantized + self.dec_pos_embedding  # add decoder positional encoding
        x = self.transformer_decoder(x)         # -> (batch, input_dim, d_model)
        x = self.output_layer(x)                # -> (batch, input_dim, 1)
        x = x.transpose(1, 2)                   # -> (batch, 1, input_dim)
        return x
    
    def forward(self, x):
        """
        Full forward pass: encode, quantize, and decode.
        Returns:
          - recon_x: reconstruction (batch, 1, input_dim)
          - vq_loss: the vector quantization loss (scalar)
        """
        encoded = self.encode(x)                # (batch, input_dim, d_model)
        quantized, vq_loss = self.vector_quantizer(encoded)
        recon_x = self.decode(quantized)
        return recon_x, vq_loss

def vqvae_loss_function(recon_x, x, vq_loss, recon_weight=1.0):
    """
    Computes the VQ-VAE loss as the weighted sum of reconstruction loss and VQ loss.
    Here we use mean-squared-error for reconstruction.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    return recon_weight * recon_loss + vq_loss

def print_num_params(model):
    """
    Prints the total number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    # Load configuration from file
    config_path = os.path.join(os.path.dirname(__file__), "..","configs", "transformer_vqvae", "fraud_transformer_vqvae.config")
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    
    # Use the section for Transformer VQVAE parameters.
    # Ensure your config file has a section named [Transformer_VQVAE]
    vqvae_config = config_parser["Transformer_VQVAE"]
    
    # Instantiate the model using the configuration
    model = TransformerVQVAE(vqvae_config)
    print_num_params(model)
    
    # Retrieve training hyperparameters from the Trainer section
    train_config = config_parser["Trainer"]
    lr = train_config.getfloat("lr", fallback=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create a dummy input: batch of 8 samples, shape (8, 1, input_dim)
    dummy_input = torch.randn(8, 1, model.input_dim)
    recon, vq_loss = model(dummy_input)
    
    loss = vqvae_loss_function(recon, dummy_input, vq_loss,
                               recon_weight=model.recon_weight)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Reconstruction shape:", recon.shape)  # Expected: (8, 1, input_dim)
    print("Loss:", loss.item())

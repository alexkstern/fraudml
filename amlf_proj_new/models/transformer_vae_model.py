import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser

class TransformerVae(nn.Module):
    """
    A Variational Autoencoder (VAE) that uses a transformer-based encoder and decoder.
    The input is expected to have shape (batch, 1, input_dim) where input_dim is the sequence length.
    """
    def __init__(self, config):
        """
        Initialize the model from the configuration.d
        The config should contain keys for:
          - input_dim, hidden_dim, z_dim, recon_weight, kl_weight
          - Transformer-specific: d_model, nhead, encoder_layers, encoder_ff_dim, decoder_layers, decoder_ff_dim
        """
        super().__init__()
        # Basic parameters
        self.input_dim    = config.getint("input_dim")     # e.g., 30
        self.hidden_dim   = config.getint("hidden_dim")      # e.g., 32
        self.z_dim        = config.getint("z_dim")           # e.g., 3
        self.recon_weight = config.getfloat("recon_weight")  # e.g., 1.0
        self.kl_weight    = config.getfloat("kl_weight")     # e.g., 1.0

        # Transformer hyperparameters (with fallbacks)
        self.d_model         = config.getint("d_model", fallback=64)
        self.nhead           = config.getint("nhead", fallback=8)
        self.encoder_layers  = config.getint("encoder_layers", fallback=2)
        self.encoder_ff_dim  = config.getint("encoder_ff_dim", fallback=128)
        self.decoder_layers  = config.getint("decoder_layers", fallback=2)
        self.decoder_ff_dim  = config.getint("decoder_ff_dim", fallback=128)

        # ------------------------------
        # Encoder: Embed and process input sequence
        # ------------------------------
        # The input shape is (batch, 1, input_dim); first transpose to (batch, input_dim, 1)
        # then embed each scalar to a d_model-dimensional token.
        self.embedding = nn.Linear(1, self.d_model)
        # Learnable positional encoding (shape: (1, input_dim, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.input_dim, self.d_model))
        
        # Build the transformer encoder (using PyTorch's transformer encoder layer)
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
        
        # After the transformer encoder, perform mean pooling over the sequence dimension
        # and pass through a FC layer to get a hidden representation.
        self.enc_fc = nn.Linear(self.d_model, self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # ------------------------------
        # Decoder: Expand latent vector into a sequence and process with transformer blocks
        # ------------------------------
        # Expand latent vector to (batch, input_dim * d_model) then reshape to (batch, input_dim, d_model)
        self.dec_fc = nn.Linear(self.z_dim, self.input_dim * self.d_model)
        # Learnable positional embedding for the decoder sequence
        self.dec_pos_embedding = nn.Parameter(torch.zeros(1, self.input_dim, self.d_model))
        
        # Instead of using a traditional TransformerDecoder (which requires memory),
        # we use a stack of transformer encoder layers as a self-attention decoder.
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
        # Final linear layer to map each token back to a scalar output
        self.output_layer = nn.Linear(self.d_model, 1)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        Encoder forward pass.
        x: tensor of shape (batch, 1, input_dim)
        """
        # Rearrange to (batch, input_dim, 1)
        x = x.transpose(1, 2)
        # Embed each scalar into a d_model-dimensional token
        x = self.embedding(x)  # (batch, input_dim, d_model)
        # Add positional encoding
        x = x + self.pos_embedding  # (batch, input_dim, d_model)
        # Process through transformer encoder
        x = self.transformer_encoder(x)  # (batch, input_dim, d_model)
        # Mean pool over sequence tokens
        x = x.mean(dim=1)  # (batch, d_model)
        # Map to hidden representation
        x = F.relu(self.enc_fc(x))  # (batch, hidden_dim)
        mu = self.fc_mu(x)          # (batch, z_dim)
        logvar = self.fc_logvar(x)  # (batch, z_dim)
        return mu, logvar

    def decode(self, z):
        """
        Decoder forward pass.
        z: latent vector of shape (batch, z_dim)
        """
        batch_size = z.size(0)
        # Expand latent vector to a sequence representation
        x = F.relu(self.dec_fc(z))  # (batch, input_dim * d_model)
        x = x.view(batch_size, self.input_dim, self.d_model)  # (batch, input_dim, d_model)
        # Add decoder positional encoding
        x = x + self.dec_pos_embedding  # (batch, input_dim, d_model)
        # Process through transformer (decoder) block(s)
        x = self.transformer_decoder(x)  # (batch, input_dim, d_model)
        # Map each token to output dimension (1)
        x = self.output_layer(x)  # (batch, input_dim, 1)
        # Transpose to (batch, 1, input_dim) for consistency with input
        x = x.transpose(1, 2)

        return x

    def forward(self, x):
        """
        Full forward pass: encode -> reparameterize -> decode.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, recon_weight=1.0, kl_weight=1.0):
    """
    Computes the VAE loss as the weighted sum of reconstruction loss and KL divergence.
    Uses mean-squared-error for reconstruction loss.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_weight * recon_loss + kl_weight * kld

def print_num_params(model):
    """
    Prints the number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")


# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    # Load configuration from file
    config_path = os.path.join(os.path.dirname(__file__), "..","configs", "transformer_vae", "fraud_transformer_vae.config")
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    
    # Use the section for transformer VAE parameters.
    # (Ensure your config file has a section [Transformer_VAE])
    transformer_config = config_parser["Transformer_VAE"]

    # Instantiate the model using the configuration
    model = TransformerVae(transformer_config)
    print_num_params(model)

    # Use Trainer config for training hyperparameters (e.g., learning rate)
    train_config = config_parser["Trainer"]
    lr = train_config.getfloat("lr", fallback=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create a dummy input: batch of 8 samples, shape (8, 1, input_dim)
    dummy_input = torch.randn(8, 1, model.input_dim)
    recon, mu, logvar = model(dummy_input)
    
    loss = vae_loss_function(recon, dummy_input, mu, logvar,
                             recon_weight=model.recon_weight,
                             kl_weight=model.kl_weight)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Reconstruction shape:", recon.shape)  # Expected shape: (8, 1, input_dim)
    print("Loss:", loss.item())

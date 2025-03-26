import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser

class VectorQuantizer(nn.Module):
    """
    A simple vector quantizer module for VQ-VAE.
    It maps continuous latent representations to the nearest embedding vector in a learned codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: embeddings of shape (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)
    
    def forward(self, inputs):
        # inputs: (batch, z_dim) --- here we assume z_dim == embedding_dim
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Compute squared Euclidean distances between encoder outputs and codebook embeddings
        distances = (flat_input.pow(2).sum(1, keepdim=True) +
                     self.embedding.weight.pow(2).sum(1) -
                     2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find the nearest code for each latent vector
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)
        
        # Compute the losses:
        # 1. Codebook loss: move codebook embeddings towards encoder outputs
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        # 2. Commitment loss: encourage encoder outputs to commit to a codebook vector
        commitment_loss = F.mse_loss(inputs, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: pass gradients to encoder
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss

class ConvVQVAE(nn.Module):
    """
    A configurable 1D Convolutional VQ-VAE that reads its hyperparameters from a config file.
    The input is expected to be of shape (batch_size, 1, input_dim).
    """
    def __init__(self, config):
        """
        config: a configparser section (e.g., config['Conv_VQVAE'])
        """
        super().__init__()
        # Read basic parameters
        self.input_dim    = config.getint("input_dim")       # e.g., 30
        self.hidden_dim   = config.getint("hidden_dim")        # e.g., 32 (for the FC layer)
        self.z_dim        = config.getint("z_dim")             # e.g., 3
        self.recon_weight = config.getfloat("recon_weight")    # e.g., 1.0
        
        # Vector quantization parameters
        self.vq_commitment_cost = config.getfloat("vq_commitment_cost", fallback=0.25)
        self.num_embeddings   = config.getint("num_embeddings", fallback=512)
        # Ensure that the embedding dimension is compatible with the latent dimension.
        self.embedding_dim    = config.getint("embedding_dim", fallback=self.z_dim)
        
        # Read convolution hyperparameters (provided as comma-separated values)
        self.enc_conv_channels = [int(x) for x in config.get("enc_conv_channels").split(",") if x.strip()]
        self.dec_conv_channels = [int(x) for x in config.get("dec_conv_channels").split(",") if x.strip()]
        self.conv_kernel_size  = config.getint("conv_kernel_size", fallback=3)
        self.conv_stride_enc2  = config.getint("conv_stride_enc2", fallback=2)
        self.conv_stride_dec2  = config.getint("conv_stride_dec2", fallback=2)
        
        # For "same" padding with kernel size k, we use p = (k - 1) // 2
        p = (self.conv_kernel_size - 1) // 2

        # ------------------------------
        # Build the Encoder
        # ------------------------------
        # Input: (batch, 1, input_dim)
        self.enc_conv1 = nn.Conv1d(in_channels=1,
                                   out_channels=self.enc_conv_channels[0],
                                   kernel_size=self.conv_kernel_size,
                                   stride=1,
                                   padding=p)
        self.enc_conv2 = nn.Conv1d(in_channels=self.enc_conv_channels[0],
                                   out_channels=self.enc_conv_channels[1],
                                   kernel_size=self.conv_kernel_size,
                                   stride=self.conv_stride_enc2,
                                   padding=p)
        self.enc_conv3 = nn.Conv1d(in_channels=self.enc_conv_channels[1],
                                   out_channels=self.enc_conv_channels[2],
                                   kernel_size=self.conv_kernel_size,
                                   stride=1,
                                   padding=p)
        
        # Calculate the output length after the conv layers:
        # First layer: stride=1 → length remains input_dim.
        L1 = self.input_dim
        # Second layer: using formula L_out = floor((L_in + 2*p - kernel_size)/stride + 1)
        L2 = (L1 + 2*p - self.conv_kernel_size) // self.conv_stride_enc2 + 1
        # Third layer: stride=1 → L3 = (L2 + 2*p - kernel_size) // 1 + 1
        L3 = (L2 + 2*p - self.conv_kernel_size) // 1 + 1
        self.conv_out_length = L3
        self.flat_dim = self.enc_conv_channels[-1] * self.conv_out_length
        
        # Fully-connected layer after flattening
        self.enc_fc = nn.Linear(self.flat_dim, self.hidden_dim)
        # Map to latent representation z (of dimension z_dim)
        self.fc_z = nn.Linear(self.hidden_dim, self.z_dim)
        
        # Vector quantizer (assumes latent dimension matches embedding_dim)
        self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.vq_commitment_cost)
        
        # ------------------------------
        # Build the Decoder
        # ------------------------------
        # Fully-connected layer to expand quantized latent vector back to flat conv features
        self.dec_fc = nn.Linear(self.z_dim, self.flat_dim)
        # Reshape into (channels, conv_out_length) then use transposed convolutions.
        self.dec_deconv1 = nn.ConvTranspose1d(in_channels=self.enc_conv_channels[-1],
                                              out_channels=self.dec_conv_channels[0],
                                              kernel_size=self.conv_kernel_size,
                                              stride=1,
                                              padding=p)
        self.dec_deconv2 = nn.ConvTranspose1d(in_channels=self.dec_conv_channels[0],
                                              out_channels=self.dec_conv_channels[1],
                                              kernel_size=self.conv_kernel_size,
                                              stride=self.conv_stride_dec2,
                                              padding=p,
                                              output_padding=self.conv_stride_dec2 - 1)
        self.dec_deconv3 = nn.ConvTranspose1d(in_channels=self.dec_conv_channels[1],
                                              out_channels=self.dec_conv_channels[2],
                                              kernel_size=self.conv_kernel_size,
                                              stride=1,
                                              padding=p)
        self.dec_final = nn.ConvTranspose1d(in_channels=self.dec_conv_channels[2],
                                            out_channels=1,
                                            kernel_size=1,
                                            stride=1)
    
    def encode(self, x):
        """
        Encodes input x into a latent representation.
        x: (batch, 1, input_dim)
        Returns: latent vector z of shape (batch, z_dim)
        """
        x = F.relu(self.enc_conv1(x))  # -> (batch, enc_conv_channels[0], input_dim)
        x = F.relu(self.enc_conv2(x))  # -> (batch, enc_conv_channels[1], L2)
        x = F.relu(self.enc_conv3(x))  # -> (batch, enc_conv_channels[2], L3)
        x = x.view(x.size(0), -1)      # flatten to (batch, flat_dim)
        x = F.relu(self.enc_fc(x))     # -> (batch, hidden_dim)
        z = self.fc_z(x)               # -> (batch, z_dim)
        return z
    
    def decode(self, z_q):
        """
        Decodes quantized latent vector into a reconstruction.
        z_q: (batch, z_dim)
        Returns: reconstruction of shape (batch, 1, input_dim)
        """
        x = F.relu(self.dec_fc(z_q))  # -> (batch, flat_dim)
        x = x.view(-1, self.enc_conv_channels[-1], self.conv_out_length)  # reshape to (batch, channels, L3)
        x = F.relu(self.dec_deconv1(x))
        x = F.relu(self.dec_deconv2(x))
        x = F.relu(self.dec_deconv3(x))
        #x = torch.sigmoid(self.dec_final(x))  # final activation to [0, 1]
        x= self.dec_final(x)
        return x
    
    def forward(self, x):
        """
        Full forward pass: encode, vector quantize, and decode.
        Returns:
          - recon_x: reconstruction (batch, 1, input_dim)
          - vq_loss: vector quantization loss (scalar)
        """
        z = self.encode(x)                    # (batch, z_dim)
        z_q, vq_loss = self.vector_quantizer(z) # quantize latent vector
        recon_x = self.decode(z_q)
        return recon_x, vq_loss

def vqvae_loss_function(recon_x, x, vq_loss, recon_weight=1.0):
    """
    Computes the VQ-VAE loss as the weighted sum of reconstruction loss and VQ loss.
    Reconstruction loss is computed using MSE.
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
    config_path = os.path.join(os.path.dirname(__file__), "..","configs", "conv_vqvae", "fraud_conv_vqvae.config")
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    conv_config = config_parser["Conv_VQVAE"]
    
    # Instantiate the model using the configuration
    model = ConvVQVAE(conv_config)
    print_num_params(model)
    
    # Load trainer hyperparameters (e.g., learning rate)
    train_config = config_parser["Trainer"]
    lr = train_config.getfloat("lr", fallback=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create a dummy input: batch of 8 samples with shape (8, 1, input_dim)
    dummy_input = torch.randn(8, 1, model.input_dim)
    recon, vq_loss = model(dummy_input)
    
    loss = vqvae_loss_function(recon, dummy_input, vq_loss, recon_weight=model.recon_weight)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Reconstruction shape:", recon.shape)
    print("Loss:", loss.item())

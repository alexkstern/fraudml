import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser

class ConvVae(nn.Module):
    """
    A configurable 1D Convolutional VAE that reads its hyperparameters from a config file.
    The input is expected to be of shape (batch_size, 1, input_dim).
    """
    def __init__(self, config):
        """
        config: a configparser section (e.g., config['Conv_VAE'])
        """
        super().__init__()
        # Read basic parameters
        self.input_dim    = config.getint("input_dim")       # e.g., 30
        self.hidden_dim   = config.getint("hidden_dim")        # e.g., 32 (for the fc layer)
        self.z_dim        = config.getint("z_dim")             # e.g., 3
        self.recon_weight = config.getfloat("recon_weight")    # e.g., 1.0
        self.kl_weight    = config.getfloat("kl_weight")       # e.g., 1.0

        # Read convolution hyperparameters
        # They are provided as comma-separated values in the config file.
        #self.enc_conv_channels = list(map(int, config.get("enc_conv_channels".split(",")))
        self.enc_conv_channels = list(map(int, config.get("enc_conv_channels").split(",")))#, fallback="16,32,64").split(",")))

        self.dec_conv_channels = list(map(int, config.get("dec_conv_channels").split(",")))
        self.conv_kernel_size  = config.getint("conv_kernel_size", fallback=3)
        self.conv_stride_enc2  = config.getint("conv_stride_enc2", fallback=2)
        self.conv_stride_dec2  = config.getint("conv_stride_dec2", fallback=2)
        
        # For "same" padding, we use (kernel_size-1)//2
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

        # Calculate the output length after the conv layers
        # First layer: stride=1 => length remains input_dim.
        L1 = self.input_dim
        # Second layer: using formula L_out = floor((L_in + 2*p - kernel_size)/stride + 1)
        L2 = (L1 + 2*p - self.conv_kernel_size) // self.conv_stride_enc2 + 1
        # Third layer: stride=1 => L3 = (L2 + 2*p - kernel_size) + 1
        L3 = (L2 + 2*p - self.conv_kernel_size) // 1 + 1
        self.conv_out_length = L3  # e.g., should be 15 for default parameters
        self.flat_dim = self.enc_conv_channels[-1] * self.conv_out_length

        # Fully-connected layer after flattening
        self.enc_fc    = nn.Linear(self.flat_dim, self.hidden_dim)
        self.fc_mu     = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # ------------------------------
        # Build the Decoder
        # ------------------------------
        # Fully-connected layer to expand latent vector back to flat conv features
        self.dec_fc = nn.Linear(self.z_dim, self.flat_dim)
        # Transpose the flat feature vector into shape (channels, conv_out_length)
        # and pass it through deconvolutional layers.
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
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        Encodes input x into latent space parameters.
        Input x shape: (batch, 1, input_dim)
        """
        x = F.relu(self.enc_conv1(x))  # -> (batch, enc_conv_channels[0], input_dim)
        x = F.relu(self.enc_conv2(x))  # -> (batch, enc_conv_channels[1], L2)
        x = F.relu(self.enc_conv3(x))  # -> (batch, enc_conv_channels[2], L3)
        x = x.view(x.size(0), -1)      # flatten to (batch, flat_dim)
        x = F.relu(self.enc_fc(x))     # -> (batch, hidden_dim)
        mu = self.fc_mu(x)             # -> (batch, z_dim)
        logvar = self.fc_logvar(x)     # -> (batch, z_dim)
        return mu, logvar

    def decode(self, z):
        """
        Decodes latent vector z to reconstruct the input.
        """
        x = F.relu(self.dec_fc(z))      # -> (batch, flat_dim)
        x = x.view(-1, self.enc_conv_channels[-1], self.conv_out_length)  # reshape to (batch, channels, L3)
        x = F.relu(self.dec_deconv1(x))
        x = F.relu(self.dec_deconv2(x))
        x = F.relu(self.dec_deconv3(x))
        #x = torch.sigmoid(self.dec_final(x))  # final activation (for values in [0, 1])
        x = self.dec_final(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar



def vae_loss_function(recon_x, x, mu, logvar, recon_weight=1.0, kl_weight=1.0):
    """
    Computes the VAE loss as a weighted sum of MSE reconstruction loss and KL divergence.
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
    # Load the configuration from file
    config_path = os.path.join(os.path.dirname(__file__), "..","configs", "conv_vae", "fraud_conv_vae.config")
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    conv_config = config_parser["Conv_VAE"]

    # Instantiate the model using the configuration
    model = ConvVae(conv_config)
    print_num_params(model)

    train_config = config_parser["Trainer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.getfloat("lr"))
    
    # Create a dummy input: batch of 8 samples with shape (8, 1, input_dim)
    dummy_input = torch.randn(8, 1, model.input_dim)
    recon, mu, logvar = model(dummy_input)
    
    loss = vae_loss_function(recon, dummy_input, mu, logvar,
                             recon_weight=model.recon_weight,
                             kl_weight=model.kl_weight)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Reconstruction shape:", recon.shape)
    print("Loss:", loss.item())

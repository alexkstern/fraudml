import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import configparser

#Variational Autoencoder

#input data -> encode to hidden representation -> sample from latent space with a speciifc mean and variance
#  -> parametrization trick -> decode to output data
class VariationalAutoEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        # Load default config if none provided
        if config is None:
            config = {
                'input_dim': 30,
                'hidden_dim': 200,
                'z_dim': 10
            }
        
        # Extract dimensions from config
        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        z_dim = config['z_dim']
        
        #Encoder
        self.input_2hid=nn.Linear(input_dim,hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hid_2_mean=nn.Linear(hidden_dim,z_dim)
        self.hid_2_sigma=nn.Linear(hidden_dim,z_dim)
        
        #in loss function we want the two linear layers to become standard gaussian so that the latent space is gaussian
        #this is done by minimizing the KL divergence between the two distributions

        #Decoder
        self.z_2_hid=nn.Linear(z_dim,hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.hid_2_output=nn.Linear(hidden_dim,input_dim)

    #Encoder
    def encode(self, x):
        #q_phi(z given x)
        #firt layer
        hid=self.input_2hid(x)
        #second layer (relu)
        hid=self.bn1(hid)
        hid=F.relu(hid)
        #mean and variance
        mu=self.hid_2_mean(hid)
        sigma=self.hid_2_sigma(hid)
        return mu,sigma

    def reparametrize(self,mu,sigma):
        std=torch.exp(0.5*sigma) #this ensures that the variance is positive
        eps=torch.randn_like(std)
        return mu+eps*std

    def decode(self,z):
        #p_theta(x given z)
        #recunstruct the data
        hid=self.z_2_hid(z)
        #apply relu
        hid=self.bn2(hid)
        hid=F.relu(hid)
        out=self.hid_2_output(hid)
        return out

    def forward(self,x):
        mu,sigma=self.encode(x)
        x_reparametrized=self.reparametrize(mu,sigma)
        x_reconstructed=self.decode(x_reparametrized)
        return x_reconstructed,mu,sigma

def load_config(config_path):
    """
    Load configuration from a .config file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Convert string values to appropriate types
        config_dict = {}
        if 'VAE' in config:
            for key, value in config['VAE'].items():
                # Try to convert to int or float if possible
                try:
                    if '.' in value:
                        config_dict[key] = float(value)
                    else:
                        config_dict[key] = int(value)
                except ValueError:
                    config_dict[key] = value
        return config_dict
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default configuration instead.")
        return None
    
def vae_loss_function(reconstructed_x, x, mu, sigma, config=None):
    # Apply weights from config if available
    if config is None:
        recon_weight = 1.0
        kl_weight = 1.0
    else:
        recon_weight = config.get('recon_weight', 1.0)
        kl_weight = config.get('kl_weight', 1.0)
    
    # Reconstruction loss (using MSE)
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')  # or reduction='mean'

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    return recon_weight * recon_loss + kl_weight * kl_divergence

# function to calculate the reconstruction error 
def reconstruction_error(model, data_loader, device):
    #calculate the minumum and maximum reconstruction error
    
    model.eval()
    total_error = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0].to(device)  # Assuming batch is (data, label)
            recon_batch, _, _ = model(batch)
            total_error += F.mse_loss(recon_batch, batch, reduction='sum').item()
    
    #Calculate the minimum reconstruction error
    return total_error / len(data_loader.dataset)

def print_model_param_count(model):
    """
    Print the number of parameters in the model
    
    Args:
        model: The PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
#do one forward pass, of a dummy tensor
if __name__ == '__main__':
    # Load config from file
    config_path = 'configs/fraud_vae/vae_test.config'
    config = load_config(config_path)
    print(f"Loaded configuration: {config if config else 'Using defaults'}")


    # Initialize model with config
    model = VariationalAutoEncoder(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(print_model_param_count(model))
    model.to(device)
    model.eval()
    
    # Get dimensions from config for validation
    input_dim = 30 if config is None else config.get('input_dim', 30)
    z_dim = 10 if config is None else config.get('z_dim', 10)
    
    with torch.no_grad():
        x = torch.randn(10, input_dim).to(device)
        x_reconstructed, mu, sigma = model(x)
        assert x_reconstructed.shape == x.shape
        assert mu.shape == (10, z_dim)
        assert sigma.shape == (10, z_dim)
        print("Forward pass successful")
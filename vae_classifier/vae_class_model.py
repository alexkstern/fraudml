import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
import os

#Variational Autoencoder

#input data -> encode to hidden representation -> sample from latent space with a speciifc mean and variance
#  -> parametrization trick -> decode to output data
class VariationalAutoEncoder_with_classifier(nn.Module):
    def __init__(self,input_dim=30,hidden_dim=200,z_dim=10,class_hidden_dim=None):
        super().__init__()
        
        # If class_hidden_dim not provided, use z_dim
        if class_hidden_dim is None:
            class_hidden_dim = z_dim
        
        #Encoder
        self.input_2hid=nn.Linear(input_dim,hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hid_2_mean=nn.Linear(hidden_dim,z_dim)
        self.hid_2_sigma=nn.Linear(hidden_dim,z_dim)
        
        #in loss function we want the two linear layers to become standard gaussian so that the latent space is gaussian
        #this is done by minimizing the KL divergence between the two distributions
        
        #Classifier
        self.z_2_class_hid=nn.Linear(z_dim,z_dim*10)  # Using z_dim instead of class_hidden_dim
        self.class_norm = nn.LayerNorm(z_dim*10)  # Added layer normalization
        self.class_hid_2_output=nn.Linear(z_dim*10,2)

        #Decoder
        self.z_2_hid_2_hid=nn.Linear(z_dim,hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.hid_2_output=nn.Linear(hidden_dim,input_dim)

    def classify(self,z):
        hid=self.z_2_class_hid(z)
        hid=self.class_norm(hid)  # Added layer normalization
        hid=F.relu(hid)
        out=self.class_hid_2_output(hid)
        return out

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
        hid=self.z_2_hid_2_hid(z)
        #apply relu
        hid=self.bn3(hid)
        hid=F.relu(hid)
        out=self.hid_2_output(hid)
        return out

    def forward(self,x):
        mu,sigma=self.encode(x)
        z=self.reparametrize(mu,sigma)
        x_reconstructed=self.decode(z)
        x_class=self.classify(z)
        return x_reconstructed,mu,sigma,x_class
    
def vae_loss_function(reconstructed_x, x, mu, sigma,y,x_class):
    # Reconstruction loss (using MSE)
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')  # or reduction='mean'

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    #classification loss
    #cross entropy loss
    class_weights = torch.tensor([1.0, 100.0], device=x_class.device)  # Higher weight for minority class
    cross_entropy_loss=F.cross_entropy(x_class, y, weight=class_weights)

    return recon_loss + kl_divergence + cross_entropy_loss

# function to calculate the reconstruction error 
def reconstruction_error(model, data_loader, device):
    #calculate the minumum and maximum reconstruction error
    
    model.eval()
    total_error = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0].to(device)  # Assuming batch is (data, label)
            recon_batch, _, _, _ = model(batch)
            total_error += F.mse_loss(recon_batch, batch, reduction='sum').item()
    
    #Calculate the minimum reconstruction error
    return total_error / len(data_loader.dataset)


def load_config(config_path):
    """Load configuration from file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config_path = "../configs/vae_classifier/vae_classifier.config"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        print("Using default model parameters instead.")
        input_dim = 30
        hidden_dim = 200
        z_dim = 10
    else:
        config = load_config(config_path)
        
        # Extract model parameters from config
        input_dim = int(config['VAE']['input_dim'])
        hidden_dim = int(config['VAE']['hidden_dim'])
        z_dim = int(config['VAE']['z_dim'])
    
    print(f"Initializing VAE with: input_dim={input_dim}, hidden_dim={hidden_dim}, z_dim={z_dim}")
    
    # Initialize the model with config parameters
    model = VariationalAutoEncoder_with_classifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        z_dim=z_dim
    ).to(device)
    
    # Create a random input tensor for testing
    batch_size = 4
    random_input = torch.randn(batch_size, input_dim).to(device)
    
    # Perform a forward pass
    print("Performing forward pass...")
    reconstructed, mu, sigma, class_output = model(random_input)
    
    # Print output shapes and sample values
    print(f"\nInput shape: {random_input.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Class output shape: {class_output.shape}")
    
    # Print sample values
    print("\nSample values:")
    print(f"Input sample: {random_input[0, :5]}")
    print(f"Reconstructed sample: {reconstructed[0, :5]}")
    print(f"Mu sample: {mu[0, :5]}")
    print(f"Sigma sample: {sigma[0, :5]}")
    print(f"Class output: {class_output}")
    
    # Compute loss for the random batch (assuming all labels are 0)
    fake_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
    
    loss = vae_loss_function(
        reconstructed, 
        random_input, 
        mu, 
        sigma,
        fake_labels,
        class_output
    )
    
    print(f"\nLoss value: {loss.item()}")
    
    print("\nModel summary:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
import torch

def extract_recon_loss(model, batch, device):
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        recon, mu, logvar = model(batch)
        recon_loss = torch.nn.functional.mse_loss(recon, batch, reduction='sum').item()
    return recon_loss
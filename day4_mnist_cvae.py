import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class ConditionalVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, num_classes: int, hidden_dims: List = None):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        modules = []

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims
        curr_in_channels = in_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(curr_in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            curr_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        test_input = torch.zeros(1, in_channels, 64, 64)
        with torch.no_grad():
            test_output = self.encoder(test_input)
            self.last_shape = test_output.shape[1:]
            flat_dim = test_output.view(1, -1).size(1)

        self.condition_encoder = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )


        combined_dim = flat_dim + 128
        
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_var = nn.Linear(combined_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + num_classes, flat_dim)

        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def encode(self, input, condition):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        condition_encoded = self.condition_encoder(condition)
        

        combined = torch.cat([result, condition_encoded], dim=1)
        
        mu = self.fc_mu(combined)
        log_var = self.fc_var(combined)
        return mu, log_var

    def decode(self, z, condition):

        z_conditioned = torch.cat([z, condition], dim=1)
        
        result = self.decoder_input(z_conditioned)
        result = result.view(-1, *self.last_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, condition):
        mu, log_var = self.encode(input, condition)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, condition), input, mu, log_var

    def loss_function(self, recons, input, mu, log_var, kld_weight=1.0):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var, dim=1))
        total_loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': total_loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach()
        }


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_loader = DataLoader(mnist_data, batch_size=32, shuffle=True)


model = ConditionalVAE(in_channels=1, latent_dim=64, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(20):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kld = 0
    for x, labels in mnist_loader:
        x = x.to(device)
        

        condition = F.one_hot(labels, num_classes=10).float().to(device)
        
        optimizer.zero_grad()
        recons, _, mu, logvar = model(x, condition)
        loss_dict = model.loss_function(recons, x, mu, logvar)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += loss_dict['Reconstruction_Loss'].item()
        total_kld += loss_dict['KLD'].item()

    avg_loss = total_loss / len(mnist_loader)
    avg_recon = total_recon / len(mnist_loader)
    avg_kld = total_kld / len(mnist_loader)
    print(f"[Epoch {epoch+1}] Total Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KLD: {avg_kld:.4f}")


model.eval()
with torch.no_grad():

    fig, axes = plt.subplots(10, 8, figsize=(16, 20))
    
    for digit in range(10):

        condition = F.one_hot(torch.tensor([digit] * 8), num_classes=10).float().to(device)
        z = torch.randn(8, 64).to(device)
        
        generated = model.decode(z, condition)
        
        for i in range(8):
            axes[digit, i].imshow(generated[i].cpu().squeeze(), cmap='gray')
            axes[digit, i].axis('off')
            if i == 0:
                axes[digit, i].set_ylabel(f"Digit {digit}", fontsize=12)
    
    plt.suptitle("Conditional Generation of MNIST Digits", fontsize=16)
    plt.tight_layout()
    plt.savefig("conditional_mnist_generation.png", dpi=300, bbox_inches='tight')
    plt.show()

model.eval()
with torch.no_grad():
    images, labels = next(iter(mnist_loader))
    images = images.to(device)
    condition = F.one_hot(labels, num_classes=10).float().to(device)
    recons, _, _, _ = model(images, condition)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title(f"Label: {labels[i].item()}")
    axes[1, i].imshow(recons[i].cpu().squeeze(), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_ylabel("Original", fontsize=12)
axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
plt.suptitle("Original vs Reconstructed MNIST Images", fontsize=14)
plt.tight_layout()
plt.savefig("mnist_reconstruction.png")
plt.show()

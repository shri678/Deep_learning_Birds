"""
VAE for Bird Phylogenetic Tree Construction

This module implements a Variational Autoencoder (VAE) specifically designed
for learning meaningful representations of bird spectrograms to construct
phylogenetic trees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class BirdSpectrogramDataset(Dataset):
    """Dataset class for bird spectrograms with species labels."""
    
    def __init__(self, data_dir, metadata_path, species_list=None, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to spectrogram directory
            metadata_path: Path to metadata CSV
            species_list: List of species to include (None for all)
            transform: Optional transforms
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.metadata = pd.read_csv(metadata_path)
        
        # Get all spectrogram files
        self.file_paths = []
        self.labels = []
        self.species_to_idx = {}
        
        if species_list is None:
            species_list = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        
        for idx, species in enumerate(species_list):
            self.species_to_idx[species] = idx
            species_dir = self.data_dir / species
            if species_dir.exists():
                for file_path in species_dir.glob('*.png'):
                    self.file_paths.append(file_path)
                    self.labels.append(idx)
        
        print(f"Loaded {len(self.file_paths)} spectrograms from {len(species_list)} species")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load spectrogram
        img_path = self.file_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        species_name = img_path.parent.name
        
        return img, label, species_name

class VAE(nn.Module):
    """Variational Autoencoder for bird spectrograms."""
    
    def __init__(self, input_shape=(1, 128, 256), latent_dim=32, hidden_dims=[64, 128, 256]):
        """
        Initialize VAE.
        
        Args:
            input_shape: Input image shape (channels, height, width)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for encoder/decoder
        """
        super(VAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1 x 128 x 256
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 64 x 64 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 32 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 16 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 8 x 16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Calculate the size after convolutions
        self.encoded_size = self._get_encoded_size()
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.encoded_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.encoded_size)
        
        self.decoder = nn.Sequential(
            # Input: 512 x 8 x 16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 16 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 32 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 64 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 1 x 128 x 256
        )
        
    def _get_encoded_size(self):
        """Calculate the size of the encoded representation."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            encoded = self.encoder(dummy_input)
            return encoded.view(1, -1).size(1)
    
    def encode(self, x):
        """Encode input to latent space."""
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space."""
        decoded = self.fc_decode(z)
        # Reshape to match encoder output shape (512 x 8 x 16)
        decoded = decoded.view(decoded.size(0), 512, 8, 16)
        decoded = self.decoder(decoded)
        return torch.sigmoid(decoded)
    
    def forward(self, x):
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class PhylogenyAnalyzer:
    """Analyzer for constructing phylogenetic trees from VAE representations."""
    
    def __init__(self, vae_model, device):
        self.vae_model = vae_model
        self.device = device
        self.species_representations = {}
        
    def extract_species_representations(self, dataloader):
        """Extract mean latent representations for each species."""
        self.vae_model.eval()
        species_embeddings = {}
        
        with torch.no_grad():
            for batch_imgs, batch_labels, species_names in dataloader:
                batch_imgs = batch_imgs.to(self.device)
                
                # Get latent representations
                mu, logvar = self.vae_model.encode(batch_imgs)
                
                for i, species in enumerate(species_names):
                    if species not in species_embeddings:
                        species_embeddings[species] = []
                    species_embeddings[species].append(mu[i].cpu().numpy())
        
        # Compute mean representation for each species
        for species, embeddings in species_embeddings.items():
            self.species_representations[species] = np.mean(embeddings, axis=0)
        
        return self.species_representations
    
    def construct_phylogenetic_tree(self, method='ward'):
        """Construct phylogenetic tree using hierarchical clustering."""
        if not self.species_representations:
            raise ValueError("Must extract species representations first")
        
        # Convert to matrix
        species_names = list(self.species_representations.keys())
        embeddings_matrix = np.array([self.species_representations[species] for species in species_names])
        
        # Compute distance matrix
        distances = pdist(embeddings_matrix, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Hierarchical clustering
        linkage_matrix = linkage(distances, method=method)
        
        return linkage_matrix, species_names, distance_matrix
    
    def visualize_phylogenetic_tree(self, linkage_matrix, species_names, save_path=None):
        """Visualize the phylogenetic tree."""
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=species_names, orientation='top', 
                  distance_sort='descending', show_leaf_counts=True)
        plt.title('Phylogenetic Tree from VAE Latent Representations')
        plt.xlabel('Species')
        plt.ylabel('Distance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_latent_space(self, dataloader, save_path=None):
        """Visualize latent space using t-SNE."""
        self.vae_model.eval()
        all_embeddings = []
        all_labels = []
        all_species = []
        
        with torch.no_grad():
            for batch_imgs, batch_labels, species_names in dataloader:
                batch_imgs = batch_imgs.to(self.device)
                mu, _ = self.vae_model.encode(batch_imgs)
                
                all_embeddings.append(mu.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_species.extend(species_names)
        
        embeddings = np.vstack(all_embeddings)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=all_labels, cmap='tab20', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of VAE Latent Space')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return embeddings_2d

def train_vae(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda'):
    """Train the VAE model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training

        print(f"Training epoch {epoch}... for {len(train_loader)} batches")
        model.train()
        train_loss = 0

        for batch_idx, (batch_imgs, _, _) in enumerate(train_loader):
            print(f"Training batch {batch_idx}...")
            batch_imgs = batch_imgs.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_imgs)
            
            # VAE loss: reconstruction + KL divergence
            recon_loss = F.mse_loss(recon, batch_imgs, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_imgs, _, _ in val_loader:
                batch_imgs = batch_imgs.to(device)
                recon, mu, logvar = model(batch_imgs)
                
                recon_loss = F.mse_loss(recon, batch_imgs, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                
                val_loss += loss.item()
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return train_losses, val_losses

def main():
    """Main function to run the VAE phylogenetic analysis."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    data_dir = Path("Data/phaethornis_goodspecs")
    metadata_path = "Data/phaethornis_metadata.csv"
    
    # Create dataset
    dataset = BirdSpectrogramDataset(data_dir, metadata_path)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize VAE
    vae = VAE(input_shape=(1, 128, 256), latent_dim=32).to(device)
    
    # Train VAE
    print("Training VAE...")
    train_losses, val_losses = train_vae(vae, train_loader, val_loader, epochs=50, device=device)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Training Progress')
    plt.show()
    
    # Phylogenetic analysis
    print("Performing phylogenetic analysis...")
    analyzer = PhylogenyAnalyzer(vae, device)
    
    # Extract species representations
    species_reps = analyzer.extract_species_representations(train_loader)
    
    # Construct phylogenetic tree
    linkage_matrix, species_names, distance_matrix = analyzer.construct_phylogenetic_tree()
    
    # Visualize results
    analyzer.visualize_phylogenetic_tree(linkage_matrix, species_names, 
                                       save_path="phylogenetic_tree.png")
    analyzer.visualize_latent_space(train_loader, save_path="latent_space_tsne.png")
    
    # Save model
    torch.save(vae.state_dict(), "vae_bird_phylogeny.pth")
    print("Model saved as 'vae_bird_phylogeny.pth'")
    
    return vae, analyzer

if __name__ == "__main__":
    vae_model, phylogeny_analyzer = main()

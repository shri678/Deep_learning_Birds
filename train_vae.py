"""
Simplified training script for VAE phylogenetic analysis.
Run this script to train the VAE and generate phylogenetic trees.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vae_phylogeny import BirdSpectrogramDataset, VAE, PhylogenyAnalyzer, train_vae
from pathlib import Path

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    data_dir = Path("Data/phaethornis_goodspecs")
    metadata_path = "Data/phaethornis_metadata.csv"
    
    # Check if data exists
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} not found!")
        print("Please run the data preparation notebooks first.")
        return
    
    # Create dataset
    print("Loading dataset...")
    dataset = BirdSpectrogramDataset(data_dir, metadata_path)
    
    if len(dataset) == 0:
        print("Error: No data found in dataset!")
        return
    
    # Split data (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize VAE
    print("Initializing VAE...")
    vae = VAE(input_shape=(1, 128, 256), latent_dim=32).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train VAE
    print("Training VAE...")
    train_losses, val_losses = train_vae(vae, train_loader, val_loader, epochs=5, device=device)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Training Progress')
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Phylogenetic analysis
    print("Performing phylogenetic analysis...")
    analyzer = PhylogenyAnalyzer(vae, device)
    
    # Extract species representations
    species_reps = analyzer.extract_species_representations(train_loader)
    print(f"Extracted representations for {len(species_reps)} species")
    
    # Construct phylogenetic tree
    linkage_matrix, species_names, distance_matrix = analyzer.construct_phylogenetic_tree()
    
    # Visualize results
    print("Generating visualizations...")
    analyzer.visualize_phylogenetic_tree(linkage_matrix, species_names, 
                                       save_path="phylogenetic_tree.png")
    analyzer.visualize_latent_space(train_loader, save_path="latent_space_tsne.png")
    
    # Save model
    torch.save(vae.state_dict(), "vae_bird_phylogeny.pth")
    print("Model saved as 'vae_bird_phylogeny.pth'")
    
    print("Training complete! Check the generated PNG files for results.")
    
    return vae, analyzer

if __name__ == "__main__":
    main()

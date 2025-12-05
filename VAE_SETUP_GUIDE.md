# VAE Setup Guide for Bird Phylogenetic Analysis

This guide will help you set up and run the Variational Autoencoder (VAE) for constructing phylogenetic trees from bird spectrograms.

## Prerequisites

1. **Data Preparation**: Ensure you have run the data preparation notebooks:
   - `notebooks/data_download.ipynb` - Downloads audio files and metadata
   - `notebooks/melspecs.ipynb` - Creates spectrograms from audio files

2. **Environment Setup**: Make sure you have the required packages installed.

## Quick Start

### Option 1: Run the Training Script (Recommended)

```bash
# Activate your virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install additional dependencies
pip install scikit-learn scipy seaborn

# Run the training script
python train_vae.py
```

This will:
- Train the VAE model
- Generate phylogenetic trees
- Create visualization plots
- Save the trained model

### Option 2: Use the Jupyter Notebook (Interactive)

```bash
# Start Jupyter Lab
jupyter lab

# Open notebooks/vae_phylogeny_analysis.ipynb
```

## Understanding the VAE Approach

### What is a VAE?
A Variational Autoencoder is a type of neural network that learns to:
1. **Encode**: Compress spectrograms into a lower-dimensional latent space
2. **Decode**: Reconstruct spectrograms from latent representations
3. **Learn meaningful representations**: The latent space captures important features

### How it works for phylogeny:
1. **Training**: VAE learns to compress spectrograms while preserving important acoustic features
2. **Representation**: Each species gets a mean latent representation from its spectrograms
3. **Phylogenetic Analysis**: Species are clustered based on similarity in latent space
4. **Tree Construction**: Hierarchical clustering creates phylogenetic trees

## Model Architecture

The VAE uses:
- **Encoder**: Convolutional layers that compress 128x256 spectrograms to 32-dimensional latent space
- **Decoder**: Transposed convolutional layers that reconstruct spectrograms
- **Loss Function**: Combines reconstruction error with KL divergence for meaningful latent space

## Key Parameters

- **Latent Dimension**: 32 (adjustable in the code)
- **Input Shape**: 1x128x256 (grayscale spectrograms)
- **Training Epochs**: 30 (adjustable)
- **Batch Size**: 16

## Output Files

After running, you'll get:
- `vae_bird_phylogeny.pth`: Trained model weights
- `phylogenetic_tree.png`: Dendrogram visualization
- `latent_space_tsne.png`: t-SNE plot of latent space
- `training_curves.png`: Training progress

## Interpreting Results

### Phylogenetic Tree
- **Y-axis**: Distance between species in latent space
- **Clustering**: Similar species cluster together
- **Branches**: Show evolutionary relationships

### Latent Space Visualization
- **t-SNE Plot**: Shows how spectrograms cluster in 2D
- **Colors**: Different colors represent different species
- **Clusters**: Tight clusters indicate similar acoustic features

## Customization

### Adjusting Model Parameters

Edit `vae_phylogeny.py` to modify:

```python
# Change latent dimension
vae = VAE(input_shape=(1, 128, 256), latent_dim=64)  # Increase for more complex representations

# Modify training parameters
train_losses, val_losses = train_vae(vae, train_loader, val_loader, epochs=50, device=device)
```

### Changing Phylogenetic Analysis

```python
# Use different clustering method
linkage_matrix, species_names, distance_matrix = analyzer.construct_phylogenetic_tree(method='complete')

# Adjust t-SNE parameters
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
   ```python
   device = torch.device('cpu')  # Force CPU usage
   ```

2. **Data not found**: Ensure spectrograms are generated first
   ```bash
   # Run the melspecs notebook first
   jupyter lab notebooks/melspecs.ipynb
   ```

3. **Poor reconstruction**: Increase training epochs or adjust model architecture

### Performance Tips

1. **Use GPU**: Training is much faster with CUDA
2. **Adjust batch size**: Larger batches for more stable training
3. **Monitor loss**: Ensure both reconstruction and KL loss decrease

## Advanced Usage

### Custom Species Selection

```python
# Train on specific species only
species_list = ['aethopygus', 'anthophilus', 'atrimentalis']
dataset = BirdSpectrogramDataset(data_dir, metadata_path, species_list=species_list)
```

### Different Input Sizes

```python
# For different spectrogram sizes
vae = VAE(input_shape=(1, 64, 128), latent_dim=16)
```

### Ensemble Methods

```python
# Train multiple VAEs and average representations
models = []
for i in range(3):
    vae = VAE(input_shape=(1, 128, 256), latent_dim=32)
    # Train each model
    models.append(vae)
```

## Next Steps

1. **Compare with known phylogeny**: Validate against established bird taxonomy
2. **Feature analysis**: Examine what acoustic features the VAE learns
3. **Cross-validation**: Test robustness across different data splits
4. **Species discovery**: Use for identifying new species relationships

## References

- [VAE Paper](https://arxiv.org/abs/1312.6114)
- [Bird Phylogeny Studies](https://www.nature.com/articles/nature21071)
- [Acoustic Phylogeny](https://www.pnas.org/doi/10.1073/pnas.1901517116)

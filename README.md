# Deep Learning Bird Phylogeny Project

This project uses deep learning methods to construct phylogenetic trees from spectrograms of bird species from the Phaethornis genus.

## Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   - Open `notebooks/data_download.ipynb` and run all cells
   - Open `notebooks/melspecs.ipynb` and run all cells

3. **VAE Phylogenetic Analysis**:
   ```bash
   # Option 1: Run training script
   python train_vae.py
   
   # Option 2: Interactive analysis
   jupyter lab notebooks/vae_phylogeny_analysis.ipynb
   ```

## Project Structure

- `Data/`: Contains bird audio files, spectrograms, and metadata
- `notebooks/`: Jupyter notebooks for data preparation and analysis
- `vae_phylogeny.py`: Main VAE implementation for phylogenetic analysis
- `train_vae.py`: Training script for the VAE model
- `VAE_SETUP_GUIDE.md`: Detailed setup and usage instructions

## Goal

Use deep learning methods (specifically Variational Autoencoders) to construct phylogenetic trees from spectrograms of bird species. The approach learns meaningful acoustic representations that capture evolutionary relationships between species.

## Key Features

- **Data Processing**: Downloads and processes bird audio from Xeno-Canto
- **Spectrogram Generation**: Creates high-quality mel-spectrograms from audio
- **VAE Training**: Learns compressed representations of acoustic features
- **Phylogenetic Analysis**: Constructs phylogenetic trees from learned representations
- **Visualization**: Interactive plots and dendrograms

## Species Included

The project includes 32 species from the Phaethornis genus, with preprocessed spectrograms ready for analysis.
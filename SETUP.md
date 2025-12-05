# Local Setup Guide

This document outlines the changes made to convert the Google Colab notebooks to work in a local environment, and any potential issues you may encounter.

## Changes Made

### 1. **data_download.ipynb**
- ✅ Removed Google Drive mounting code
- ✅ Removed `!pip install` command (dependencies moved to `requirements.txt`)
- ✅ Updated all file paths to use local `Data/` directory
- ✅ Added missing import: `from urllib.request import urlretrieve`
- ✅ Changed relative paths to use `pathlib` properly

### 2. **melspecs.ipynb**
- ✅ Removed Google Drive mounting code
- ✅ Replaced `cv2_imshow` (Colab-specific) with custom `show_spec()` function using matplotlib
- ✅ Updated all file paths to use local `Data/` directory
- ✅ Converted shell commands (`!du`) to Python subprocess calls
- ✅ Added proper imports at the beginning

### 3. **requirements.txt**
- ✅ Created comprehensive dependency file with all necessary packages

## Installation Instructions

1. **Activate your virtual environment** (if not already activated):
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks**:
   - Open Jupyter: `jupyter notebook`
   - Navigate to the `notebooks/` folder
   - Run notebooks in order: `data_download.ipynb` first, then `melspecs.ipynb`

## Potential Issues & Missing Information

### ⚠️ Issues You May Encounter:

1. **Metadata CSV Already Exists**
   - **Issue**: You already have `Data/phaethornis_metadata.csv` downloaded
   - **Impact**: If you re-run `data_download.ipynb`, it will overwrite this file
   - **Solution**: You can skip the metadata download cells if you already have the data

2. **Audio Files Already Downloaded**
   - **Issue**: You already have all MP3 files in `Data/phaethornis_audio/`
   - **Impact**: Re-downloading will duplicate files or overwrite existing ones
   - **Solution**: You can skip the audio download section (Cell 20) in `data_download.ipynb`

3. **Image Files Already Downloaded**
   - **Issue**: You already have spectrograms in `Data/phaethornis_images/`
   - **Impact**: Re-downloading will duplicate/overwrite
   - **Solution**: You can skip the image download section (Cell 16) in `data_download.ipynb`

4. **FastAI Installation Time**
   - **Issue**: FastAI and PyTorch are large packages (~2-3GB)
   - **Impact**: First-time installation may take 10-15 minutes
   - **Solution**: Be patient during `pip install`

5. **Memory Usage (melspecs.ipynb)**
   - **Issue**: Processing all audio files to create spectrograms is memory-intensive
   - **Impact**: May consume significant RAM and take considerable time (original: ~15 min on Colab)
   - **Solution**: Consider processing in batches if you run into memory issues

6. **Missing Spectrogram Directories**
   - **Issue**: `phaethornis_goodspecs/` and `phaethornis_noisyspecs/` don't exist yet
   - **Impact**: These will be created when you run `melspecs.ipynb`
   - **Solution**: No action needed - the notebook creates these automatically

7. **Display in Jupyter**
   - **Issue**: The `show_spec()` function uses matplotlib instead of Colab's `cv2_imshow`
   - **Impact**: Images may display differently than in Colab
   - **Solution**: If images don't display inline, ensure you have `%matplotlib inline` at the top of the notebook

8. **Path Separator Compatibility**
   - **Issue**: Using `pathlib` should handle macOS paths correctly, but some older code uses string concatenation
   - **Impact**: Minimal - all critical paths have been updated
   - **Solution**: If you see path errors, check that `pathlib.Path` objects are being used

## Recommended Workflow

Since you already have the data downloaded:

1. **Skip `data_download.ipynb`** - You already have all the data
2. **Run `melspecs.ipynb`** - This will create the custom spectrograms from your existing MP3 files
3. **Use `birds.ipynb`** - For your model training (if this is your main notebook)

## Verification Checklist

Before running the notebooks, verify:
- [ ] Virtual environment is activated
- [ ] All dependencies installed: `pip list | grep -E '(pandas|librosa|fastai|torch)'`
- [ ] Data directory exists: `ls ../Data/`
- [ ] Audio files exist: `ls ../Data/phaethornis_audio/`
- [ ] Metadata exists: `ls ../Data/phaethornis_metadata.csv`

## Additional Notes

### File Structure Expected:
```
Deep_learning_Birds/
├── Data/
│   ├── phaethornis_metadata.csv
│   ├── phaethornis_audio/
│   │   ├── aethopygus/
│   │   ├── anthophilus/
│   │   └── ... (27 species folders)
│   ├── phaethornis_images/
│   ├── phaethornis_goodspecs/  (will be created by melspecs.ipynb)
│   └── phaethornis_noisyspecs/ (will be created by melspecs.ipynb)
├── notebooks/
│   ├── data_download.ipynb
│   └── melspecs.ipynb
├── requirements.txt
└── venv/
```

### Performance Tips:
- Close other memory-intensive applications when running `melspecs.ipynb`
- Consider using a GPU if available (PyTorch will detect automatically)
- Monitor disk space - spectrograms will require additional storage

## Need Help?

If you encounter issues not covered here, check:
1. Console output for specific error messages
2. That file paths resolve correctly: `pathlib.Path('../Data').resolve()`
3. Python version compatibility: Requires Python 3.8+



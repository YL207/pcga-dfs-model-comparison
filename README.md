# PCGA-DFS Model Comparison

This repository contains a Python script for training and comparing deep learning models for heavy metal concentration prediction from digital fingerprint spectra (DFS) images.

## File

- `model_comparison.py`: model training and evaluation script. It currently includes a ResNet50-CBAM model and placeholder configurations for other comparison models.

## Main Functions

- Loads DFS image data and concentration labels.
- Applies optional log transformation and label standardization.
- Trains deep learning models for multi-output regression.
- Evaluates model performance using RMSE, R2, MAE, and relative error thresholds.
- Exports prediction results, performance metrics, scatter plots, and trained model weights.

## Requirements

The script requires Python and the following packages:

```bash
pip install numpy pandas matplotlib opencv-python joblib torch torchvision scikit-learn openpyxl
```

## Usage

Before running the script, update the paths in `GLOBAL_CONFIG`:

```python
GLOBAL_CONFIG = {
    "IMAGE_FOLDER": "path/to/dfs/images",
    "LABEL_FILE": "path/to/labels.xlsx",
    "BASE_SAVE_PATH": "path/to/output/results",
}
```

Then run:

```bash
python model_comparison.py
```

## Notes

- The input labels should be stored in an Excel file and should match the DFS image filenames.
- The default model configuration trains `ResNet50_CBAM`.
- GPU acceleration is used automatically when CUDA is available.

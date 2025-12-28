# Model Training Scripts

This directory contains scripts for training the storm motion prediction model.

## Files

- **`data_loader.py`**: Loads and preprocesses storm data from JSON files
- **`model.py`**: Original Gradient Boosting model implementation
- **`model_improved.py`**: Improved model with XGBoost support and better hyperparameters
- **`train_model.py`**: Main training script

## Usage

Train the model:
```bash
python src/model/train_model.py --data_dir /path/to/data --output_dir models
```

The script will:
1. Load storm data with 38 features (34 base + 4 engineered)
2. Apply StandardScaler normalization
3. Train XGBoost model with 1000 estimators
4. Save model and scaler to `models/`

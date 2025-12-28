# Gradient Boosted Regression Model

This directory contains the XGBoost-based storm motion prediction model.

## Files

- **`data_loader.py`**: Loads and preprocesses storm data (38 features)
- **`model.py`**: Original sklearn GradientBoostingRegressor
- **`model_improved.py`**: XGBoost implementation with validation monitoring
- **`train_model.py`**: Training script

## Performance

- **Overall MAE**: 6.39 m/s (with 31 m/s sanity check)
- **Improvement over baseline**: 19.8%
- **Erratic storms**: 6.61 m/s MAE (20.4% better than baseline)
- **Smooth storms**: 4.19 m/s MAE (6.8% better than baseline)

## Usage

```bash
python src/model/gbr/train_model.py --data_dir /path/to/data --output_dir models
```

## Features

- 31 environmental features (CAPE, shear, wind, radar, probabilities)
- 3 motion features (dx, dy, dt)
- 4 engineered features (velocity magnitude, direction, interactions)
- **Total: 38 features**

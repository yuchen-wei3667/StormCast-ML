# Model Training Scripts

This directory contains different model implementations for storm motion prediction.

## Structure

### `gbr/` - Gradient Boosted Regression (Current)
XGBoost-based model with 38 features
- **MAE**: 6.39 m/s
- **Status**: ✅ Production ready
- See [gbr/README.md](gbr/README.md) for details

### `lstm/` - LSTM Neural Network (Planned)
Temporal sequence model for improved predictions
- **Expected MAE**: 4.5-5.5 m/s
- **Status**: 🚧 To be implemented
- See [lstm/README.md](lstm/README.md) for details

## Quick Start

### Train GBR Model
```bash
python src/model/gbr/train_model.py --data_dir /path/to/data --output_dir models
```

### Train LSTM Model (Coming Soon)
```bash
python src/model/lstm/train_model.py --data_dir /path/to/data --output_dir models
```

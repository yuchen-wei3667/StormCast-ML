# GRU Model for Storm Motion Prediction

This directory contains the GRU (Gated Recurrent Unit) based storm motion prediction model, optimized for CPU inference.

## Features

### Architecture
- **Input**: Sequences of 5 scans (configurable)
- **Features**: 44 per timestep
  - 31 environmental (CAPE, shear, wind, radar, probabilities)
  - 3 motion (dx, dy, dt)
  - 4 engineered (velocity mag/dir, interactions)
  - 6 bounding box (x_min, y_min, x_max, y_max, area, aspect_ratio)
- **GRU layers**: 2 layers (64 + 32 units)
- **Dropout**: 0.3 for regularization
- **Output**: (u, v) velocity prediction

### Training Features
- ✅ Early stopping (patience=15)
- ✅ Learning rate reduction (patience=5)
- ✅ Verbose train/val loss monitoring
- ✅ Model checkpointing (saves best model)
- ✅ CSV logging of training history

### Expected Performance
- **Target MAE**: 4.5-5.5 m/s (vs GBR: 6.39 m/s)
- **Improvement**: 15-30% over baseline
- **Better on erratic storms**: Captures temporal patterns

## Files

- **`gru_data_loader.py`**: Sequence data loading with bounding box
- **`gru_model.py`**: GRU architecture and callbacks
- **`train_gru.py`**: Training script

## Usage

### Install TensorFlow
```bash
conda install tensorflow
```

### Train Model
```bash
python src/model/lstm/train_gru.py \
  --data_dir /path/to/data \
  --output_dir models \
  --sequence_length 5 \
  --batch_size 64 \
  --epochs 100
```

### Monitor Training
The script will print train/val loss every epoch:
```
Epoch 1: loss=450.23, val_loss=425.67, mae=15.32, val_mae=14.89
Epoch 2: loss=380.45, val_loss=395.12, mae=13.21, val_mae=13.45
...
```

## Status

✅ **Implemented** - Ready for training and evaluation

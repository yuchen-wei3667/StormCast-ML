# StormCast-ML Training Instructions

## Overview

This guide provides comprehensive instructions for training the StormCast-ML LSTM model for storm cell motion prediction. The model learns to predict storm cell motion vectors (vx, vy) based on historical storm data and environmental conditions.

## Modified Training Data

The training script has been updated to include **all 4 data files** from the TrainingData directory:

1. `TrainingData/stormcells_Central_20250315.json`
2. `TrainingData/stormcells_Midwest_20240506.json`
3. `TrainingData/stormcells_SE_20251125.json`
4. `TrainingData/stormcells_TX_20251123.json`

This provides a more diverse and comprehensive dataset covering different geographic regions and weather patterns.

## Quick Start Training

### Basic Training

To train the model with default parameters:

```bash
python src/train_model.py
```

This will:
- Load all 4 training data files
- Train for 300 epochs with batch size 32
- Use a maximum sequence length of 20
- Apply learning rate of 0.0005
- Split data 80/20 for training/validation
- Save the best model to `trained_storm_lstm.pth`

### Custom Training Parameters

You can modify the training parameters in `src/train_model.py` by editing the function call:

```python
model = train_model(
    json_files, 
    epochs=300,           # Number of training epochs
    batch_size=32,        # Training batch size
    max_seq_len=20,       # Maximum sequence length for LSTM
    learning_rate=0.0005  # Learning rate for optimizer
)
```

## Training Parameters Explained

### Epochs
- **Recommended range**: 200-500 epochs
- **Too few**: Model may underfit
- **Too many**: Risk of overfitting (watch validation loss)

### Batch Size
- **Default**: 32
- **Larger batches** (64, 128): Faster training, more memory usage
- **Smaller batches** (16, 8): Slower but potentially better generalization

### Learning Rate
- **Default**: 0.0005
- **Higher** (0.001, 0.005): Faster convergence, risk of overshooting
- **Lower** (0.0001, 0.00001): Slower but more stable convergence

### Max Sequence Length
- **Default**: 20
- **Longer sequences**: More context, harder to train
- **Shorter sequences**: Less context, easier to train

## Training Process

### 1. Data Preparation
The training script:
- Loads all JSON files from TrainingData directory
- Extracts storm history sequences for each storm cell
- Creates input sequences with features: `[dx, dy, dt, EBShear, SRW46km, MeanWind_1-3kmAGL]`
- Calculates target velocity vectors `[vx, vy]` from next timestep data
- Pads sequences to uniform length
- Splits data 80/20 for training/validation

### 2. Model Architecture
- **Input Size**: 6 features per timestep
- **Hidden Size**: 64 LSTM hidden units
- **Num Layers**: 2 LSTM layers
- **Output Size**: 2 (vx, vy velocity components)

### 3. Training Loop
- Uses Adam optimizer with specified learning rate
- Mean Squared Error (MSE) loss function
- Trains for specified number of epochs
- Validates on held-out data every epoch
- Saves best model based on validation loss

## Monitoring Training Progress

The training script outputs progress information:

```
Training samples: XXXX
Validation samples: XXXX
Epoch [20/300], Train Loss: 0.XXXX, Val Loss: 0.XXXX
```

### Key Metrics to Watch:
- **Train Loss**: Should generally decrease
- **Validation Loss**: Should decrease then stabilize
- **Gap between train/val loss**: Large gap indicates overfitting

## Advanced Training Strategies

### Learning Rate Scheduling
For better convergence, consider implementing learning rate scheduling:

```python
# Add to training loop
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
# Call after optimizer.step()
scheduler.step()
```

### Early Stopping
Prevent overfitting by stopping when validation loss stops improving:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Data Augmentation
Increase training data diversity by adding synthetic variations:

```python
# Example: Add noise to velocity targets
noise = torch.randn_like(targets) * 0.01  # Small random noise
augmented_targets = targets + noise
```

## Validation and Testing

### Running Predictions
After training, test the model:

```bash
python src/predict_and_compare.py
```

### Loading Trained Models
To use a pre-trained model:

```python
from src.model import StormCellLSTM

model = StormCellLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=2)
model.load_trained_weights('trained_storm_lstm.pth')
model.eval()
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Reduce max sequence length
   - Use gradient accumulation

2. **Poor Performance**
   - Increase training epochs
   - Adjust learning rate
   - Check data quality
   - Try different model architecture

3. **Overfitting**
   - Add dropout layers
   - Reduce model complexity
   - Use early stopping
   - Increase regularization

4. **Slow Training**
   - Use larger batch sizes
   - Reduce sequence length
   - Use GPU acceleration if available

### Performance Expectations

With the full dataset, you should expect:
- **Training samples**: Several thousand sequences
- **Training time**: 10-60 minutes (depending on hardware)
- **Final validation loss**: Typically 0.1-1.0 range
- **Convergence**: Most improvement in first 100-200 epochs

## File Structure After Training

```
StormCast-ML/
├── trained_storm_lstm.pth      # Saved model weights
├── TrainingData/               # Training data files
│   ├── stormcells_Central_20250315.json
│   ├── stormcells_Midwest_20240506.json
│   ├── stormcells_SE_20251125.json
│   └── stormcells_TX_20251123.json
├── src/
│   ├── train_model.py          # Training script (modified)
│   ├── model.py                # Model architecture
│   ├── data_loader.py          # Data preprocessing
│   └── ...
└── TRAINING_INSTRUCTIONS.md    # This file
```

## Next Steps

After training:

1. **Evaluate model performance** on test data
2. **Analyze prediction accuracy** for different storm types
3. **Fine-tune hyperparameters** based on validation results
4. **Deploy model** for real-time storm prediction
5. **Consider ensemble methods** with multiple trained models

## Support

For issues or questions:
- Check the model architecture in `src/model.py`
- Review data preprocessing in `src/data_loader.py`
- Examine prediction script in `src/predict_and_compare.py`
- Refer to PyTorch documentation for deep learning specifics
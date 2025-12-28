# LSTM Model (To Be Implemented)

This directory will contain the LSTM-based storm motion prediction model.

## Planned Features

### Architecture
- **Input**: Sequence of last N scans (e.g., 5-10 scans)
- **LSTM layers**: 2-3 layers with 64-128 units each
- **Output**: Multi-step predictions (next 1-5 scans)

### Expected Benefits
- **Temporal learning**: Captures storm acceleration and turning patterns
- **Better erratic storm prediction**: Expected 15-25% improvement over GBR
- **Multi-step forecasting**: Predict 5, 10, 15 minutes ahead
- **Expected MAE**: 4.5-5.5 m/s (vs current 6.39 m/s)

## Implementation Plan

1. **Data preparation**
   - Restructure data into sequences (storm tracks)
   - Handle variable-length sequences
   - Create multi-step targets

2. **Model architecture**
   - LSTM encoder for temporal features
   - Dense decoder for predictions
   - Dropout for regularization

3. **Training**
   - GPU acceleration (CUDA)
   - Learning rate scheduling
   - Early stopping on validation loss

4. **Evaluation**
   - Compare against GBR baseline
   - Analyze performance by forecast horizon
   - Test on erratic vs smooth storms

## Status

🚧 **Not yet implemented** - GBR model serves as baseline

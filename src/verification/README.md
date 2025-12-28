# Verification Scripts

This directory contains scripts for evaluating and analyzing the trained model.

## Files

- **`infer.py`**: Run inference on validation set with detailed metrics
- **`baseline_comparison.py`**: Compare model against simple motion extrapolation baseline
- **`erratic_vs_smooth.py`**: Compare performance on erratic vs smooth storm motion

## Usage

### Full Inference
```bash
python src/verification/infer.py --data_dir /path/to/data --model_dir models --output_dir results
```

### Baseline Comparison
```bash
python src/verification/baseline_comparison.py --data_dir /path/to/data --n_history 5
```

### Erratic vs Smooth Motion Analysis
```bash
python src/verification/erratic_vs_smooth.py --data_dir /path/to/data --model_dir models
```

## Results

All scripts include a 31 m/s sanity check to filter unrealistic velocities.

Current performance (with sanity check):
- **ML Model MAE**: 6.39 m/s
- **Baseline MAE**: 7.97 m/s
- **Improvement**: 19.8%

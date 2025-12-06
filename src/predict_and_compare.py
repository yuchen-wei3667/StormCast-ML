import json
import torch
import numpy as np
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import StormCellLSTM

def compute_baseline_prediction(storm_history, n=4):
    """
    Compute baseline prediction by calculating vx and vy from dx, dy, dt
    for the last n entries, then averaging them.
    Returns (vx, vy) as numpy array.
    """
    if len(storm_history) < n:
        n = len(storm_history)
    
    if n == 0:
        return np.array([0.0, 0.0])
    
    # Get the last n entries
    last_n = storm_history[-n:]
    
    # Calculate vx and vy from dx, dy, dt
    vx_values = []
    vy_values = []
    
    for point in last_n:
        dx = point.get('dx', 0)
        dy = point.get('dy', 0)
        dt = point.get('dt', 1)  # default to 1 to avoid division by zero
        
        if dt > 0:
            vx_values.append(dx / dt)
            vy_values.append(dy / dt)
    
    # Average them
    avg_vx = np.mean(vx_values) if vx_values else 0.0
    avg_vy = np.mean(vy_values) if vy_values else 0.0
    
    return np.array([avg_vx, avg_vy])

def compute_deviation(predicted, baseline):
    """
    Compute the deviation between predicted and baseline vectors.
    Returns magnitude of difference and percentage difference.
    """
    diff = predicted - baseline
    magnitude = np.linalg.norm(diff)
    
    baseline_magnitude = np.linalg.norm(baseline)
    if baseline_magnitude > 0:
        percent_diff = (magnitude / baseline_magnitude) * 100
    else:
        percent_diff = float('inf') if magnitude > 0 else 0.0
    
    return magnitude, percent_diff, diff

def main():
    # Load sample data
    json_file = 'TrainingData/stormcells_TX_20251123.json'
    
    output_lines = []
    output_lines.append(f"Loading data from {json_file}...\n")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        output_lines.append(f"Error: File {json_file} not found.")
        print('\n'.join(output_lines))
        return

    features = data.get('features', [])
    if not features:
        output_lines.append("No features found in JSON.")
        print('\n'.join(output_lines))
        return
        
    output_lines.append(f"Found {len(features)} storm cells.\n")
    
    # Initialize model
    model = StormCellLSTM()
    output_lines.append("Model initialized.\n")
    output_lines.append("=" * 80)
    
    # Process first 5 storm cells (or fewer if less available)
    num_samples = min(5, len(features))
    
    for i in range(num_samples):
        sample_cell = features[i]
        cell_id = sample_cell.get('id', 'Unknown')
        storm_history = sample_cell.get('storm_history', [])
        
        output_lines.append(f"\nStorm Cell ID: {cell_id}")
        output_lines.append(f"History Length: {len(storm_history)} timesteps")
        
        if len(storm_history) == 0:
            output_lines.append("  No history available for prediction.\n")
            output_lines.append("-" * 80)
            continue
        
        # Get LSTM prediction
        lstm_prediction = model.predict_from_json(sample_cell)
        
        if lstm_prediction is None:
            output_lines.append("  LSTM prediction failed (insufficient history).")
            lstm_prediction = np.array([0.0, 0.0])
        
        # Get baseline prediction (average of last 4 vectors)
        baseline_prediction = compute_baseline_prediction(storm_history, n=4)
        
        # Compute deviation
        magnitude, percent_diff, diff = compute_deviation(lstm_prediction, baseline_prediction)
        
        # Print results
        output_lines.append(f"\n  LSTM Prediction:      vx = {lstm_prediction[0]:8.2f}, vy = {lstm_prediction[1]:8.2f}")
        output_lines.append(f"  Baseline (avg last 4): vx = {baseline_prediction[0]:8.2f}, vy = {baseline_prediction[1]:8.2f}")
        output_lines.append(f"\n  Difference:           Δvx = {diff[0]:8.2f}, Δvy = {diff[1]:8.2f}")
        output_lines.append(f"  Deviation Magnitude:  {magnitude:.2f}")
        
        if percent_diff != float('inf'):
            output_lines.append(f"  Percent Difference:   {percent_diff:.2f}%")
        else:
            output_lines.append(f"  Percent Difference:   N/A (baseline is zero)")
        
        output_lines.append("-" * 80)
    
    output_lines.append("\nComparison complete!")
    
    # Print to console
    full_output = '\n'.join(output_lines)
    print(full_output)
    
    # Also save to file
    with open('prediction_results.txt', 'w', encoding='utf-8') as f:
        f.write(full_output)
    print("\nResults saved to prediction_results.txt")

if __name__ == "__main__":
    main()

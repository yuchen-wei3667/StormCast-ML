"""
Baseline comparison: Simple motion vector extrapolation
Predicts next scan motion by averaging the last N scans
"""
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import load_storm_data
import glob
import json

def load_storm_sequences(base_path, n_history=5):
    """
    Load storm data with temporal sequences for extrapolation
    
    Returns:
        X: Current scan features (for fair comparison with ML model)
        y_true: True next scan velocities
        y_baseline: Baseline predictions from averaged past motion
    """
    all_raw_entries = []
    
    # Load all data
    date_folders = glob.glob(os.path.join(base_path, "*"))
    print(f"Found {len(date_folders)} potential date folders in {base_path}")
    
    for date_folder in date_folders:
        if not os.path.isdir(date_folder):
            continue
            
        cells_dir = os.path.join(date_folder, "cells")
        if not os.path.exists(cells_dir):
            continue
            
        json_files = glob.glob(os.path.join(cells_dir, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                input_data = data if isinstance(data, list) else [data]
                all_raw_entries.extend(input_data)
                    
            except Exception as e:
                continue

    # Group by ID
    grouped_data = {}
    for entry in all_raw_entries:
        id_val = entry.get('id')
        if id_val is None:
            continue
        if id_val not in grouped_data:
            grouped_data[id_val] = []
        grouped_data[id_val].append(entry)
    
    X_list = []
    y_true_list = []
    y_baseline_list = []
    skipped = 0
    used = 0
    
    # Process each storm track
    for id_val, history in grouped_data.items():
        # Sort by timestamp
        history.sort(key=lambda x: x.get('timestamp', ''))
        
        # Need at least n_history + 1 scans (n_history for averaging, 1 for target)
        if len(history) < n_history + 1:
            skipped += len(history)
            continue
        
        # For each valid position in the sequence
        for i in range(n_history, len(history)):
            # Get last n_history scans for averaging
            past_scans = history[i-n_history:i]
            current_scan = history[i-1]  # Most recent scan
            next_scan = history[i]  # Target scan
            
            # Check validity
            if not all('dx' in s and 'dy' in s and 'dt' in s for s in [current_scan, next_scan]):
                skipped += 1
                continue
            
            if not all('dx' in s and 'dy' in s and 'dt' in s for s in past_scans):
                skipped += 1
                continue
            
            # Calculate true target velocity (from next scan)
            dx_next = float(next_scan['dx'])
            dy_next = float(next_scan['dy'])
            dt_next = float(next_scan['dt'])
            
            if dt_next == 0:
                skipped += 1
                continue
            
            u_true = dx_next / dt_next
            v_true = dy_next / dt_next
            
            # Calculate baseline prediction (average of past n_history velocities)
            past_velocities = []
            for scan in past_scans:
                dx = float(scan['dx'])
                dy = float(scan['dy'])
                dt = float(scan['dt'])
                if dt > 0:
                    past_velocities.append([dx/dt, dy/dt])
            
            if len(past_velocities) < n_history:
                skipped += 1
                continue
            
            # Average the past velocities
            avg_velocity = np.mean(past_velocities, axis=0)
            u_baseline = avg_velocity[0]
            v_baseline = avg_velocity[1]
            
            # Store current scan features (for context, though not used in baseline)
            dx_curr = float(current_scan['dx'])
            dy_curr = float(current_scan['dy'])
            dt_curr = float(current_scan['dt'])
            
            X_list.append([dx_curr, dy_curr, dt_curr])
            y_true_list.append([u_true, v_true])
            y_baseline_list.append([u_baseline, v_baseline])
            used += 1
    
    print(f"Loaded {used} valid sequences. Skipped {skipped} invalid entries.")
    
    return np.array(X_list), np.array(y_true_list), np.array(y_baseline_list)

def compute_velocity_errors(y_true, y_pred, name="Model"):
    """Compute velocity-specific error metrics"""
    u_true, v_true = y_true[:, 0], y_true[:, 1]
    u_pred, v_pred = y_pred[:, 0], y_pred[:, 1]
    
    mag_true = np.sqrt(u_true**2 + v_true**2)
    mag_pred = np.sqrt(u_pred**2 + v_pred**2)
    
    dir_true = np.arctan2(v_true, u_true) * 180 / np.pi
    dir_pred = np.arctan2(v_pred, u_pred) * 180 / np.pi
    
    dir_error = np.abs(dir_true - dir_pred)
    dir_error = np.minimum(dir_error, 360 - dir_error)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n{name} Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  U-velocity MAE: {mean_absolute_error(u_true, u_pred):.4f} m/s")
    print(f"  V-velocity MAE: {mean_absolute_error(v_true, v_pred):.4f} m/s")
    print(f"  Speed MAE: {mean_absolute_error(mag_true, mag_pred):.4f} m/s")
    print(f"  Direction MAE: {np.mean(dir_error):.2f} degrees")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'u_mae': mean_absolute_error(u_true, u_pred),
        'v_mae': mean_absolute_error(v_true, v_pred),
        'mag_mae': mean_absolute_error(mag_true, mag_pred),
        'dir_mae': np.mean(dir_error)
    }

def compare_baseline(data_dir, n_history=5):
    """Compare baseline extrapolation with ground truth"""
    print(f"Loading data with {n_history}-scan history...")
    X, y_true, y_baseline = load_storm_sequences(data_dir, n_history=n_history)
    
    print(f"\nTotal valid samples: {len(X)}")
    
    print("\n" + "="*60)
    print(f"BASELINE COMPARISON: {n_history}-Scan Average Extrapolation")
    print("="*60)
    
    # Compute baseline errors
    baseline_metrics = compute_velocity_errors(y_true, y_baseline, 
                                               name=f"Baseline ({n_history}-scan avg)")
    
    return baseline_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline motion extrapolation")
    parser.add_argument("--data_dir", required=True, help="Path to validation data")
    parser.add_argument("--n_history", type=int, default=5, 
                       help="Number of past scans to average (default: 5)")
    
    args = parser.parse_args()
    
    compare_baseline(args.data_dir, args.n_history)

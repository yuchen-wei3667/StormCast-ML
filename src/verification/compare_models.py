"""
Comprehensive model comparison script
Compares GBR, GRU, and Baseline models on the same validation set
"""
import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model', 'gbr'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model', 'lstm'))

from data_loader import load_storm_data
from gru_data_loader import load_sequences
from gru_model import velocity_aware_huber_loss, directional_error_deg

def load_gbr_model(model_dir):
    """Load GBR model and scaler"""
    with open(os.path.join(model_dir, "gb_storm_motion.pkl"), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def load_gru_model(model_dir, filename="gru_storm_motion.keras"):
    """Load GRU model and scaler"""
    import tensorflow as tf
    custom_objects = {
        "velocity_aware_huber_loss": velocity_aware_huber_loss,
        "directional_error_deg": directional_error_deg
    }
    model = tf.keras.models.load_model(os.path.join(model_dir, filename), custom_objects=custom_objects)
    
    scaler_filename = filename.replace('.keras', '_scaler.pkl')
    with open(os.path.join(model_dir, scaler_filename), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def compute_baseline_predictions(data_dir, n_history=5, max_velocity=31):
    """Compute baseline predictions using simple extrapolation"""
    import glob
    import json
    
    all_raw_entries = []
    date_folders = glob.glob(os.path.join(data_dir, "*"))
    
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
            except:
                continue
    
    # Group by ID
    grouped_data = {}
    for entry in all_raw_entries:
        storm_id = entry.get('id')
        if storm_id is None:
            continue
        if storm_id not in grouped_data:
            grouped_data[storm_id] = []
        grouped_data[storm_id].append(entry)
    
    y_true_list = []
    y_baseline_list = []
    
    for storm_id, track in grouped_data.items():
        track.sort(key=lambda x: x.get('timestamp', ''))
        
        if len(track) < n_history + 1:
            continue
        
        for i in range(n_history, len(track)):
            past_scans = track[i-n_history:i]
            next_scan = track[i]
            
            # Calculate baseline
            past_velocities = []
            for scan in past_scans:
                dx, dy, dt = scan.get('dx'), scan.get('dy'), scan.get('dt')
                if dx is None or dy is None or dt is None or dt == 0:
                    break
                past_velocities.append([float(dx)/float(dt), float(dy)/float(dt)])
            
            if len(past_velocities) < n_history:
                continue
            
            avg_velocity = np.mean(past_velocities, axis=0)
            
            # Get true target
            dx_next = next_scan.get('dx')
            dy_next = next_scan.get('dy')
            dt_next = next_scan.get('dt')
            
            if dx_next is None or dy_next is None or dt_next is None or dt_next == 0:
                continue
            
            u_true = float(dx_next) / float(dt_next)
            v_true = float(dy_next) / float(dt_next)
            
            # Sanity check
            if np.sqrt(u_true**2 + v_true**2) > max_velocity:
                continue
            if np.sqrt(avg_velocity[0]**2 + avg_velocity[1]**2) > max_velocity:
                continue
            
            y_true_list.append([u_true, v_true])
            y_baseline_list.append(avg_velocity)
    
    return np.array(y_true_list), np.array(y_baseline_list)

def evaluate_model(y_true, y_pred, model_name):
    """Compute comprehensive metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    u_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    v_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    
    r2_u = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_v = r2_score(y_true[:, 1], y_pred[:, 1])
    
    # Magnitude and direction
    mag_true = np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2)
    mag_pred = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)
    mag_mae = mean_absolute_error(mag_true, mag_pred)
    
    dir_true = np.arctan2(y_true[:, 1], y_true[:, 0]) * 180 / np.pi
    dir_pred = np.arctan2(y_pred[:, 1], y_pred[:, 0]) * 180 / np.pi
    dir_error = np.abs(dir_true - dir_pred)
    dir_error = np.minimum(dir_error, 360 - dir_error)
    dir_mae = np.mean(dir_error)
    
    return {
        'name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'u_mae': u_mae,
        'v_mae': v_mae,
        'r2_u': r2_u,
        'r2_v': r2_v,
        'mag_mae': mag_mae,
        'dir_mae': dir_mae,
        'n_samples': len(y_true)
    }

def compare_all_models(data_dir, model_dir, output_dir="results"):
    """Compare all three models"""
    print("="*80)
    print("MODEL COMPARISON: GBR vs GRU vs BASELINE")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    gbr_model, gbr_scaler = load_gbr_model(model_dir)
    gru_model, gru_scaler = load_gru_model(model_dir)
    
    # Load data for GBR
    print("\nLoading GBR validation data...")
    X_gbr, y_gbr = load_storm_data(data_dir)
    X_gbr_scaled = gbr_scaler.transform(X_gbr)
    
    # Filter for sanity check
    mag_gbr = np.sqrt(y_gbr[:, 0]**2 + y_gbr[:, 1]**2)
    valid_mask_gbr = mag_gbr <= 31
    X_gbr_scaled = X_gbr_scaled[valid_mask_gbr]
    y_gbr = y_gbr[valid_mask_gbr]
    
    # Predict GBR with timing
    print("Running GBR predictions...")
    import time
    start_time = time.time()
    y_gbr_pred = gbr_model.predict(X_gbr_scaled)
    gbr_time = time.time() - start_time
    gbr_time_per_sample = gbr_time / len(X_gbr_scaled) * 1000  # ms
    
    # Load data for GRU (use sequence_length=7 to match trained model)
    print("\nLoading GRU validation data...")
    X_gru, y_gru, _ = load_sequences(data_dir, sequence_length=7, max_velocity=31)
    
    # Predict GRU with timing
    print("Running GRU predictions...")
    start_time = time.time()
    y_gru_pred = gru_model.predict(X_gru, verbose=0)
    gru_time = time.time() - start_time
    gru_time_per_sample = gru_time / len(X_gru) * 1000  # ms
    
    # Compute baseline with timing
    print("\nComputing baseline predictions...")
    start_time = time.time()
    y_baseline_true, y_baseline_pred = compute_baseline_predictions(data_dir)
    baseline_time = time.time() - start_time
    baseline_time_per_sample = baseline_time / len(y_baseline_true) * 1000  # ms
    
    # Evaluate all models
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    results = []
    results.append(evaluate_model(y_gbr, y_gbr_pred, "GBR (XGBoost)"))
    results.append(evaluate_model(y_gru, y_gru_pred, "GRU (Temporal)"))
    results.append(evaluate_model(y_baseline_true, y_baseline_pred, "Baseline (5-scan avg)"))
    
    # Add timing info
    results[0]['time_ms'] = gbr_time_per_sample
    results[1]['time_ms'] = gru_time_per_sample
    results[2]['time_ms'] = baseline_time_per_sample
    
    # Print comparison table
    print(f"\n{'Model':<20} {'Samples':<10} {'MAE':>8} {'RMSE':>8} {'U-MAE':>8} {'V-MAE':>8} {'Dir-MAE':>10}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<20} {r['n_samples']:<10} {r['mae']:>8.2f} {r['rmse']:>8.2f} "
              f"{r['u_mae']:>8.2f} {r['v_mae']:>8.2f} {r['dir_mae']:>10.1f}°")
    
    # Print inference times
    print("\n" + "="*80)
    print("INFERENCE PERFORMANCE")
    print("="*80)
    print(f"\n{'Model':<20} {'Total Time':>12} {'Per Sample':>15} {'Throughput':>15}")
    print("-"*80)
    for r in results:
        total_time = r['time_ms'] * r['n_samples'] / 1000  # seconds
        throughput = r['n_samples'] / total_time if total_time > 0 else 0
        print(f"{r['name']:<20} {total_time:>10.2f}s {r['time_ms']:>12.4f} ms {throughput:>12.0f} samples/s")
    
    # Calculate improvements
    print("\n" + "="*80)
    print("IMPROVEMENTS OVER BASELINE")
    print("="*80)
    baseline_mae = results[2]['mae']
    for r in results[:2]:
        improvement = (baseline_mae - r['mae']) / baseline_mae * 100
        print(f"{r['name']:<20} {improvement:>6.1f}% better")
    
    # Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, "model_comparison.txt"), 'w') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'Samples':>10} {'Time/Sample':>15}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r['name']:<20} {r['mae']:>8.2f} {r['rmse']:>8.2f} {r['n_samples']:>10} {r['time_ms']:>12.4f} ms\n")
        
        f.write("\n\nIMPROVEMENTS OVER BASELINE\n")
        f.write("-"*80 + "\n")
        baseline_mae = results[2]['mae']
        for r in results[:2]:
            improvement = (baseline_mae - r['mae']) / baseline_mae * 100
            f.write(f"{r['name']:<20} {improvement:>6.1f}% better\n")
    
    # Create comparison plot
    create_comparison_plot(results, output_dir)
    
    print(f"\nResults saved to {output_dir}/model_comparison.txt")
    print(f"Plot saved to {output_dir}/model_comparison.png")

def create_comparison_plot(results, output_dir):
    """Create bar chart comparing models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = [r['name'] for r in results]
    maes = [r['mae'] for r in results]
    rmses = [r['rmse'] for r in results]
    
    colors = ['#2E86AB', '#A23B72', '#F77F00']
    
    # MAE comparison
    bars1 = ax1.bar(models, maes, color=colors, alpha=0.8)
    ax1.set_ylabel('Mean Absolute Error (m/s)', fontsize=12)
    ax1.set_title('Model Comparison - MAE', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, mae in zip(bars1, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # RMSE comparison
    bars2 = ax2.bar(models, rmses, color=colors, alpha=0.8)
    ax2.set_ylabel('Root Mean Squared Error (m/s)', fontsize=12)
    ax2.set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, rmse in zip(bars2, rmses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all models")
    parser.add_argument("--data_dir", required=True, help="Path to validation data")
    parser.add_argument("--model_dir", default="models", help="Directory with trained models")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    compare_all_models(args.data_dir, args.model_dir, args.output_dir)

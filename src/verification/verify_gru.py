"""
Dedicated verification script for StormCast GRU models.
Evaluates a trained model on a specific verification data folder.
"""
import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model', 'lstm'))

from gru_data_loader import load_sequences
from gru_model import velocity_aware_huber_loss, directional_error_deg, StormCastGRU, combined_loss

def verify_model(data_dir, model_path, sequence_length=8, residual=False):
    """
    Run evaluation on the specified data directory.
    """
    print("="*80)
    print(f"VERIFICATION: {os.path.basename(model_path)}")
    print("="*80)
    
    # 1. Load Model
    print(f"Loading model from {model_path}...")
    custom_objects = {
        "velocity_aware_huber_loss": velocity_aware_huber_loss,
        "directional_error_deg": directional_error_deg,
        "StormCastGRU": StormCastGRU,
        "combined_loss": combined_loss
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # 2. Load Scalers
    scaler_path = model_path.replace('.keras', '_scaler.pkl')
    if not os.path.exists(scaler_path):
        # Try finding it in the same directory if model_path is absolute
        scaler_path = os.path.join(os.path.dirname(model_path), os.path.basename(model_path).replace('.keras', '_scaler.pkl'))
        
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}")
        return
        
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    scaler_x = scalers['x']
    scaler_y = scalers['y']
    print("Scalers loaded successfully.")

    # 3. Load Verification Data
    print(f"Loading verification data from {data_dir}...")
    X_raw, y_true_scaled_or_absolute, ids = load_sequences(
        data_dir, 
        sequence_length=sequence_length, 
        residual=residual
    )
    
    if len(X_raw) == 0:
        print("Error: No sequences loaded. Check data_dir and sequence_length.")
        return
        
    print(f"Loaded {len(X_raw)} sequences.")
    
    # Filter outliers as per training protocol
    v_mags = np.sqrt(y_true_scaled_or_absolute[:, 0]**2 + y_true_scaled_or_absolute[:, 1]**2)
    threshold = 40.0 if residual else 80.0
    valid_mask = v_mags < threshold
    X_raw = X_raw[valid_mask]
    y_true_scaled_or_absolute = y_true_scaled_or_absolute[valid_mask]
    print(f"Filtered {np.sum(~valid_mask)} extreme outliers. Remaining: {len(X_raw)}")

    # 4. Preprocess
    n_samples, seq_len, n_features = X_raw.shape
    X_2d = X_raw.reshape(n_samples * seq_len, n_features)
    X_scaled_2d = scaler_x.transform(X_2d)
    X_scaled = X_scaled_2d.reshape(n_samples, seq_len, n_features)
    
    # 5. Predict
    print("Running predictions...")
    y_pred_scaled = model.predict(X_scaled, verbose=1)
    
    # 6. Post-process (Inverse Transform)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    if residual:
        # Reconstruct absolute velocities
        # u, v are at index 46, 47 in the features
        u_idx, v_idx = 46, 47
        curr_u = X_raw[:, -1, u_idx]
        curr_v = X_raw[:, -1, v_idx]
        
        y_true_abs = np.zeros_like(y_true_scaled_or_absolute)
        y_true_abs[:, 0] = y_true_scaled_or_absolute[:, 0] + curr_u
        y_true_abs[:, 1] = y_true_scaled_or_absolute[:, 1] + curr_v
        
        y_pred_abs = np.zeros_like(y_pred)
        y_pred_abs[:, 0] = y_pred[:, 0] + curr_u
        y_pred_abs[:, 1] = y_pred[:, 1] + curr_v
    else:
        # y_true_scaled_or_absolute is absolute but scaled? 
        # Actually load_sequences returns unscaled targets.
        # Wait, in train_gru.py, we split X_scaled and y_scaled.
        # load_sequences returns UNscaled y.
        y_true_abs = y_true_scaled_or_absolute
        y_pred_abs = y_pred

    # 7. Calculate Metrics
    mae = np.mean(np.abs(y_true_abs - y_pred_abs))
    u_mae = np.mean(np.abs(y_true_abs[:, 0] - y_pred_abs[:, 0]))
    v_mae = np.mean(np.abs(y_true_abs[:, 1] - y_pred_abs[:, 1]))
    
    # Directional MAE
    theta_true = np.degrees(np.arctan2(y_true_abs[:, 1], y_true_abs[:, 0]))
    theta_pred = np.degrees(np.arctan2(y_pred_abs[:, 1], y_pred_abs[:, 0]))
    diff = np.abs(theta_true - theta_pred)
    diff = np.where(diff > 180, 360 - diff, diff)
    dir_mae = np.mean(diff)
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true_abs - y_pred_abs)**2))
    
    print("\n" + "="*40)
    print("VERIFICATION RESULTS")
    print("="*40)
    print(f"Samples Evaluated: {len(y_true_abs)}")
    print(f"Speed MAE:        {mae:.4f} m/s")
    print(f"  U-Component MAE: {u_mae:.4f} m/s")
    print(f"  V-Component MAE: {v_mae:.4f} m/s")
    print(f"RMSE:             {rmse:.4f} m/s")
    print(f"Directional MAE:  {dir_mae:.2f} degrees")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify GRU Model on a data folder")
    parser.add_argument("--data_dir", required=True, help="Path to verification data (JSON folder structure)")
    parser.add_argument("--model_path", required=True, help="Full path to the .keras model file")
    parser.add_argument("--sequence_length", type=int, default=8, help="Lookback sequence length (default: 8 for v5)")
    parser.add_argument("--residual", action="store_true", help="Set if model was trained with --residual (v5+)")
    
    args = parser.parse_args()
    
    verify_model(
        args.data_dir, 
        args.model_path, 
        args.sequence_length, 
        args.residual
    )

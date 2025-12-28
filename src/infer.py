"""
Inference script to evaluate trained model on validation set
"""
import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from data_loader import load_storm_data

def load_model_and_scaler(model_dir):
    """Load trained model and scaler"""
    model_path = os.path.join(model_dir, "gb_storm_motion.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def compute_velocity_errors(y_true, y_pred):
    """Compute velocity-specific error metrics"""
    # Separate u and v components
    u_true, v_true = y_true[:, 0], y_true[:, 1]
    u_pred, v_pred = y_pred[:, 0], y_pred[:, 1]
    
    # Compute magnitude and direction
    mag_true = np.sqrt(u_true**2 + v_true**2)
    mag_pred = np.sqrt(u_pred**2 + v_pred**2)
    
    dir_true = np.arctan2(v_true, u_true) * 180 / np.pi  # degrees
    dir_pred = np.arctan2(v_pred, u_pred) * 180 / np.pi
    
    # Direction error (handle wraparound)
    dir_error = np.abs(dir_true - dir_pred)
    dir_error = np.minimum(dir_error, 360 - dir_error)
    
    return {
        'u_mae': mean_absolute_error(u_true, u_pred),
        'v_mae': mean_absolute_error(v_true, v_pred),
        'u_rmse': np.sqrt(mean_squared_error(u_true, u_pred)),
        'v_rmse': np.sqrt(mean_squared_error(v_true, v_pred)),
        'mag_mae': mean_absolute_error(mag_true, mag_pred),
        'mag_rmse': np.sqrt(mean_squared_error(mag_true, mag_pred)),
        'dir_mae': np.mean(dir_error),
        'r2_u': r2_score(u_true, u_pred),
        'r2_v': r2_score(v_true, v_pred)
    }

def analyze_errors_by_magnitude(y_true, y_pred):
    """Analyze errors for different storm motion magnitudes"""
    mag_true = np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2)
    
    # Define magnitude bins
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 100)]
    
    print("\n=== Error Analysis by Storm Speed ===")
    for low, high in bins:
        mask = (mag_true >= low) & (mag_true < high)
        if np.sum(mask) == 0:
            continue
        
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        mae = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        
        print(f"Speed {low}-{high} m/s ({np.sum(mask)} samples):")
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}")

def infer_on_validation(data_dir, model_dir, output_dir="results"):
    """Run inference on validation set and compute detailed metrics"""
    print(f"Loading model from {model_dir}...")
    model, scaler = load_model_and_scaler(model_dir)
    
    print(f"Loading validation data from {data_dir}...")
    X, y = load_storm_data(data_dir)
    
    if len(X) == 0:
        print("No valid data found. Exiting.")
        return
    
    print(f"Total validation samples: {len(X)}")
    
    # Normalize features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_scaled)
    
    # Sanity check: filter out unrealistic velocities (> 31 m/s)
    mag_true = np.sqrt(y[:, 0]**2 + y[:, 1]**2)
    mag_pred = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)
    
    valid_mask = (mag_true <= 31) & (mag_pred <= 31)
    n_filtered = np.sum(~valid_mask)
    
    if n_filtered > 0:
        print(f"\nFiltered {n_filtered} samples with velocity > 31 m/s (sanity check)")
        y = y[valid_mask]
        y_pred = y_pred[valid_mask]
        X = X[valid_mask]
        print(f"Remaining samples: {len(y)}")
    
    # Compute overall metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*50)
    print("VALIDATION SET RESULTS")
    print("="*50)
    print(f"Total Samples: {len(X)}")
    print(f"\nOverall Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    
    # Compute velocity-specific metrics
    vel_metrics = compute_velocity_errors(y, y_pred)
    
    print(f"\nComponent-wise Metrics:")
    print(f"  U-velocity MAE: {vel_metrics['u_mae']:.4f} m/s")
    print(f"  V-velocity MAE: {vel_metrics['v_mae']:.4f} m/s")
    print(f"  U-velocity RMSE: {vel_metrics['u_rmse']:.4f} m/s")
    print(f"  V-velocity RMSE: {vel_metrics['v_rmse']:.4f} m/s")
    print(f"  U-velocity R²: {vel_metrics['r2_u']:.4f}")
    print(f"  V-velocity R²: {vel_metrics['r2_v']:.4f}")
    
    print(f"\nMagnitude & Direction:")
    print(f"  Speed MAE: {vel_metrics['mag_mae']:.4f} m/s")
    print(f"  Speed RMSE: {vel_metrics['mag_rmse']:.4f} m/s")
    print(f"  Direction MAE: {vel_metrics['dir_mae']:.2f} degrees")
    
    # Error analysis by magnitude
    analyze_errors_by_magnitude(y, y_pred)
    
    # Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_file = os.path.join(output_dir, "validation_results.txt")
    with open(results_file, 'w') as f:
        f.write("VALIDATION SET RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Total Samples: {len(X)}\n\n")
        f.write(f"Overall MAE: {mae:.4f}\n")
        f.write(f"Overall RMSE: {rmse:.4f}\n")
        f.write(f"Overall MSE: {mse:.4f}\n\n")
        f.write(f"U-velocity MAE: {vel_metrics['u_mae']:.4f} m/s\n")
        f.write(f"V-velocity MAE: {vel_metrics['v_mae']:.4f} m/s\n")
        f.write(f"Speed MAE: {vel_metrics['mag_mae']:.4f} m/s\n")
        f.write(f"Direction MAE: {vel_metrics['dir_mae']:.2f} degrees\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Create scatter plots
    create_plots(y, y_pred, output_dir)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        **vel_metrics
    }

def create_plots(y_true, y_pred, output_dir):
    """Create visualization plots"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # U-velocity scatter
    axes[0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.3, s=1)
    axes[0].plot([-50, 50], [-50, 50], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('True U-velocity (m/s)')
    axes[0].set_ylabel('Predicted U-velocity (m/s)')
    axes[0].set_title('U-velocity Predictions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # V-velocity scatter
    axes[1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.3, s=1)
    axes[1].plot([-50, 50], [-50, 50], 'r--', label='Perfect prediction')
    axes[1].set_xlabel('True V-velocity (m/s)')
    axes[1].set_ylabel('Predicted V-velocity (m/s)')
    axes[1].set_title('V-velocity Predictions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'velocity_predictions.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plots saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on validation set")
    parser.add_argument("--data_dir", required=True, help="Path to validation data")
    parser.add_argument("--model_dir", default="models", help="Directory containing trained model")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    infer_on_validation(args.data_dir, args.model_dir, args.output_dir)

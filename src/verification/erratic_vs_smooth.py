"""
Compare model performance on erratic vs smooth storm motion
Erratic motion = high variance in velocity over time
Smooth motion = low variance in velocity over time
"""
import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model', 'gbr'))
from data_loader import load_storm_data
import glob
import json

def load_storm_sequences_with_variance(base_path, n_history=5):
    """
    Load storm data and calculate motion variance to classify erratic vs smooth
    
    Returns:
        Data for smooth storms and erratic storms separately
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
    
    smooth_data = {'X': [], 'y_true': [], 'y_baseline': []}
    erratic_data = {'X': [], 'y_true': [], 'y_baseline': []}
    
    # Process each storm track
    for id_val, history in grouped_data.items():
        history.sort(key=lambda x: x.get('timestamp', ''))
        
        if len(history) < n_history + 1:
            continue
        
        # Calculate variance for this storm track
        velocities = []
        for scan in history:
            if 'dx' in scan and 'dy' in scan and 'dt' in scan:
                dt = float(scan['dt'])
                if dt > 0:
                    u = float(scan['dx']) / dt
                    v = float(scan['dy']) / dt
                    velocities.append([u, v])
        
        if len(velocities) < n_history:
            continue
        
        # Calculate variance in velocity
        velocities = np.array(velocities)
        velocity_variance = np.var(velocities, axis=0).mean()  # Average variance across u and v
        
        # Classify as erratic (high variance) or smooth (low variance)
        # Threshold: median variance across all storms
        is_erratic = velocity_variance > 50  # Adjust threshold as needed
        
        # Process sequences for this storm
        for i in range(n_history, len(history)):
            past_scans = history[i-n_history:i]
            current_scan = history[i-1]
            next_scan = history[i]
            
            if not all('dx' in s and 'dy' in s and 'dt' in s for s in [current_scan, next_scan]):
                continue
            
            if not all('dx' in s and 'dy' in s and 'dt' in s for s in past_scans):
                continue
            
            # Calculate true target velocity
            dx_next = float(next_scan['dx'])
            dy_next = float(next_scan['dy'])
            dt_next = float(next_scan['dt'])
            
            if dt_next == 0:
                continue
            
            u_true = dx_next / dt_next
            v_true = dy_next / dt_next
            
            # Sanity check
            mag_true = np.sqrt(u_true**2 + v_true**2)
            if mag_true > 31:
                continue
            
            # Calculate baseline prediction
            past_velocities = []
            for scan in past_scans:
                dx = float(scan['dx'])
                dy = float(scan['dy'])
                dt = float(scan['dt'])
                if dt > 0:
                    past_velocities.append([dx/dt, dy/dt])
            
            if len(past_velocities) < n_history:
                continue
            
            avg_velocity = np.mean(past_velocities, axis=0)
            u_baseline = avg_velocity[0]
            v_baseline = avg_velocity[1]
            
            # Sanity check baseline
            mag_baseline = np.sqrt(u_baseline**2 + v_baseline**2)
            if mag_baseline > 31:
                continue
            
            # Get current scan motion for features
            dx_curr = float(current_scan['dx'])
            dy_curr = float(current_scan['dy'])
            dt_curr = float(current_scan['dt'])
            
            # Extract all features properly (matching data_loader.py)
            # Base features from properties
            features_to_extract = [
                'SRW46km', 'MeanWind_1-3kmAGL', 'EBShear',
                'EchoTop18', 'EchoTop30', 'PrecipRate', 'VILDensity', 
                'RALA', 'VII', 'ProbSevere', 'ProbWind', 'ProbHail', 'ProbTor',
                'MLCAPE', 'MUCAPE', 'MLCIN', 'DCAPE', 'CAPE_M10M30', 'LCL',
                'Wetbulb_0C_Hgt', 'LLLR', 'MLLR', 'SRH01km', 'SRH02km', 'LJA',
                'CompRef', 'Ref10', 'Ref20', 'MESH', 'H50_Above_0C', 'EchoTop50', 'VIL'
            ]
            
            # Extract base features
            features = []
            properties = current_scan.get('properties', {})
            valid_features = True
            
            for feat in features_to_extract:
                val = None
                if feat in current_scan:
                    val = current_scan[feat]
                elif feat in properties:
                    val = properties[feat]
                
                if val is None:
                    valid_features = False
                    break
                features.append(float(val))
            
            if not valid_features:
                continue
            
            # Add dx, dy, dt
            features.extend([dx_curr, dy_curr, dt_curr])
            
            # Feature engineering (matching data_loader.py)
            velocity_mag = np.sqrt(dx_curr**2 + dy_curr**2) / dt_curr if dt_curr > 0 else 0
            velocity_dir = np.arctan2(dy_curr, dx_curr)
            
            mlcape = float(current_scan.get('properties', {}).get('MLCAPE', 0))
            ebshear = float(current_scan.get('properties', {}).get('EBShear', 0))
            vil = float(current_scan.get('properties', {}).get('VIL', 0))
            precip_rate = float(current_scan.get('properties', {}).get('PrecipRate', 0))
            
            cape_shear = mlcape * ebshear / 1000.0
            vil_precip = vil * precip_rate / 100.0
            
            features.extend([velocity_mag, velocity_dir, cape_shear, vil_precip])
            
            # Now features should have 38 elements total
            target_data = erratic_data if is_erratic else smooth_data
            target_data['X'].append(features)
            target_data['y_true'].append([u_true, v_true])
            target_data['y_baseline'].append([u_baseline, v_baseline])
    
    # Convert to arrays
    for data in [smooth_data, erratic_data]:
        for key in data:
            data[key] = np.array(data[key])
    
    print(f"\nSmooth storms: {len(smooth_data['X'])} samples")
    print(f"Erratic storms: {len(erratic_data['X'])} samples")
    
    return smooth_data, erratic_data

def evaluate_model_on_category(model, scaler, X, y_true, category_name):
    """Evaluate ML model on a category"""
    if len(X) == 0:
        print(f"\nNo samples for {category_name}")
        return
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\n{category_name} - ML Model:")
    print(f"  MAE: {mae:.4f} m/s")
    print(f"  RMSE: {rmse:.4f} m/s")
    
    return {'mae': mae, 'rmse': rmse}

def evaluate_baseline_on_category(y_true, y_baseline, category_name):
    """Evaluate baseline on a category"""
    if len(y_true) == 0:
        print(f"\nNo samples for {category_name}")
        return
    
    mae = mean_absolute_error(y_true, y_baseline)
    rmse = np.sqrt(mean_squared_error(y_true, y_baseline))
    
    print(f"\n{category_name} - Baseline:")
    print(f"  MAE: {mae:.4f} m/s")
    print(f"  RMSE: {rmse:.4f} m/s")
    
    return {'mae': mae, 'rmse': rmse}

def compare_erratic_vs_smooth(data_dir, model_dir, n_history=5):
    """Main comparison function"""
    print("Loading model and scaler...")
    model_path = os.path.join(model_dir, "gb_storm_motion.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"\nLoading storm sequences with motion variance analysis...")
    smooth_data, erratic_data = load_storm_sequences_with_variance(data_dir, n_history)
    
    print("\n" + "="*70)
    print("ERRATIC vs SMOOTH MOTION COMPARISON")
    print("="*70)
    
    # Evaluate on smooth storms
    print("\n" + "-"*70)
    print("SMOOTH MOTION STORMS (Low velocity variance)")
    print("-"*70)
    smooth_ml = evaluate_model_on_category(model, scaler, smooth_data['X'], 
                                           smooth_data['y_true'], "Smooth Storms")
    smooth_baseline = evaluate_baseline_on_category(smooth_data['y_true'], 
                                                    smooth_data['y_baseline'], "Smooth Storms")
    
    # Evaluate on erratic storms
    print("\n" + "-"*70)
    print("ERRATIC MOTION STORMS (High velocity variance)")
    print("-"*70)
    erratic_ml = evaluate_model_on_category(model, scaler, erratic_data['X'], 
                                            erratic_data['y_true'], "Erratic Storms")
    erratic_baseline = evaluate_baseline_on_category(erratic_data['y_true'], 
                                                     erratic_data['y_baseline'], "Erratic Storms")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if smooth_ml and smooth_baseline:
        smooth_improvement = (smooth_baseline['mae'] - smooth_ml['mae']) / smooth_baseline['mae'] * 100
        print(f"\nSmooth Storms:")
        print(f"  ML Model MAE: {smooth_ml['mae']:.4f} m/s")
        print(f"  Baseline MAE: {smooth_baseline['mae']:.4f} m/s")
        print(f"  Improvement: {smooth_improvement:.1f}%")
    
    if erratic_ml and erratic_baseline:
        erratic_improvement = (erratic_baseline['mae'] - erratic_ml['mae']) / erratic_baseline['mae'] * 100
        print(f"\nErratic Storms:")
        print(f"  ML Model MAE: {erratic_ml['mae']:.4f} m/s")
        print(f"  Baseline MAE: {erratic_baseline['mae']:.4f} m/s")
        print(f"  Improvement: {erratic_improvement:.1f}%")
    
    # Create visualizations
    create_scatter_plots(smooth_data, erratic_data, model, scaler)

def create_scatter_plots(smooth_data, erratic_data, model, scaler):
    """Create scatter plots comparing predictions for smooth vs erratic storms"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Smooth storms - ML Model
    if len(smooth_data['X']) > 0:
        X_smooth_scaled = scaler.transform(smooth_data['X'])
        y_smooth_pred = model.predict(X_smooth_scaled)
        
        axes[0, 0].scatter(smooth_data['y_true'][:, 0], y_smooth_pred[:, 0], 
                          alpha=0.4, s=10, c='#2E86AB', label='U-velocity')
        axes[0, 0].scatter(smooth_data['y_true'][:, 1], y_smooth_pred[:, 1], 
                          alpha=0.4, s=10, c='#F77F00', label='V-velocity')
        axes[0, 0].plot([-30, 30], [-30, 30], 'r--', linewidth=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('True Velocity (m/s)', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Velocity (m/s)', fontsize=11)
        axes[0, 0].set_title('Smooth Storms - ML Model', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(-30, 30)
        axes[0, 0].set_ylim(-30, 30)
    
    # Smooth storms - Baseline
    if len(smooth_data['y_baseline']) > 0:
        axes[0, 1].scatter(smooth_data['y_true'][:, 0], smooth_data['y_baseline'][:, 0], 
                          alpha=0.4, s=10, c='#2E86AB', label='U-velocity')
        axes[0, 1].scatter(smooth_data['y_true'][:, 1], smooth_data['y_baseline'][:, 1], 
                          alpha=0.4, s=10, c='#F77F00', label='V-velocity')
        axes[0, 1].plot([-30, 30], [-30, 30], 'r--', linewidth=2, label='Perfect prediction')
        axes[0, 1].set_xlabel('True Velocity (m/s)', fontsize=11)
        axes[0, 1].set_ylabel('Predicted Velocity (m/s)', fontsize=11)
        axes[0, 1].set_title('Smooth Storms - Baseline', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(-30, 30)
        axes[0, 1].set_ylim(-30, 30)
    
    # Erratic storms - ML Model
    if len(erratic_data['X']) > 0:
        X_erratic_scaled = scaler.transform(erratic_data['X'])
        y_erratic_pred = model.predict(X_erratic_scaled)
        
        axes[1, 0].scatter(erratic_data['y_true'][:, 0], y_erratic_pred[:, 0], 
                          alpha=0.3, s=5, c='#2E86AB', label='U-velocity')
        axes[1, 0].scatter(erratic_data['y_true'][:, 1], y_erratic_pred[:, 1], 
                          alpha=0.3, s=5, c='#F77F00', label='V-velocity')
        axes[1, 0].plot([-30, 30], [-30, 30], 'r--', linewidth=2, label='Perfect prediction')
        axes[1, 0].set_xlabel('True Velocity (m/s)', fontsize=11)
        axes[1, 0].set_ylabel('Predicted Velocity (m/s)', fontsize=11)
        axes[1, 0].set_title('Erratic Storms - ML Model', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(-30, 30)
        axes[1, 0].set_ylim(-30, 30)
    
    # Erratic storms - Baseline
    if len(erratic_data['y_baseline']) > 0:
        axes[1, 1].scatter(erratic_data['y_true'][:, 0], erratic_data['y_baseline'][:, 0], 
                          alpha=0.3, s=5, c='#2E86AB', label='U-velocity')
        axes[1, 1].scatter(erratic_data['y_true'][:, 1], erratic_data['y_baseline'][:, 1], 
                          alpha=0.3, s=5, c='#F77F00', label='V-velocity')
        axes[1, 1].plot([-30, 30], [-30, 30], 'r--', linewidth=2, label='Perfect prediction')
        axes[1, 1].set_xlabel('True Velocity (m/s)', fontsize=11)
        axes[1, 1].set_ylabel('Predicted Velocity (m/s)', fontsize=11)
        axes[1, 1].set_title('Erratic Storms - Baseline', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(-30, 30)
        axes[1, 1].set_ylim(-30, 30)
    
    plt.tight_layout()
    plt.savefig('results/erratic_vs_smooth_scatter.png', dpi=150, bbox_inches='tight')
    print(f"\nScatter plots saved to results/erratic_vs_smooth_scatter.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare erratic vs smooth motion")
    parser.add_argument("--data_dir", required=True, help="Path to validation data")
    parser.add_argument("--model_dir", default="models", help="Directory containing trained model")
    parser.add_argument("--n_history", type=int, default=5, help="Number of past scans")
    
    args = parser.parse_args()
    
    compare_erratic_vs_smooth(args.data_dir, args.model_dir, args.n_history)

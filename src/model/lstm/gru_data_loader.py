"""
Sequence data loader for GRU model
Loads storm tracks as temporal sequences with bounding box features
"""
import os
import glob
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def extract_bounding_box_features(entry):
    """Extract bounding box features from storm entry"""
    bbox = entry.get('bbox', entry.get('properties', {}).get('bbox', None))
    
    if bbox is None or len(bbox) < 4:
        return None
    
    # Handle nested list format [[x_min, y_min], [x_max, y_max]]
    if isinstance(bbox[0], list):
        if len(bbox) < 2 or len(bbox[0]) < 2 or len(bbox[1]) < 2:
            return None
        x_min, y_min = float(bbox[0][0]), float(bbox[0][1])
        x_max, y_max = float(bbox[1][0]), float(bbox[1][1])
    else:
        # Handle flat list format [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    
    # Calculate derived features
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    aspect_ratio = width / height if height > 0 else 0
    
    return [x_min, y_min, x_max, y_max, area, aspect_ratio]

def load_sequences(base_path, sequence_length=5, max_velocity=31):
    """
    Load storm data as temporal sequences
    
    Args:
        base_path: Path to data directory
        sequence_length: Number of scans in each sequence
        max_velocity: Maximum velocity for sanity check
        
    Returns:
        X_sequences: (n_samples, sequence_length, n_features)
        y: (n_samples, 2) - target velocities
        scaler: Fitted StandardScaler
    """
    # Feature list (38 + 6 bbox = 44 features)
    features_to_extract = [
        'SRW46km', 'MeanWind_1-3kmAGL', 'EBShear',
        'EchoTop18', 'EchoTop30', 'PrecipRate', 'VILDensity', 
        'RALA', 'VII', 'ProbSevere', 'ProbWind', 'ProbHail', 'ProbTor',
        'MLCAPE', 'MUCAPE', 'MLCIN', 'DCAPE', 'CAPE_M10M30', 'LCL',
        'Wetbulb_0C_Hgt', 'LLLR', 'MLLR', 'SRH01km', 'SRH02km', 'LJA',
        'CompRef', 'Ref10', 'Ref20', 'MESH', 'H50_Above_0C', 'EchoTop50', 'VIL'
    ]
    
    # Load all data
    all_raw_entries = []
    date_folders = glob.glob(os.path.join(base_path, "*"))
    print(f"Found {len(date_folders)} potential date folders")
    
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
    
    # Group by storm ID
    grouped_data = {}
    for entry in all_raw_entries:
        storm_id = entry.get('id')
        if storm_id is None:
            continue
        if storm_id not in grouped_data:
            grouped_data[storm_id] = []
        grouped_data[storm_id].append(entry)
    
    print(f"Found {len(grouped_data)} unique storm tracks")
    
    # Create sequences
    sequences = []
    targets = []
    skipped = 0
    
    for storm_id, track in grouped_data.items():
        # Sort by timestamp
        track.sort(key=lambda x: x.get('timestamp', ''))
        
        # Need at least sequence_length + 1 scans
        if len(track) < sequence_length + 1:
            skipped += len(track)
            continue
        
        # Create sequences
        for i in range(len(track) - sequence_length):
            sequence_scans = track[i:i+sequence_length]
            target_scan = track[i+sequence_length]
            
            # Extract features for each scan in sequence
            sequence_features = []
            valid_sequence = True
            
            for scan in sequence_scans:
                # Extract base features
                scan_features = []
                properties = scan.get('properties', {})
                
                for feat in features_to_extract:
                    val = scan.get(feat, properties.get(feat, None))
                    if val is None:
                        valid_sequence = False
                        break
                    scan_features.append(float(val))
                
                if not valid_sequence:
                    break
                
                # Add motion features
                dx = scan.get('dx')
                dy = scan.get('dy')
                dt = scan.get('dt')
                
                if dx is None or dy is None or dt is None or dt == 0:
                    valid_sequence = False
                    break
                
                dx, dy, dt = float(dx), float(dy), float(dt)
                scan_features.extend([dx, dy, dt])
                
                # Feature engineering
                velocity_mag = np.sqrt(dx**2 + dy**2) / dt
                velocity_dir = np.arctan2(dy, dx)
                
                mlcape = float(properties.get('MLCAPE', 0))
                ebshear = float(properties.get('EBShear', 0))
                vil = float(properties.get('VIL', 0))
                precip_rate = float(properties.get('PrecipRate', 0))
                
                cape_shear = mlcape * ebshear / 1000.0
                vil_precip = vil * precip_rate / 100.0
                
                scan_features.extend([velocity_mag, velocity_dir, cape_shear, vil_precip])
                
                # Add bounding box features
                bbox_features = extract_bounding_box_features(scan)
                if bbox_features is None:
                    valid_sequence = False
                    break
                scan_features.extend(bbox_features)
                
                sequence_features.append(scan_features)
            
            if not valid_sequence:
                skipped += 1
                continue
            
            # Extract target
            target_dx = target_scan.get('dx')
            target_dy = target_scan.get('dy')
            target_dt = target_scan.get('dt')
            
            if target_dx is None or target_dy is None or target_dt is None or target_dt == 0:
                skipped += 1
                continue
            
            target_dx, target_dy, target_dt = float(target_dx), float(target_dy), float(target_dt)
            u_target = target_dx / target_dt
            v_target = target_dy / target_dt
            
            # Sanity check
            mag = np.sqrt(u_target**2 + v_target**2)
            if mag > max_velocity:
                skipped += 1
                continue
            
            sequences.append(sequence_features)
            targets.append([u_target, v_target])
    
    print(f"Created {len(sequences)} sequences, skipped {skipped} invalid entries")
    
    if len(sequences) == 0:
        raise ValueError("No valid sequences found!")
    
    # Convert to numpy arrays
    X_sequences = np.array(sequences)  # (n_samples, seq_len, n_features)
    y = np.array(targets)  # (n_samples, 2)
    
    print(f"Sequence shape: {X_sequences.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features per timestep: {X_sequences.shape[2]}")
    
    # Normalize features
    # Reshape to (n_samples * seq_len, n_features) for scaling
    n_samples, seq_len, n_features = X_sequences.shape
    X_flat = X_sequences.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    
    # Reshape back to sequences
    X_sequences_scaled = X_flat_scaled.reshape(n_samples, seq_len, n_features)
    
    return X_sequences_scaled, y, scaler

if __name__ == "__main__":
    # Test data loading
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        X, y, scaler = load_sequences(data_dir)
        print(f"\nSuccessfully loaded {len(X)} sequences!")
        print(f"Feature dimensions: {X.shape}")

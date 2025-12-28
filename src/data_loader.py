import json
import os
import glob
import numpy as np
from pathlib import Path

def load_storm_data(base_path, features_to_extract=None):
    """
    Load storm data from the specified base path.
    Expected structure: base_path/{date}/cells/*.json
    
    Args:
        base_path (str): Path to the root directory containing date folders.
        features_to_extract (list): List of feature names to extract from properties.
                                    Defaults to ['SRW46km', 'MeanWind_1-3kmAGL', 'EBShear'].
                                    Also extracts 'dx', 'dy', 'dt' by default.
    
    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target matrix (u, v velocities).
    """
    if features_to_extract is None:
        features_to_extract = ['SRW46km', 'MeanWind_1-3kmAGL', 'EBShear']
    
    # Always include dx, dy, dt in features as requested
    required_keys = ['dx', 'dy', 'dt']
    all_features_list = features_to_extract + required_keys
    
    X_list = []
    y_list = []
    
    # search for date folders
    date_folders = glob.glob(os.path.join(base_path, "*"))
    
    print(f"Found {len(date_folders)} potential date folders in {base_path}")
    
    count = 0
    skipped = 0
    
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
                
                # We expect a list of storm objects or a single dict? 
                # The user example showed a list of dicts: [ {...}, {...} ]
                
                input_data = data
                if isinstance(data, dict):
                    input_data = [data]
                
                for entry in input_data:
                    # Check for required keys
                    if not all(key in entry for key in required_keys):
                        skipped += 1
                        continue
                    
                    # Extract Features
                    # Some features are in 'properties', some in root (dx, dy, dt)
                    features = []
                    properties = entry.get('properties', {})
                    
                    valid_entry = True
                    for feat in all_features_list:
                        val = None
                        if feat in entry:
                            val = entry[feat]
                        elif feat in properties:
                            val = properties[feat]
                        
                        if val is None:
                            valid_entry = False
                            break
                        features.append(float(val))
                    
                    if not valid_entry:
                        skipped += 1
                        continue
                        
                    # Calculate Targets (Velocity)
                    dx = float(entry['dx'])
                    dy = float(entry['dy'])
                    dt = float(entry['dt'])
                    
                    if dt == 0:
                        skipped += 1
                        continue
                        
                    u = dx / dt
                    v = dy / dt
                    
                    X_list.append(features)
                    y_list.append([u, v])
                    count += 1
                    
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue

    print(f"Loaded {count} samples. Skipped {skipped} entries due to missing keys or invalid data.")
    
    return np.array(X_list), np.array(y_list)

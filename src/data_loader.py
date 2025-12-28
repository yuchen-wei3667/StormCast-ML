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
        features_to_extract = [
            'SRW46km', 'MeanWind_1-3kmAGL', 'EBShear',
            'EchoTop18', 'EchoTop30', 'PrecipRate', 'VILDensity', 
            'RALA', 'VII', 'ProbSevere', 'ProbWind', 'ProbHail', 'ProbTor',
            'MLCAPE', 'MUCAPE', 'MLCIN', 'DCAPE', 'CAPE_M10M30', 'LCL',
            'Wetbulb_0C_Hgt', 'LLLR', 'MLLR', 'SRH01km', 'SRH02km', 'LJA',
            'CompRef', 'Ref10', 'Ref20', 'MESH', 'H50_Above_0C', 'EchoTop50', 'VIL'
        ]
    
    # Always include dx, dy, dt in features as requested
    required_keys = ['dx', 'dy', 'dt']
    all_features_list = features_to_extract + required_keys
    
    # Store all raw entries first to handle grouping
    all_raw_entries = []
    
    # search for date folders
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
                
                input_data = data
                if isinstance(data, dict):
                    input_data = [data]
                
                all_raw_entries.extend(input_data)
                    
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
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
    y_list = []
    skipped = 0
    used_pairs = 0
    
    # Process each group
    for id_val, history in grouped_data.items():
        # Sort by timestamp
        # Assume timestamp is in ISO format which sorts correctly as string
        history.sort(key=lambda x: x.get('timestamp', ''))
        
        for i in range(len(history) - 1):
            current_entry = history[i]
            next_entry = history[i+1]
            
            # Check if current entry is valid (has input features)
            if not all(key in current_entry for key in required_keys):
                skipped += 1
                continue
                
            # Check if next entry is valid (has target info)
            if not all(key in next_entry for key in required_keys):
                skipped += 1
                continue
                
            # Extract Input Features from CURRENT entry
            features = []
            properties = current_entry.get('properties', {})
            
            valid_input = True
            for feat in all_features_list:
                val = None
                if feat in current_entry:
                    val = current_entry[feat]
                elif feat in properties:
                    val = properties[feat]
                
                if val is None:
                    valid_input = False
                    break
                features.append(float(val))
            
            if not valid_input:
                skipped += 1
                continue
                
            # Calculate Target from NEXT entry
            # Target is the velocity of the next scan
            dx_next = float(next_entry['dx'])
            dy_next = float(next_entry['dy'])
            dt_next = float(next_entry['dt'])
            
            if dt_next == 0:
                skipped += 1
                continue
                
            u_target = dx_next / dt_next
            v_target = dy_next / dt_next
            
            X_list.append(features)
            y_list.append([u_target, v_target])
            used_pairs += 1

    print(f"Loaded {used_pairs} training pairs. Skipped {skipped} invalid or unpaired entries.")
    
    return np.array(X_list), np.array(y_list)

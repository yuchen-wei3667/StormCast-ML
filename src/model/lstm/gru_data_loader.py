import json
import os
import glob
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default features to extract
DEFAULT_FEATURES = [
    'MeanWind_1-3kmAGL', 'SRW46km', 'EBShear', 'SRH01km', 'SRH02km',
    'LLLR', 'MLLR', 'MLCAPE', 'MUCAPE', 'MLCIN', 'DCAPE', 'LCL',
    'Wetbulb_0C_Hgt', 'CompRef', 'Ref10', 'Ref20', 'EchoTop18',
    'EchoTop30', 'EchoTop50', 'VIL', 'VILDensity', 'PrecipRate',
    'H50_Above_0C', 'MESH'
]

REQUIRED_KEYS = ['dx', 'dy', 'dt']

def _extract_property_features(entry, feature_names):
    """Extracts scalar property features from an entry."""
    features = []
    properties = entry.get('properties', {})
    
    for feat in feature_names:
        val = properties.get(feat)
        if val is None:
            val = entry.get(feat)  # Fallback to top level
        
        if val is None:
            return None # Indicate missing feature
        features.append(float(val))
        
    return features

def _calculate_geometry_features(entry):
    """Calculates geometric measurements from centroid and bbox."""
    # 1. Centroid
    centroid = entry.get('centroid') # [lon, lat]
    if not centroid or len(centroid) < 2:
        return None
    
    centroid_lon = float(centroid[0])
    centroid_lat = float(centroid[1])
    
    # 2. Bbox
    bbox = entry.get('bbox') # List of (lat, lon) points
    if not bbox:
        return None
        
    try:
        lats = np.array([float(p[0]) for p in bbox])
        lons = np.array([float(p[1]) for p in bbox])
        
        min_lat, max_lat = np.min(lats), np.max(lats)
        min_lon, max_lon = np.min(lons), np.max(lons)
        
        width = max_lon - min_lon
        height = max_lat - min_lat
        
        if width == 0 or height == 0:
            return None

        # Basic Features
        aspect_ratio = width / height
        # Area estimation (Shoelace formula)
        # 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
        polygon_area = 0.5 * np.abs(np.dot(lons, np.roll(lats, 1)) - np.dot(lats, np.roll(lons, 1)))
        
        # Perimeter estimation
        dx = np.diff(lons)
        dy = np.diff(lats)
        perimeter = np.sum(np.sqrt(dx**2 + dy**2)) + \
                    np.sqrt((lons[-1]-lons[0])**2 + (lats[-1]-lats[0])**2)
                    
        # Compactness: 4 * pi * Area / Perimeter^2 (Isoperimetric quotient)
        compactness = (4 * np.pi * polygon_area / (perimeter**2)) if perimeter > 0 else 0
        
        # Orientation (PCA)
        coords = np.vstack([lons - np.mean(lons), lats - np.mean(lats)])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        
        # Major axis index
        major_idx = np.argmax(evals)
        major_axis = evecs[:, major_idx]
        
        orientation = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
        
        return [centroid_lon, centroid_lat, width, height, polygon_area, aspect_ratio, compactness, orientation]
        
    except Exception as e:
        # logging.debug(f"Geometry calculation failed: {e}")
        return None

def load_sequences(base_path, sequence_length=1, features_to_extract=None):
    """
    Load storm data sequences from the specified base path.
    Structure: base_path/{date}/cells/*.json
    """
    if features_to_extract is None:
        features_to_extract = DEFAULT_FEATURES
        
    all_raw_entries = []
    
    # 1. Collect all data
    date_folders = glob.glob(os.path.join(base_path, "*"))
    logging.info(f"Found {len(date_folders)} potential date folders in {base_path}")
    
    for date_folder in date_folders:
        if not os.path.isdir(date_folder): continue
            
        cells_dir = os.path.join(date_folder, "cells")
        if not os.path.exists(cells_dir): continue
            
        json_files = glob.glob(os.path.join(cells_dir, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    all_raw_entries.extend(data)
                elif isinstance(data, dict):
                    all_raw_entries.append(data)
            except Exception as e:
                logging.warning(f"Error reading {json_file}: {e}")

    # 2. Group by ID
    grouped_data = {}
    for entry in all_raw_entries:
        storm_id = entry.get('id')
        if storm_id:
            grouped_data.setdefault(storm_id, []).append(entry)
            
    # 3. Process Sequences
    X_list, y_list, ids_list = [], [], []
    skipped_missing, skipped_short = 0, 0
    
    logging.info(f"Processing {len(grouped_data)} storm tracks...")
    
    for storm_id, track in grouped_data.items():
        track.sort(key=lambda x: x.get('timestamp', ''))
        
        if len(track) < sequence_length + 1:
            skipped_short += len(track)
            continue
            
        for i in range(len(track) - sequence_length):
            sequence_entries = track[i : i + sequence_length]
            target_entry = track[i + sequence_length]
            
            # Target Validation
            if not all(k in target_entry for k in REQUIRED_KEYS) or target_entry['dt'] == 0:
                skipped_missing += 1
                continue
                
            # Sequence Feature Extraction
            seq_features = []
            valid_sequence = True
            
            for entry in sequence_entries:
                # Basic Checks
                if not all(k in entry for k in REQUIRED_KEYS):
                    valid_sequence = False
                    break
                    
                # A. Properties
                props = _extract_property_features(entry, features_to_extract)
                if props is None:
                    valid_sequence = False
                    break
                    
                # B. Spatial
                spatial = [float(entry['dx']), float(entry['dy']), float(entry['dt'])]
                
                # C. Geometry
                geom = _calculate_geometry_features(entry)
                if geom is None:
                    valid_sequence = False
                    break
                    
                # Combine
                step_features = props + spatial + geom
                seq_features.append(step_features)
            
            if not valid_sequence:
                skipped_missing += 1
                continue
                
            # Compute Target
            u_target = float(target_entry['dx']) / float(target_entry['dt'])
            v_target = float(target_entry['dy']) / float(target_entry['dt'])
            
            # Filter stationary storms (velocity < 1.8 m/s)
            velocity_mag = np.sqrt(u_target**2 + v_target**2)
            if velocity_mag < 1.8:
                continue
            
            X_list.append(seq_features)
            y_list.append([u_target, v_target])
            ids_list.append(storm_id)
            
    logging.info(f"Loaded {len(X_list)} sequences. Skipped {skipped_missing} missing/invalid, {skipped_short} too short.")
    
    return np.array(X_list), np.array(y_list), ids_list

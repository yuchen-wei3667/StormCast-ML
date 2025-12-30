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
    'H50_Above_0C', 'MESH', 'RALA', 'VII', 'ProbSevere', 'ProbWind', 
    'ProbHail', 'ProbTor', 'CAPE_M10M30', 'LJA'
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
    if not bbox or len(bbox) < 3: # Need at least 3 points for area
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
        polygon_area = 0.5 * np.abs(np.dot(lons, np.roll(lats, 1)) - np.dot(lats, np.roll(lons, 1)))
        
        # Perimeter estimation
        dx = np.diff(lons)
        dy = np.diff(lats)
        perimeter = np.sum(np.sqrt(dx**2 + dy**2)) + \
                    np.sqrt((lons[-1]-lons[0])**2 + (lats[-1]-lats[0])**2)
                    
        # Compactness: 4 * pi * Area / Perimeter^2
        compactness = (4 * np.pi * polygon_area / (perimeter**2)) if perimeter > 0 else 0
        
        # Orientation (PCA)
        # Use a small epsilon to avoid singular covariance
        coords = np.vstack([lons - np.mean(lons), lats - np.mean(lats)])
        if coords.shape[1] < 2:
            orientation = 0.0
        else:
            cov = np.cov(coords)
            if np.all(np.isfinite(cov)) and np.any(cov != 0):
                evals, evecs = np.linalg.eig(cov)
                major_idx = np.argmax(evals)
                major_axis = evecs[:, major_idx]
                orientation = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
            else:
                orientation = 0.0
        
        return [centroid_lon, centroid_lat, width, height, polygon_area, aspect_ratio, compactness, orientation]
        
    except Exception as e:
        return None

def _extract_engineered_features(entry):
    """Calculates derived meteorological and kinematic features."""
    props = entry.get('properties', {})
    
    # 1. Storm Intensity Interaction
    mlcape = float(props.get('MLCAPE', 0))
    ebshear = float(props.get('EBShear', 0))
    vil = float(props.get('VIL', 0))
    precip_rate = float(props.get('PrecipRate', 0))
    
    cape_shear = mlcape * ebshear / 1000.0
    vil_precip = vil * precip_rate / 100.0
    
    # 2. Current movement (Persistence basis)
    dx = float(entry.get('dx', 0))
    dy = float(entry.get('dy', 0))
    dt = float(entry.get('dt', 1))
    
    u = dx / dt
    v = dy / dt
    velocity_mag = np.sqrt(u**2 + v**2)
    velocity_dir = np.degrees(np.arctan2(v, u))
    
    # Return u, v as well to make identity learning easier
    return [cape_shear, vil_precip, u, v, velocity_mag, velocity_dir]

def load_sequences(base_path, sequence_length=1, features_to_extract=None, residual=False):
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
            
        # Try both the 'cells' subfolder and the root of the date folder
        cells_dir = os.path.join(date_folder, "cells")
        if os.path.exists(cells_dir):
            json_files = glob.glob(os.path.join(cells_dir, "*.json"))
        else:
            json_files = glob.glob(os.path.join(date_folder, "*.json"))
        
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
            
        # 1. Pre-calculate features for the whole track
        track_processed_features = []
        
        for entry in track:
            # Basic Checks
            if not all(k in entry for k in REQUIRED_KEYS):
                track_processed_features.append(None)
                continue
                
            props = _extract_property_features(entry, features_to_extract)
            if props is None:
                track_processed_features.append(None)
                continue
                
            spatial = [float(entry['dx']), float(entry['dy']), float(entry['dt'])]
            geom = _calculate_geometry_features(entry)
            if geom is None:
                track_processed_features.append(None)
                continue
                
            eng = _extract_engineered_features(entry)
            
            step_features = props + spatial + geom + eng
            track_processed_features.append(step_features)
            
        # 2. Create Sequences
        for i in range(len(track) - sequence_length):
            target_idx = i + sequence_length
            target_entry = track[target_idx]
            
            if not all(k in target_entry for k in REQUIRED_KEYS) or target_entry['dt'] == 0:
                skipped_missing += 1
                continue
 
            # Check sequence validity
            seq_features = []
            valid_sequence = True
            
            for j in range(sequence_length):
                feat = track_processed_features[i + j]
                if feat is None:
                    valid_sequence = False
                    break
                seq_features.append(feat)
                
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
                
            if residual:
                # Current velocity is from the last step of the sequence
                curr_entry = track[target_idx - 1]
                u_curr = float(curr_entry['dx']) / float(curr_entry['dt'])
                v_curr = float(curr_entry['dy']) / float(curr_entry['dt'])
                y_val = [u_target - u_curr, v_target - v_curr]
            else:
                y_val = [u_target, v_target]
            
            X_list.append(seq_features)
            y_list.append(y_val)
            ids_list.append(storm_id)
            
    logging.info(f"Loaded {len(X_list)} sequences. Skipped {skipped_missing} missing/invalid, {skipped_short} too short.")
    return np.array(X_list), np.array(y_list), ids_list

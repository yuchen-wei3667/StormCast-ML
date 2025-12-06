import json
import torch
import sys
import os

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import StormCellLSTM

def main():
    # Load sample data
    json_file = 'TrainingData/stormcells_TX_20251123.json'
    print(f"Loading data from {json_file}...")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_file} not found.")
        return

    features = data.get('features', [])
    if not features:
        print("No features found in JSON.")
        return
        
    print(f"Found {len(features)} storm cells.")
    
    # Initialize model
    model = StormCellLSTM()
    print("Model initialized.")
    
    # Pick a sample
    sample_idx = 0
    sample_cell = features[sample_idx]
    print(f"Testing on storm cell ID: {sample_cell.get('id')}")
    
    # Run prediction
    prediction = model.predict_from_json(sample_cell)
    
    if prediction is None:
        print("Prediction failed (insufficient history).")
    else:
        print(f"Predicted Motion Vector (vx, vy): {prediction}")
        print("Verification Successful!")

if __name__ == "__main__":
    main()

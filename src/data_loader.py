import json
import torch
import numpy as np
from torch.utils.data import Dataset

class StormCellDataset(Dataset):
    def __init__(self, json_file, sequence_length=10):
        self.data = self.load_data(json_file)
        self.sequence_length = sequence_length

    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data['features']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx]
        storm_history = feature.get('storm_history', [])
        
        # Extract features
        # We need to handle cases where history is shorter than sequence_length
        # For now, let's assume we just take the available history up to sequence_length
        # and pad if necessary, or just return what we have and let the collate_fn handle it
        # But for a simple LSTM, fixed length is easier.
        
        # Features to extract:
        # vx, vy, max_refl, VIL, ProbSevere, FlashRate, EchoTop18
        
        sequences = []
        targets = []
        
        # Sort history by timestamp just in case
        storm_history.sort(key=lambda x: x['timestamp'])
        
        for point in storm_history:
            # Input features
            feats = [
                float(point.get('dx', 0)),
                float(point.get('dy', 0)),
                float(point.get('dt', 0)),
                float(point.get('EBShear', 0)),
                float(point.get('SRW46km', 0)),
                float(point.get('MeanWind_1-3kmAGL', 0)),
            ]
            sequences.append(feats)
            
            # Target is the NEXT step's vx, vy. 
            # But wait, the prompt asks to "output predicted motion vectors".
            # Usually this means predicting the motion at the current step or the next step.
            # Let's assume we want to predict the CURRENT motion vector given the history up to this point?
            # Or predict the NEXT motion vector?
            # Given "predict storm cell motion", usually implies future motion.
            # Let's set target as the vx, vy of the *next* timestep.
            # However, for the purpose of this specific request "takes in the JSON input directly and outputs predicted motion vectors",
            # it might mean inference mode.
            
        # For training, we would create sliding windows. 
        # For this specific file, let's just make it capable of parsing a single storm cell's history 
        # and returning a tensor suitable for the model.
        
        # Let's return the full history as a sequence
        return torch.tensor(sequences, dtype=torch.float32)

def parse_storm_cell_json(json_data):
    """
    Parses a single storm cell feature from the JSON and returns a tensor.
    """
    storm_history = json_data.get('storm_history', [])
    storm_history.sort(key=lambda x: x['timestamp'])
    
    sequences = []
    for point in storm_history:
        feats = [
            float(point.get('dx', 0)),
            float(point.get('dy', 0)),
            float(point.get('dt', 0)),
            float(point.get('EBShear', 0)),
            float(point.get('SRW46km', 0)),
            float(point.get('MeanWind_1-3kmAGL', 0)),
        ]
        sequences.append(feats)
        
    if not sequences:
        return torch.empty(0, 6)
        
    return torch.tensor(sequences, dtype=torch.float32).unsqueeze(0) # Add batch dimension

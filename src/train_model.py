import json
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import StormCellLSTM

def create_training_data(json_files):
    """
    Load storm cell data from multiple files and create training sequences.
    Input: sequence of (dx, dy, dt)
    Target: (vx, vy) calculated from next timestep's dx, dy, dt
    """
    X_train = []
    y_train = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {json_file} not found, skipping.")
            continue
        
        features = data.get('features', [])
        
        for feature in features:
            storm_history = feature.get('storm_history', [])
            
            if len(storm_history) < 2:
                continue
            
            # Sort by timestamp
            storm_history.sort(key=lambda x: x['timestamp'])
            
            # Create sequences
            for i in range(len(storm_history) - 1):
                # Input: all history up to point i
                sequence = []
                for j in range(i + 1):
                    point = storm_history[j]
                    feats = [
                        float(point.get('dx', 0)),
                        float(point.get('dy', 0)),
                        float(point.get('dt', 1)),
                        float(point.get('EBShear', 0)),
                        float(point.get('SRW46km', 0)),
                        float(point.get('MeanWind_1-3kmAGL', 0)),
                    ]
                    sequence.append(feats)
                
                # Target: velocity at next timestep (i+1)
                next_point = storm_history[i + 1]
                dx_next = float(next_point.get('dx', 0))
                dy_next = float(next_point.get('dy', 0))
                dt_next = float(next_point.get('dt', 1))
                
                if dt_next > 0:
                    vx_target = dx_next / dt_next
                    vy_target = dy_next / dt_next
                else:
                    vx_target = 0.0
                    vy_target = 0.0
                
                X_train.append(sequence)
                y_train.append([vx_target, vy_target])
    
    return X_train, y_train

def pad_sequences(sequences, max_len=None):
    """Pad sequences to same length"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with zeros
            padding = [[0, 0, 0, 0, 0, 0]] * (max_len - len(seq))
            padded.append(padding + seq)
        else:
            padded.append(seq[-max_len:])  # Take last max_len
    
    return padded

def train_model(json_files, epochs=300, batch_size=32, max_seq_len=20, learning_rate=0.001):
    """Train the LSTM model"""
    print("Loading and preparing training data...")
    X_train, y_train = create_training_data(json_files)
    
    print(f"Total training samples: {len(X_train)}")
    
    # Pad sequences
    X_train_padded = pad_sequences(X_train, max_len=max_seq_len)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_train_padded, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    print(f"Input shape: {X_tensor.shape}")
    print(f"Target shape: {y_tensor.shape}")
    
    # Split into train and validation (80/20)
    split_idx = int(0.8 * len(X_tensor))
    X_train_split = X_tensor[:split_idx]
    y_train_split = y_tensor[:split_idx]
    X_val = X_tensor[split_idx:]
    y_val = y_tensor[split_idx:]
    
    print(f"Training samples: {len(X_train_split)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    model = StormCellLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=2)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        indices = torch.randperm(len(X_train_split))
        
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train_split), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train_split[batch_indices]
            batch_y = y_train_split[batch_indices]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'trained_storm_lstm.pth')
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Model saved to 'trained_storm_lstm.pth'")
    
    return model

def main():
    json_files = [
        'TrainingData/stormcells_TX_20251123.json',
        'TrainingData/stormcells_SE_20251125.json'
    ]
    
    print("=" * 80)
    print("STORM CELL LSTM TRAINING - IMPROVED VERSION")
    print("=" * 80)
    print()
    
    model = train_model(json_files, epochs=300, batch_size=32, max_seq_len=20, learning_rate=0.0005)
    
    print("\n" + "=" * 80)
    print("Training complete! Run predict_and_compare.py to see the results.")
    print("=" * 80)

if __name__ == "__main__":
    main()

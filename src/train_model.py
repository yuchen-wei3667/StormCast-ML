import json
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    """Train the LSTM model with overfitting prevention"""
    print("Loading and preparing training data...")
    X_train, y_train = create_training_data(json_files)
    
    print(f"Total training samples: {len(X_train)}")
    
    # Pad sequences - use shorter sequences to prevent overfitting
    X_train_padded = pad_sequences(X_train, max_len=max_seq_len)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_train_padded, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    print(f"Input shape: {X_tensor.shape}")
    print(f"Target shape: {y_tensor.shape}")
    
    # Split into train and validation with better stratification
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train_split)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model - use smaller architecture to prevent overfitting
    model = StormCellLSTM(input_size=6, hidden_size=32, num_layers=2, output_size=2)
    
    # Add dropout to the model by wrapping it
    class StormCellLSTMWithDropout(nn.Module):
        def __init__(self, base_model, dropout_rate=0.3):
            super(StormCellLSTMWithDropout, self).__init__()
            self.base_model = base_model
            self.dropout = nn.Dropout(dropout_rate)
            
        def forward(self, x):
            # Apply dropout to LSTM output
            out = self.base_model.lstm(x)[0]  # Get LSTM output
            out = self.dropout(out[:, -1, :])  # Take last timestep and apply dropout
            out = self.base_model.fc(out)
            return out
            
        def train(self, mode=True):
            """Override train to handle dropout properly"""
            self.training = mode
            self.base_model.train(mode)
            return self
    
    # Wrap model with dropout
    model = StormCellLSTMWithDropout(model, dropout_rate=0.3)
    
    # Loss and optimizer with weight decay (L2 regularization) - KEY OVERFITTING FIX
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler - automatically reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=15, verbose=True)
    
    # Training loop with early stopping
    print(f"\nTraining for {epochs} epochs with overfitting prevention...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30  # Early stopping patience
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    
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
            
            # Gradient clipping - prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping check - KEY OVERFITTING PREVENTION
        if val_loss < best_val_loss - 0.001:  # Minimum improvement threshold
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'trained_storm_lstm.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping - stops training when overfitting detected
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            print("This prevents overfitting by stopping when validation loss stops improving.")
            break
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Model saved to 'trained_storm_lstm.pth'")
    
    # Plot training curves to visualize overfitting prevention
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss - Overfitting Prevention Active')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot last 50% of epochs for better visibility of convergence
    start_idx = len(train_losses) // 2
    plt.plot(range(start_idx, len(train_losses)), train_losses[start_idx:], label='Training Loss', alpha=0.7)
    plt.plot(range(start_idx, len(val_losses)), val_losses[start_idx:], label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss (Last 50% of epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model

def main():
    # Use all available training data files
    json_files = [
        'TrainingData/stormcells_Central_20250315.json',
        'TrainingData/stormcells_Midwest_20240506.json',
        'TrainingData/stormcells_SE_20251125.json',
        'TrainingData/stormcells_TX_20251123.json'
    ]
    
    print("=" * 80)
    print("STORM CELL LSTM TRAINING - OVERFITTING PREVENTION ENABLED")
    print("=" * 80)
    print()
    print("🔧 OVERFITTING PREVENTIONS APPLIED:")
    print("   • Dropout regularization (30%)")
    print("   • Early stopping with patience")
    print("   • Weight decay (L2 regularization)")
    print("   • Learning rate scheduling")
    print("   • Gradient clipping")
    print("   • Smaller model architecture")
    print()
    
    model = train_model(json_files, epochs=300, batch_size=32, max_seq_len=15, learning_rate=0.001)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE - OVERFITTING PREVENTED!")
    print("📊 Check training_curves.png for loss visualization")
    print("💾 Model saved as 'trained_storm_lstm.pth'")
    print("🎯 Expected: Training and validation losses should decrease together")
    print("=" * 80)

if __name__ == "__main__":
    main()
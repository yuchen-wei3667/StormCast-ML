import json
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import StormCellLSTM

def create_enhanced_training_data(json_files, augment_data=True, noise_factor=0.05):
    """
    Enhanced data creation with feature engineering and optional data augmentation
    """
    X_train = []
    y_train = []
    
    # Statistics for normalization
    all_features = []
    all_velocities = []
    
    print("Loading and processing storm data...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {json_file} not found, skipping.")
            continue
        
        features = data.get('features', [])
        print(f"Processing {json_file}: {len(features)} storm cells")
        
        for feature in features:
            storm_history = feature.get('storm_history', [])
            
            if len(storm_history) < 2:
                continue
            
            # Sort by timestamp
            storm_history.sort(key=lambda x: x['timestamp'])
            
            # Create sequences
            for i in range(len(storm_history) - 1):
                # Input: enhanced sequence with engineered features
                sequence = []
                for j in range(i + 1):
                    point = storm_history[j]
                    
                    # Original features
                    dx = float(point.get('dx', 0))
                    dy = float(point.get('dy', 0))
                    dt = float(point.get('dt', 1))
                    eb_shear = float(point.get('EBShear', 0))
                    srw = float(point.get('SRW46km', 0))
                    mean_wind = float(point.get('MeanWind_1-3kmAGL', 0))
                    
                    # Engineered features
                    speed = np.sqrt(dx**2 + dy**2) if dt > 0 else 0
                    direction = np.arctan2(dy, dx) if speed > 0 else 0
                    
                    # Normalize time (relative position in sequence)
                    time_normalized = j / max(len(storm_history) - 1, 1)
                    
                    # Weather features interaction
                    wind_shear_ratio = srw / (mean_wind + 1e-6)
                    momentum = speed * mean_wind
                    
                    feats = [
                        dx, dy, dt, eb_shear, srw, mean_wind,  # Original
                        speed, direction, time_normalized,     # Engineered
                        wind_shear_ratio, momentum             # Additional
                    ]
                    
                    sequence.append(feats)
                    
                    # Store for statistics
                    all_features.append(feats)
                
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
                all_velocities.append([vx_target, vy_target])
                
                # Data augmentation: add noise to create variations
                if augment_data and random.random() < 0.3:  # 30% augmentation rate
                    augmented_sequence = []
                    for feats in sequence:
                        # Add small random noise to features
                        noisy_feats = []
                        for feat in feats:
                            noise = np.random.normal(0, noise_factor * abs(feat) if feat != 0 else noise_factor)
                            noisy_feats.append(feat + noise)
                        augmented_sequence.append(noisy_feats)
                    
                    # Add slight noise to targets
                    vx_aug = vx_target + np.random.normal(0, noise_factor * abs(vx_target) if vx_target != 0 else noise_factor)
                    vy_aug = vy_target + np.random.normal(0, noise_factor * abs(vy_target) if vy_target != 0 else noise_factor)
                    
                    X_train.append(augmented_sequence)
                    y_train.append([vx_aug, vy_aug])
    
    print(f"Total samples (including augmented): {len(X_train)}")
    
    # Calculate feature statistics for normalization
    feature_stats = {
        'mean': np.mean(all_features, axis=0),
        'std': np.std(all_features, axis=0),
        'velocity_mean': np.mean(all_velocities, axis=0),
        'velocity_std': np.std(all_velocities, axis=0)
    }
    
    return X_train, y_train, feature_stats

def normalize_features(sequences, stats, is_target=False):
    """Normalize features using computed statistics"""
    normalized = []
    
    if is_target:
        # Handle target normalization (2D vectors: [vx, vy])
        mean, std = stats['velocity_mean'], stats['velocity_std']
        for target in sequences:
            normalized_target = []
            for i, val in enumerate(target):
                if std[i] > 1e-6:
                    normalized_target.append((val - mean[i]) / std[i])
                else:
                    normalized_target.append(val - mean[i])
            normalized.append(normalized_target)
    else:
        # Handle feature sequence normalization (sequences of 11D vectors)
        mean, std = stats['mean'], stats['std']
        for seq in sequences:
            normalized_seq = []
            for feats in seq:
                normalized_feats = []
                for i, feat in enumerate(feats):
                    if std[i] > 1e-6:
                        normalized_feats.append((feat - mean[i]) / std[i])
                    else:
                        normalized_feats.append(feat - mean[i])
                normalized_seq.append(normalized_feats)
            normalized.append(normalized_seq)
    
    return normalized

def pad_sequences(sequences, max_len=None):
    """Pad sequences to same length"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with mean values
            padding = [[0] * len(seq[0])] * (max_len - len(seq))
            padded.append(padding + seq)
        else:
            padded.append(seq[-max_len:])  # Take last max_len
    
    return padded

class AttentionLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=2, output_size=2, dropout=0.3):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights with better initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        for name, param in self.attention.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        # Reshape for attention: (sequence_length, batch_size, hidden_size)
        lstm_out = lstm_out.transpose(0, 1)
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Reshape back: (batch_size, sequence_length, hidden_size)
        attended_out = attended_out.transpose(0, 1)
        
        # Apply layer normalization
        attended_out = self.layer_norm(attended_out + lstm_out.transpose(0, 1))
        
        # Take the last output with attention weighting
        final_output = attended_out[:, -1, :]
        
        # Pass through classifier
        x = self.dropout(final_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x, attention_weights

def train_with_cross_validation(json_files, n_folds=5, epochs=200, batch_size=64, max_seq_len=15):
    """Train model with cross-validation for better validation"""
    print("=" * 80)
    print("ADVANCED LSTM TRAINING WITH CROSS-VALIDATION")
    print("=" * 80)
    
    # Create enhanced training data
    X_train, y_train, feature_stats = create_enhanced_training_data(json_files, augment_data=True)
    
    # Normalize features
    X_normalized = normalize_features(X_train, feature_stats)
    y_normalized = normalize_features(y_train, feature_stats, is_target=True)
    
    # Pad sequences
    X_padded = pad_sequences(X_normalized, max_len=max_seq_len)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_padded, dtype=torch.float32)
    y_tensor = torch.tensor(y_normalized, dtype=torch.float32)
    
    print(f"Final dataset shape: {X_tensor.shape}")
    
    # Cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_train_losses = []
    all_val_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tensor)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        X_fold_train, X_fold_val = X_tensor[train_idx], X_tensor[val_idx]
        y_fold_train, y_fold_val = y_tensor[train_idx], y_tensor[val_idx]
        
        # Create model for this fold
        model = AttentionLSTM(input_size=11, hidden_size=64, num_layers=2, 
                            output_size=2, dropout=0.4)
        
        # Loss and optimizer with advanced settings
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3, 
                                    betas=(0.9, 0.999), eps=1e-8)
        
        # Advanced learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Training for this fold
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 30
        
        for epoch in range(epochs):
            # Training
            model.train()
            indices = torch.randperm(len(X_fold_train))
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_fold_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_fold_train[batch_indices]
                batch_y = y_fold_train[batch_indices]
                
                # Forward pass
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Advanced gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs, _ = model(X_fold_val)
                val_loss = criterion(val_outputs, y_fold_val).item()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                patience_counter = 0
                if fold == 0:  # Save best model from first fold
                    torch.save(model.state_dict(), 'trained_storm_lstm_advanced.pth')
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        fold_results.append({
            'train_loss': train_losses,
            'val_loss': val_losses,
            'best_val_loss': best_val_loss
        })
        
        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)
        
        print(f"Fold {fold + 1} best validation loss: {best_val_loss:.4f}")
    
    # Plot combined results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for i, result in enumerate(fold_results):
        plt.plot(result['train_loss'], alpha=0.6, label=f'Fold {i+1} Train')
        plt.plot(result['val_loss'], alpha=0.6, linestyle='--', label=f'Fold {i+1} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss by Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    avg_train_losses = np.array([result['train_loss'][-10:] for result in fold_results]).mean(axis=0)
    avg_val_losses = np.array([result['val_loss'][-10:] for result in fold_results]).mean(axis=0)
    plt.plot(avg_train_losses, label='Avg Train Loss')
    plt.plot(avg_val_losses, label='Avg Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Loss Across Folds (Last 10 epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    best_val_losses = [result['best_val_loss'] for result in fold_results]
    plt.bar(range(1, len(best_val_losses) + 1), best_val_losses)
    plt.xlabel('Fold')
    plt.ylabel('Best Validation Loss')
    plt.title('Best Validation Loss by Fold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Average best validation loss: {np.mean(best_val_losses):.4f} ± {np.std(best_val_losses):.4f}")
    print(f"Best fold validation loss: {np.min(best_val_losses):.4f}")
    print(f"Worst fold validation loss: {np.max(best_val_losses):.4f}")
    print(f"Model saved as 'trained_storm_lstm_advanced.pth'")
    print("=" * 80)

def main():
    json_files = [
        'TrainingData/stormcells_Central_20250315.json',
        'TrainingData/stormcells_Midwest_20240506.json',
        'TrainingData/stormcells_SE_20251125.json',
        'TrainingData/stormcells_TX_20251123.json'
    ]
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    train_with_cross_validation(json_files, n_folds=5, epochs=150, batch_size=64, max_seq_len=12)

if __name__ == "__main__":
    main()
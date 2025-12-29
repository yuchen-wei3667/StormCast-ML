"""
Training script for GRU storm motion prediction model
"""
import argparse
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gru_data_loader import load_sequences
from gru_model import create_gru_model, get_callbacks

def train_gru(data_dir, output_dir="models", model_filename="gru_storm_motion.keras", 
              sequence_length=5, batch_size=64, epochs=100, val_split=0.2):
    """
    Train GRU model on storm sequences
    
    Args:
        data_dir: Path to training data
        output_dir: Directory to save model
        model_filename: Filename for the saved model
        sequence_length: Number of scans in sequence
        batch_size: Training batch size
        epochs: Maximum number of epochs
        val_split: Fraction of data for validation
    """
    print("="*70)
    print("GRU STORM MOTION PREDICTION - TRAINING")
    print("="*70)
    
    # Load data
    print(f"\nLoading sequences from {data_dir}...")
    X, y, scaler = load_sequences(data_dir, sequence_length=sequence_length)
    
    print(f"\nDataset info:")
    print(f"  Total sequences: {len(X)}")
    print(f"  Sequence length: {X.shape[1]}")
    print(f"  Features per timestep: {X.shape[2]}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} sequences")
    print(f"  Validation: {len(X_val)} sequences")
    
    # Create model with improved architecture
    print(f"\nCreating refined GRU model...")
    model = create_gru_model(
        sequence_length=X.shape[1],
        n_features=X.shape[2],
        gru_units_1=128,  # Increased from 64
        gru_units_2=64,   # Increased from 32
        dropout_rate=0.3,
        l2_reg=0.001,     # Reduced from 0.01
        use_bidirectional=True,
        use_attention=True
    )
    
    model.summary()
    
    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, model_filename)
    scaler_filename = model_filename.replace('.keras', '_scaler.pkl')
    scaler_path = os.path.join(output_dir, scaler_filename)
    
    # Get callbacks
    callback_list = get_callbacks(model_path, patience_early=15, patience_lr=5)
    
    # Train model
    print(f"\n{'='*70}")
    print("TRAINING")
    print("="*70)
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")
    print(f"Early stopping patience: 15")
    print(f"Learning rate reduction patience: 5")
    print("="*70 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callback_list,
        verbose=0  # We use custom verbose callback
    )
    
    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print("="*70)
    
    train_loss, train_mae, train_mse, train_rmse = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae, val_mse, val_rmse = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\nTraining Set:")
    print(f"  Loss (MSE): {train_loss:.4f}")
    print(f"  MAE: {train_mae:.4f} m/s")
    print(f"  RMSE: {train_rmse:.4f} m/s")
    
    print(f"\nValidation Set:")
    print(f"  Loss (MSE): {val_loss:.4f}")
    print(f"  MAE: {val_mae:.4f} m/s")
    print(f"  RMSE: {val_rmse:.4f} m/s")
    
    # Example prediction
    print(f"\n{'='*70}")
    print("EXAMPLE PREDICTION")
    print("="*70)
    
    idx = 0
    sample_seq = X_val[idx:idx+1]
    true_vel = y_val[idx]
    pred_vel = model.predict(sample_seq, verbose=0)[0]
    
    print(f"\nSample {idx}:")
    print(f"  True velocity (u, v): [{true_vel[0]:.2f}, {true_vel[1]:.2f}] m/s")
    print(f"  Predicted velocity (u, v): [{pred_vel[0]:.2f}, {pred_vel[1]:.2f}] m/s")
    print(f"  Error: {np.sqrt(np.sum((true_vel - pred_vel)**2)):.2f} m/s")
    
    print(f"\n{'='*70}")
    print("Done!")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU storm motion model")
    parser.add_argument("--data_dir", required=True, help="Path to training data")
    parser.add_argument("--output_dir", default="models", help="Output directory")
    parser.add_argument("--model_filename", default="gru_storm_motion.keras", 
                        help="Filename for the saved model")
    parser.add_argument("--sequence_length", type=int, default=7, 
                       help="Number of scans in sequence (default: 7)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--val_split", type=float, default=0.2, 
                        help="Fraction of data for validation (default: 0.2)")
    
    args = parser.parse_args()
    
    train_gru(
        args.data_dir,
        args.output_dir,
        args.model_filename,
        args.sequence_length,
        args.batch_size,
        args.epochs,
        args.val_split
    )

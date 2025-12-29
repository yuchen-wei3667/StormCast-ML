"""
Training script for optimized GRU model
Uses aggressive optimization techniques
"""
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from gru_data_loader import load_sequences
from gru_optimized import create_optimized_gru_model, compile_optimized_model
from gru_model import get_callbacks

def train_optimized_gru(data_dir, output_dir="models", sequence_length=7, 
                        batch_size=32, epochs=150):
    """Train optimized GRU model"""
    
    print("="*70)
    print("OPTIMIZED GRU - MAXIMUM PERFORMANCE")
    print("="*70)
    
    # Load data
    print(f"\nLoading sequences...")
    X, y, scaler = load_sequences(data_dir, sequence_length=sequence_length)
    
    print(f"\nDataset: {len(X)} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Create optimized model
    print(f"\nCreating optimized model...")
    model = create_optimized_gru_model(
        sequence_length=X.shape[1],
        n_features=X.shape[2]
    )
    
    # Calculate total training steps for learning rate schedule
    steps_per_epoch = len(X_train) // batch_size
    total_steps = steps_per_epoch * epochs
    
    model = compile_optimized_model(model, total_steps=total_steps)
    model.summary()
    
    # Prepare output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, "gru_optimized.keras")
    scaler_path = os.path.join(output_dir, "gru_optimized_scaler.pkl")
    
    # Get callbacks with longer patience
    callback_list = get_callbacks(model_path, patience_early=20, patience_lr=7)
    
    # Train
    print(f"\n{'='*70}")
    print("TRAINING - OPTIMIZED MODEL")
    print("="*70)
    print(f"Batch size: {batch_size} (smaller for better gradients)")
    print(f"Max epochs: {epochs}")
    print(f"Learning rate: Cosine annealing with warmup")
    print(f"Loss: Velocity-aware (MSE + magnitude + direction)")
    print("="*70 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callback_list,
        verbose=0
    )
    
    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    
    # Evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print("="*70)
    
    train_results = model.evaluate(X_train, y_train, verbose=0)
    val_results = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\nTraining Set:")
    print(f"  Loss: {train_results[0]:.4f}")
    print(f"  MAE: {train_results[1]:.4f} m/s")
    print(f"  RMSE: {train_results[2]:.4f} m/s")
    
    print(f"\nValidation Set:")
    print(f"  Loss: {val_results[0]:.4f}")
    print(f"  MAE: {val_results[1]:.4f} m/s ⭐")
    print(f"  RMSE: {val_results[2]:.4f} m/s")
    
    # Example
    print(f"\n{'='*70}")
    print("EXAMPLE PREDICTION")
    print("="*70)
    
    idx = 0
    pred = model.predict(X_val[idx:idx+1], verbose=0)[0]
    true = y_val[idx]
    
    print(f"\nTrue: [{true[0]:.2f}, {true[1]:.2f}] m/s")
    print(f"Pred: [{pred[0]:.2f}, {pred[1]:.2f}] m/s")
    print(f"Error: {np.sqrt(np.sum((true - pred)**2)):.2f} m/s")
    
    print(f"\n{'='*70}")
    print("🎯 OPTIMIZATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train optimized GRU")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--sequence_length", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    
    args = parser.parse_args()
    
    train_optimized_gru(
        args.data_dir,
        args.output_dir,
        args.sequence_length,
        args.batch_size,
        args.epochs
    )

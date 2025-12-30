import argparse
import os
import pickle
import logging
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Local imports
from gru_data_loader import load_sequences
from gru_model import create_gru_model, StormCastGRU

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LoggingModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom ModelCheckpoint that uses logging.info instead of print.
    """
    def __init__(self, filepath, monitor='val_loss'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.best = np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(f"Can save best model only with {self.monitor} available, skipping.")
            return

        if current < self.best:
            logging.info(f"Epoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, saving model to {self.filepath}")
            self.best = current
            self.model.save(self.filepath)

class MetricLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logging.info(f"Epoch {epoch + 1}: "
                     f"Train Loss: {logs.get('loss'):.4f}, "
                     f"Val Loss: {logs.get('val_loss'):.4f}, "
                     f"Val Velocity MAE: {logs.get('val_mae'):.4f}, "
                     f"Val Directional MAE: {logs.get('val_directional_error_deg'):.4f}")

def train_gru(data_dir, output_dir="models", model_name="gru_storm_motion.keras", sequence_length=4, residual=False, loss_type='combined'):
    """
    Main training pipeline for StormCast GRU (v5).
    """
    logging.info(f"Loading data from {data_dir} with sequence_length={sequence_length}, residual={residual}...")
    
    # 1. Load Data
    X, y, ids = load_sequences(data_dir, sequence_length=sequence_length, residual=residual)
    
    if len(X) == 0:
        logging.error("No valid data found. Exiting.")
        return
        
    logging.info(f"Data Loaded. Total samples: {len(X)}")
    
    # Filter extreme outliers (> 80 m/s for absolute, or > 40 m/s for delta)
    v_mags = np.sqrt(y[:, 0]**2 + y[:, 1]**2)
    threshold = 40.0 if residual else 80.0
    valid_mask = v_mags < threshold
    X = X[valid_mask]
    y = y[valid_mask]
    logging.info(f"Filtered {np.sum(~valid_mask)} extreme outliers. Remaining: {len(X)}")
    
    # 2. Preprocessing
    n_samples, seq_len, n_features = X.shape
    X_2d = X.reshape(n_samples * seq_len, n_features)
    
    logging.info("Normalizing features...")
    scaler_x = StandardScaler()
    X_scaled_2d = scaler_x.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(n_samples, seq_len, n_features)
    
    logging.info("Normalizing targets...")
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # 3. Split Data (85% Train, 15% Validation)
    X_train, X_val, y_train, y_val_scaled = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=42)
    
    # 4. Initialize Model
    logging.info(f"Initializing GRU Model (v5 architecture, loss={loss_type})...")
    model = create_gru_model(input_shape=(seq_len, n_features), gru_units=[128, 64], loss=loss_type)
    model.summary(print_fn=logging.info)
    
    # 5. Setup Callbacks
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, model_name)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        LoggingModelCheckpoint(filepath=model_path, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        MetricLogger()
    ]
    
    # 6. Train
    logging.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val_scaled),
        epochs=150,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Save Artifacts
    scaler_path = os.path.join(output_dir, model_name.replace('.keras', '_scaler.pkl'))
    with open(scaler_path, 'wb') as f:
        pickle.dump({'x': scaler_x, 'y': scaler_y}, f)
        
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Scalers saved to {scaler_path}")
    
    # 8. Final Evaluation
    logging.info("Running final evaluation on Validation Set (unscaled results)...")
    y_pred_scaled = model.predict(X_val, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_val_scaled)
    
    # Reconstruct absolute velocity if needed for final metrics comparison
    if residual:
        # Get unscaled input features to extract current velocity
        X_val_2d = X_val.reshape(-1, n_features)
        X_val_unscaled_2d = scaler_x.inverse_transform(X_val_2d)
        X_val_unscaled = X_val_unscaled_2d.reshape(-1, seq_len, n_features)
        
        # u, v are features at index 46, 47 (eng features 2, 3)
        u_idx, v_idx = 46, 47
        curr_u = X_val_unscaled[:, -1, u_idx]
        curr_v = X_val_unscaled[:, -1, v_idx]
        
        y_true_abs = np.zeros_like(y_true)
        y_true_abs[:, 0] = y_true[:, 0] + curr_u
        y_true_abs[:, 1] = y_true[:, 1] + curr_v
        
        y_pred_abs = np.zeros_like(y_pred)
        y_pred_abs[:, 0] = y_pred[:, 0] + curr_u
        y_pred_abs[:, 1] = y_pred[:, 1] + curr_v
    else:
        y_true_abs = y_true
        y_pred_abs = y_pred

    mae_val = np.mean(np.abs(y_true_abs - y_pred_abs))
    
    # Directional Error on absolute velocity
    u_true, v_true = y_true_abs[:, 0], y_true_abs[:, 1]
    u_pred, v_pred = y_pred_abs[:, 0], y_pred_abs[:, 1]
    theta_true = np.degrees(np.arctan2(v_true, u_true))
    theta_pred = np.degrees(np.arctan2(v_pred, u_pred))
    diff = np.abs(theta_true - theta_pred)
    diff = np.where(diff > 180, 360 - diff, diff)
    dir_mae_val = np.mean(diff)
    
    logging.info(f"Final Validation Results (Unscaled Absolute Velocity):")
    logging.info(f"  Speed MAE: {mae_val:.4f} m/s")
    logging.info(f"  Directional MAE: {dir_mae_val:.4f} degrees")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StormCast GRU Model")
    parser.add_argument("--data_dir", required=True, help="Path to training data directory")
    parser.add_argument("--model_dir", default="models", help="Directory to save the trained model")
    parser.add_argument("--model_name", default="gru_storm_motion.keras", help="Name of saved model file")
    parser.add_argument("--sequence_length", type=int, default=8, help="Features lookback sequence length")
    parser.add_argument("--residual", action="store_true", help="Predict velocity change (residual)")
    parser.add_argument("--loss_type", default="combined", choices=["combined", "mse"], help="Loss function")
    
    args = parser.parse_args()
    
    train_gru(args.data_dir, args.model_dir, args.model_name, args.sequence_length, args.residual, args.loss_type)

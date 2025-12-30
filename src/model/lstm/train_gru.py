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

def train_gru(data_dir, output_dir="models", model_name="gru_storm_motion.keras"):
    """
    Main training pipeline for StormCast GRU.
    """
    logging.info(f"Loading data from {data_dir}...")
    
    # 1. Load Data
    X, y, ids = load_sequences(data_dir, sequence_length=1)
    
    if len(X) == 0:
        logging.error("No valid data found. Exiting.")
        return
        
    logging.info(f"Data Loaded. Total samples: {len(X)}")
    logging.info(f"Input shape: {X.shape}")
    
    # 2. Preprocessing
    # Flatten sequence dim for scaling (samples, seq_len, features) -> (samples * seq_len, features)
    n_samples, seq_len, n_features = X.shape
    X_2d = X.reshape(n_samples * seq_len, n_features)
    
    logging.info("Normalizing (scaling) features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled_2d = scaler.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(n_samples, seq_len, n_features)
    
    # 3. Split Data (85% Train, 15% Validation)
    logging.info("Splitting data 85% Train, 15% Validation...")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
    
    logging.info(f"Training Samples: {len(X_train)}")
    logging.info(f"Validation Samples: {len(X_val)}")
    
    # 4. Initialize Model
    logging.info("Initializing GRU Model...")
    model = create_gru_model(input_shape=(seq_len, n_features), gru_units=64)
    model.summary(print_fn=logging.info)
    
    # 5. Setup Callbacks
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_path = os.path.join(output_dir, model_name)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        LoggingModelCheckpoint(
            filepath=model_path,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        MetricLogger()
    ]
    
    # 6. Train
    logging.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Save Artifacts
    scaler_path = os.path.join(output_dir, model_name.replace('.keras', '_scaler.pkl'))
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Scaler saved to {scaler_path}")
    
    # 8. Final Evaluation
    logging.info("Running final evaluation on Validation Set...")
    val_loss, val_mae, val_dir_mae = model.evaluate(X_val, y_val, verbose=0)
    logging.info(f"Final Validation Results:")
    logging.info(f"  Loss: {val_loss:.4f}")
    logging.info(f"  MAE: {val_mae:.4f} m/s")
    logging.info(f"  Directional MAE: {val_dir_mae:.4f} degrees")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StormCast GRU Model")
    parser.add_argument("--data_dir", required=True, help="Path to training data directory")
    parser.add_argument("--model_dir", default="models", help="Directory to save the trained model")
    parser.add_argument("--model_name", default="gru_storm_motion.keras", help="Name of saved model file")
    
    args = parser.parse_args()
    
    train_gru(args.data_dir, args.model_dir, args.model_name)

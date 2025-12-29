"""
GRU Model for Storm Motion Prediction
Optimized for CPU inference
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np

def create_gru_model(sequence_length=5, n_features=44, 
                     gru_units_1=64, gru_units_2=32,
                     dropout_rate=0.3, l2_reg=0.01):
    """
    Create GRU model optimized for CPU inference
    
    Args:
        sequence_length: Number of timesteps in sequence
        n_features: Number of features per timestep
        gru_units_1: Units in first GRU layer
        gru_units_2: Units in second GRU layer
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=(sequence_length, n_features), name='sequence_input')
    
    # First GRU layer with return_sequences=True
    x = layers.GRU(
        gru_units_1,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='gru_1'
    )(inputs)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    # Second GRU layer with return_sequences=False
    x = layers.GRU(
        gru_units_2,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='gru_2'
    )(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(l2_reg),
                     name='dense_1')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout_3')(x)
    
    # Output layer (u, v velocities)
    outputs = layers.Dense(2, activation='linear', name='output')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='storm_gru')
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

class VerboseTrainingCallback(callbacks.Callback):
    """Custom callback to print train/val loss every epoch"""
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        train_mae = logs.get('mae', 0)
        val_mae = logs.get('val_mae', 0)
        
        print(f"Epoch {epoch+1}: "
              f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"mae={train_mae:.4f}, val_mae={val_mae:.4f}")

def get_callbacks(model_path, patience_early=15, patience_lr=5):
    """
    Get training callbacks
    
    Args:
        model_path: Path to save best model
        patience_early: Patience for early stopping
        patience_lr: Patience for learning rate reduction
        
    Returns:
        List of callbacks
    """
    callback_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_early,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Save best model
        callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # CSV logger
        callbacks.CSVLogger(
            model_path.replace('.keras', '_history.csv'),
            append=False
        ),
        
        # Verbose training
        VerboseTrainingCallback()
    ]
    
    return callback_list

if __name__ == "__main__":
    # Test model creation
    model = create_gru_model()
    model.summary()
    
    print("\nModel created successfully!")
    print(f"Total parameters: {model.count_params():,}")

"""
GRU Model for Storm Motion Prediction
Optimized for CPU inference
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np

def velocity_aware_huber_loss(y_true, y_pred, delta=1.0):
    """
    Custom Huber loss that penalizes direction and magnitude errors
    """
    # 1. Huber Loss (Base stability)
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    huber = tf.where(is_small_error, squared_loss, linear_loss)
    # Per-entry base loss
    base_loss_per_entry = tf.reduce_mean(huber, axis=-1)
    
    # 2. Magnitude (Speed) Error
    mag_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1))
    mag_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1))
    mag_error_per_entry = tf.square(mag_true - mag_pred)
    
    # 3. Directional (Cosine) Penalty
    # Normalize vectors
    unit_true = y_true / (tf.expand_dims(mag_true, -1) + 1e-7)
    unit_pred = y_pred / (tf.expand_dims(mag_pred, -1) + 1e-7)
    inner_product = tf.reduce_sum(unit_true * unit_pred, axis=-1)
    
    # Cosine distance: 0 (same direction) to 2 (opposite)
    cosine_dist = 1.0 - inner_product
    
    # Masking logic: Stationary storms (< 2 m/s) are handled externally
    # We remove them from total loss calculation
    is_moving = tf.cast(mag_true >= 2.0, tf.float32)
    num_moving = tf.reduce_sum(is_moving)
    
    # Directional weight (only for moving storms)
    directional_weight = tf.clip_by_value(mag_true / 10.0, 0.5, 3.0)
    
    # Combine per-entry
    total_loss_per_entry = (
        base_loss_per_entry + 
        0.5 * mag_error_per_entry + 
        0.4 * cosine_dist * directional_weight
    ) * is_moving
    
    # Average only over moving entries
    return tf.reduce_sum(total_loss_per_entry) / (num_moving + 1e-7)

def directional_error_deg(y_true, y_pred):
    """
    Calculate directional error in degrees (ignoring storms < 2 m/s)
    """
    mag_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1) + 1e-7)
    mag_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1) + 1e-7)
    
    unit_true = y_true / (tf.expand_dims(mag_true, -1) + 1e-7)
    unit_pred = y_pred / (tf.expand_dims(mag_pred, -1) + 1e-7)
    
    cosine_sim = tf.reduce_sum(unit_true * unit_pred, axis=-1)
    cosine_sim = tf.clip_by_value(cosine_sim, -1.0, 1.0)
    
    angle_rad = tf.acos(cosine_sim)
    
    # Mask out errors for slow-moving storms (< 2 m/s) to avoid misleading noise
    is_fast_enough = tf.cast(mag_true >= 2.0, tf.float32)
    return (angle_rad * (180.0 / np.pi)) * is_fast_enough

def create_gru_model(sequence_length=5, n_features=44, 
                     gru_units_1=128, gru_units_2=64,
                     dropout_rate=0.3, l2_reg=0.0001,
                     use_bidirectional=True, use_attention=True):
    """
    Create highly refined GRU model with LayerNorm and Residuals
    """
    inputs = layers.Input(shape=(sequence_length, n_features), name='sequence_input')
    
    # Layer norm for input stabilization
    x = layers.LayerNormalization(name='input_norm')(inputs)
    
    # First GRU layer
    gru_1_layer = layers.GRU(
        gru_units_1,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        recurrent_dropout=0.1,
        name='gru_1_core'
    )
    
    if use_bidirectional:
        x = layers.Bidirectional(gru_1_layer, name='bidirectional_1')(x)
    else:
        x = gru_1_layer(x)
    
    x = layers.LayerNormalization(name='ln_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    # Attention mechanism
    if use_attention:
        # Improved dot-product style attention
        query = layers.Dense(x.shape[-1], name='attn_query')(x)
        attention_weights = layers.Softmax(axis=1, name='attn_weights')(query)
        x = layers.Multiply(name='attn_context')([x, attention_weights])
    
    # Second GRU layer
    gru_2_input = x
    gru_2_layer = layers.GRU(
        gru_units_2,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        recurrent_dropout=0.1,
        name='gru_2_core'
    )
    
    if use_bidirectional:
        x = layers.Bidirectional(gru_2_layer, name='bidirectional_2')(x)
    else:
        x = gru_2_layer(x)
    
    x = layers.LayerNormalization(name='ln_2')(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)
    
    # Refined Dense layers (96 -> 48 as requested)
    x = layers.Dense(96, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name='dense_96')(x)
    x = layers.LayerNormalization(name='ln_3')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout_3')(x)
    
    x = layers.Dense(48, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name='dense_48')(x)
    
    # Output layer
    outputs = layers.Dense(2, activation='linear', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='storm_gru_refined_v2')
    
    # Use AdamW if available (standard in modern Keras 3/TF 2.15+)
    try:
        opt = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001, clipnorm=1.0)
    except AttributeError:
        opt = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(
        optimizer=opt,
        loss=velocity_aware_huber_loss,
        metrics=['mae', directional_error_deg, keras.metrics.RootMeanSquaredError(name='rmse')]
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
        train_dir = logs.get('directional_error_deg', 0)
        val_dir = logs.get('val_directional_error_deg', 0)
        
        print(f"Epoch {epoch+1}: "
              f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"mae={train_mae:.4f}, val_mae={val_mae:.4f}, "
              f"dir_err={train_dir:.2f}°, val_dir_err={val_dir:.2f}°")

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

import tensorflow as tf
import numpy as np

def velocity_aware_huber_loss(y_true, y_pred, delta=1.0, velocity_weight=0.1):
    """
    Computes a weighted Huber loss where high-velocity samples contribute more.
    
    Args:
        y_true: True values (batch_size, 2)
        y_pred: Predicted values (batch_size, 2)
        delta: Huber loss threshold
        velocity_weight: Weighting factor for velocity magnitude
        
    Returns:
        Weighted mean Huber loss
    """
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    huber = tf.where(is_small_error, squared_loss, linear_loss)
    
    # Calculate magnitude of true velocity for weighting
    velocity_mag = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1))
    weight = 1.0 + velocity_weight * velocity_mag
    
    return tf.reduce_mean(huber * tf.expand_dims(weight, -1))

def directional_error_deg(y_true, y_pred):
    """
    Computes the mean absolute directional error in degrees.
    """
    u_true, v_true = y_true[:, 0], y_true[:, 1]
    u_pred, v_pred = y_pred[:, 0], y_pred[:, 1]
    
    theta_true = tf.math.atan2(v_true, u_true) * 180.0 / np.pi
    theta_pred = tf.math.atan2(v_pred, u_pred) * 180.0 / np.pi
    
    diff = tf.abs(theta_true - theta_pred)
    # Handle wrap-around (e.g. 359 vs 1 degree is 2 degrees diff, not 358)
    diff = tf.minimum(diff, 360.0 - diff)
    
    return tf.reduce_mean(diff)

def combined_loss(y_true, y_pred):
    """
    Combines Velocity-Aware Huber Loss and Directional Cosine Loss.
    """
    # 1. Velocity Magnitude Loss (Huber)
    v_loss = velocity_aware_huber_loss(y_true, y_pred, velocity_weight=0.05)
    
    # 2. Directional Loss (1 - Cosine Similarity)
    # y = [u, v]
    u_true, v_true = y_true[:, 0], y_true[:, 1]
    u_pred, v_pred = y_pred[:, 0], y_pred[:, 1]
    
    norm_true = tf.sqrt(tf.square(u_true) + tf.square(v_true)) + 1e-7
    norm_pred = tf.sqrt(tf.square(u_pred) + tf.square(v_pred)) + 1e-7
    
    u_true_n = u_true / norm_true
    v_true_n = v_true / norm_true
    u_pred_n = u_pred / norm_pred
    v_pred_n = v_pred / norm_pred
    
    cos_sim = u_true_n * u_pred_n + v_true_n * v_pred_n
    dir_loss = 1.0 - cos_sim
    dir_loss = tf.reduce_mean(dir_loss)
    
    # Weight directional loss lower to prioritize velocity magnitude accuracy in v4
    return v_loss + 2.0 * dir_loss

class StormCastGRU(tf.keras.Model):
    """
    Custom GRU model for Storm Motion prediction (v5 Architecture).
    Focus: Regularization to prevent overfitting on noisy meteorological data.
    """
    def __init__(self, gru_units=[128, 64], dense_units=[256, 128, 64], dropout_rate=0.4, l2_reg=1e-4, **kwargs):
        super(StormCastGRU, self).__init__(**kwargs)
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        reg = tf.keras.regularizers.l2(l2_reg)
        
        # Stacked GRU Layers
        self.gru_layers = []
        self.bn_gru_layers = []
        
        for i, units in enumerate(gru_units):
            ret_seq = (i < len(gru_units) - 1)
            self.gru_layers.append(tf.keras.layers.GRU(
                units, return_sequences=ret_seq, name=f"gru_{i}",
                kernel_regularizer=reg, recurrent_regularizer=reg
            ))
            self.bn_gru_layers.append(tf.keras.layers.BatchNormalization(name=f"bn_gru_{i}"))
        
        # Dense Blocks
        self.dense_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        
        for i, units in enumerate(dense_units):
            self.dense_layers.append(tf.keras.layers.Dense(
                units, activation='elu', name=f"dense_{i}",
                kernel_regularizer=reg
            ))
            self.bn_layers.append(tf.keras.layers.BatchNormalization(name=f"bn_{i}"))
            self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i}"))
            
        self.out = tf.keras.layers.Dense(2, name="output_velocity")
        
    def call(self, inputs, training=False):
        x = inputs
        for gru, bn in zip(self.gru_layers, self.bn_gru_layers):
            x = gru(x)
            x = bn(x, training=training)
        
        for dense, bn, dropout in zip(self.dense_layers, self.bn_layers, self.dropout_layers):
            x = dense(x)
            x = bn(x, training=training)
            x = dropout(x, training=training)
            
        return self.out(x)
        
    def get_config(self):
        config = super(StormCastGRU, self).get_config()
        config.update({
            "gru_units": self.gru_units,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg
        })
        return config
    
    def compile_model(self, learning_rate=0.001, loss='combined'):
        """Helper to compile with custom or standard loss."""
        if loss == 'combined':
            loss_fn = combined_loss
        else:
            loss_fn = 'mse'
            
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=['mae', directional_error_deg]
        )

def create_gru_model(input_shape, gru_units=[128, 64], loss='combined'):
    """
    Factory function to instantiate and build the StormCastGRU model.
    """
    model = StormCastGRU(gru_units=gru_units)
    
    # Build
    dummy_x = tf.zeros((1, *input_shape))
    _ = model(dummy_x)
    
    model.compile_model(loss=loss)
    return model

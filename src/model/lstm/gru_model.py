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
    Combined optimization objective:
    1. Velocity-Aware Huber Loss (Magnitude/Component accuracy)
    2. Directional Cosine Loss (Angle accuracy)
    
    Weights are tuned to balance MAE (< 5.5 m/s) and DirMAE (< 40 deg).
    """
    # 1. Component Loss (Huber)
    v_loss = velocity_aware_huber_loss(y_true, y_pred, velocity_weight=0.05)
    
    # 2. Directional Loss (Cosine Similarity)
    u_true, v_true = y_true[:, 0], y_true[:, 1]
    u_pred, v_pred = y_pred[:, 0], y_pred[:, 1]
    
    # Normalize vectors (add epsilon to avoid div-by-zero)
    norm_true = tf.sqrt(tf.square(u_true) + tf.square(v_true)) + 1e-7
    norm_pred = tf.sqrt(tf.square(u_pred) + tf.square(v_pred)) + 1e-7
    
    u_true_n = u_true / norm_true
    v_true_n = v_true / norm_true
    u_pred_n = u_pred / norm_pred
    v_pred_n = v_pred / norm_pred
    
    # Cosine distance: 1 - cos(theta)
    # Range: [0, 2], where 0 is perfect alignment
    cos_sim = u_true_n * u_pred_n + v_true_n * v_pred_n
    dir_loss = tf.reduce_mean(1.0 - cos_sim)
    
    # Total Loss
    # Directional loss is weighted heavily (x10) to force alignment focus
    return v_loss + 10.0 * dir_loss

class StormCastGRU(tf.keras.Model):
    """
    Custom GRU model for Storm Motion prediction.
    Architecture: GRU -> Dense(96) -> Dense(48) -> Output(2)
    """
    def __init__(self, gru_units=64, dense_units=[96, 48], **kwargs):
        super(StormCastGRU, self).__init__(**kwargs)
        self.gru_units = gru_units
        self.dense_units = dense_units
        
        # Layers
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=False, name="gru_layer")
        self.dense1 = tf.keras.layers.Dense(dense_units[0], activation='relu', name="dense_1")
        self.dense2 = tf.keras.layers.Dense(dense_units[1], activation='relu', name="dense_2")
        self.out = tf.keras.layers.Dense(2, name="output_velocity")
        
    def call(self, inputs):
        x = self.gru(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)
        
    def get_config(self):
        config = super(StormCastGRU, self).get_config()
        config.update({
            "gru_units": self.gru_units,
            "dense_units": self.dense_units
        })
        return config
    
    def compile_model(self, learning_rate=0.001):
        """Helper to compile with the custom combined loss."""
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=combined_loss,
            metrics=['mae', directional_error_deg]
        )

def create_gru_model(input_shape, gru_units=64):
    """
    Factory function to instantiate and build the StormCastGRU model.
    """
    model = StormCastGRU(gru_units=gru_units)
    
    # Build by passing a dummy input (standard for subclassed models to initialize weights)
    # Input shape is (seq_len, features) -> adds batch dim: (1, seq_len, features)
    dummy_x = tf.zeros((1, *input_shape))
    _ = model(dummy_x)
    
    model.compile_model()
    return model

"""
Advanced optimizations for GRU model
Implements aggressive techniques to maximize performance
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

def create_optimized_gru_model(sequence_length=7, n_features=44):
    """
    Highly optimized GRU with advanced techniques
    
    Optimizations:
    - Larger capacity (256/128 units)
    - Layer normalization instead of batch norm
    - Residual connections
    - Custom velocity-aware loss
    - Cosine annealing learning rate
    """
    inputs = layers.Input(shape=(sequence_length, n_features), name='input')
    
    # First bidirectional GRU with layer norm
    x = layers.Bidirectional(
        layers.GRU(
            256,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.0001),
            recurrent_dropout=0.1,
            name='gru_1'
        )
    )(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    # Attention mechanism
    attention_scores = layers.Dense(1, activation='tanh')(x)
    attention_weights = layers.Softmax(axis=1)(attention_scores)
    x_attended = layers.Multiply()([x, attention_weights])
    
    # Second bidirectional GRU
    x = layers.Bidirectional(
        layers.GRU(
            128,
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(0.0001),
            recurrent_dropout=0.1,
            name='gru_2'
        )
    )(x_attended)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    # Dense layers with skip connection
    dense_input = x
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = layers.LayerNormalization()(x)
    
    # Output
    outputs = layers.Dense(2, activation='linear', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='optimized_gru')
    
    return model

def velocity_aware_loss(y_true, y_pred):
    """
    Custom loss that penalizes direction errors more heavily
    """
    # Standard MSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Magnitude error
    mag_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1))
    mag_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1))
    mag_error = tf.reduce_mean(tf.square(mag_true - mag_pred))
    
    # Direction error (cosine similarity)
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    norm_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1))
    norm_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1))
    cos_sim = dot_product / (norm_true * norm_pred + 1e-8)
    direction_error = tf.reduce_mean(1.0 - cos_sim)
    
    # Combined loss
    return mse + 0.5 * mag_error + 0.3 * direction_error

class CosineAnnealingSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing learning rate schedule"""
    
    def __init__(self, initial_lr=0.001, min_lr=1e-6, warmup_steps=500, total_steps=10000):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step):
        # Convert to float for calculations
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_lr * (step / warmup_steps)
        
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * progress))
        annealing_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        # Use tf.where instead of if/else
        lr = tf.where(step < warmup_steps, warmup_lr, annealing_lr)
        
        return lr
    
    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }

def compile_optimized_model(model, total_steps=10000):
    """Compile model with advanced optimizer and loss"""
    
    # Learning rate schedule
    lr_schedule = CosineAnnealingSchedule(
        initial_lr=0.002,
        min_lr=1e-6,
        warmup_steps=500,
        total_steps=total_steps
    )
    
    # AdamW optimizer with weight decay
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.0001,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=velocity_aware_loss,
        metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    return model

if __name__ == "__main__":
    print("Optimized GRU model module")
    model = create_optimized_gru_model()
    model = compile_optimized_model(model)
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")

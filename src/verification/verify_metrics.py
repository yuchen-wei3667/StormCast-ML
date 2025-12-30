
import numpy as np
import tensorflow as tf

def directional_error_deg_keras_style(y_true, y_pred):
    mag_true = np.sqrt(np.sum(np.square(y_true), axis=-1) + 1e-7)
    mag_pred = np.sqrt(np.sum(np.square(y_pred), axis=-1) + 1e-7)
    
    unit_true = y_true / (np.expand_dims(mag_true, -1) + 1e-7)
    unit_pred = y_pred / (np.expand_dims(mag_pred, -1) + 1e-7)
    
    cosine_sim = np.sum(unit_true * unit_pred, axis=-1)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    angle_rad = np.arccos(cosine_sim)
    angle_deg = angle_rad * 180.0 / np.pi
    
    is_fast_enough = (mag_true >= 2.0).astype(float)
    return np.mean(angle_deg * is_fast_enough)

def directional_error_deg_honest_style(y_true, y_pred):
    mag_true = np.sqrt(np.sum(np.square(y_true), axis=-1) + 1e-7)
    mag_pred = np.sqrt(np.sum(np.square(y_pred), axis=-1) + 1e-7)
    
    moving_mask = mag_true >= 2.0
    if not np.any(moving_mask):
        return 0.0
        
    y_t_moving = y_true[moving_mask]
    y_p_moving = y_pred[moving_mask]
    
    unit_true = y_t_moving / (np.expand_dims(np.sqrt(np.sum(y_t_moving**2, axis=1)), -1) + 1e-7)
    unit_pred = y_p_moving / (np.expand_dims(np.sqrt(np.sum(y_p_moving**2, axis=1)), -1) + 1e-7)
    
    cosine_sim = np.sum(unit_true * unit_pred, axis=-1)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    angle_rad = np.arccos(cosine_sim)
    return np.mean(angle_rad * 180.0 / np.pi)

# Example data
y_true = np.array([[10, 0], [0, 10], [1, 0], [0, 1]]) # 2 moving, 2 stationary
y_pred = np.array([[10, 10], [10, 10], [10, 10], [10, 10]]) # 45 deg error for all

keras_val = directional_error_deg_keras_style(y_true, y_pred)
honest_val = directional_error_deg_honest_style(y_true, y_pred)

print(f"Keras Style (Averaged over all): {keras_val:.1f}")
print(f"Honest Style (Averaged over moving): {honest_val:.1f}")
print(f"Ratio: {keras_val/honest_val:.2f}")

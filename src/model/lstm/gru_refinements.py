"""
Advanced GRU model with additional refinements
This version includes data augmentation and ensemble techniques
"""
import numpy as np
from gru_model import create_gru_model
from gru_data_loader import load_sequences
from sklearn.model_selection import train_test_split
import pickle

def augment_sequences(X, y, noise_level=0.01):
    """
    Augment training data with small noise
    
    Args:
        X: Input sequences
        y: Target values
        noise_level: Standard deviation of Gaussian noise
        
    Returns:
        Augmented X and y
    """
    X_aug = X + np.random.normal(0, noise_level, X.shape)
    return X_aug, y

def create_ensemble_models(n_models=3, **model_kwargs):
    """
    Create ensemble of GRU models with different initializations
    
    Args:
        n_models: Number of models in ensemble
        **model_kwargs: Arguments for create_gru_model
        
    Returns:
        List of models
    """
    models = []
    for i in range(n_models):
        model = create_gru_model(**model_kwargs)
        models.append(model)
    return models

def ensemble_predict(models, X):
    """
    Make predictions using ensemble averaging
    
    Args:
        models: List of trained models
        X: Input data
        
    Returns:
        Averaged predictions
    """
    predictions = []
    for model in models:
        pred = model.predict(X, verbose=0)
        predictions.append(pred)
    
    return np.mean(predictions, axis=0)

if __name__ == "__main__":
    print("Advanced GRU refinements module loaded")
    print("Available functions:")
    print("  - augment_sequences(): Add noise for data augmentation")
    print("  - create_ensemble_models(): Create model ensemble")
    print("  - ensemble_predict(): Ensemble prediction")

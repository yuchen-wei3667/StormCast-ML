import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Import our new modules
from data_loader import load_storm_data
from model import create_gb_model

def train_model(data_dir, output_dir="models", model_name="gb_storm_motion.pkl"):
    print(f"Loading data from {data_dir}...")
    
    # Load data
    X, y = load_storm_data(data_dir)
    
    if len(X) == 0:
        print("No valid data found. Exiting.")
        return
        
    print(f"Data Loaded. X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    print("Initializing Gradient Boosting Model...")
    model = create_gb_model()
    
    # Train
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Save model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_path = os.path.join(output_dir, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save a verification prediction
    print("\n--- Example Prediction ---")
    print(f"Input Features: {X_test[0]}")
    print(f"True Motion (u, v): {y_test[0]}")
    print(f"Predicted Motion (u, v): {y_pred[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gradient Boosting Model for Storm Motion")
    parser.add_argument("--data_dir", required=True, help="Path to directory containing date folders (e.g. /path/to/data)")
    parser.add_argument("--output_dir", default="models", help="Directory to save the trained model")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_dir)

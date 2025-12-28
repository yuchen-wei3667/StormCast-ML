import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our modules (now in same directory)
from data_loader import load_storm_data

# Flag to use improved model
USE_IMPROVED_MODEL = True

if USE_IMPROVED_MODEL:
    try:
        from model_improved import create_gb_model
        print("Using improved model with XGBoost and better hyperparameters")
    except ImportError:
        from model import create_gb_model
        USE_IMPROVED_MODEL = False
        print("Improved model not available, using standard model")
else:
    from model import create_gb_model

def train_model(data_dir, output_dir="models", model_name="gb_storm_motion.pkl"):
    print(f"Loading data from {data_dir}...")
    
    # Load data
    X, y = load_storm_data(data_dir)
    
    if len(X) == 0:
        print("No valid data found. Exiting.")
        return
        
    print(f"Data Loaded. Total samples: {len(X)}")
    
    # Split data into Training and Verification
    print("Splitting data into Training (80%) and Verification (20%) sets...")
    X_train, X_verification, y_train, y_verification = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Samples: {len(X_train)}")
    print(f"Verification Samples: {len(X_verification)}")
    
    # Normalize features using StandardScaler
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_verification_scaled = scaler.transform(X_verification)
    
    # Split training data further for internal validation monitoring
    X_train_fit, X_train_val, y_train_fit, y_train_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=42
    )
    
    # Create model with validation data for monitoring
    print("Initializing Gradient Boosting Model (Improved Version)...")
    if USE_IMPROVED_MODEL:
        model = create_gb_model(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            X_val=X_train_val,
            y_val=y_train_val,
            use_xgboost=True
        )
    else:
        model = create_gb_model(X_val=X_train_val, y_val=y_train_val)
    
    # Train
    print("Training model...")
    # Fit each estimator separately to enable monitoring
    for idx, estimator in enumerate(model.estimators_):
        output_name = "u-velocity" if idx == 0 else "v-velocity"
        print(f"\n=== Training {output_name} ===")
        estimator.fit(X_train_fit, y_train_fit[:, idx])
    
    
    # Compute validation loss on internal validation set
    print("\n--- Internal Validation Scores (from training) ---")
    y_train_val_pred = model.predict(X_train_val)
    train_val_mse = mean_squared_error(y_train_val, y_train_val_pred)
    train_val_mae = mean_absolute_error(y_train_val, y_train_val_pred)
    
    for idx, estimator in enumerate(model.estimators_):
        output_name = "u-velocity" if idx == 0 else "v-velocity"
        print(f"\n{output_name}:")
        if hasattr(estimator, 'train_score_'):
            final_train_loss = estimator.train_score_[-1]
            print(f"  Final Train Loss: {final_train_loss:.4f}")
            print(f"  Total iterations: {len(estimator.train_score_)}")
            
            # Compute validation loss for this output
            y_val_single = y_train_val[:, idx]
            y_pred_single = y_train_val_pred[:, idx]
            val_mse_single = mean_squared_error(y_val_single, y_pred_single)
            print(f"  Validation MSE: {val_mse_single:.4f}")
    
    print(f"\nOverall Internal Validation:")
    print(f"  MSE: {train_val_mse:.4f}")
    print(f"  MAE: {train_val_mae:.4f}")
    
    # Evaluate
    print("Evaluating model on Verification set...")
    y_pred = model.predict(X_verification_scaled)
    
    mse = mean_squared_error(y_verification, y_pred)
    mae = mean_absolute_error(y_verification, y_pred)
    
    print(f"\nVerification Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Save model and scaler
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_path = os.path.join(output_dir, model_name)
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Save a verification prediction
    print("\n--- Example Verification Prediction ---")
    print(f"Input Features (first 6): {X_verification[0][:6]}")
    print(f"True Motion (u, v): {y_verification[0]}")
    print(f"Predicted Motion (u, v): {y_pred[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gradient Boosting Model for Storm Motion")
    parser.add_argument("--data_dir", required=True, help="Path to directory containing date folders (e.g. /path/to/data)")
    parser.add_argument("--output_dir", default="models", help="Directory to save the trained model")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_dir)

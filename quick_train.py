#!/usr/bin/env python3
"""
Quick training script for StormCast-ML
This script provides a simple interface to train the model with different configurations.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.train_model import train_model

def main():
    print("=" * 80)
    print("STORMCAST-ML QUICK TRAINING")
    print("=" * 80)
    print()
    
    # All training data files
    json_files = [
        'TrainingData/stormcells_Central_20250315.json',
        'TrainingData/stormcells_Midwest_20240506.json', 
        'TrainingData/stormcells_SE_20251125.json',
        'TrainingData/stormcells_TX_20251123.json'
    ]
    
    print("Training Configuration:")
    print(f"- Data files: {len(json_files)} files")
    print("- Model: LSTM (6 input features -> 2 velocity outputs)")
    print("- Default parameters: 300 epochs, batch size 32, sequence length 20")
    print()
    
    # Check if files exist
    existing_files = []
    for file in json_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✓ Found: {file}")
        else:
            print(f"✗ Missing: {file}")
    
    if not existing_files:
        print("\nERROR: No training data files found!")
        return
    
    print(f"\nUsing {len(existing_files)} training files")
    print()
    
    # Start training
    print("Starting training...")
    print("-" * 80)
    
    try:
        model = train_model(
            existing_files,
            epochs=300,           # You can modify these parameters
            batch_size=32,        # Batch size for training
            max_seq_len=20,       # Maximum sequence length
            learning_rate=0.0005  # Learning rate
        )
        
        print("-" * 80)
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Model saved to: trained_storm_lstm.pth")
        print("\nNext steps:")
        print("1. Run 'python src/predict_and_compare.py' to test predictions")
        print("2. Use 'python src/run_model.py' for new storm data prediction")
        print("3. Check TRAINING_INSTRUCTIONS.md for detailed guidance")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main()
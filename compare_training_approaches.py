#!/usr/bin/env python3
"""
StormCast-ML Training Comparison Script

This script allows you to easily compare different training approaches:
1. Original (overfitting model)
2. Improved (overfitting resolved)
3. Advanced (state-of-the-art techniques)

Usage:
    python compare_training_approaches.py --approach improved
    python compare_training_approaches.py --approach advanced
"""

import argparse
import sys
import os
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_training_approach(approach):
    """Run the specified training approach"""
    
    approaches = {
        'original': {
            'file': 'src/train_model.py',
            'description': 'Original model (with overfitting)',
            'expected_time': '10-15 minutes'
        },
        'improved': {
            'file': 'src/train_model_improved.py', 
            'description': 'Improved model (overfitting resolved)',
            'expected_time': '5-10 minutes'
        },
        'advanced': {
            'file': 'src/train_model_advanced.py',
            'description': 'Advanced model (state-of-the-art)',
            'expected_time': '30-45 minutes (5-fold CV)'
        }
    }
    
    if approach not in approaches:
        print(f"❌ Unknown approach: {approach}")
        print(f"Available approaches: {list(approaches.keys())}")
        return False
    
    config = approaches[approach]
    training_file = config['file']
    
    print("=" * 80)
    print(f"🚀 STARTING {approach.upper()} TRAINING APPROACH")
    print("=" * 80)
    print(f"📁 Script: {training_file}")
    print(f"📝 Description: {config['description']}")
    print(f"⏱️  Expected time: {config['expected_time']}")
    print("=" * 80)
    
    if not os.path.exists(training_file):
        print(f"❌ Training script not found: {training_file}")
        return False
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run([sys.executable, training_file], 
                              capture_output=False, 
                              text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print(f"✅ {approach.upper()} TRAINING COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Actual time: {duration/60:.1f} minutes")
            print("=" * 80)
            return True
        else:
            print(f"\n❌ Training failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False

def compare_approaches():
    """Show comparison of different approaches"""
    print("=" * 80)
    print("STORMCAST-ML TRAINING APPROACH COMPARISON")
    print("=" * 80)
    
    comparison_data = [
        ("Original", "Basic LSTM without regularization", "❌ Severe overfitting", "300 epochs", "Large model"),
        ("Improved", "LSTM + dropout + early stopping", "✅ Overfitting resolved", "~100 epochs", "Smaller model"),
        ("Advanced", "Attention + cross-validation + features", "✅ Best performance", "5×150 epochs", "State-of-the-art")
    ]
    
    print(f"{'Approach':<10} {'Description':<30} {'Overfitting':<20} {'Training Time':<15} {'Model Size':<15}")
    print("-" * 90)
    
    for approach, desc, overfit, time_req, model_size in comparison_data:
        print(f"{approach:<10} {desc:<30} {overfit:<20} {time_req:<15} {model_size:<15}")
    
    print("\n📊 EXPECTED VALIDATION LOSS RANGES:")
    print("   • Original: 695+ (getting worse)")
    print("   • Improved: 615-625 (steady improvement)")
    print("   • Advanced: 580-620 (robust cross-validation)")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("   • Quick test: Use 'improved' approach")
    print("   • Production: Use 'advanced' approach")
    print("   • Research: Try 'advanced' for publication-quality results")

def main():
    parser = argparse.ArgumentParser(description='StormCast-ML Training Comparison')
    parser.add_argument('--approach', choices=['original', 'improved', 'advanced', 'compare'],
                      default='compare', help='Training approach to use')
    parser.add_argument('--list', action='store_true', 
                      help='List available approaches and their descriptions')
    
    args = parser.parse_args()
    
    if args.list or args.approach == 'compare':
        compare_approaches()
        print("\n" + "=" * 80)
        print("TO RUN A SPECIFIC APPROACH:")
        print("python compare_training_approaches.py --approach improved")
        print("python compare_training_approaches.py --approach advanced")
        print("=" * 80)
        return
    
    success = run_training_approach(args.approach)
    
    if success:
        print("\n🎉 Training completed! Check the generated files:")
        print("   • Model weights: trained_storm_lstm_*.pth")
        print("   • Training curves: training_curves_*.png")
        print("   • Results summary: Check terminal output")
        
        if args.approach == 'improved':
            print("\n📈 Next steps:")
            print("   1. Run predictions: python src/run_model.py")
            print("   2. Compare results: python src/predict_and_compare.py")
            print("   3. Try advanced approach for better performance")
        
        elif args.approach == 'advanced':
            print("\n🔬 Advanced analysis:")
            print("   1. Check cross-validation results in advanced_training_results.png")
            print("   2. Examine feature importance and attention weights")
            print("   3. Consider ensemble methods with multiple folds")
    else:
        print("\n💡 Troubleshooting:")
        print("   1. Check that all training data files exist in TrainingData/")
        print("   2. Verify PyTorch and required packages are installed")
        print("   3. Try the 'improved' approach first if 'advanced' fails")
        print("   4. Check TRAINING_IMPROVEMENTS_SUMMARY.md for detailed explanations")

if __name__ == "__main__":
    main()
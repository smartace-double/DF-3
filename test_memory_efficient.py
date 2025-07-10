#!/usr/bin/env python3
"""
Test script for memory-efficient training and cross-validation.

This script tests the memory-efficient implementations to ensure they prevent
"Killed" errors during training.
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from training.cross_validation import run_walk_forward_cv, get_example_config
from train_modular import ModularTrainer

def test_memory_efficient_cv():
    """Test memory-efficient cross-validation."""
    print("="*60)
    print("Testing Memory-Efficient Cross-Validation")
    print("="*60)
    
    # Get a smaller test configuration
    config = get_example_config()
    config.update({
        'model_type': 'lightgbm',  # Use LightGBM for faster testing
        'epochs': 2,  # Minimal epochs for testing
        'batch_size': 64,  # Small batch size
        'lookback': 12,  # Smaller lookback
        'horizon': 12,
        'bias': 15000,
        'save_dir': 'models/test_cv'
    })
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        results = run_walk_forward_cv(config, save_models=False, verbose=True)
        print(f"\nTest PASSED!")
        print(f"Results: {results.mean_score:.4f} ± {results.std_score:.4f}")
        return True
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficient_training():
    """Test memory-efficient regular training."""
    print("="*60)
    print("Testing Memory-Efficient Regular Training")
    print("="*60)
    
    # Create a test configuration file
    config = {
        'mode': 'precog',
        'model_type': 'lightgbm',
        'dataset_path': 'datasets/complete_dataset_20250709_152829.csv',
        'lookback': 12,
        'horizon': 12,
        'scaler_type': 'standard',
        'bias': 15000,
        'batch_size': 64,
        'epochs': 2,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'patience': 2,
        'grad_clip': 1.0,
        'mixed_precision': True,
        'use_gpu': True,
        'save_dir': 'models/test_training',
        # Model-specific parameters
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'num_boost_round': 100,
        'early_stopping_rounds': 50
    }
    
    # Save test config
    config_path = 'config/test_memory_efficient.json'
    os.makedirs('config', exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    
    try:
        # Test training
        trainer = ModularTrainer(config_path)
        success = trainer.run_full_pipeline()
        
        if success:
            print(f"\nTest PASSED!")
            print(f"Training completed successfully")
            return True
        else:
            print(f"\nTest FAILED: Training pipeline returned False")
            return False
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test config
        if os.path.exists(config_path):
            os.remove(config_path)

def main():
    """Run all memory efficiency tests."""
    print("Starting Memory Efficiency Tests...")
    
    tests = [
        ("Cross-Validation", test_memory_efficient_cv),
        ("Regular Training", test_memory_efficient_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Running {test_name} Test")
        print(f"{'='*80}")
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print(f"\n{'='*80}")
    print(f"Test Summary: {passed}/{total} tests passed")
    print(f"{'='*80}")
    
    if passed == total:
        print("🎉 All tests PASSED! Memory-efficient training is working correctly.")
        return True
    else:
        print("⚠️  Some tests FAILED. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
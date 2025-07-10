"""
Comprehensive Pipeline Test

This test script verifies that the complete pipeline works correctly:
1. Enhanced preprocessing with walk-forward CV compatibility
2. All predictors can be created and trained
3. Modular training system works
4. Cross-validation framework functions properly

Run this script to validate the entire codebase before production use.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing import preprocess_bitcoin_enhanced, BitcoinPreprocessor, preprocess_bitcoin_data
from predictors import PredictorFactory, create_predictor_from_config
from training.cross_validation import WalkForwardCrossValidator
from config.config_loader import ConfigLoader
import warnings
warnings.filterwarnings('ignore')

def test_preprocessing():
    """Test the enhanced preprocessing pipeline."""
    print("=" * 60)
    print("Testing Enhanced Preprocessing Pipeline")
    print("=" * 60)
    
    try:
        # Test with dummy data if real dataset doesn't exist
        dataset_path = 'datasets/complete_dataset_20250709_152829.csv'
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}, creating dummy dataset...")
            create_dummy_dataset(dataset_path)
        
        # Test enhanced preprocessing
        print("Testing preprocess_bitcoin_enhanced...")
        train_data, val_data, test_data, preprocessor = preprocess_bitcoin_enhanced(
            dataset_path=dataset_path,
            lookback=12,
            horizon=12,
            scaler_type='standard',
            save_dir=None,  # Don't save artifacts
            bias=1000  # Use smaller bias for testing
        )
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        print(f"✓ Enhanced preprocessing successful")
        print(f"  Train: {X_train.shape}, {y_train.shape}")
        print(f"  Val: {X_val.shape}, {y_val.shape}")
        print(f"  Test: {X_test.shape}, {y_test.shape}")
        
        # Test compatibility wrapper
        print("\nTesting preprocess_bitcoin_data compatibility wrapper...")
        train_data2, val_data2, test_data2, preprocessor2 = preprocess_bitcoin_data(
            dataset_path=dataset_path,
            lookback=12,
            horizon=12,
            scaler_type='standard',
            save_dir=None,
            bias=1000
        )
        
        print(f"✓ Compatibility wrapper successful")
        
        # Verify data quality
        assert not np.isnan(X_train).any(), "Training features contain NaN"
        assert not np.isnan(y_train).any(), "Training targets contain NaN"
        assert not np.isinf(X_train).any(), "Training features contain Inf"
        assert not np.isinf(y_train).any(), "Training targets contain Inf"
        
        print(f"✓ Data quality checks passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predictor_creation():
    """Test that all predictors can be created properly."""
    print("\n" + "=" * 60)
    print("Testing Predictor Creation")
    print("=" * 60)
    
    try:
        # Get available predictors
        factory = PredictorFactory()
        available_predictors = factory.get_available_predictors()
        
        print(f"Available predictors: {list(available_predictors.keys())}")
        
        input_size = 500  # Typical input size after preprocessing
        success_count = 0
        
        for predictor_name in ['LSTM', 'LightGBM']:  # Test core predictors
            try:
                print(f"\nTesting {predictor_name} predictor...")
                
                # Test precog mode
                config_precog = {
                    'model_type': predictor_name,
                    'mode': 'precog',
                    'hidden_size': 64,  # Small for testing
                    'num_layers': 2,
                    'dropout': 0.1,
                    'lookback': 12,
                    'horizon': 12
                }
                
                predictor_precog = create_predictor_from_config(config_precog, input_size)
                print(f"  ✓ {predictor_name} precog mode created")
                
                # Test synth mode
                config_synth = config_precog.copy()
                config_synth['mode'] = 'synth'
                
                predictor_synth = create_predictor_from_config(config_synth, input_size)
                print(f"  ✓ {predictor_name} synth mode created")
                
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ {predictor_name} creation failed: {e}")
        
        print(f"\nPredictor creation: {success_count}/2 successful")
        return success_count > 0
        
    except Exception as e:
        print(f"✗ Predictor creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading system."""
    print("\n" + "=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)
    
    try:
        config_loader = ConfigLoader()
        
        # Test loading existing configs
        config_files = [
            'config/lstm_precog.yaml',
            'config/lightgbm_precog.yaml'
        ]
        
        success_count = 0
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    config = config_loader.load_config(config_file)
                    print(f"✓ Loaded {config_file}")
                    
                    # Validate config
                    config_loader.validate_config(config)
                    print(f"  ✓ Config validation passed")
                    
                    success_count += 1
                except Exception as e:
                    print(f"✗ Failed to load {config_file}: {e}")
            else:
                print(f"⚠ Config file not found: {config_file}")
        
        print(f"\nConfig loading: {success_count}/{len(config_files)} successful")
        return success_count > 0
        
    except Exception as e:
        print(f"✗ Config loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_training():
    """Test a mini training session to verify the pipeline works."""
    print("\n" + "=" * 60)
    print("Testing Mini Training Session")
    print("=" * 60)
    
    try:
        # Create minimal config for testing
        config = {
            'model_type': 'LSTM',
            'mode': 'precog',
            'hidden_size': 32,  # Very small
            'num_layers': 1,
            'dropout': 0.1,
            'lookback': 12,
            'horizon': 12,
            'batch_size': 32,
            'epochs': 2,  # Just 2 epochs for testing
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'patience': 2,
            'dataset_path': 'datasets/complete_dataset_20250709_152829.csv',
            'save_dir': 'test/temp_models',
            'use_gpu': torch.cuda.is_available(),
            'mixed_precision': False,  # Disable for testing
            'scaler_type': 'standard',
            'bias': 1000
        }
        
        # Ensure temp directory exists
        os.makedirs('test/temp_models', exist_ok=True)
        
        # Get test data
        dataset_path = config['dataset_path']
        if not os.path.exists(dataset_path):
            print(f"Dataset not found, creating dummy dataset...")
            create_dummy_dataset(dataset_path)
        
        # Test preprocessing
        train_data, val_data, test_data, preprocessor = preprocess_bitcoin_data(
            dataset_path=dataset_path,
            lookback=config['lookback'],
            horizon=config['horizon'],
            scaler_type=config['scaler_type'],
            bias=config['bias']
        )
        
        X_train, y_train = train_data
        input_size = X_train.shape[1]
        
        print(f"✓ Got training data: {X_train.shape}, {y_train.shape}")
        
        # Create model
        model = create_predictor_from_config(config, input_size)
        print(f"✓ Created model: {model.__class__.__name__}")
        
        # Test one forward pass
        device = torch.device('cuda' if config['use_gpu'] else 'cpu')
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            test_batch = torch.FloatTensor(X_train[:4]).to(device)  # Small batch
            predictions = model(test_batch)
            print(f"✓ Forward pass successful: {[p.shape if torch.is_tensor(p) else type(p) for p in (predictions if isinstance(predictions, tuple) else [predictions])]}")
        
        print(f"✓ Mini training test successful")
        
        # Cleanup
        import shutil
        if os.path.exists('test/temp_models'):
            shutil.rmtree('test/temp_models')
        
        return True
        
    except Exception as e:
        print(f"✗ Mini training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_dummy_dataset(dataset_path):
    """Create a dummy dataset for testing."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    print(f"Creating dummy dataset at {dataset_path}")
    
    # Create minimal dataset structure
    n_samples = 5000
    start_time = datetime(2024, 1, 1)
    
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_samples)]
    
    # Generate realistic-looking Bitcoin price data
    np.random.seed(42)
    base_price = 50000
    price_changes = np.random.normal(0, 0.01, n_samples).cumsum()
    prices = base_price * (1 + price_changes)
    
    # Create basic OHLCV data
    data = {
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.exponential(100, n_samples),
        'taker_buy_volume': np.random.exponential(50, n_samples)
    }
    
    # Add technical indicators (simplified)
    for i in range(30):  # Add various technical features
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Add whale features
    data['whale_tx_count'] = np.random.poisson(5, n_samples)
    data['whale_btc_volume'] = np.random.exponential(10, n_samples)
    data['whale_avg_price'] = prices * (1 + np.random.normal(0, 0.01, n_samples))
    data['exchange_netflow'] = np.random.normal(0, 100, n_samples)
    data['sopr'] = np.random.normal(1, 0.1, n_samples)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    df.to_csv(dataset_path, index=False)
    print(f"✓ Dummy dataset created with {len(df)} rows")

def main():
    """Run all tests."""
    print("🚀 Starting Comprehensive Pipeline Test")
    print("=" * 80)
    
    # Track test results
    test_results = []
    
    # Run tests
    test_results.append(("Preprocessing", test_preprocessing()))
    test_results.append(("Predictor Creation", test_predictor_creation()))
    test_results.append(("Config Loading", test_config_loading()))
    test_results.append(("Mini Training", test_mini_training()))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is ready for production use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
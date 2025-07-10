#!/usr/bin/env python3
"""
Train All Predictors Script

This script trains all available predictors using the modular training system.
It will run through each configuration file and train the corresponding model.

Usage:
    python train_all_predictors.py [--mode train|test] [--predictors LSTM,LightGBM,...]

Options:
    --mode: 'train' (default) to train models, 'test' to just test configurations
    --predictors: Comma-separated list of predictors to train (default: all)
    --configs-only: Only train precog mode configs
    --use-cv: Use walk-forward cross-validation instead of simple training
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import yaml
import time
from datetime import datetime

def get_available_configs() -> List[str]:
    """Get all available configuration files."""
    config_dir = Path('config')
    if not config_dir.exists():
        return []
    
    config_files = []
    for config_file in config_dir.glob('*.yaml'):
        if config_file.name != 'config_loader.py':  # Skip the Python file
            config_files.append(str(config_file))
    
    return sorted(config_files)

def parse_config_file(config_path: str) -> Dict[str, Any]:
    """Parse a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error parsing {config_path}: {e}")
        return {}

def filter_configs(config_files: List[str], 
                  predictors: List[str] = None, 
                  modes: List[str] = None) -> List[str]:
    """Filter configuration files based on predictors and modes."""
    if not predictors and not modes:
        return config_files
    
    filtered = []
    for config_file in config_files:
        config = parse_config_file(config_file)
        if not config:
            continue
        
        # Check predictor type
        model_type = config.get('model_type', '').lower()
        if predictors and not any(p.lower() in model_type for p in predictors):
            continue
        
        # Check mode
        mode = config.get('mode', '').lower()
        if modes and mode not in [m.lower() for m in modes]:
            continue
        
        filtered.append(config_file)
    
    return filtered

def run_training(config_file: str, use_cv: bool = False) -> bool:
    """Run training for a specific configuration."""
    print(f"\n{'='*80}")
    print(f"Training with config: {config_file}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        if use_cv:
            # Use walk-forward cross-validation
            from training.cross_validation import WalkForwardCrossValidator
            
            # Load config
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Create cross-validator
            cv = WalkForwardCrossValidator(config, save_models=True, verbose=True)
            
            # Run cross-validation
            results = cv.run_walk_forward_cv()
            
            print(f"Cross-validation completed:")
            print(f"  Mean Score: {results.mean_score:.4f} ± {results.std_score:.4f}")
            print(f"  Best Fold: {results.best_fold}")
            print(f"  Best Score: {results.best_score:.4f}")
            
        else:
            # Use modular training
            cmd = [sys.executable, 'train_modular.py', '--config', config_file, '--mode', 'train']
            
            print(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Training completed successfully!")
                if result.stdout:
                    print("Output:")
                    print(result.stdout[-1000:])  # Last 1000 chars
            else:
                print("❌ Training failed!")
                if result.stderr:
                    print("Error:")
                    print(result.stderr[-1000:])  # Last 1000 chars
                return False
        
        elapsed = time.time() - start_time
        print(f"Training time: {elapsed:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration(config_file: str) -> bool:
    """Test a configuration without training."""
    print(f"\nTesting config: {config_file}")
    
    try:
        # Parse config
        config = parse_config_file(config_file)
        if not config:
            print("  ❌ Failed to parse config")
            return False
        
        # Check required fields
        required_fields = ['model_type', 'mode']
        for field in required_fields:
            if field not in config:
                print(f"  ❌ Missing required field: {field}")
                return False
        
        # Test config loading with train_modular.py
        cmd = [sys.executable, 'train_modular.py', '--config', config_file, '--mode', 'test']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Config test passed")
            return True
        else:
            print(f"  ❌ Config test failed: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train all Bitcoin price predictors')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='Mode: train models or test configurations')
    parser.add_argument('--predictors', type=str,
                        help='Comma-separated list of predictors (e.g., LSTM,LightGBM)')
    parser.add_argument('--configs-only', action='store_true',
                        help='Only train precog mode configs')
    parser.add_argument('--use-cv', action='store_true',
                        help='Use walk-forward cross-validation')
    parser.add_argument('--config-filter', type=str,
                        help='Filter configs by substring in filename')
    
    args = parser.parse_args()
    
    print("🚀 Bitcoin Price Predictor Training Script")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Use Cross-Validation: {args.use_cv}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get available configs
    config_files = get_available_configs()
    if not config_files:
        print("❌ No configuration files found in config/ directory")
        return 1
    
    print(f"\nFound {len(config_files)} configuration files:")
    for config_file in config_files:
        print(f"  - {config_file}")
    
    # Filter configs
    predictors = args.predictors.split(',') if args.predictors else None
    modes = ['precog'] if args.configs_only else None
    
    filtered_configs = filter_configs(config_files, predictors, modes)
    
    # Apply additional filter
    if args.config_filter:
        filtered_configs = [c for c in filtered_configs if args.config_filter.lower() in c.lower()]
    
    if predictors:
        print(f"\nFiltering for predictors: {predictors}")
    if modes:
        print(f"Filtering for modes: {modes}")
    if args.config_filter:
        print(f"Filtering for substring: {args.config_filter}")
    
    print(f"\nSelected {len(filtered_configs)} configurations to process:")
    for config_file in filtered_configs:
        config = parse_config_file(config_file)
        model_type = config.get('model_type', 'Unknown')
        mode = config.get('mode', 'Unknown')
        print(f"  - {config_file} ({model_type} - {mode})")
    
    if not filtered_configs:
        print("❌ No configurations match the filter criteria")
        return 1
    
    # Confirm with user
    if args.mode == 'train':
        print(f"\n⚠️  This will train {len(filtered_configs)} models.")
        print("This may take a significant amount of time and computational resources.")
        response = input("Continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Training cancelled.")
            return 0
    
    # Process configurations
    results = []
    total_start_time = time.time()
    
    for i, config_file in enumerate(filtered_configs, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(filtered_configs)}: {config_file}")
        print(f"{'='*80}")
        
        if args.mode == 'train':
            success = run_training(config_file, args.use_cv)
        else:
            success = test_configuration(config_file)
        
        results.append((config_file, success))
    
    # Print summary
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Total configurations processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Total time: {total_elapsed:.1f} seconds")
    
    print(f"\nDetailed results:")
    for config_file, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        config = parse_config_file(config_file)
        model_type = config.get('model_type', 'Unknown')
        mode = config.get('mode', 'Unknown')
        print(f"  {status} {config_file} ({model_type} - {mode})")
    
    if successful == total:
        print(f"\n🎉 All {total} configurations processed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - successful} configurations failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
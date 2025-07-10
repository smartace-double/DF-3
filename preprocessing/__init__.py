"""
Preprocessing Package

This package contains comprehensive data preprocessing utilities for Bitcoin price prediction:
- precog_preprocess: Enhanced preprocessing module with BitcoinPreprocessor class and interval constraints
"""

from typing import Optional
from .precog_preprocess import (
    BitcoinPreprocessor, 
    preprocess_bitcoin_enhanced,
    enforce_interval_constraints,
    calculate_interval_metrics,
    postprocess_predictions
)

# Compatibility wrapper for train_modular.py
def preprocess_bitcoin_data(dataset_path: str,
                           lookback: int = 12,
                           horizon: int = 12,
                           scaler_type: str = 'standard',
                           save_dir: Optional[str] = None,
                           bias: int = 15000):
    """
    Compatibility wrapper for preprocess_bitcoin_enhanced.
    
    This function provides the same interface expected by train_modular.py
    but uses the enhanced preprocessing pipeline internally.
    """
    return preprocess_bitcoin_enhanced(
        dataset_path=dataset_path,
        lookback=lookback,
        horizon=horizon,
        scaler_type=scaler_type,
        save_dir=save_dir,
        bias=bias
    )

__all__ = [
    'BitcoinPreprocessor',
    'preprocess_bitcoin_enhanced',
    'preprocess_bitcoin_data',  # Add compatibility function
    'enforce_interval_constraints',
    'calculate_interval_metrics',
    'postprocess_predictions'
] 
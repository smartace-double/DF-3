"""
Training Framework Package

This package contains comprehensive training utilities for Bitcoin price prediction models:
- cross_validation: Time-series cross-validation framework
- optuna_optimization: Hyperparameter optimization with Optuna
"""

from .cross_validation import CrossValidator, CrossValidationResults, run_cross_validation
from .optuna_optimization import OptunaOptimizer, optimize_hyperparameters

__all__ = [
    'CrossValidator',
    'CrossValidationResults', 
    'run_cross_validation',
    'OptunaOptimizer',
    'optimize_hyperparameters'
] 
"""
Training Framework Package

This package contains comprehensive training utilities for Bitcoin price prediction models:
- cross_validation: Walk-forward time-series cross-validation framework
- optuna_optimization: Hyperparameter optimization with Optuna
"""

from .cross_validation import WalkForwardCrossValidator, WalkForwardResults, run_walk_forward_cv
from .optuna_optimization import OptunaOptimizer, optimize_hyperparameters

__all__ = [
    'WalkForwardCrossValidator',
    'WalkForwardResults', 
    'run_walk_forward_cv',
    'OptunaOptimizer',
    'optimize_hyperparameters'
] 
"""
Loss Functions Package

This package contains loss and evaluation functions for different prediction modes:
- precog_loss: Point and interval predictions for precog subnet challenge
- synth_loss: Detailed close, low, high predictions for each timestep
"""

from .precog_loss import precog_loss, evaluate_precog
from .synth_loss import synth_loss, evaluate_synth

__all__ = [
    'precog_loss',
    'evaluate_precog',
    'synth_loss',
    'evaluate_synth'
] 
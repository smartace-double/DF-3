"""
Base Predictor Class

This module defines the base interface for all Bitcoin price predictors.
All predictor implementations should inherit from this base class.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np

class BaseBitcoinPredictor(nn.Module, ABC):
    """
    Abstract base class for Bitcoin price predictors.
    
    This class defines the interface that all predictors must implement.
    It supports two prediction modes:
    - 'precog': Point and interval predictions for the precog subnet challenge
    - 'synth': Detailed close, low, high predictions for each timestep
    """
    
    def __init__(self, input_size: int, config: Dict[str, Any]):
        """
        Initialize the base predictor.
        
        Args:
            input_size: Number of input features
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        
        self.input_size = input_size
        self.config = config
        self.mode = config.get('mode', 'precog')  # 'precog' or 'synth'
        
        # Validate mode
        if self.mode not in ['precog', 'synth']:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'precog' or 'synth'")
        
        # Store common configuration
        self.hidden_size = config.get('hidden_size', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        self.use_layer_norm = config.get('use_layer_norm', True)
        self.activation = config.get('activation', 'SiLU')
        
        # Initialize activation function
        self.act_fn = self._get_activation_function(self.activation)
        
        # Initialize the model architecture
        self._build_model()
        
    def _get_activation_function(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            'SiLU': nn.SiLU(),
            'GELU': nn.GELU(),
            'Mish': nn.Mish(),
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh()
        }
        
        if activation_name not in activation_map:
            raise ValueError(f"Unsupported activation: {activation_name}")
            
        return activation_map[activation_name]
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            
        Returns:
            Model predictions. Format depends on mode:
            - precog mode: (point_pred, interval_pred)
            - synth mode: (detailed_pred,)
        """
        pass
    
    def get_output_size(self) -> int:
        """Get the expected output size based on mode."""
        if self.mode == 'precog':
            return 3  # point_pred (1) + interval_pred (2)
        elif self.mode == 'synth':
            return 36  # 12 timesteps * 3 values (close, low, high)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging and saving."""
        return {
            'model_type': self.__class__.__name__,
            'input_size': self.input_size,
            'mode': self.mode,
            'config': self.config,
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_model(self, path: str, additional_info: Optional[Dict[str, Any]] = None):
        """Save model state and configuration."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'config': self.config
        }
        
        if additional_info:
            save_dict.update(additional_info)
            
        torch.save(save_dict, path)
    
    @classmethod
    def load_model(cls, path: str, input_size: Optional[int] = None):
        """Load model from saved state."""
        checkpoint = torch.load(path, map_location='cpu')
        
        config = checkpoint['config']
        model_input_size = input_size if input_size is not None else checkpoint['model_info']['input_size']
        
        # Create model instance
        model = cls(model_input_size, config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint.get('model_info', {})
    
    def print_model_summary(self):
        """Print a summary of the model architecture."""
        info = self.get_model_info()
        print(f"\n{'='*50}")
        print(f"Model Summary: {info['model_type']}")
        print(f"{'='*50}")
        print(f"Mode: {info['mode']}")
        print(f"Input Size: {info['input_size']}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Num Layers: {self.num_layers}")
        print(f"Activation: {self.activation}")
        print(f"Dropout: {self.dropout}")
        print(f"Layer Norm: {self.use_layer_norm}")
        print(f"Total Parameters: {info['parameter_count']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"{'='*50}\n") 
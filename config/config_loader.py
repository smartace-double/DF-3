"""
Configuration Loader

This module provides utilities for loading and validating configuration files
for Bitcoin price prediction models.
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json

class ConfigLoader:
    """
    Configuration loader for Bitcoin price prediction models.
    
    This class provides methods to load, validate, and merge configuration files
    from YAML format.
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        'model_type': 'LSTM',
        'mode': 'precog',
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.1,
        'activation': 'SiLU',
        'use_layer_norm': True,
        'bidirectional': False,
        'lr': 1e-4,
        'batch_size': 512,
        'epochs': 10,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'patience': 5,
        'dataset_path': 'datasets/complete_dataset_20250709_152829.csv',
        'lookback': 72,  # 6 hours * 12 (5-minute intervals)
        'horizon': 12,   # 1 hour * 12 (5-minute intervals)
        'train_split': 0.85,
        'val_split': 0.10,
        'test_split': 0.05,
        'save_dir': 'models',
        'use_gpu': True,
        'mixed_precision': True,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True
    }
    
    # Required configuration keys
    REQUIRED_KEYS = ['model_type', 'mode']
    
    # Valid values for specific keys
    VALID_VALUES = {
        'model_type': ['LSTM', 'lstm', 'LightGBM', 'lightgbm', 'lgb', 'TFT', 'tft', 'TCN', 'tcn', 'GARCH', 'garch'],
        'mode': ['precog', 'synth'],
        'activation': ['SiLU', 'GELU', 'Mish', 'ReLU', 'Tanh'],
    }
    
    def __init__(self, config_dir: str = 'config'):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            # Try relative to config directory
            config_path = self.config_dir / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        # Merge with defaults
        config = self.merge_with_defaults(config)
        
        # Validate configuration
        self.validate_config(config)
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: Union[str, Path]):
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save the configuration file
        """
        config_path = Path(config_path)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save YAML file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with default values.
        
        Args:
            config: User configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        merged_config = self.DEFAULT_CONFIG.copy()
        merged_config.update(config)
        return merged_config
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required keys
        for key in self.REQUIRED_KEYS:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Check valid values
        for key, valid_values in self.VALID_VALUES.items():
            if key in config and config[key] not in valid_values:
                raise ValueError(f"Invalid value for {key}: {config[key]}. "
                               f"Valid values: {valid_values}")
        
        # Check data types and ranges
        validations = {
            'hidden_size': (int, lambda x: x > 0),
            'num_layers': (int, lambda x: x > 0),
            'dropout': (float, lambda x: 0 <= x <= 1),
            'lr': (float, lambda x: x > 0),
            'batch_size': (int, lambda x: x > 0),
            'epochs': (int, lambda x: x > 0),
            'weight_decay': (float, lambda x: x >= 0),
            'grad_clip': (float, lambda x: x > 0),
            'patience': (int, lambda x: x > 0),
            'lookback': (int, lambda x: x > 0),
            'horizon': (int, lambda x: x > 0),
            'train_split': (float, lambda x: 0 < x < 1),
            'val_split': (float, lambda x: 0 < x < 1),
            'test_split': (float, lambda x: 0 < x < 1),
        }
        
        for key, (expected_type, validator) in validations.items():
            if key in config:
                value = config[key]
                if not isinstance(value, expected_type):
                    raise ValueError(f"Invalid type for {key}: expected {expected_type.__name__}, "
                                   f"got {type(value).__name__}")
                if not validator(value):
                    raise ValueError(f"Invalid value for {key}: {value}")
        
        # Check that splits sum to 1
        if 'train_split' in config and 'val_split' in config and 'test_split' in config:
            total_split = config['train_split'] + config['val_split'] + config['test_split']
            if abs(total_split - 1.0) > 1e-6:
                raise ValueError(f"Train/val/test splits must sum to 1.0, got {total_split}")
    
    def get_config_template(self) -> Dict[str, Any]:
        """
        Get a template configuration with comments.
        
        Returns:
            Template configuration dictionary
        """
        return {
            '# Model Configuration': None,
            'model_type': 'LSTM',  # Model architecture type
            'mode': 'precog',   # 'precog' or 'synth'
            
            '# Model Architecture': None,
            'hidden_size': 256,    # Hidden dimension size
            'num_layers': 3,       # Number of LSTM layers
            'dropout': 0.1,        # Dropout rate
            'activation': 'SiLU',  # Activation function
            'use_layer_norm': True, # Use layer normalization
            'bidirectional': False, # Use bidirectional LSTM
            
            '# Training Configuration': None,
            'lr': 1e-4,            # Learning rate
            'batch_size': 512,     # Batch size
            'epochs': 10,          # Number of training epochs
            'weight_decay': 1e-5,  # Weight decay
            'grad_clip': 1.0,      # Gradient clipping threshold
            'patience': 5,         # Early stopping patience
            
            '# Data Configuration': None,
            'dataset_path': 'datasets/complete_dataset_20250709_152829.csv',
            'lookback': 72,        # Input sequence length (6 hours)
            'horizon': 12,         # Output sequence length (1 hour)
            'train_split': 0.85,   # Training data split
            'val_split': 0.10,     # Validation data split
            'test_split': 0.05,    # Test data split
            
            '# System Configuration': None,
            'save_dir': 'models',  # Model save directory
            'use_gpu': True,       # Use GPU if available
            'mixed_precision': True, # Use mixed precision training
            'num_workers': 4,      # Number of data loader workers
            'pin_memory': True,    # Pin memory for GPU transfer
            'persistent_workers': True, # Keep workers alive
        }
    
    def create_config_template(self, config_path: Union[str, Path]):
        """
        Create a configuration template file.
        
        Args:
            config_path: Path to save the template file
        """
        template = self.get_config_template()
        
        # Remove comment keys and create clean config
        clean_config = {k: v for k, v in template.items() if not k.startswith('#')}
        
        self.save_config(clean_config, config_path)
    
    def list_configs(self) -> List[str]:
        """
        List available configuration files.
        
        Returns:
            List of configuration file names
        """
        config_files = []
        for file_path in self.config_dir.glob('*.yaml'):
            config_files.append(file_path.name)
        for file_path in self.config_dir.glob('*.yml'):
            config_files.append(file_path.name)
        return sorted(config_files)
    
    def print_config(self, config: Dict[str, Any]):
        """
        Print configuration in a formatted way.
        
        Args:
            config: Configuration dictionary to print
        """
        print(f"\n{'='*60}")
        print("Configuration")
        print(f"{'='*60}")
        
        sections = {
            'Model Configuration': ['model_type', 'mode'],
            'Model Architecture': ['hidden_size', 'num_layers', 'dropout', 'activation', 
                                 'use_layer_norm', 'bidirectional'],
            'Training Configuration': ['lr', 'batch_size', 'epochs', 'weight_decay', 
                                     'grad_clip', 'patience'],
            'Data Configuration': ['dataset_path', 'lookback', 'horizon', 'train_split', 
                                 'val_split', 'test_split'],
            'System Configuration': ['save_dir', 'use_gpu', 'mixed_precision', 'num_workers', 
                                   'pin_memory', 'persistent_workers']
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                if key in config:
                    value = config[key]
                    print(f"  {key}: {value}")
        
        print(f"\n{'='*60}\n")


# Convenience functions
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_config(config_path)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Convenience function to save configuration.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    loader = ConfigLoader()
    loader.save_config(config, config_path)


def create_config_template(config_path: Union[str, Path]):
    """
    Convenience function to create configuration template.
    
    Args:
        config_path: Path to save template file
    """
    loader = ConfigLoader()
    loader.create_config_template(config_path) 
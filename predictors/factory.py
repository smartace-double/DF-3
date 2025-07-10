"""
Predictor Factory

This module provides a factory for creating different types of Bitcoin price predictors
based on configuration specifications.
"""

from typing import Dict, Any, Optional, Type
from .base_predictor import BaseBitcoinPredictor
from .LSTM_predictor import LSTMBitcoinPredictor
from .lightgbm_predictor import LightGBMBitcoinPredictor
from .tft_predictor import TFTPredictor
from .tcn_predictor import TCNPredictor
from .garch_predictor import GARCHPredictor

class PredictorFactory:
    """
    Factory class for creating Bitcoin price predictors.
    
    This factory allows easy instantiation of different predictor types
    based on configuration dictionaries.
    """
    
    # Registry of available predictors
    _predictors: Dict[str, Type[BaseBitcoinPredictor]] = {
        'LSTM': LSTMBitcoinPredictor,
        'lstm': LSTMBitcoinPredictor,  # Case insensitive
        'LightGBM': LightGBMBitcoinPredictor,
        'lightgbm': LightGBMBitcoinPredictor,
        'lgb': LightGBMBitcoinPredictor,
        'TFT': TFTPredictor,
        'tft': TFTPredictor,  # Case insensitive
        'TCN': TCNPredictor,
        'tcn': TCNPredictor,  # Case insensitive
        'GARCH': GARCHPredictor,
        'garch': GARCHPredictor,  # Case insensitive
    }
    
    @classmethod
    def register_predictor(cls, name: str, predictor_class: Type[BaseBitcoinPredictor]):
        """
        Register a new predictor type.
        
        Args:
            name: Name identifier for the predictor
            predictor_class: Class that implements BaseBitcoinPredictor
        """
        if not issubclass(predictor_class, BaseBitcoinPredictor):
            raise ValueError(f"Predictor class must inherit from BaseBitcoinPredictor")
        
        cls._predictors[name] = predictor_class
        cls._predictors[name.lower()] = predictor_class  # Case insensitive
    
    @classmethod
    def get_available_predictors(cls) -> Dict[str, Type[BaseBitcoinPredictor]]:
        """Get all available predictor types."""
        # Return only unique predictors (remove lowercase duplicates)
        unique_predictors = {}
        for name, predictor_class in cls._predictors.items():
            if name.isupper() or name.islower():
                unique_predictors[name] = predictor_class
        return unique_predictors
    
    @classmethod
    def create_predictor(cls, 
                        predictor_type: str, 
                        input_size: int, 
                        config: Dict[str, Any]) -> BaseBitcoinPredictor:
        """
        Create a predictor instance.
        
        Args:
            predictor_type: Type of predictor to create (e.g., 'LSTM')
            input_size: Number of input features
            config: Configuration dictionary with model parameters
            
        Returns:
            Configured predictor instance
            
        Raises:
            ValueError: If predictor type is not registered
        """
        if predictor_type not in cls._predictors:
            available = list(cls.get_available_predictors().keys())
            raise ValueError(f"Unknown predictor type: {predictor_type}. "
                           f"Available types: {available}")
        
        predictor_class = cls._predictors[predictor_type]
        return predictor_class(input_size, config)
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any], input_size: int) -> BaseBitcoinPredictor:
        """
        Create a predictor from a complete configuration dictionary.
        
        Args:
            config: Configuration dictionary that must contain 'model_type' key
            input_size: Number of input features
            
        Returns:
            Configured predictor instance
            
        Raises:
            ValueError: If 'model_type' is not specified in config
        """
        if 'model_type' not in config:
            raise ValueError("Configuration must contain 'model_type' key")
        
        predictor_type = config['model_type']
        return cls.create_predictor(predictor_type, input_size, config)
    
    @classmethod
    def load_predictor(cls, 
                      model_path: str, 
                      predictor_type: Optional[str] = None,
                      input_size: Optional[int] = None) -> BaseBitcoinPredictor:
        """
        Load a predictor from a saved model file.
        
        Args:
            model_path: Path to saved model file
            predictor_type: Type of predictor (if not in saved file)
            input_size: Input size (if not in saved file)
            
        Returns:
            Loaded predictor instance
        """
        import torch
        
        # Load checkpoint to get model info
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine predictor type
        if predictor_type is None:
            if 'model_info' in checkpoint and 'model_type' in checkpoint['model_info']:
                model_type = checkpoint['model_info']['model_type']
                # Convert class name to factory key
                if model_type == 'LSTMBitcoinPredictor':
                    predictor_type = 'LSTM'
                else:
                    predictor_type = model_type
            else:
                raise ValueError("Could not determine predictor type from saved model. "
                               "Please specify predictor_type parameter.")
        
        # Determine input size
        if input_size is None:
            if 'model_info' in checkpoint and 'input_size' in checkpoint['model_info']:
                input_size = checkpoint['model_info']['input_size']
            else:
                raise ValueError("Could not determine input size from saved model. "
                               "Please specify input_size parameter.")
        
        # Get predictor class and load model
        if predictor_type not in cls._predictors:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        
        predictor_class = cls._predictors[predictor_type]
        return predictor_class.load_model(model_path, input_size)[0]
    
    @classmethod
    def print_available_predictors(cls):
        """Print information about available predictors."""
        predictors = cls.get_available_predictors()
        
        print(f"\n{'='*60}")
        print("Available Predictor Types")
        print(f"{'='*60}")
        
        for name, predictor_class in predictors.items():
            print(f"• {name}: {predictor_class.__name__}")
            print(f"  Description: {predictor_class.__doc__.strip().split('.')[0] if predictor_class.__doc__ else 'No description'}")
        
        print(f"{'='*60}\n")


# Convenience functions for common operations
def create_lstm_predictor(input_size: int, config: Dict[str, Any]) -> BaseBitcoinPredictor:
    """
    Convenience function to create LSTM predictor.
    
    Args:
        input_size: Number of input features
        config: Configuration dictionary
        
    Returns:
        LSTM predictor instance
    """
    return PredictorFactory.create_predictor('LSTM', input_size, config)


def create_lightgbm_predictor(input_size: int, config: Dict[str, Any]) -> BaseBitcoinPredictor:
    """
    Convenience function to create LightGBM predictor.
    
    Args:
        input_size: Number of input features
        config: Configuration dictionary
        
    Returns:
        LightGBM predictor instance
    """
    return PredictorFactory.create_predictor('LightGBM', input_size, config)


def create_tft_predictor(input_size: int, config: Dict[str, Any]) -> BaseBitcoinPredictor:
    """
    Convenience function to create TFT predictor.
    
    Args:
        input_size: Number of input features
        config: Configuration dictionary
        
    Returns:
        TFT predictor instance
    """
    return PredictorFactory.create_predictor('TFT', input_size, config)


def create_tcn_predictor(input_size: int, config: Dict[str, Any]) -> BaseBitcoinPredictor:
    """
    Convenience function to create TCN predictor.
    
    Args:
        input_size: Number of input features
        config: Configuration dictionary
        
    Returns:
        TCN predictor instance
    """
    return PredictorFactory.create_predictor('TCN', input_size, config)


def create_garch_predictor(input_size: int, config: Dict[str, Any]) -> BaseBitcoinPredictor:
    """
    Convenience function to create GARCH predictor.
    
    Args:
        input_size: Number of input features
        config: Configuration dictionary
        
    Returns:
        GARCH predictor instance
    """
    return PredictorFactory.create_predictor('GARCH', input_size, config)


def create_predictor_from_config(config: Dict[str, Any], input_size: int) -> BaseBitcoinPredictor:
    """
    Convenience function to create predictor from config.
    
    Args:
        config: Configuration dictionary with 'model_type' key
        input_size: Number of input features
        
    Returns:
        Predictor instance
    """
    return PredictorFactory.create_from_config(config, input_size)


def load_predictor(model_path: str, 
                  predictor_type: Optional[str] = None,
                  input_size: Optional[int] = None) -> BaseBitcoinPredictor:
    """
    Convenience function to load predictor from file.
    
    Args:
        model_path: Path to saved model file
        predictor_type: Type of predictor (auto-detected if None)
        input_size: Input size (auto-detected if None)
        
    Returns:
        Loaded predictor instance
    """
    return PredictorFactory.load_predictor(model_path, predictor_type, input_size) 
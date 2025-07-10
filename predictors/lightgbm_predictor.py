"""
LightGBM Bitcoin Price Predictor

This module implements a LightGBM-based predictor for Bitcoin price forecasting.
LightGBM is a gradient boosting framework that is particularly effective for
tabular data and can work well with time series features.

It supports both precog and synth modes:
- 'precog': Point and interval predictions for the precog subnet challenge
- 'synth': Detailed close, low, high predictions for each timestep
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import pickle
import json
import warnings

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM is required for this predictor. Install with: pip install lightgbm")

from .base_predictor import BaseBitcoinPredictor

warnings.filterwarnings('ignore')

class LightGBMBitcoinPredictor(BaseBitcoinPredictor):
    """
    LightGBM-based Bitcoin price predictor.
    
    This predictor uses LightGBM for both precog and synth modes.
    Unlike neural networks, LightGBM works with flattened features rather than sequences.
    """
    
    def __init__(self, input_size: int, config: Dict[str, Any]):
        """
        Initialize the LightGBM predictor.
        
        Args:
            input_size: Number of input features (will be flattened from sequences)
            config: Configuration dictionary
        """
        super().__init__(input_size, config)
        
        # LightGBM specific parameters
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.1),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'min_child_samples': config.get('min_child_samples', 20),
            'max_depth': config.get('max_depth', -1),
            'reg_alpha': config.get('reg_alpha', 0.0),
            'reg_lambda': config.get('reg_lambda', 0.0),
            'random_state': config.get('random_state', 42),
            'n_jobs': config.get('n_jobs', -1),
            'verbose': -1
        }
        
        # Training parameters
        self.num_boost_round = config.get('num_boost_round', 1000)
        self.early_stopping_rounds = config.get('early_stopping_rounds', 100)
        self.lookback = config.get('lookback', 72)
        
        # Models for each target
        self.models = {}
        self.is_trained = False
        
    def _build_model(self):
        """Build the model architecture (not needed for LightGBM)."""
        pass
    
    def parameters(self, recurse: bool = True):
        """Override parameters method since LightGBM doesn't use PyTorch parameters."""
        # Return empty parameter list since LightGBM doesn't use PyTorch parameters
        return []
    
    def train(self, mode: bool = True):
        """Override train method since LightGBM doesn't use PyTorch training."""
        # LightGBM models are already trained, so this is a no-op
        return self
    
    def eval(self):
        """Override eval method since LightGBM doesn't use PyTorch evaluation."""
        # LightGBM models are always in eval mode
        return self
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare features for LightGBM training.
        
        Args:
            X: Input features [batch_size, flattened_features]
                The features are already flattened by the preprocessor in the format:
                [per_timestep_features * lookback + static_features]
            
        Returns:
            Features with additional statistical measures
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input features X must be a numpy array")
        
        if X.size == 0:
            raise ValueError("Input features X cannot be empty")
        
        batch_size = X.shape[0]
        
        # Define feature dimensions from preprocessing stats
        n_per_timestep = 29  # From preprocessing stats
        lookback = 72  # From config
        n_static = 6  # From preprocessing stats
        
        # Validate input dimensions
        expected_features = n_per_timestep * lookback + n_static
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features "
                f"({n_per_timestep} per-timestep features × {lookback} timesteps + {n_static} static features) "
                f"but got {X.shape[1]} features"
            )
        
        # Add statistical features
        X_features = []
        for i in range(batch_size):
            sample_features = []
            
            # Add original flattened features
            sample_features.extend(X[i].tolist())  # Convert to list to avoid numpy array issues
            
            # Extract per-timestep features
            per_timestep_data = X[i, :n_per_timestep * lookback].reshape(lookback, n_per_timestep)
            
            # Calculate statistics for each per-timestep feature
            for feat_idx in range(n_per_timestep):
                feat_seq = per_timestep_data[:, feat_idx]
                
                # Basic statistics
                sample_features.extend([
                    float(np.mean(feat_seq)),  # Convert numpy types to Python float
                    float(np.std(feat_seq)),
                    float(np.min(feat_seq)),
                    float(np.max(feat_seq)),
                    float(np.median(feat_seq)),
                    float(feat_seq[-1]),  # Last value
                    float(feat_seq[-1] - feat_seq[0]),  # Change from first to last
                    float(np.sum(feat_seq > 0)),  # Count of positive values
                ])
                
                # Trend features
                if len(feat_seq) > 1:
                    sample_features.append(float(feat_seq[-1] - feat_seq[-2]))  # Last change
                    sample_features.append(float(np.mean(np.diff(feat_seq))))  # Average change
                else:
                    sample_features.extend([0.0, 0.0])
            
            X_features.append(sample_features)
        
        return np.array(X_features, dtype=np.float32)
    
    def _train_target_model(self, X: np.ndarray, y: np.ndarray, target_name: str, 
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train a LightGBM model for a specific target.
        
        Args:
            X: Training features
            y: Training targets
            target_name: Name of the target
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        # Create LightGBM datasets
        train_data = lgb.Dataset(X, label=y)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train model
        callbacks = []
        if self.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
        
        model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self.models[target_name] = model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the LightGBM models.
        
        Args:
            X: Training features [batch_size, sequence_length, features]
            y: Training targets [batch_size, n_targets]
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        # Prepare features
        X_features = self._prepare_features(X)
        if X_val is not None:
            X_val_features = self._prepare_features(X_val)
        else:
            X_val_features = None
        
        # Train separate models for each target
        n_targets = y.shape[1]
        
        if self.mode == 'precog':
            # For precog mode, we need to derive point and interval targets
            # from the detailed targets
            
            # y has shape [batch_size, 36] with detailed targets
            # Reshape to [batch_size, 12, 3] for easier processing
            y_detailed = y.reshape(-1, 12, 3)  # [close, low, high] for each timestep
            
            # Extract precog targets:
            # 1. Point prediction = close price at last timestep
            point_targets = y_detailed[:, -1, 0]  # close at timestep 11
            
            # 2. Interval prediction = [min_low, max_high] over all timesteps
            min_targets = y_detailed[:, :, 1].min(axis=1)  # min of all lows
            max_targets = y_detailed[:, :, 2].max(axis=1)  # max of all highs
            
            # Train models for precog targets
            self._train_target_model(X_features, point_targets, 'point',
                                   X_val_features, y_val[:, -1, 0].reshape(-1, 12, 3)[:, -1, 0] if X_val_features is not None else None)
            
            self._train_target_model(X_features, min_targets, 'interval_min',
                                   X_val_features, y_val.reshape(-1, 12, 3)[:, :, 1].min(axis=1) if X_val_features is not None else None)
            
            self._train_target_model(X_features, max_targets, 'interval_max',
                                   X_val_features, y_val.reshape(-1, 12, 3)[:, :, 2].max(axis=1) if X_val_features is not None else None)
        
        elif self.mode == 'synth':
            # For synth mode, train a model for each target
            for target_idx in range(n_targets):
                target_name = f'target_{target_idx}'
                y_target = y[:, target_idx]
                
                y_val_target = y_val[:, target_idx] if y_val is not None else None
                
                self._train_target_model(X_features, y_target, target_name,
                                       X_val_features, y_val_target)
        
        self.is_trained = True
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the LightGBM models.
        
        Args:
            x: Input tensor [batch_size, sequence_length, features]
            
        Returns:
            Predictions based on mode
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Convert to numpy and prepare features
        x_np = x.detach().cpu().numpy()
        X_features = self._prepare_features(x_np)
        
        batch_size = x_np.shape[0]
        
        if self.mode == 'precog':
            # Get predictions from precog models
            point_pred = self.models['point'].predict(X_features)
            interval_min = self.models['interval_min'].predict(X_features)
            interval_max = self.models['interval_max'].predict(X_features)
            
            # Ensure interval_min <= interval_max
            interval_min = np.minimum(interval_min, interval_max - 1e-6)
            interval_max = np.maximum(interval_max, interval_min + 1e-6)
            
            # Convert back to tensors
            point_pred = torch.FloatTensor(point_pred).to(x.device)
            interval_pred = torch.stack([
                torch.FloatTensor(interval_min).to(x.device),
                torch.FloatTensor(interval_max).to(x.device)
            ], dim=1)
            
            return point_pred, interval_pred
        
        elif self.mode == 'synth':
            # Get predictions from all synth models
            predictions = []
            for target_name in sorted(self.models.keys()):
                pred = self.models[target_name].predict(X_features)
                predictions.append(pred)
            
            # Stack predictions and reshape to [batch_size, 12, 3]
            all_preds = np.stack(predictions, axis=1)  # [batch_size, 36]
            detailed_pred = all_preds.reshape(batch_size, 12, 3)  # [batch_size, 12, 3]
            
            # Convert to tensor
            detailed_pred = torch.FloatTensor(detailed_pred).to(x.device)
            
            return (detailed_pred,)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def save_model(self, path: str, additional_info: Optional[Dict[str, Any]] = None):
        """Save the LightGBM models and configuration."""
        save_dict = {
            'model_info': self.get_model_info(),
            'config': self.config,
            'lgb_params': self.lgb_params,
            'is_trained': self.is_trained,
            'mode': self.mode
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        # Save models separately
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM models
        models_info = {}
        for target_name, model in self.models.items():
            model_path = model_dir / f'{target_name}_model.txt'
            model.save_model(str(model_path))
            models_info[target_name] = str(model_path)
        
        save_dict['models_info'] = models_info
        
        # Save configuration
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
    
    @classmethod
    def load_model(cls, path: str, input_size: Optional[int] = None):
        """Load a saved LightGBM model."""
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        config = checkpoint['config']
        model_input_size = input_size if input_size is not None else checkpoint['model_info']['input_size']
        
        # Create model instance
        model = cls(model_input_size, config)
        
        # Load LightGBM models
        models_info = checkpoint['models_info']
        for target_name, model_path in models_info.items():
            lgb_model = lgb.Booster(model_file=model_path)
            model.models[target_name] = lgb_model
        
        model.is_trained = checkpoint['is_trained']
        model.lgb_params = checkpoint['lgb_params']
        
        return model, checkpoint.get('model_info', {})
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = super().get_model_info()
        
        # Add LightGBM specific information
        lgb_info = {
            'lgb_params': self.lgb_params,
            'num_models': len(self.models),
            'model_names': list(self.models.keys()),
            'is_trained': self.is_trained
        }
        
        # Add feature importance if models are trained
        if self.is_trained:
            feature_importance = {}
            for target_name, model in self.models.items():
                try:
                    importance = model.feature_importance(importance_type='gain')
                    feature_importance[target_name] = importance.tolist()
                except:
                    feature_importance[target_name] = []
            
            lgb_info['feature_importance'] = feature_importance
        
        base_info.update(lgb_info)
        return base_info
    
    def print_model_summary(self):
        """Print a summary of the LightGBM model."""
        info = self.get_model_info()
        print(f"\n{'='*60}")
        print(f"Model Summary: {info['model_type']}")
        print(f"{'='*60}")
        print(f"Mode: {info['mode']}")
        print(f"Input Size: {info['input_size']}")
        print(f"Num Models: {info['num_models']}")
        print(f"Model Names: {info['model_names']}")
        print(f"Is Trained: {info['is_trained']}")
        print(f"\nLightGBM Parameters:")
        for key, value in self.lgb_params.items():
            print(f"  {key}: {value}")
        
        if self.is_trained and 'feature_importance' in info:
            print(f"\nFeature Importance (top 10 per model):")
            for target_name, importance in info['feature_importance'].items():
                if len(importance) > 0:
                    top_indices = np.argsort(importance)[-10:][::-1]
                    print(f"  {target_name}:")
                    for i, idx in enumerate(top_indices):
                        print(f"    Feature {idx}: {importance[idx]:.4f}")
        
        print(f"{'='*60}\n")
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance for all models."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance_dict = {}
        for target_name, model in self.models.items():
            try:
                importance = model.feature_importance(importance_type='gain')
                importance_dict[target_name] = importance
            except:
                importance_dict[target_name] = np.array([])
        
        return importance_dict 
"""
Ensemble Module for Bitcoin Price Prediction

This module implements advanced ensemble methods for combining multiple predictors
with adaptive weighting and dynamic model selection capabilities.

Key Features:
- Adaptive weighting ensemble with dynamic weight updates
- Dynamic model selection based on recent performance
- Multiple ensemble strategies (weighted average, stacking, voting)
- Uncertainty quantification and confidence intervals
- Support for both precog and synth modes
- Real-time performance monitoring and adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
import math
from collections import deque
import warnings

from predictors import PredictorFactory
from losses.precog_loss import precog_loss, evaluate_precog
from losses.synth_loss import synth_loss, evaluate_synth


class AdaptiveWeightingEnsemble(nn.Module):
    """Adaptive weighting ensemble with dynamic weight updates"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Ensemble configuration
        self.prediction_mode = config.get('prediction_mode', 'precog')
        self.num_models = config.get('num_models', 3)
        self.ensemble_method = config.get('ensemble_method', 'adaptive_weighting')
        
        # Adaptive weighting parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.adaptation_window = config.get('adaptation_window', 100)
        
        # Dynamic model selection parameters
        self.selection_threshold = config.get('selection_threshold', 0.1)
        self.min_models = config.get('min_models', 2)
        self.max_models = config.get('max_models', 5)
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.adaptation_window)
        self.weight_history = deque(maxlen=self.adaptation_window)
        
        # Initialize ensemble weights
        self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        self.optimizer = torch.optim.Adam([self.weights], 
                                         lr=self.learning_rate, 
                                         weight_decay=self.weight_decay)
        
        # Model registry
        self.models = {}
        self.model_performances = {}
        self.active_models = []
        
    def add_model(self, model_name: str, model: nn.Module, initial_weight: float = 1.0):
        """Add a model to the ensemble"""
        if len(self.models) >= self.max_models:
            warnings.warn(f"Maximum number of models ({self.max_models}) reached")
            return
        
        self.models[model_name] = model
        self.model_performances[model_name] = deque(maxlen=self.adaptation_window)
        
        # Update weights
        if len(self.models) == 1:
            self.weights = nn.Parameter(torch.tensor([initial_weight]))
        else:
            # Add new weight and renormalize
            new_weights = torch.cat([self.weights.data, torch.tensor([initial_weight])])
            new_weights = F.softmax(new_weights, dim=0)
            self.weights = nn.Parameter(new_weights)
            
            # Reinitialize optimizer
            self.optimizer = torch.optim.Adam([self.weights], 
                                             lr=self.learning_rate, 
                                             weight_decay=self.weight_decay)
        
        self.active_models.append(model_name)
        self._update_model_selection()
    
    def remove_model(self, model_name: str):
        """Remove a model from the ensemble"""
        if model_name in self.models:
            del self.models[model_name]
            del self.model_performances[model_name]
            
            if model_name in self.active_models:
                self.active_models.remove(model_name)
            
            # Update weights
            if len(self.models) > 0:
                # Remove weight and renormalize
                model_idx = list(self.models.keys()).index(model_name)
                new_weights = torch.cat([
                    self.weights.data[:model_idx],
                    self.weights.data[model_idx+1:]
                ])
                new_weights = F.softmax(new_weights, dim=0)
                self.weights = nn.Parameter(new_weights)
                
                # Reinitialize optimizer
                self.optimizer = torch.optim.Adam([self.weights], 
                                                 lr=self.learning_rate, 
                                                 weight_decay=self.weight_decay)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            
        Returns:
            Ensemble predictions based on prediction mode
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all active models
        model_predictions = {}
        model_weights = []
        
        for i, model_name in enumerate(self.active_models):
            if model_name in self.models:
                model = self.models[model_name]
                try:
                    predictions = model(x)
                    model_predictions[model_name] = predictions
                    model_weights.append(self.weights[i])
                except Exception as e:
                    print(f"Warning: Model {model_name} failed: {e}")
                    continue
        
        if not model_predictions:
            raise ValueError("No valid model predictions")
        
        # Apply ensemble method
        if self.ensemble_method == 'adaptive_weighting':
            return self._adaptive_weighting_ensemble(model_predictions, model_weights)
        elif self.ensemble_method == 'stacking':
            return self._stacking_ensemble(model_predictions, model_weights)
        elif self.ensemble_method == 'voting':
            return self._voting_ensemble(model_predictions, model_weights)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _adaptive_weighting_ensemble(self, model_predictions: Dict[str, Tuple], 
                                   model_weights: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Adaptive weighting ensemble method"""
        if self.prediction_mode == 'precog':
            # Combine point predictions
            point_predictions = []
            interval_predictions = []
            
            for model_name, predictions in model_predictions.items():
                point_pred, interval_pred = predictions
                point_predictions.append(point_pred)
                interval_predictions.append(interval_pred)
            
            # Weighted average
            weights = torch.stack(model_weights)
            weights = F.softmax(weights, dim=0)
            
            ensemble_point = torch.zeros_like(point_predictions[0])
            ensemble_interval = torch.zeros_like(interval_predictions[0])
            
            for i, (point_pred, interval_pred) in enumerate(zip(point_predictions, interval_predictions)):
                ensemble_point += weights[i] * point_pred
                ensemble_interval += weights[i] * interval_pred
            
            return ensemble_point, ensemble_interval
            
        elif self.prediction_mode == 'synth':
            # Combine detailed predictions
            detailed_predictions = []
            
            for model_name, predictions in model_predictions.items():
                detailed_pred = predictions[0]  # First element is detailed prediction
                detailed_predictions.append(detailed_pred)
            
            # Weighted average
            weights = torch.stack(model_weights)
            weights = F.softmax(weights, dim=0)
            
            ensemble_detailed = torch.zeros_like(detailed_predictions[0])
            
            for i, detailed_pred in enumerate(detailed_predictions):
                ensemble_detailed += weights[i] * detailed_pred
            
            return ensemble_detailed
    
    def _stacking_ensemble(self, model_predictions: Dict[str, Tuple], 
                          model_weights: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Stacking ensemble method with meta-learner"""
        # This would require a meta-learner to be trained
        # For now, fall back to adaptive weighting
        return self._adaptive_weighting_ensemble(model_predictions, model_weights)
    
    def _voting_ensemble(self, model_predictions: Dict[str, Tuple], 
                        model_weights: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Voting ensemble method"""
        if self.prediction_mode == 'precog':
            # Median voting for point predictions
            point_predictions = []
            interval_predictions = []
            
            for model_name, predictions in model_predictions.items():
                point_pred, interval_pred = predictions
                point_predictions.append(point_pred)
                interval_predictions.append(interval_pred)
            
            # Median for point prediction
            point_stack = torch.stack(point_predictions, dim=0)
            ensemble_point = torch.median(point_stack, dim=0)[0]
            
            # Mean for interval prediction
            interval_stack = torch.stack(interval_predictions, dim=0)
            ensemble_interval = torch.mean(interval_stack, dim=0)
            
            return ensemble_point, ensemble_interval
            
        elif self.prediction_mode == 'synth':
            # Median voting for detailed predictions
            detailed_predictions = []
            
            for model_name, predictions in model_predictions.items():
                detailed_pred = predictions[0]
                detailed_predictions.append(detailed_pred)
            
            detailed_stack = torch.stack(detailed_predictions, dim=0)
            ensemble_detailed = torch.median(detailed_stack, dim=0)[0]
            
            return ensemble_detailed
    
    def update_weights(self, targets: torch.Tensor, predictions: Tuple[torch.Tensor, ...]):
        """Update ensemble weights based on recent performance"""
        if self.prediction_mode == 'precog':
            point_pred, interval_pred = predictions
            loss = precog_loss(point_pred, interval_pred, targets)
        elif self.prediction_mode == 'synth':
            detailed_pred = predictions[0]
            loss = synth_loss(detailed_pred, targets)
        
        # Update weights using gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Ensure weights are positive and sum to 1
        with torch.no_grad():
            self.weights.data = F.softmax(self.weights.data, dim=0)
        
        # Track performance
        self.performance_history.append(loss.item())
        self.weight_history.append(self.weights.data.clone())
        
        # Update model selection
        self._update_model_selection()
    
    def _update_model_selection(self):
        """Update active model selection based on performance"""
        if len(self.performance_history) < 10:  # Need some history
            return
        
        # Calculate recent performance for each model
        model_scores = {}
        for model_name in self.models.keys():
            if model_name in self.model_performances and len(self.model_performances[model_name]) > 0:
                recent_performance = np.mean(list(self.model_performances[model_name])[-10:])
                model_scores[model_name] = recent_performance
        
        # Select best performing models
        if model_scores:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
            best_models = [model for model, score in sorted_models[:self.max_models]]
            
            # Update active models
            self.active_models = best_models
            
            # Update weights for active models only
            if len(self.active_models) > 0:
                active_weights = []
                for model_name in self.active_models:
                    if model_name in self.models:
                        model_idx = list(self.models.keys()).index(model_name)
                        active_weights.append(self.weights.data[model_idx])
                
                if active_weights:
                    new_weights = torch.stack(active_weights)
                    new_weights = F.softmax(new_weights, dim=0)
                    
                    # Update weights for active models
                    for i, model_name in enumerate(self.active_models):
                        model_idx = list(self.models.keys()).index(model_name)
                        self.weights.data[model_idx] = new_weights[i]
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble information"""
        return {
            'ensemble_method': self.ensemble_method,
            'prediction_mode': self.prediction_mode,
            'num_models': len(self.models),
            'active_models': self.active_models,
            'current_weights': self.weights.data.tolist(),
            'performance_history': list(self.performance_history),
            'adaptation_window': self.adaptation_window,
            'selection_threshold': self.selection_threshold
        }


class DynamicModelSelection:
    """Dynamic model selection based on performance and diversity"""
    
    def __init__(self, config: Dict[str, Any]):
        self.prediction_mode = config.get('prediction_mode', 'precog')
        self.selection_strategy = config.get('selection_strategy', 'performance_diversity')
        self.max_models = config.get('max_models', 5)
        self.min_models = config.get('min_models', 2)
        self.performance_window = config.get('performance_window', 50)
        self.diversity_threshold = config.get('diversity_threshold', 0.3)
        
        # Model registry
        self.available_models = {}
        self.model_performances = {}
        self.model_predictions = {}
        self.selected_models = []
        
    def add_model(self, model_name: str, model: nn.Module):
        """Add a model to the selection pool"""
        self.available_models[model_name] = model
        self.model_performances[model_name] = deque(maxlen=self.performance_window)
        self.model_predictions[model_name] = deque(maxlen=self.performance_window)
    
    def update_performance(self, model_name: str, performance: float, predictions: torch.Tensor):
        """Update model performance"""
        if model_name in self.model_performances:
            self.model_performances[model_name].append(performance)
            self.model_predictions[model_name].append(predictions.detach().cpu().numpy())
    
    def select_models(self) -> List[str]:
        """Select models based on strategy"""
        if self.selection_strategy == 'performance_diversity':
            return self._performance_diversity_selection()
        elif self.selection_strategy == 'ensemble_diversity':
            return self._ensemble_diversity_selection()
        elif self.selection_strategy == 'adaptive':
            return self._adaptive_selection()
        else:
            return list(self.available_models.keys())[:self.max_models]
    
    def _performance_diversity_selection(self) -> List[str]:
        """Select models based on performance and diversity"""
        if len(self.available_models) <= self.min_models:
            return list(self.available_models.keys())
        
        # Calculate average performance for each model
        model_scores = {}
        for model_name in self.available_models.keys():
            if len(self.model_performances[model_name]) > 0:
                avg_performance = np.mean(list(self.model_performances[model_name]))
                model_scores[model_name] = avg_performance
        
        # Sort by performance
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
        
        # Select top performing models
        selected = [model for model, score in sorted_models[:self.max_models]]
        
        # Add diverse models if needed
        if len(selected) < self.max_models:
            remaining = [model for model in self.available_models.keys() if model not in selected]
            selected.extend(remaining[:self.max_models - len(selected)])
        
        return selected
    
    def _ensemble_diversity_selection(self) -> List[str]:
        """Select models based on ensemble diversity"""
        if len(self.available_models) <= self.min_models:
            return list(self.available_models.keys())
        
        # Calculate prediction diversity
        model_diversities = {}
        for model_name in self.available_models.keys():
            if len(self.model_predictions[model_name]) > 0:
                # Calculate prediction variance as diversity measure
                predictions = np.array(list(self.model_predictions[model_name]))
                diversity = np.var(predictions)
                model_diversities[model_name] = diversity
        
        # Sort by diversity
        sorted_models = sorted(model_diversities.items(), key=lambda x: x[1], reverse=True)
        
        # Select diverse models
        selected = [model for model, diversity in sorted_models[:self.max_models]]
        
        return selected
    
    def _adaptive_selection(self) -> List[str]:
        """Adaptive selection based on recent performance trends"""
        if len(self.available_models) <= self.min_models:
            return list(self.available_models.keys())
        
        # Calculate performance trends
        model_trends = {}
        for model_name in self.available_models.keys():
            if len(self.model_performances[model_name]) >= 10:
                recent_performance = list(self.model_performances[model_name])[-10:]
                trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
                model_trends[model_name] = trend
        
        # Select models with improving trends
        improving_models = [model for model, trend in model_trends.items() if trend < 0]
        
        if len(improving_models) >= self.min_models:
            return improving_models[:self.max_models]
        else:
            # Fall back to performance-based selection
            return self._performance_diversity_selection()


class EnsembleTrainer:
    """Trainer for ensemble models with cross-validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prediction_mode = config.get('prediction_mode', 'precog')
        self.ensemble = AdaptiveWeightingEnsemble(config)
        self.model_selector = DynamicModelSelection(config)
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 256)
        self.epochs = config.get('epochs', 100)
        self.validation_split = config.get('validation_split', 0.2)
        
    def add_model(self, model_name: str, model: nn.Module, initial_weight: float = 1.0):
        """Add a model to the ensemble"""
        self.ensemble.add_model(model_name, model, initial_weight)
        self.model_selector.add_model(model_name, model)
    
    def train_ensemble(self, train_loader, val_loader):
        """Train the ensemble"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble.to(device)
        
        # Training loop
        for epoch in range(self.epochs):
            self.ensemble.train()
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                predictions = self.ensemble(x)
                
                # Update weights
                self.ensemble.update_weights(y, predictions)
                
                # Update model selection
                if batch_idx % 10 == 0:
                    self.model_selector.select_models()
            
            # Validation
            if epoch % 5 == 0:
                val_loss = self._validate(val_loader, device)
                print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
    
    def _validate(self, val_loader, device):
        """Validate ensemble performance"""
        self.ensemble.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                predictions = self.ensemble(x)
                
                if self.prediction_mode == 'precog':
                    loss = precog_loss(predictions[0], predictions[1], y)
                else:
                    loss = synth_loss(predictions[0], y)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Make ensemble predictions"""
        self.ensemble.eval()
        with torch.no_grad():
            return self.ensemble(x)
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble information"""
        ensemble_info = self.ensemble.get_ensemble_info()
        ensemble_info.update({
            'model_selector': self.model_selector.selection_strategy,
            'selected_models': self.model_selector.selected_models,
            'available_models': list(self.model_selector.available_models.keys())
        })
        return ensemble_info


def create_ensemble_from_config(config: Dict[str, Any]) -> EnsembleTrainer:
    """Create ensemble from configuration"""
    trainer = EnsembleTrainer(config)
    
    # Add models based on configuration
    model_configs = config.get('models', [])
    
    for model_config in model_configs:
        model_type = model_config['type']
        model_params = model_config.get('params', {})
        
        # Create model using factory
        model = PredictorFactory.create_predictor(
            model_type, 
            input_size=config.get('input_size', 20),
            config=model_params
        )
        
        trainer.add_model(
            model_config['name'],
            model,
            initial_weight=model_config.get('initial_weight', 1.0)
        )
    
    return trainer 
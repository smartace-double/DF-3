"""
GARCH Predictor for Volatility Prediction

This module implements GARCH-based predictors for Bitcoin price volatility prediction.
GARCH models are specifically designed for volatility forecasting and are well-suited
for financial time series with time-varying volatility.

Key Features:
- GARCH(1,1) model for basic volatility prediction
- GARCH(p,q) model with configurable orders
- EGARCH model for asymmetric volatility effects
- Volatility clustering and mean reversion
- Support for precog mode (volatility intervals)
- Integration with precog loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import math
from scipy import stats
from scipy.optimize import minimize

from predictors.base_predictor import BaseBitcoinPredictor
from losses.precog_loss import precog_loss, evaluate_precog


class GARCHModel:
    """Base GARCH model implementation"""
    
    def __init__(self, p: int = 1, q: int = 1, model_type: str = 'garch'):
        """
        Initialize GARCH model
        
        Args:
            p: Order of GARCH terms
            q: Order of ARCH terms
            model_type: Type of GARCH model ('garch', 'egarch')
        """
        self.p = p
        self.q = q
        self.model_type = model_type
        self.fitted = False
        self.params = None
        
    def fit(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Fit GARCH model to return series
        
        Args:
            returns: Return series (log returns)
            
        Returns:
            Dictionary with fitted parameters and model info
        """
        if self.model_type == 'garch':
            return self._fit_garch(returns)
        elif self.model_type == 'egarch':
            return self._fit_egarch(returns)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _fit_garch(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fit standard GARCH(p,q) model"""
        n = len(returns)
        
        # Initial parameter estimates
        omega = np.var(returns) * 0.1
        alpha = np.ones(self.q) * 0.1 / self.q
        beta = np.ones(self.p) * 0.8 / self.p
        
        # Parameter bounds
        bounds = [(1e-8, None)] + [(0, 1)] * self.q + [(0, 1)] * self.p
        
        # Constraint: sum of alpha + beta < 1
        def constraint(params):
            return 1 - np.sum(params[1:1+self.q]) - np.sum(params[1+self.q:])
        
        # Objective function (negative log-likelihood)
        def objective(params):
            return self._garch_loglikelihood(params, returns)
        
        # Optimization
        initial_params = np.concatenate([[omega], alpha, beta])
        result = minimize(objective, initial_params, bounds=bounds, 
                        constraints={'type': 'ineq', 'fun': constraint})
        
        if not result.success:
            raise ValueError("GARCH optimization failed")
        
        self.params = result.x
        self.fitted = True
        
        return {
            'omega': result.x[0],
            'alpha': result.x[1:1+self.q],
            'beta': result.x[1+self.q:],
            'loglikelihood': -result.fun,
            'aic': 2 * result.fun + 2 * (1 + self.p + self.q),
            'bic': 2 * result.fun + np.log(n) * (1 + self.p + self.q)
        }
    
    def _fit_egarch(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fit EGARCH(p,q) model"""
        n = len(returns)
        
        # Initial parameter estimates
        omega = 0.0
        alpha = np.ones(self.q) * 0.1 / self.q
        gamma = np.ones(self.q) * 0.05 / self.q
        beta = np.ones(self.p) * 0.9 / self.p
        
        # Parameter bounds
        bounds = [(None, None)] + [(None, None)] * self.q + [(None, None)] * self.q + [(None, 1)] * self.p
        
        # Objective function
        def objective(params):
            return self._egarch_loglikelihood(params, returns)
        
        # Optimization
        initial_params = np.concatenate([[omega], alpha, gamma, beta])
        result = minimize(objective, initial_params, bounds=bounds)
        
        if not result.success:
            raise ValueError("EGARCH optimization failed")
        
        self.params = result.x
        self.fitted = True
        
        return {
            'omega': result.x[0],
            'alpha': result.x[1:1+self.q],
            'gamma': result.x[1+self.q:1+2*self.q],
            'beta': result.x[1+2*self.q:],
            'loglikelihood': -result.fun,
            'aic': 2 * result.fun + 2 * (1 + 2*self.q + self.p),
            'bic': 2 * result.fun + np.log(n) * (1 + 2*self.q + self.p)
        }
    
    def _garch_loglikelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Compute GARCH log-likelihood"""
        omega = params[0]
        alpha = params[1:1+self.q]
        beta = params[1+self.q:]
        
        n = len(returns)
        h = np.zeros(n)  # Conditional variance
        
        # Initialize with unconditional variance
        h[0] = omega / (1 - np.sum(alpha) - np.sum(beta))
        
        # Compute conditional variance
        for t in range(1, n):
            arch_term = np.sum(alpha * returns[t-self.q:t]**2)
            garch_term = np.sum(beta * h[t-self.p:t])
            h[t] = omega + arch_term + garch_term
        
        # Log-likelihood
        loglik = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + returns**2 / h)
        return -loglik
    
    def _egarch_loglikelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Compute EGARCH log-likelihood"""
        omega = params[0]
        alpha = params[1:1+self.q]
        gamma = params[1+self.q:1+2*self.q]
        beta = params[1+2*self.q:]
        
        n = len(returns)
        h = np.zeros(n)  # Conditional variance
        log_h = np.zeros(n)  # Log conditional variance
        
        # Initialize
        log_h[0] = omega / (1 - np.sum(beta))
        
        # Compute conditional variance
        for t in range(1, n):
            arch_term = np.sum(alpha * returns[t-self.q:t] / np.sqrt(h[t-self.q:t]))
            leverage_term = np.sum(gamma * (np.abs(returns[t-self.q:t]) / np.sqrt(h[t-self.q:t]) - np.sqrt(2/np.pi)))
            garch_term = np.sum(beta * log_h[t-self.p:t])
            log_h[t] = omega + arch_term + leverage_term + garch_term
            h[t] = np.exp(log_h[t])
        
        # Log-likelihood
        loglik = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + returns**2 / h)
        return -loglik
    
    def forecast(self, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast volatility
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Tuple of (point_forecast, confidence_intervals)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if self.model_type == 'garch':
            return self._garch_forecast(horizon)
        elif self.model_type == 'egarch':
            return self._egarch_forecast(horizon)
    
    def _garch_forecast(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """GARCH volatility forecast"""
        omega = self.params[0]
        alpha = self.params[1:1+self.q]
        beta = self.params[1+self.q:]
        
        # Unconditional variance
        unconditional_var = omega / (1 - np.sum(alpha) - np.sum(beta))
        
        # Multi-step forecast
        forecast_var = np.zeros(horizon)
        forecast_var[0] = unconditional_var
        
        for h in range(1, horizon):
            forecast_var[h] = omega + np.sum(alpha) * unconditional_var + np.sum(beta) * forecast_var[h-1]
        
        # Convert to volatility (standard deviation)
        forecast_vol = np.sqrt(forecast_var)
        
        # Confidence intervals (assuming normal distribution)
        confidence_intervals = np.column_stack([
            forecast_vol * 0.5,  # Lower bound
            forecast_vol * 2.0   # Upper bound
        ])
        
        return forecast_vol, confidence_intervals
    
    def _egarch_forecast(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """EGARCH volatility forecast"""
        omega = self.params[0]
        alpha = self.params[1:1+self.q]
        gamma = self.params[1+self.q:1+2*self.q]
        beta = self.params[1+2*self.q:]
        
        # Unconditional log-variance
        unconditional_logvar = omega / (1 - np.sum(beta))
        
        # Multi-step forecast
        forecast_logvar = np.zeros(horizon)
        forecast_logvar[0] = unconditional_logvar
        
        for h in range(1, horizon):
            forecast_logvar[h] = omega + np.sum(beta) * forecast_logvar[h-1]
        
        # Convert to volatility
        forecast_vol = np.sqrt(np.exp(forecast_logvar))
        
        # Confidence intervals
        confidence_intervals = np.column_stack([
            forecast_vol * 0.5,
            forecast_vol * 2.0
        ])
        
        return forecast_vol, confidence_intervals


class GARCHPredictor(BaseBitcoinPredictor):
    """GARCH predictor for volatility prediction (precog mode only)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # GARCH model parameters
        self.p = config.get('p', 1)
        self.q = config.get('q', 1)
        self.model_type = config.get('model_type', 'garch')  # 'garch' or 'egarch'
        
        # Prediction parameters
        self.prediction_mode = config.get('prediction_mode', 'precog')
        if self.prediction_mode != 'precog':
            raise ValueError("GARCH predictor only supports precog mode")
        
        # Volatility prediction parameters
        self.volatility_window = config.get('volatility_window', 20)
        self.forecast_horizon = config.get('forecast_horizon', 12)
        self.confidence_level = config.get('confidence_level', 0.95)
        
        # Model storage
        self.garch_models = {}
        self.fitted = False
        
    def fit(self, data: np.ndarray) -> None:
        """
        Fit GARCH models to the data
        
        Args:
            data: Price data [batch_size, seq_len, features]
        """
        if len(data.shape) != 3:
            raise ValueError("Data must be 3D: [batch_size, seq_len, features]")
        
        batch_size, seq_len, features = data.shape
        
        # Extract price data (assuming first feature is price)
        prices = data[:, :, 0]  # [batch_size, seq_len]
        
        # Compute returns
        returns = np.diff(np.log(prices), axis=1)  # [batch_size, seq_len-1]
        
        # Fit GARCH model for each batch
        for i in range(batch_size):
            model = GARCHModel(p=self.p, q=self.q, model_type=self.model_type)
            try:
                model.fit(returns[i])
                self.garch_models[i] = model
            except Exception as e:
                print(f"Warning: Failed to fit GARCH model for batch {i}: {e}")
                # Use simple volatility model as fallback
                self.garch_models[i] = self._create_fallback_model(returns[i])
        
        self.fitted = True
    
    def _create_fallback_model(self, returns: np.ndarray) -> GARCHModel:
        """Create a simple volatility model as fallback"""
        class SimpleVolModel:
            def __init__(self):
                self.fitted = True
                self.params = None
            
            def forecast(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
                # Simple volatility forecast using rolling window
                vol = np.std(returns[-self.volatility_window:])
                forecast_vol = np.full(horizon, vol)
                confidence_intervals = np.column_stack([
                    forecast_vol * 0.5,
                    forecast_vol * 2.0
                ])
                return forecast_vol, confidence_intervals
        
        return SimpleVolModel()
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict volatility using fitted GARCH models
        
        Args:
            data: Price data [batch_size, seq_len, features]
            
        Returns:
            Tuple of (point_predictions, interval_predictions)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        batch_size, seq_len, features = data.shape
        
        # Extract price data
        prices = data[:, :, 0]  # [batch_size, seq_len]
        
        # Compute returns for volatility estimation
        returns = np.diff(np.log(prices), axis=1)  # [batch_size, seq_len-1]
        
        point_predictions = []
        interval_predictions = []
        
        for i in range(batch_size):
            if i in self.garch_models:
                model = self.garch_models[i]
                try:
                    # Get current price for point prediction
                    current_price = prices[i, -1]
                    
                    # Forecast volatility
                    vol_forecast, vol_intervals = model.forecast(self.forecast_horizon)
                    
                    # Convert volatility to price predictions
                    # Point prediction: current_price * (1 + mean_return)
                    mean_return = np.mean(returns[i, -self.volatility_window:])
                    point_pred = current_price * (1 + mean_return)
                    
                    # Interval prediction based on volatility
                    vol_std = np.std(vol_forecast)
                    interval_min = current_price * (1 - 2 * vol_std)
                    interval_max = current_price * (1 + 2 * vol_std)
                    
                    point_predictions.append(point_pred)
                    interval_predictions.append([interval_min, interval_max])
                    
                except Exception as e:
                    print(f"Warning: Failed to predict for batch {i}: {e}")
                    # Fallback predictions
                    current_price = prices[i, -1]
                    point_predictions.append(current_price)
                    interval_predictions.append([current_price * 0.9, current_price * 1.1])
            else:
                # Fallback for unfitted models
                current_price = prices[i, -1]
                point_predictions.append(current_price)
                interval_predictions.append([current_price * 0.9, current_price * 1.1])
        
        return np.array(point_predictions), np.array(interval_predictions)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for GARCH predictor
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            
        Returns:
            Tuple of (point_predictions, interval_predictions)
        """
        # Convert to numpy for GARCH processing
        data_np = x.detach().cpu().numpy()
        
        # Fit model if not already fitted
        if not self.fitted:
            self.fit(data_np)
        
        # Make predictions
        point_pred, interval_pred = self.predict(data_np)
        
        # Convert back to tensors
        point_tensor = torch.FloatTensor(point_pred).to(x.device)
        interval_tensor = torch.FloatTensor(interval_pred).to(x.device)
        
        return point_tensor, interval_tensor
    
    def compute_loss(self, predictions: Tuple[torch.Tensor, ...], targets: torch.Tensor) -> torch.Tensor:
        """Compute loss using precog loss function"""
        point_pred, interval_pred = predictions
        return precog_loss(point_pred, interval_pred, targets)
    
    def evaluate(self, predictions: Tuple[torch.Tensor, ...], targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        point_pred, interval_pred = predictions
        return evaluate_precog(point_pred, interval_pred, targets)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': 'GARCH',
            'prediction_mode': self.prediction_mode,
            'model_type': self.model_type,
            'p': self.p,
            'q': self.q,
            'volatility_window': self.volatility_window,
            'forecast_horizon': self.forecast_horizon,
            'confidence_level': self.confidence_level,
            'fitted_models': len(self.garch_models),
            'architecture': {
                'garch_order': f"GARCH({self.p},{self.q})",
                'model_type': self.model_type,
                'volatility_estimation': 'rolling_window'
            }
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for GARCH model"""
        if not self.fitted or len(self.garch_models) == 0:
            return {'feature_importance': 'Not available - model not fitted'}
        
        # Analyze GARCH parameters for feature importance
        alphas = []
        betas = []
        
        for model in self.garch_models.values():
            if hasattr(model, 'params') and model.params is not None:
                if self.model_type == 'garch':
                    alphas.append(model.params[1:1+self.q])
                    betas.append(model.params[1+self.q:])
                elif self.model_type == 'egarch':
                    alphas.append(model.params[1:1+self.q])
                    betas.append(model.params[1+2*self.q:])
        
        if alphas and betas:
            avg_alpha = np.mean(alphas, axis=0)
            avg_beta = np.mean(betas, axis=0)
            
            return {
                'arch_importance': avg_alpha.tolist(),
                'garch_importance': avg_beta.tolist(),
                'persistence': np.sum(avg_alpha) + np.sum(avg_beta)
            }
        else:
            return {'feature_importance': 'Not available - no valid models'}


class EGARCHPredictor(GARCHPredictor):
    """EGARCH predictor with asymmetric volatility effects"""
    
    def __init__(self, config: Dict[str, Any]):
        # Override model type to EGARCH
        config['model_type'] = 'egarch'
        super().__init__(config)
        
        # EGARCH-specific parameters
        self.leverage_effect = config.get('leverage_effect', True)
        self.asymmetric_vol = config.get('asymmetric_vol', True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get EGARCH-specific model information"""
        base_info = super().get_model_info()
        base_info.update({
            'model_name': 'EGARCH',
            'leverage_effect': self.leverage_effect,
            'asymmetric_volatility': self.asymmetric_vol,
            'architecture': {
                'garch_order': f"EGARCH({self.p},{self.q})",
                'model_type': 'egarch',
                'leverage_terms': self.q,
                'asymmetric_effects': True
            }
        })
        return base_info


class GARCHEnsemble(GARCHPredictor):
    """Ensemble of multiple GARCH models for robust volatility prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Ensemble parameters
        self.ensemble_models = config.get('ensemble_models', ['garch', 'egarch'])
        self.ensemble_weights = config.get('ensemble_weights', None)
        
        # Initialize multiple GARCH models
        self.models = {}
        for model_type in self.ensemble_models:
            model_config = config.copy()
            model_config['model_type'] = model_type
            self.models[model_type] = GARCHPredictor(model_config)
    
    def fit(self, data: np.ndarray) -> None:
        """Fit all ensemble models"""
        for model_type, model in self.models.items():
            try:
                model.fit(data)
            except Exception as e:
                print(f"Warning: Failed to fit {model_type} model: {e}")
        
        self.fitted = True
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble prediction"""
        predictions = []
        
        for model_type, model in self.models.items():
            if model.fitted:
                try:
                    point_pred, interval_pred = model.predict(data)
                    predictions.append((point_pred, interval_pred))
                except Exception as e:
                    print(f"Warning: Failed to predict with {model_type} model: {e}")
        
        if not predictions:
            raise ValueError("No valid ensemble predictions")
        
        # Combine predictions (simple average for now)
        point_preds = np.array([pred[0] for pred in predictions])
        interval_preds = np.array([pred[1] for pred in predictions])
        
        # Weighted average if weights provided
        if self.ensemble_weights is not None and len(self.ensemble_weights) == len(predictions):
            weights = np.array(self.ensemble_weights)
            weights = weights / np.sum(weights)
            
            ensemble_point = np.average(point_preds, axis=0, weights=weights)
            ensemble_interval = np.average(interval_preds, axis=0, weights=weights)
        else:
            ensemble_point = np.mean(point_preds, axis=0)
            ensemble_interval = np.mean(interval_preds, axis=0)
        
        return ensemble_point, ensemble_interval
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble model information"""
        base_info = super().get_model_info()
        base_info.update({
            'model_name': 'GARCH_Ensemble',
            'ensemble_models': self.ensemble_models,
            'ensemble_weights': self.ensemble_weights,
            'num_models': len(self.models),
            'fitted_models': sum(1 for m in self.models.values() if m.fitted)
        })
        return base_info 
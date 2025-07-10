"""
Temporal Fusion Transformer (TFT) Predictor

This module implements a TFT predictor for Bitcoin price prediction supporting both precog and synth modes.
The TFT architecture combines LSTM encoders with multi-head attention mechanisms for temporal modeling.

Key Features:
- Variable Selection Networks (VSNs) for feature importance
- Gating mechanisms for adaptive feature selection
- Multi-head attention for temporal dependencies
- Quantile prediction for uncertainty estimation
- Support for both precog (point + interval) and synth (detailed) predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import math

from predictors.base_predictor import BaseBitcoinPredictor
from losses.precog_loss import precog_loss, evaluate_precog
from losses.synth_loss import synth_loss, evaluate_synth


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for feature selection and gating mechanisms"""
    
    def __init__(self, input_size: int, hidden_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        
        self.linear = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated linear unit: GLU(x) = σ(Wx + b) ⊙ (Vx + c)"""
        x = self.linear(x)
        x = self.dropout(x)
        gate, candidate = x.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        return gate * candidate


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for adaptive feature selection"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Flattened grn for variable selection
        self.flattened_grn = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Variable selection weights
        self.weight_network = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_size] or [batch_size, input_size]
            context: Optional context tensor for conditioning
            
        Returns:
            selected_features: Weighted input features
            weights: Variable selection weights
        """
        # Ensure input is 2D for processing
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, input_size = original_shape
            x_flat = x.view(batch_size * seq_len, input_size)
        else:
            x_flat = x
            
        # Apply GRN to get selection context
        grn_output = self.flattened_grn(x_flat, context)
        
        # Generate variable selection weights
        weights = self.weight_network(grn_output)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        # Apply variable selection
        selected_features = x_flat * weights
        
        # Reshape back to original shape
        if len(original_shape) == 3:
            selected_features = selected_features.view(batch_size, seq_len, input_size)
            weights = weights.view(batch_size, seq_len, input_size)
        
        return selected_features, weights


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network with skip connections"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1, 
                 use_layer_norm: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layer_norm = use_layer_norm
        
        # Primary processing layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanism
        self.gate = GatedLinearUnit(hidden_size, hidden_size, dropout)
        
        # Skip connection projection if dimensions don't match
        if input_size != hidden_size:
            self.skip_connection = nn.Linear(input_size, hidden_size)
        else:
            self.skip_connection = None
            
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            context: Optional context tensor (unused in this implementation)
            
        Returns:
            Output tensor after gated residual processing
        """
        # Store original input for skip connection
        residual = x
        
        # Primary processing
        x = self.linear1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        x = F.elu(x)
        
        # Apply gating
        x = self.gate(x)
        
        # Skip connection with dimension matching
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
            
        # Add residual connection
        x = x + residual
        
        # Layer normalization
        if self.use_layer_norm:
            x = self.layer_norm(x)
            
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for temporal modeling"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = query.shape
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scaled dot-product attention mechanism"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output


class TemporalFusionTransformer(nn.Module):
    """Core TFT architecture with encoder-decoder structure"""
    
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 8,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Variable selection for temporal features
        self.temporal_vsn = VariableSelectionNetwork(input_size, hidden_size, dropout)
        
        # LSTM encoder-decoder
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Gated residual networks
        self.grn_layers = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Encoded features [batch_size, seq_len, hidden_size]
        """
        # Variable selection for temporal features
        x, _ = self.temporal_vsn(x)
        
        # LSTM encoding
        encoder_output, (hidden, cell) = self.encoder(x)
        
        # Initialize decoder with encoder final state
        decoder_input = encoder_output[:, -1:, :]  # Use last encoder output
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Expand decoder output to match sequence length
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        decoder_output = decoder_output.repeat(1, seq_len, 1)
        
        # Combine encoder and decoder outputs
        combined = encoder_output + decoder_output
        
        # Apply attention and GRN layers
        for attention, grn, layer_norm in zip(self.attention_layers, self.grn_layers, self.layer_norms):
            # Multi-head attention
            attention_output = attention(combined, combined, combined)
            combined = layer_norm(combined + self.dropout(attention_output))
            
            # Gated residual network
            grn_output = grn(combined)
            combined = layer_norm(combined + self.dropout(grn_output))
        
        return combined


class TFTPredictor(BaseBitcoinPredictor):
    """TFT predictor supporting both precog and synth modes"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model architecture parameters
        self.input_size = config.get('input_size', 20)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        self.use_layer_norm = config.get('use_layer_norm', True)
        
        # Prediction parameters
        self.prediction_mode = config.get('prediction_mode', 'precog')
        self.quantiles = config.get('quantiles', [0.1, 0.5, 0.9])
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build the TFT model architecture"""
        # Core TFT transformer
        self.tft_core = TemporalFusionTransformer(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Prediction heads based on mode
        if self.prediction_mode == 'precog':
            # Point prediction head
            self.point_head = nn.Sequential(
                GatedResidualNetwork(self.hidden_size, self.hidden_size, self.dropout),
                nn.Linear(self.hidden_size, 1)
            )
            
            # Interval prediction head (min, max)
            self.interval_head = nn.Sequential(
                GatedResidualNetwork(self.hidden_size, self.hidden_size, self.dropout),
                nn.Linear(self.hidden_size, 2)
            )
            
        elif self.prediction_mode == 'synth':
            # Detailed prediction head for each time step
            self.detailed_head = nn.Sequential(
                GatedResidualNetwork(self.hidden_size, self.hidden_size, self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size // 2, 36)  # 12 time steps × 3 values (close, low, high)
            )
        
        # Quantile prediction heads for uncertainty estimation
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                GatedResidualNetwork(self.hidden_size, self.hidden_size, self.dropout),
                nn.Linear(self.hidden_size, 1)
            )
            for _ in self.quantiles
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through TFT model
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Model predictions based on prediction mode
        """
        # Core TFT processing
        tft_output = self.tft_core(x)
        
        # Use final time step for prediction
        final_features = tft_output[:, -1, :]
        
        if self.prediction_mode == 'precog':
            # Point and interval predictions
            point_pred = self.point_head(final_features).squeeze(-1)
            interval_pred = self.interval_head(final_features)
            
            # Ensure interval ordering (min < max)
            interval_pred = torch.sort(interval_pred, dim=-1)[0]
            
            # Quantile predictions for uncertainty
            quantile_preds = [head(final_features).squeeze(-1) for head in self.quantile_heads]
            
            return point_pred, interval_pred, quantile_preds
            
        elif self.prediction_mode == 'synth':
            # Detailed predictions for each time step
            detailed_pred = self.detailed_head(final_features)
            detailed_pred = detailed_pred.view(-1, 12, 3)  # [batch_size, 12, 3]
            
            # Quantile predictions for uncertainty
            quantile_preds = [head(final_features).squeeze(-1) for head in self.quantile_heads]
            
            return detailed_pred, quantile_preds
    
    def compute_loss(self, predictions: Tuple[torch.Tensor, ...], targets: torch.Tensor) -> torch.Tensor:
        """Compute loss based on prediction mode"""
        if self.prediction_mode == 'precog':
            point_pred, interval_pred, quantile_preds = predictions
            return precog_loss(point_pred, interval_pred, targets)
        elif self.prediction_mode == 'synth':
            detailed_pred, quantile_preds = predictions
            return synth_loss(detailed_pred, targets)
    
    def evaluate(self, predictions: Tuple[torch.Tensor, ...], targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.prediction_mode == 'precog':
            point_pred, interval_pred, quantile_preds = predictions
            return evaluate_precog(point_pred, interval_pred, targets)
        elif self.prediction_mode == 'synth':
            detailed_pred, quantile_preds = predictions
            return evaluate_synth(detailed_pred, targets)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'TFT',
            'prediction_mode': self.prediction_mode,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'quantiles': self.quantiles,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'tft_core': str(self.tft_core),
                'prediction_heads': len(self.quantile_heads) + (2 if self.prediction_mode == 'precog' else 1)
            }
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from variable selection networks"""
        # This would require storing attention weights during forward pass
        # For now, return placeholder
        return {
            'temporal_features': 1.0,
            'attention_weights': 'Available during inference',
            'variable_selection': 'Available during inference'
        }


class TFTWithUncertainty(TFTPredictor):
    """TFT predictor with enhanced uncertainty quantification"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Enhanced uncertainty parameters
        self.uncertainty_method = config.get('uncertainty_method', 'quantile')
        self.monte_carlo_samples = config.get('monte_carlo_samples', 10)
        
        # Additional uncertainty head
        if self.uncertainty_method == 'gaussian':
            self.uncertainty_head = nn.Sequential(
                GatedResidualNetwork(self.hidden_size, self.hidden_size, self.dropout),
                nn.Linear(self.hidden_size, 1),
                nn.Softplus()  # Ensure positive variance
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass with uncertainty quantification"""
        if self.uncertainty_method == 'monte_carlo' and self.training:
            # Monte Carlo dropout for uncertainty estimation
            outputs = []
            for _ in range(self.monte_carlo_samples):
                output = super().forward(x)
                outputs.append(output)
            
            # Return mean and variance
            if self.prediction_mode == 'precog':
                point_preds = torch.stack([out[0] for out in outputs], dim=0)
                interval_preds = torch.stack([out[1] for out in outputs], dim=0)
                
                point_mean = point_preds.mean(dim=0)
                point_var = point_preds.var(dim=0)
                interval_mean = interval_preds.mean(dim=0)
                interval_var = interval_preds.var(dim=0)
                
                return point_mean, interval_mean, point_var, interval_var
            else:
                detailed_preds = torch.stack([out[0] for out in outputs], dim=0)
                detailed_mean = detailed_preds.mean(dim=0)
                detailed_var = detailed_preds.var(dim=0)
                
                return detailed_mean, detailed_var
        else:
            # Standard forward pass
            return super().forward(x) 
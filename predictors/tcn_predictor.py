"""
Temporal Convolutional Network (TCN) Predictor

This module implements a TCN predictor for Bitcoin price prediction supporting both precog and synth modes.
The TCN architecture uses dilated convolutions to capture long-range temporal dependencies efficiently.

Key Features:
- Dilated convolutions for efficient long-range dependency modeling
- Residual connections for stable training
- Temporal block architecture with skip connections
- Causal convolutions to prevent information leakage
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


class Chomp1d(nn.Module):
    """Chomp1d removes the extra padding added by causal convolution"""
    
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal Block: The basic building block of TCN
    
    Contains:
    - Dilated causal convolution
    - Weight normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, 
                 dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        
        # First dilated convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Weight normalization
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.conv2 = nn.utils.weight_norm(self.conv2)
        if self.downsample is not None:
            self.downsample = nn.utils.weight_norm(self.downsample)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using normal distribution"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal block
        
        Args:
            x: Input tensor [batch_size, n_inputs, seq_len]
            
        Returns:
            Output tensor [batch_size, n_outputs, seq_len]
        """
        # First convolution path
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second convolution path
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network
    
    Stacks multiple temporal blocks with increasing dilation factors
    to capture dependencies at different time scales.
    """
    
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 2, 
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding for causal convolution
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN
        
        Args:
            x: Input tensor [batch_size, num_inputs, seq_len]
            
        Returns:
            Output tensor [batch_size, num_channels[-1], seq_len]
        """
        return self.network(x)


class AttentionPool(nn.Module):
    """Attention-based pooling for temporal features"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Pooled tensor [batch_size, input_size]
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention pooling
        pooled = torch.sum(x * attention_weights, dim=1)  # [batch_size, input_size]
        
        return pooled


class TCNPredictor(BaseBitcoinPredictor):
    """TCN predictor supporting both precog and synth modes"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model architecture parameters
        self.input_size = config.get('input_size', 20)
        self.num_channels = config.get('num_channels', [64, 128, 256])
        self.kernel_size = config.get('kernel_size', 3)
        self.dropout = config.get('dropout', 0.2)
        
        # Prediction parameters
        self.prediction_mode = config.get('prediction_mode', 'precog')
        self.pooling_method = config.get('pooling_method', 'attention')  # 'last', 'mean', 'max', 'attention'
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build the TCN model architecture"""
        # Core TCN network
        self.tcn = TemporalConvNet(
            num_inputs=self.input_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )
        
        # Feature dimension after TCN
        tcn_output_size = self.num_channels[-1]
        
        # Pooling layer
        if self.pooling_method == 'attention':
            self.pooling = AttentionPool(tcn_output_size, tcn_output_size // 2)
        elif self.pooling_method == 'adaptive':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            self.pooling = None
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(tcn_output_size, tcn_output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(tcn_output_size, tcn_output_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        feature_size = tcn_output_size // 2
        
        # Prediction heads based on mode
        if self.prediction_mode == 'precog':
            # Point prediction head
            self.point_head = nn.Sequential(
                nn.Linear(feature_size, feature_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(feature_size // 2, 1)
            )
            
            # Interval prediction head (min, max)
            self.interval_head = nn.Sequential(
                nn.Linear(feature_size, feature_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(feature_size // 2, 2)
            )
            
        elif self.prediction_mode == 'synth':
            # Detailed prediction head for each time step
            self.detailed_head = nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(feature_size, feature_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(feature_size // 2, 36)  # 12 time steps × 3 values (close, low, high)
            )
        
        # Uncertainty quantification heads
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(feature_size // 2, 1),
            nn.Softplus()  # Ensure positive values
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through TCN model
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Model predictions based on prediction mode
        """
        # TCN expects input in format [batch_size, input_size, seq_len]
        x = x.transpose(1, 2)  # [batch_size, input_size, seq_len]
        
        # Forward through TCN
        tcn_output = self.tcn(x)  # [batch_size, num_channels[-1], seq_len]
        
        # Apply pooling
        if self.pooling_method == 'last':
            pooled_features = tcn_output[:, :, -1]  # Use last time step
        elif self.pooling_method == 'mean':
            pooled_features = torch.mean(tcn_output, dim=2)
        elif self.pooling_method == 'max':
            pooled_features = torch.max(tcn_output, dim=2)[0]
        elif self.pooling_method == 'attention':
            # Transpose for attention pooling
            tcn_output = tcn_output.transpose(1, 2)  # [batch_size, seq_len, num_channels[-1]]
            pooled_features = self.pooling(tcn_output)
        elif self.pooling_method == 'adaptive':
            pooled_features = self.pooling(tcn_output).squeeze(-1)
        else:
            pooled_features = tcn_output[:, :, -1]  # Default to last
        
        # Feature projection
        features = self.feature_projection(pooled_features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(features).squeeze(-1)
        
        if self.prediction_mode == 'precog':
            # Point and interval predictions
            point_pred = self.point_head(features).squeeze(-1)
            interval_pred = self.interval_head(features)
            
            # Ensure interval ordering (min < max)
            interval_pred = torch.sort(interval_pred, dim=-1)[0]
            
            return point_pred, interval_pred, uncertainty
            
        elif self.prediction_mode == 'synth':
            # Detailed predictions for each time step
            detailed_pred = self.detailed_head(features)
            detailed_pred = detailed_pred.view(-1, 12, 3)  # [batch_size, 12, 3]
            
            return detailed_pred, uncertainty
    
    def compute_loss(self, predictions: Tuple[torch.Tensor, ...], targets: torch.Tensor) -> torch.Tensor:
        """Compute loss based on prediction mode"""
        if self.prediction_mode == 'precog':
            point_pred, interval_pred, uncertainty = predictions
            base_loss = precog_loss(point_pred, interval_pred, targets)
            
            # Add uncertainty regularization
            uncertainty_loss = torch.mean(uncertainty)  # Encourage small uncertainty
            total_loss = base_loss + 0.01 * uncertainty_loss
            
            return total_loss
            
        elif self.prediction_mode == 'synth':
            detailed_pred, uncertainty = predictions
            base_loss = synth_loss(detailed_pred, targets)
            
            # Add uncertainty regularization
            uncertainty_loss = torch.mean(uncertainty)  # Encourage small uncertainty
            total_loss = base_loss + 0.01 * uncertainty_loss
            
            return total_loss
    
    def evaluate(self, predictions: Tuple[torch.Tensor, ...], targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.prediction_mode == 'precog':
            point_pred, interval_pred, uncertainty = predictions
            metrics = evaluate_precog(point_pred, interval_pred, targets)
            metrics['avg_uncertainty'] = torch.mean(uncertainty).item()
            return metrics
            
        elif self.prediction_mode == 'synth':
            detailed_pred, uncertainty = predictions
            metrics = evaluate_synth(detailed_pred, targets)
            metrics['avg_uncertainty'] = torch.mean(uncertainty).item()
            return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'TCN',
            'prediction_mode': self.prediction_mode,
            'input_size': self.input_size,
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'pooling_method': self.pooling_method,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'tcn_layers': len(self.num_channels),
                'receptive_field': self.calculate_receptive_field(),
                'dilation_factors': [2**i for i in range(len(self.num_channels))],
                'pooling_method': self.pooling_method
            }
        }
    
    def calculate_receptive_field(self) -> int:
        """Calculate the receptive field of the TCN"""
        receptive_field = 1
        for i in range(len(self.num_channels)):
            dilation = 2 ** i
            receptive_field += (self.kernel_size - 1) * dilation
        return receptive_field
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from attention weights (if using attention pooling)"""
        if self.pooling_method == 'attention' and hasattr(self, '_last_attention_weights'):
            return {
                'attention_weights': self._last_attention_weights.detach().cpu().numpy(),
                'temporal_importance': 'Available from attention weights'
            }
        else:
            return {
                'feature_importance': 'Not available for this pooling method',
                'receptive_field': self.calculate_receptive_field()
            }


class TCNWithResidualConnections(TCNPredictor):
    """TCN predictor with enhanced residual connections and skip connections"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Additional residual connection parameters
        self.use_global_skip = config.get('use_global_skip', True)
        self.skip_connection_factor = config.get('skip_connection_factor', 0.1)
        
        # Build enhanced model
        self._build_enhanced_model()
    
    def _build_enhanced_model(self):
        """Build enhanced TCN with additional skip connections"""
        # Build base model first
        super()._build_model()
        
        # Add global skip connection if enabled
        if self.use_global_skip:
            self.global_skip = nn.Linear(self.input_size, self.num_channels[-1])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Enhanced forward pass with skip connections"""
        # Store original input for skip connection
        original_x = x.clone()
        
        # Standard TCN forward pass
        x_transposed = x.transpose(1, 2)
        tcn_output = self.tcn(x_transposed)
        
        # Add global skip connection
        if self.use_global_skip:
            # Pool original input to single vector
            pooled_input = torch.mean(original_x, dim=1)  # [batch_size, input_size]
            skip_connection = self.global_skip(pooled_input)  # [batch_size, num_channels[-1]]
            
            # Add skip connection to TCN output
            skip_connection = skip_connection.unsqueeze(-1)  # [batch_size, num_channels[-1], 1]
            tcn_output = tcn_output + self.skip_connection_factor * skip_connection
        
        # Continue with standard pooling and prediction
        if self.pooling_method == 'last':
            pooled_features = tcn_output[:, :, -1]
        elif self.pooling_method == 'mean':
            pooled_features = torch.mean(tcn_output, dim=2)
        elif self.pooling_method == 'max':
            pooled_features = torch.max(tcn_output, dim=2)[0]
        elif self.pooling_method == 'attention':
            tcn_output = tcn_output.transpose(1, 2)
            pooled_features = self.pooling(tcn_output)
        elif self.pooling_method == 'adaptive':
            pooled_features = self.pooling(tcn_output).squeeze(-1)
        else:
            pooled_features = tcn_output[:, :, -1]
        
        # Feature projection
        features = self.feature_projection(pooled_features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(features).squeeze(-1)
        
        if self.prediction_mode == 'precog':
            point_pred = self.point_head(features).squeeze(-1)
            interval_pred = self.interval_head(features)
            interval_pred = torch.sort(interval_pred, dim=-1)[0]
            return point_pred, interval_pred, uncertainty
        elif self.prediction_mode == 'synth':
            detailed_pred = self.detailed_head(features)
            detailed_pred = detailed_pred.view(-1, 12, 3)
            return detailed_pred, uncertainty 
"""
LSTM Bitcoin Price Predictor

This module implements an LSTM-based Bitcoin price predictor that supports both
challenge mode (point + interval predictions) and detailed mode (full timestep predictions).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base_predictor import BaseBitcoinPredictor

class LSTMBitcoinPredictor(BaseBitcoinPredictor):
    """
    LSTM-based Bitcoin price predictor.
    
    This predictor uses LSTM layers to encode temporal features and supports
    two prediction modes:
    - 'precog': Point and interval predictions for the precog subnet challenge
    - 'synth': Detailed close, low, high predictions for each timestep
    """
    
    def __init__(self, input_size: int, config: Dict[str, Any]):
        """
        Initialize the LSTM predictor.
        
        Args:
            input_size: Number of input features
            config: Configuration dictionary with model parameters
        """
        super().__init__(input_size, config)
        
    def _build_model(self):
        """Build the LSTM model architecture."""
        # LSTM encoder for all features
        self.feature_encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.config.get('bidirectional', False)
        )
        
        # Adjust hidden size for bidirectional LSTM
        encoder_output_size = self.hidden_size * (2 if self.config.get('bidirectional', False) else 1)
        
        # Layer normalization
        if self.use_layer_norm:
            self.feature_norm = nn.LayerNorm(encoder_output_size)
        
        # Build prediction heads based on mode
        if self.mode == 'precog':
            self._build_precog_heads(encoder_output_size)
        elif self.mode == 'synth':
            self._build_synth_heads(encoder_output_size)
    
    def _build_precog_heads(self, encoder_output_size: int):
        """Build prediction heads for precog mode."""
        # Point prediction head (exact price 1 hour ahead)
        self.point_head = nn.Sequential(
            nn.Linear(encoder_output_size, encoder_output_size // 2),
            self.act_fn,
            nn.Dropout(self.dropout),
            nn.Linear(encoder_output_size // 2, 1)
        )
        
        # Interval prediction head (min and max prices)
        self.interval_head = nn.Sequential(
            nn.Linear(encoder_output_size, encoder_output_size // 2),
            self.act_fn,
            nn.Dropout(self.dropout),
            nn.Linear(encoder_output_size // 2, 2)  # [min, max]
        )
    
    def _build_synth_heads(self, encoder_output_size: int):
        """Build prediction heads for synth mode."""
        # Detailed prediction head for close, low, high at each timestep
        self.detailed_head = nn.Sequential(
            nn.Linear(encoder_output_size, encoder_output_size),
            self.act_fn,
            nn.Dropout(self.dropout),
            nn.Linear(encoder_output_size, encoder_output_size // 2),
            self.act_fn,
            nn.Dropout(self.dropout),
            nn.Linear(encoder_output_size // 2, 3 * 12)  # 3 values * 12 timesteps
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the LSTM predictor.
        
        Args:
            x: Input tensor of shape [batch_size, flattened_features] or [batch_size, sequence_length, input_size]
            
        Returns:
            Model predictions based on mode:
            - precog mode: (point_pred, interval_pred)
            - synth mode: (detailed_pred,)
        """
        # Handle flattened input by reshaping to sequence format
        if x.dim() == 2:
            # For flattened input, we need to reshape to [batch_size, sequence_length, features_per_timestep]
            total_features = x.shape[1]
            
            # Get the expected input size from the LSTM layer
            lstm_input_size = self.feature_encoder.input_size
            
            # Calculate lookback from config or infer from dimensions
            lookback = self.config.get('lookback', 12)
            
            # Simple approach: divide total features by lookback to get features per timestep
            if total_features % lookback == 0:
                features_per_timestep = total_features // lookback
                x = x.view(-1, lookback, features_per_timestep)
            else:
                # If doesn't divide evenly, use the LSTM's expected input size
                # and take the first lookback * lstm_input_size features
                expected_total = lookback * lstm_input_size
                if total_features >= expected_total:
                    x_trimmed = x[:, :expected_total]
                    x = x_trimmed.view(-1, lookback, lstm_input_size)
                else:
                    # If not enough features, pad with zeros
                    padding_needed = expected_total - total_features
                    x_padded = torch.cat([x, torch.zeros(x.shape[0], padding_needed, device=x.device)], dim=1)
                    x = x_padded.view(-1, lookback, lstm_input_size)
        
        # Encode features with LSTM
        encoded, _ = self.feature_encoder(x)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            encoded = self.feature_norm(encoded)
        
        # Use the last timestep for prediction
        final_encoding = encoded[:, -1]
        
        # Generate predictions based on mode
        if self.mode == 'precog':
            return self._forward_precog(final_encoding)
        elif self.mode == 'synth':
            return self._forward_synth(final_encoding)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _forward_precog(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for precog mode."""
        # Point prediction
        point_pred = self.point_head(encoded).squeeze(-1)
        
        # Interval prediction
        interval_pred = self.interval_head(encoded)
        
        # Ensure min < max in interval by sorting
        interval_pred = torch.sort(interval_pred, dim=-1)[0]
        
        return point_pred, interval_pred
    
    def _forward_synth(self, encoded: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward pass for synth mode."""
        # Detailed predictions
        detailed_pred = self.detailed_head(encoded)
        
        # Reshape to [batch_size, 12, 3] for [timestep, (close, low, high)]
        detailed_pred = detailed_pred.view(-1, 12, 3)
        
        # Ensure consistency: low <= close <= high for each timestep
        detailed_pred = self._enforce_price_consistency(detailed_pred)
        
        return (detailed_pred,)
    
    def _enforce_price_consistency(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Ensure price consistency: low <= close <= high for each timestep.
        
        Args:
            predictions: Tensor of shape [batch_size, 12, 3] with [close, low, high]
            
        Returns:
            Tensor with enforced consistency
        """
        # Sort each timestep to ensure low <= close <= high
        # Note: The current order is [close, low, high], so we need to reorder
        close = predictions[:, :, 0]  # Shape: [batch_size, 12]
        low = predictions[:, :, 1]    # Shape: [batch_size, 12]
        high = predictions[:, :, 2]   # Shape: [batch_size, 12]
        
        # Ensure low <= close <= high
        low = torch.minimum(low, close)
        high = torch.maximum(high, close)
        
        # Stack back in original order [close, low, high]
        return torch.stack([close, low, high], dim=-1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get extended model information."""
        base_info = super().get_model_info()
        
        # Add LSTM-specific information
        lstm_info = {
            'architecture': 'LSTM',
            'bidirectional': self.config.get('bidirectional', False),
            'encoder_layers': self.num_layers,
            'encoder_hidden_size': self.hidden_size,
            'encoder_dropout': self.dropout if self.num_layers > 1 else 0,
        }
        
        base_info.update(lstm_info)
        return base_info
    
    def print_model_summary(self):
        """Print detailed model summary."""
        info = self.get_model_info()
        print(f"\n{'='*60}")
        print(f"LSTM Model Summary: {info['model_type']}")
        print(f"{'='*60}")
        print(f"Mode: {info['mode']}")
        print(f"Input Size: {info['input_size']}")
        print(f"Architecture: {info['architecture']}")
        print(f"Bidirectional: {info['bidirectional']}")
        print(f"Encoder Layers: {info['encoder_layers']}")
        print(f"Hidden Size: {info['encoder_hidden_size']}")
        print(f"Activation: {self.activation}")
        print(f"Dropout: {self.dropout}")
        print(f"Layer Norm: {self.use_layer_norm}")
        print(f"Total Parameters: {info['parameter_count']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        
        if self.mode == 'precog':
            print(f"Output: Point prediction (1) + Interval prediction (2)")
        elif self.mode == 'synth':
            print(f"Output: Detailed predictions (12 timesteps × 3 values)")
        
        print(f"{'='*60}\n")


# Factory function for easy instantiation
def create_lstm_predictor(input_size: int, config: Dict[str, Any]) -> LSTMBitcoinPredictor:
    """
    Factory function to create LSTM predictor.
    
    Args:
        input_size: Number of input features
        config: Configuration dictionary
        
    Returns:
        Configured LSTM predictor
    """
    return LSTMBitcoinPredictor(input_size, config) 
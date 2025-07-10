"""
Synth Loss Functions

This module contains loss and evaluation functions specific to the synth mode.
The synth mode predicts detailed close, low, high prices for each 5-minute timestep
over a 1-hour period (12 timesteps total).

This provides full temporal resolution for analysis and research purposes.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

def synth_loss(detailed_pred: torch.Tensor, 
               targets: torch.Tensor, 
               scaler: Optional[Any] = None) -> torch.Tensor:
    """
    Loss function for synth mode (detailed predictions).
    
    Args:
        detailed_pred: Detailed predictions [batch_size, 12, 3] for [close, low, high]
        targets: Target tensor [batch_size, 36] with detailed predictions
        scaler: Optional scaler for inverse transformation
    
    Returns:
        MSE loss for detailed predictions
    """
    # Reshape targets to [batch_size, 12, 3] to match detailed_pred
    detailed_targets = targets.view(-1, 12, 3)  # [batch_size, 12, 3] for [close, low, high]
    
    # Check for NaN values
    if torch.isnan(detailed_pred).any() or torch.isnan(detailed_targets).any():
        print(f"Warning: NaN detected in synth_loss")
        print(f"  detailed_pred shape: {detailed_pred.shape}")
        print(f"  detailed_targets shape: {detailed_targets.shape}")
        print(f"  NaN in pred: {torch.isnan(detailed_pred).sum().item()}")
        print(f"  NaN in targets: {torch.isnan(detailed_targets).sum().item()}")
        return torch.tensor(float('inf'), device=detailed_pred.device)
    
    # Apply scaler inverse transformation if provided
    if scaler is not None:
        # Cache scaler parameters on GPU to avoid repeated CPU-GPU transfers
        if not hasattr(synth_loss, '_scaler_cache'):
            synth_loss._scaler_cache = {
                'mean': torch.FloatTensor(scaler.mean_).to(detailed_pred.device),
                'scale': torch.FloatTensor(scaler.scale_).to(detailed_pred.device),
                'price_indices': torch.tensor([3, 1, 2], device=detailed_pred.device)  # [close, low, high]
            }
        
        cache = synth_loss._scaler_cache
        mean_ = cache['mean']
        scale_ = cache['scale']
        price_indices = cache['price_indices']
        
        # Inverse transform predictions and targets
        detailed_pred_unscaled = detailed_pred.clone()
        detailed_targets_unscaled = detailed_targets.clone()
        
        for i, price_idx in enumerate(price_indices):
            detailed_pred_unscaled[:, :, i] = (detailed_pred[:, :, i] * scale_[price_idx] + mean_[price_idx])
            detailed_targets_unscaled[:, :, i] = (detailed_targets[:, :, i] * scale_[price_idx] + mean_[price_idx])
        
        # Use unscaled values for loss calculation
        detailed_pred = detailed_pred_unscaled
        detailed_targets = detailed_targets_unscaled
    
    # Calculate MSE loss for detailed predictions
    mse_loss = F.mse_loss(detailed_pred, detailed_targets)
    
    # Add consistency penalty: ensure low <= close <= high for each timestep
    close_prices = detailed_pred[:, :, 0]  # [batch_size, 12]
    low_prices = detailed_pred[:, :, 1]    # [batch_size, 12]
    high_prices = detailed_pred[:, :, 2]   # [batch_size, 12]
    
    # Penalty for violating low <= close <= high
    low_violation = torch.relu(low_prices - close_prices)  # low should be <= close
    high_violation = torch.relu(close_prices - high_prices)  # close should be <= high
    
    consistency_penalty = (low_violation.mean() + high_violation.mean()) * 0.1
    
    # Add temporal smoothness penalty to encourage realistic price movements
    close_diff = torch.abs(close_prices[:, 1:] - close_prices[:, :-1])
    smoothness_penalty = close_diff.mean() * 0.01
    
    total_loss = mse_loss + consistency_penalty + smoothness_penalty
    
    return total_loss


def evaluate_synth(model, data_loader, device, scaler=None) -> Dict[str, float]:
    """
    Comprehensive evaluation system for synth mode.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        scaler: Optional scaler for inverse transformation
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_detailed_preds = []
    all_targets = []
    
    # Progress bar for evaluation
    eval_pbar = tqdm(data_loader, desc='Synth Evaluation', leave=False, ncols=100)
    
    with torch.no_grad():
        for batch_X, batch_y in eval_pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Check input data
            if torch.isnan(batch_X).any():
                print(f"Warning: NaN detected in evaluation input features")
                continue
            
            # Get detailed predictions
            detailed_pred = model(batch_X)[0]  # First element from tuple
            
            # Check model output
            if torch.isnan(detailed_pred).any():
                print(f"Warning: NaN detected in evaluation model predictions")
                continue
            
            # Synth loss calculation
            loss = synth_loss(detailed_pred, batch_y, scaler)
            
            # Collect predictions for detailed analysis
            all_detailed_preds.append(detailed_pred.cpu())
            all_targets.append(batch_y.cpu())
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss in evaluation: {loss.item()}")
                continue
                
            total_loss += loss.item()
            
            # Update progress bar
            eval_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Check if we have any valid predictions
    if len(all_detailed_preds) == 0:
        print("Warning: No valid predictions collected during evaluation")
        return {'total_loss': float('inf'), 'mse': float('inf'), 'mae': float('inf')}
    
    # Concatenate all predictions
    all_detailed_preds = torch.cat(all_detailed_preds, dim=0)  # [N, 12, 3]
    all_targets = torch.cat(all_targets, dim=0)  # [N, 36]
    
    # Reshape targets to match predictions
    detailed_targets = all_targets.view(-1, 12, 3)  # [N, 12, 3]
    
    # Basic data validation
    print(f"  SYNTH EVALUATION DATA:")
    print(f"    Predictions shape: {all_detailed_preds.shape}")
    print(f"    Targets shape: {all_targets.shape}")
    print(f"    Detailed targets shape: {detailed_targets.shape}")
    print(f"    Pred range: [{all_detailed_preds.min().item():.2f}, {all_detailed_preds.max().item():.2f}]")
    print(f"    Target range: [{detailed_targets.min().item():.2f}, {detailed_targets.max().item():.2f}]")
    
    # Calculate comprehensive metrics
    mse = F.mse_loss(all_detailed_preds, detailed_targets).item()
    mae = torch.abs(all_detailed_preds - detailed_targets).mean().item()
    rmse = torch.sqrt(torch.mean((all_detailed_preds - detailed_targets) ** 2)).item()
    
    # Per-timestep analysis
    timestep_mse = []
    timestep_mae = []
    for t in range(12):
        t_mse = F.mse_loss(all_detailed_preds[:, t, :], detailed_targets[:, t, :]).item()
        t_mae = torch.abs(all_detailed_preds[:, t, :] - detailed_targets[:, t, :]).mean().item()
        timestep_mse.append(t_mse)
        timestep_mae.append(t_mae)
    
    # Per-variable analysis (close, low, high)
    variable_names = ['close', 'low', 'high']
    variable_mse = []
    variable_mae = []
    for v in range(3):
        v_mse = F.mse_loss(all_detailed_preds[:, :, v], detailed_targets[:, :, v]).item()
        v_mae = torch.abs(all_detailed_preds[:, :, v] - detailed_targets[:, :, v]).mean().item()
        variable_mse.append(v_mse)
        variable_mae.append(v_mae)
    
    # Consistency analysis
    pred_close = all_detailed_preds[:, :, 0]
    pred_low = all_detailed_preds[:, :, 1]
    pred_high = all_detailed_preds[:, :, 2]
    
    consistency_violations = (
        (pred_low > pred_close).sum().item() + 
        (pred_close > pred_high).sum().item()
    )
    total_predictions = all_detailed_preds.shape[0] * all_detailed_preds.shape[1]
    consistency_rate = 1.0 - (consistency_violations / total_predictions)
    
    # Temporal smoothness analysis
    close_diff = torch.abs(pred_close[:, 1:] - pred_close[:, :-1])
    avg_price_change = close_diff.mean().item()
    max_price_change = close_diff.max().item()
    
    # Print detailed results
    print(f"  SYNTH DETAILED PREDICTIONS:")
    print(f"    Overall MSE: {mse:.4f}")
    print(f"    Overall MAE: {mae:.4f}")
    print(f"    Overall RMSE: {rmse:.4f}")
    
    print(f"  SYNTH PER-VARIABLE ANALYSIS:")
    for i, var_name in enumerate(variable_names):
        print(f"    {var_name.upper()} - MSE: {variable_mse[i]:.4f}, MAE: {variable_mae[i]:.4f}")
    
    print(f"  SYNTH CONSISTENCY ANALYSIS:")
    print(f"    Consistency rate: {consistency_rate:.4f}")
    print(f"    Violations: {consistency_violations}/{total_predictions}")
    
    print(f"  SYNTH TEMPORAL ANALYSIS:")
    print(f"    Avg price change: {avg_price_change:.4f}")
    print(f"    Max price change: {max_price_change:.4f}")
    
    # Return comprehensive metrics
    return {
        'total_loss': total_loss / len(data_loader),
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'close_mse': variable_mse[0],
        'low_mse': variable_mse[1],
        'high_mse': variable_mse[2],
        'close_mae': variable_mae[0],
        'low_mae': variable_mae[1],
        'high_mae': variable_mae[2],
        'consistency_rate': consistency_rate,
        'consistency_violations': consistency_violations,
        'avg_price_change': avg_price_change,
        'max_price_change': max_price_change,
        'timestep_mse': timestep_mse,
        'timestep_mae': timestep_mae,
        'num_samples': len(all_detailed_preds)
    } 
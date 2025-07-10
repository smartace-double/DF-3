"""
Precog Loss Functions

This module contains loss and evaluation functions specific to the precog challenge mode.
The precog challenge predicts:
1. Point prediction: Exact BTC price 1 hour ahead
2. Interval prediction: [min, max] range for the entire 1-hour period

These functions exactly match the precog subnet scoring system.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

EPSILON = 1e-4

def precog_loss(point_pred: torch.Tensor, 
                interval_pred: torch.Tensor, 
                targets: torch.Tensor, 
                scaler: Optional[Any] = None) -> torch.Tensor:
    """
    Loss function that exactly matches the precog subnet scoring system.
    
    Args:
        point_pred: Point predictions [batch_size]
        interval_pred: Interval predictions [batch_size, 2] (min, max)
        targets: Target tensor [batch_size, 36] with detailed predictions
        scaler: Optional scaler for inverse transformation
    
    Returns:
        Combined loss value
    """
    # Extract targets from the detailed structure
    # targets has shape [batch_size, 36] where 36 = 12 time steps * 3 values (close, low, high)
    
    # Debug: Check target shape and values
    if torch.isnan(targets).any():
        print(f"Warning: NaN detected in targets in precog_loss")
        print(f"  targets shape: {targets.shape}")
        print(f"  NaN count: {torch.isnan(targets).sum().item()}")
        return torch.tensor(float('inf'), device=point_pred.device)
    
    # Reshape targets to [batch_size, 12, 3] for easier processing
    targets_reshaped = targets.view(-1, 12, 3)  # [batch_size, 12, 3] for [close, low, high]
    
    # Extract base targets:
    # 1. Exact price 1 hour ahead = close price at the last time step (index 11)
    exact_price_target = targets_reshaped[:, 11, 0].clamp(min=EPSILON)  # close at time step 11
    
    # 2. Min price during 1-hour period = minimum of all low prices
    min_price_target = targets_reshaped[:, :, 1].min(dim=1)[0]  # min of all low prices
    
    # 3. Max price during 1-hour period = maximum of all high prices  
    max_price_target = targets_reshaped[:, :, 2].max(dim=1)[0]  # max of all high prices
    
    # 4. All close prices for inclusion factor calculation
    hour_prices = targets_reshaped[:, :, :].view(-1, 36)  # All high, low, close prices [batch_size, 36]
    # remove duplicated hour prices
    hour_prices = hour_prices.unique(dim=1)
    
    # Convert scaled predictions back to original scale for price-related features
    if scaler is not None:
        # Cache scaler parameters on GPU to avoid repeated CPU-GPU transfers
        if not hasattr(precog_loss, '_scaler_cache'):
            precog_loss._scaler_cache = {
                'mean': torch.FloatTensor(scaler.mean_).to(point_pred.device),
                'scale': torch.FloatTensor(scaler.scale_).to(point_pred.device),
                'price_indices': torch.tensor([3, 1, 2], device=point_pred.device)  # [close, high, low]
            }
        
        cache = precog_loss._scaler_cache
        mean_ = cache['mean']
        scale_ = cache['scale']
        price_indices = cache['price_indices']
        
        # Efficient GPU-based inverse transform (vectorized operations)
        point_pred_unscaled = point_pred * scale_[price_indices[0]] + mean_[price_indices[0]]
        interval_min_unscaled = interval_pred[:, 0] * scale_[price_indices[1]] + mean_[price_indices[1]]
        interval_max_unscaled = interval_pred[:, 1] * scale_[price_indices[2]] + mean_[price_indices[2]]
        
        # Use unscaled values for calculations
        point_pred = point_pred_unscaled
        interval_pred = torch.stack([interval_min_unscaled, interval_max_unscaled], dim=1)
    
    # 1. Point Prediction Loss (MAPE for exact 1-hour ahead price)
    point_mape = torch.abs(point_pred - exact_price_target) / exact_price_target.clamp(min=EPSILON)
    point_loss = point_mape.mean()

    # 2. Interval Loss Components (exactly matching reward.py logic)
    interval_min = interval_pred[:, 0]
    interval_max = interval_pred[:, 1]
    
    # Ensure valid intervals (min < max)
    interval_min = torch.minimum(interval_min, interval_max - EPSILON)
    interval_max = torch.maximum(interval_max, interval_min + EPSILON)
    
    # Calculate width factor (f_w) exactly as in reward.py
    # effective_top = min(pred_max, observed_max)
    # effective_bottom = max(pred_min, observed_min)
    # width_factor = (effective_top - effective_bottom) / (pred_max - pred_min)
    effective_top = torch.minimum(interval_max, max_price_target)
    effective_bottom = torch.maximum(interval_min, min_price_target)
    
    # Handle case where pred_max == pred_min (invalid interval) with better numerical stability
    interval_width = interval_max - interval_min
    # Add small epsilon to prevent division by zero and numerical instability
    epsilon = EPSILON
    width_factor = torch.where(
        interval_width <= epsilon,
        torch.zeros_like(interval_max),
        (effective_top - effective_bottom) / (interval_width + epsilon)
    )
    
    # Calculate inclusion factor (f_i) exactly as in reward.py
    # prices_in_bounds = sum(1 for price in hour_prices if pred_min <= price <= pred_max)
    # inclusion_factor = prices_in_bounds / len(hour_prices)
    
    # Count prices within bounds for each sample
    prices_in_bounds = torch.sum(
        (interval_min.unsqueeze(1) <= hour_prices) & (hour_prices <= interval_max.unsqueeze(1)),
        dim=1
    ).float()
    
    inclusion_factor = prices_in_bounds / hour_prices.shape[1]  # Divide by number of price points
    
    # Final interval score is the product (exactly as in reward.py)
    interval_score = inclusion_factor * width_factor
    
    # Convert to loss (higher score = lower loss)
    interval_loss = 1.0 - interval_score.mean()
    
    # Combine point and interval losses with 1:1 weight
    total_loss = 0.5 * point_loss + 0.5 * interval_loss

    # Add a penalty for invalid prediction outside the interval
    total_loss = total_loss + 10 * (point_pred < min_price_target).float().mean() + 10 * (point_pred > max_price_target).float().mean()
    
    return total_loss


def evaluate_precog(model, data_loader, device, scaler=None) -> Dict[str, float]:
    """
    Comprehensive evaluation system for precog mode that exactly matches the precog subnet scoring system.
    
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
    all_point_preds = []
    all_interval_preds = []
    all_targets = []
    
    # Progress bar for evaluation
    eval_pbar = tqdm(data_loader, desc='Precog Evaluation', leave=False, ncols=100)
    
    with torch.no_grad():
        for batch_X, batch_y in eval_pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Debug: Check input data in evaluation
            if torch.isnan(batch_X).any():
                print(f"Warning: NaN detected in evaluation input features")
                continue
            
            point_pred, interval_pred = model(batch_X)
            
            # Debug: Check model output in evaluation
            if torch.isnan(point_pred).any() or torch.isnan(interval_pred).any():
                print(f"Warning: NaN detected in evaluation model predictions")
                continue
            
            # Precog loss calculation
            loss = precog_loss(point_pred, interval_pred, batch_y, scaler)
            
            # Collect predictions for detailed analysis
            all_point_preds.append(point_pred.cpu())
            all_interval_preds.append(interval_pred.cpu())
            all_targets.append(batch_y.cpu())
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss in evaluation: {loss.item()}")
                continue
                
            total_loss += loss.item()
            
            # Update progress bar
            eval_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Check if we have any valid predictions
    if len(all_point_preds) == 0:
        print("Warning: No valid predictions collected during evaluation")
        return {'total_loss': float('inf'), 'point_mape': float('inf'), 'interval_score': 0.0}
    
    # Concatenate all predictions
    all_point_preds = torch.cat(all_point_preds, dim=0)
    all_interval_preds = torch.cat(all_interval_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Extract targets exactly as in precog_loss
    targets_reshaped = all_targets.view(-1, 12, 3)  # [batch_size, 12, 3] for [close, low, high]
    
    # Extract base targets:
    exact_price_target = targets_reshaped[:, 11, 0]  # close at time step 11
    min_price_target = targets_reshaped[:, :, 1].min(dim=1)[0]  # min of all low prices
    max_price_target = targets_reshaped[:, :, 2].max(dim=1)[0]  # max of all high prices
    hour_prices = targets_reshaped[:, :, :].view(-1, 36)  # All high, low, close prices [batch_size, 36]
    hour_prices = hour_prices.unique(dim=1)
    
    # Basic data validation
    print(f"  PRECOG EVALUATION DATA:")
    print(f"    Predictions shape: {all_point_preds.shape}, {all_interval_preds.shape}")
    print(f"    Targets shape: {all_targets.shape}")
    print(f"    Hour prices shape: {hour_prices.shape}")
    print(f"    Point pred range: [{all_point_preds.min().item():.2f}, {all_point_preds.max().item():.2f}]")
    print(f"    Interval pred range: [{all_interval_preds.min().item():.2f}, {all_interval_preds.max().item():.2f}]")
    print(f"    Exact target range: [{exact_price_target.min().item():.2f}, {exact_price_target.max().item():.2f}]")
    
    # 1. Point Prediction Analysis
    point_errors = torch.abs(all_point_preds - exact_price_target) / exact_price_target.clamp(min=EPSILON)
    avg_point_error = point_errors.mean().item()
    
    # Additional point metrics
    mae = torch.abs(all_point_preds - exact_price_target).mean().item()
    rmse = torch.sqrt(torch.mean((all_point_preds - exact_price_target) ** 2)).item()
    
    # 2. Interval Analysis
    interval_min = all_interval_preds[:, 0]
    interval_max = all_interval_preds[:, 1]
    
    # Calculate width factor and inclusion factor
    effective_top = torch.minimum(interval_max, max_price_target)
    effective_bottom = torch.maximum(interval_min, min_price_target)
    
    interval_width = interval_max - interval_min
    epsilon = EPSILON
    width_factor = torch.where(
        interval_width <= epsilon,
        torch.zeros_like(interval_max),
        (effective_top - effective_bottom) / (interval_width + epsilon)
    )
    
    prices_in_bounds = torch.sum(
        (interval_min.unsqueeze(1) <= hour_prices) & (hour_prices <= interval_max.unsqueeze(1)),
        dim=1
    ).float()
    
    inclusion_factor = prices_in_bounds / hour_prices.shape[1]
    
    # Final interval score
    interval_scores = inclusion_factor * width_factor
    avg_interval_score = interval_scores.mean().item()
    
    # 3. Comprehensive Results
    print(f"  PRECOG POINT PREDICTION (1-hour ahead):")
    print(f"    Average MAPE: {avg_point_error:.4f}")
    print(f"    MAE: {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    Target mean: {exact_price_target.mean().item():.2f}")
    print(f"    Pred mean: {all_point_preds.mean().item():.2f}")
    
    print(f"  PRECOG INTERVAL ANALYSIS (1-hour period):")
    print(f"    Average width factor: {width_factor.mean().item():.4f}")
    print(f"    Average inclusion factor: {inclusion_factor.mean().item():.4f}")
    print(f"    Average interval score: {avg_interval_score:.4f}")
    
    # Final precog subnet score
    precog_score = 0.5 * avg_point_error + 0.5 * (1.0 - avg_interval_score)
    print(f"  PRECOG SUBNET SCORE: {precog_score:.4f}")
    
    # Return comprehensive metrics
    return {
        'total_loss': total_loss / len(data_loader),
        'point_mape': avg_point_error,
        'point_mae': mae,
        'point_rmse': rmse,
        'width_factor': width_factor.mean().item(),
        'inclusion_factor': inclusion_factor.mean().item(),
        'interval_score': avg_interval_score,
        'precog_score': precog_score,
        'num_samples': len(all_point_preds)
    } 
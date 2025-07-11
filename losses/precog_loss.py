"""
Precog Loss Functions

This module contains loss and evaluation functions specific to the precog challenge mode.
The precog challenge predicts:
1. Point prediction: Exact BTC price in USD 1 hour ahead  
2. Interval prediction: [min, max] USD price range for the entire 1-hour period

The model predicts relative returns (trained on relative return targets), but these must be
converted to USD prices for loss calculation to match the challenge requirements.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

EPSILON = 1e-8

def precog_loss(point_pred: torch.Tensor, 
                interval_pred: torch.Tensor, 
                targets: torch.Tensor, 
                scaler: Optional[Any] = None,
                current_prices: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Loss function for precog challenge converting relative returns to USD prices.
    
    Args:
        point_pred: Point predictions [batch_size] (relative returns from model)
        interval_pred: Interval predictions [batch_size, 2] (relative returns from model)
        targets: Target tensor [batch_size, 36] with relative return targets
        scaler: Scaler for extracting current price reference
        current_prices: Optional tensor of current prices [batch_size] for conversion
    
    Returns:
        Combined loss value
    """
    # Debug: Check target shape and values
    if torch.isnan(targets).any():
        print(f"Warning: NaN detected in targets in precog_loss")
        print(f"  targets shape: {targets.shape}")
        print(f"  NaN count: {torch.isnan(targets).sum().item()}")
        return torch.tensor(float('inf'), device=point_pred.device)
    
    # Extract current_close from targets (37th feature, index 36) and reshape the rest
    if targets.shape[1] >= 37:
        # Extract current prices from 37th feature
        extracted_current_prices = targets[:, 36]  # Current close prices
        # Use the first 36 features for relative returns
        relative_return_targets = targets[:, :36]
    else:
        print(f"Warning: Expected 37 target features, got {targets.shape[1]}")
        relative_return_targets = targets
        extracted_current_prices = torch.ones(targets.shape[0], device=targets.device) * 50000.0  # Dummy
        raise ValueError("Stop Here due to don't have current close target")
    
    # Reshape relative return targets to [batch_size, 12, 3] for easier processing
    targets_reshaped = relative_return_targets.view(-1, 12, 3)  # [batch_size, 12, 3] for [point_return, min_return, max_return]
    
    # Extract relative return targets:
    # 1. Point return at the last time step (index 11)
    point_return_target = targets_reshaped[:, 11, 0]  # point return at time step 11
    
    # 2. Min return during 1-hour period = minimum of all min returns
    min_return_target = targets_reshaped[:, :, 1].min(dim=1)[0]  # min of all min returns
    
    # 3. Max return during 1-hour period = maximum of all max returns  
    max_return_target = targets_reshaped[:, :, 2].max(dim=1)[0]  # max of all max returns
    
    # 4. All point returns for inclusion factor calculation
    hour_returns = targets_reshaped[:, :, 0]  # All point returns [batch_size, 12]
    
    # Use extracted current prices if current_prices parameter is not provided
    if current_prices is None:
        current_prices = extracted_current_prices
    
    # Convert relative returns to USD prices
    # USD_price = current_price * (1 + relative_return)
    
    # Convert point predictions and targets
    point_pred_usd = current_prices * (1 + point_pred)
    point_target_usd = current_prices * (1 + point_return_target)
    
    # Convert interval predictions and targets
    interval_min_pred_usd = current_prices * (1 + interval_pred[:, 0])
    interval_max_pred_usd = current_prices * (1 + interval_pred[:, 1])
    min_target_usd = current_prices * (1 + min_return_target)
    max_target_usd = current_prices * (1 + max_return_target)
    
    # Convert all hour returns to USD prices for inclusion factor
    hour_prices_usd = current_prices.unsqueeze(1) * (1 + hour_returns)  # [batch_size, 12]
    
    # Ensure targets are positive for MAPE calculation
    point_target_usd = point_target_usd.clamp(min=EPSILON)
    
    # 1. Point Prediction Loss (MAPE for exact 1-hour ahead USD price)
    point_mape = torch.abs(point_pred_usd - point_target_usd) / point_target_usd.clamp(min=EPSILON)
    point_loss = point_mape.mean()

    # 2. Interval Loss Components (exactly matching reward.py logic for USD prices)
    interval_min = interval_min_pred_usd
    interval_max = interval_max_pred_usd
    
    # Ensure valid intervals (min < max)
    interval_min = torch.minimum(interval_min, interval_max - EPSILON)
    interval_max = torch.maximum(interval_max, interval_min + EPSILON)
    
    # Calculate width factor (f_w) exactly as in reward.py
    # effective_top = min(pred_max, observed_max)
    # effective_bottom = max(pred_min, observed_min)  
    # width_factor = (effective_top - effective_bottom) / (pred_max - pred_min)
    effective_top = torch.minimum(interval_max, max_target_usd)
    effective_bottom = torch.maximum(interval_min, min_target_usd)
    
    # Handle case where pred_max == pred_min (invalid interval)
    interval_width = interval_max - interval_min
    width_factor = torch.where(
        interval_width <= EPSILON,
        torch.zeros_like(interval_max),
        (effective_top - effective_bottom) / (interval_width + EPSILON)
    )
    
    # Calculate inclusion factor (f_i) exactly as in reward.py
    # Count USD prices within bounds for each sample
    prices_in_bounds = torch.sum(
        (interval_min.unsqueeze(1) <= hour_prices_usd) & (hour_prices_usd <= interval_max.unsqueeze(1)),
        dim=1
    ).float()
    
    inclusion_factor = prices_in_bounds / hour_prices_usd.shape[1]  # Divide by number of price points
    
    # Final interval score is the product (exactly as in reward.py)
    interval_score = inclusion_factor * width_factor
    
    # Convert to loss (higher score = lower loss)
    interval_loss = 1.0 - interval_score.mean()
    
    # Combine point and interval losses with equal weight
    total_loss = 0.5 * point_loss + 0.5 * interval_loss
    
    # Add penalty for predictions outside reasonable bounds (USD prices)
    # For BTC, reasonable bounds might be ±50% from current price range
    current_price_range = max_target_usd - min_target_usd
    reasonable_bound = current_price_range * 0.5
    
    out_of_bounds_penalty = (
        torch.relu(point_pred_usd - max_target_usd - reasonable_bound).mean() +
        torch.relu(min_target_usd - reasonable_bound - point_pred_usd).mean() +
        torch.relu(interval_min - min_target_usd - reasonable_bound).mean() +
        torch.relu(max_target_usd + reasonable_bound - interval_max).mean()
    )
    
    total_loss = total_loss + 0.01 * out_of_bounds_penalty
    
    # Safety check: ensure loss is not NaN or inf
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"Warning: Invalid loss detected in precog_loss")
        print(f"  point_loss: {point_loss.item()}")
        print(f"  interval_loss: {interval_loss.item()}")
        print(f"  out_of_bounds_penalty: {out_of_bounds_penalty.item()}")
        return torch.tensor(1.0, device=point_pred.device)  # Return reasonable fallback loss
    
    return total_loss


def extract_current_prices_from_targets(targets: torch.Tensor) -> torch.Tensor:
    """
    Extract current prices from the target tensor (37th feature).
    
    Args:
        targets: Target tensor [batch_size, 37] with current_close as the 37th feature
        
    Returns:
        Current prices tensor [batch_size]
    """
    if targets.shape[1] >= 37:
        # Current close is the 37th target feature (index 36)
        current_prices = targets[:, 36]  # Extract current_close from targets
        return current_prices.clamp(min=EPSILON)
    else:
        print("Warning: Targets don't have 37th feature (current_close)")
        print("Using dummy current prices (this will cause incorrect loss calculation)")
        raise ValueError("Stop Here due to don't have current close target")
        return torch.ones(targets.shape[0], device=targets.device) * 50000.0  # Dummy BTC price


def evaluate_precog(model, data_loader, device, scaler=None) -> Dict[str, float]:
    """
    Comprehensive evaluation system for precog mode using USD prices converted from relative returns.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        scaler: Scaler for extracting current price reference
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_point_preds = []
    all_interval_preds = []
    all_targets = []
    all_current_prices = []
    
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
            
            predictions = model(batch_X)
            
            # Handle different prediction formats
            if isinstance(predictions, tuple) and len(predictions) >= 2:
                point_pred, interval_pred = predictions[0], predictions[1]
            elif torch.is_tensor(predictions):
                # If single tensor, assume it needs to be split
                point_pred = predictions
                interval_pred = torch.stack([point_pred, point_pred], dim=1)  # Dummy interval
            else:
                # Handle other formats
                raise ValueError(f"Unknown prediction format: {type(predictions)}")
            
            # Debug: Check model output in evaluation
            if torch.isnan(point_pred).any() or torch.isnan(interval_pred).any():
                print(f"Warning: NaN detected in evaluation model predictions")
                continue
            
            # Extract current prices from targets (37th feature)
            current_prices = extract_current_prices_from_targets(batch_y)
            
            # Precog loss calculation
            loss = precog_loss(point_pred, interval_pred, batch_y, scaler, current_prices)
            
            # Collect predictions for detailed analysis
            all_point_preds.append(point_pred.cpu())
            all_interval_preds.append(interval_pred.cpu())
            all_targets.append(batch_y.cpu())
            all_current_prices.append(current_prices.cpu())
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss in evaluation: {loss.item()}")
                continue
                
            total_loss += loss.item()
            
            # Update progress bar
            eval_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Check if we have any valid predictions
    if len(all_point_preds) == 0:
        print("Warning: No valid predictions collected during evaluation")
        return {'total_loss': float('inf'), 'point_mae': float('inf'), 'interval_score': 0.0}
    
    # Concatenate all predictions
    all_point_preds = torch.cat(all_point_preds, dim=0)
    all_interval_preds = torch.cat(all_interval_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_current_prices = torch.cat(all_current_prices, dim=0)
    
    # Extract relative return targets and convert to USD prices
    # Use only the first 36 features (relative returns), excluding the 37th feature (current_close)
    relative_return_targets = all_targets[:, :36]
    targets_reshaped = relative_return_targets.view(-1, 12, 3)  # [batch_size, 12, 3] for [point_return, min_return, max_return]
    
    # Extract relative return targets:
    point_return_target = targets_reshaped[:, 11, 0]  # point return at time step 11
    min_return_target = targets_reshaped[:, :, 1].min(dim=1)[0]  # min of all min returns
    max_return_target = targets_reshaped[:, :, 2].max(dim=1)[0]  # max of all max returns
    hour_returns = targets_reshaped[:, :, 0]  # All point returns [batch_size, 12]
    
    # Convert to USD prices using current prices
    all_point_preds_usd = all_current_prices * (1 + all_point_preds)
    all_interval_preds_usd = torch.stack([
        all_current_prices * (1 + all_interval_preds[:, 0]),
        all_current_prices * (1 + all_interval_preds[:, 1])
    ], dim=1)
    
    point_target_usd = all_current_prices * (1 + point_return_target)
    min_target_usd = all_current_prices * (1 + min_return_target)
    max_target_usd = all_current_prices * (1 + max_return_target)
    hour_prices_usd = all_current_prices.unsqueeze(1) * (1 + hour_returns)
    
    # Basic data validation
    print(f"  PRECOG EVALUATION DATA (USD Prices from Relative Returns):")
    print(f"    Predictions shape: {all_point_preds_usd.shape}, {all_interval_preds_usd.shape}")
    print(f"    Targets shape: {all_targets.shape}")
    print(f"    Hour prices shape: {hour_prices_usd.shape}")
    print(f"    Point pred range: [${all_point_preds_usd.min().item():.2f}, ${all_point_preds_usd.max().item():.2f}]")
    print(f"    Interval pred range: [${all_interval_preds_usd.min().item():.2f}, ${all_interval_preds_usd.max().item():.2f}]")
    print(f"    Point target range: [${point_target_usd.min().item():.2f}, ${point_target_usd.max().item():.2f}]")
    print(f"    Current prices range: [${all_current_prices.min().item():.2f}, ${all_current_prices.max().item():.2f}]")
    
    # 1. Point Prediction Analysis (USD prices)
    point_errors = torch.abs(all_point_preds_usd - point_target_usd)
    avg_point_error = point_errors.mean().item()
    
    # Additional point metrics
    mae = torch.abs(all_point_preds_usd - point_target_usd).mean().item()
    rmse = torch.sqrt(torch.mean((all_point_preds_usd - point_target_usd) ** 2)).item()
    
    # Mean Absolute Percentage Error for USD prices
    mape = torch.mean(torch.abs((all_point_preds_usd - point_target_usd) / (point_target_usd + EPSILON))).item()
    
    # 2. Interval Analysis (USD prices)
    interval_min = all_interval_preds_usd[:, 0]
    interval_max = all_interval_preds_usd[:, 1]
    
    # Calculate width factor and inclusion factor
    effective_top = torch.minimum(interval_max, max_target_usd)
    effective_bottom = torch.maximum(interval_min, min_target_usd)
    
    interval_width = interval_max - interval_min
    width_factor = torch.where(
        interval_width <= EPSILON,
        torch.zeros_like(interval_max),
        (effective_top - effective_bottom) / (interval_width + EPSILON)
    )
    
    prices_in_bounds = torch.sum(
        (interval_min.unsqueeze(1) <= hour_prices_usd) & (hour_prices_usd <= interval_max.unsqueeze(1)),
        dim=1
    ).float()
    
    inclusion_factor = prices_in_bounds / hour_prices_usd.shape[1]
    
    # Final interval score
    interval_scores = inclusion_factor * width_factor
    avg_interval_score = interval_scores.mean().item()
    
    # 3. Comprehensive Results
    print(f"  PRECOG POINT PREDICTION (1-hour ahead USD price):")
    print(f"    Average MAE: ${avg_point_error:.2f}")
    print(f"    RMSE: ${rmse:.2f}")
    print(f"    MAPE: {mape:.4f} ({mape*100:.2f}%)")
    print(f"    Target mean: ${point_target_usd.mean().item():.2f}")
    print(f"    Pred mean: ${all_point_preds_usd.mean().item():.2f}")
    
    print(f"  PRECOG INTERVAL ANALYSIS (1-hour period USD prices):")
    print(f"    Average width factor: {width_factor.mean().item():.4f}")
    print(f"    Average inclusion factor: {inclusion_factor.mean().item():.4f}")
    print(f"    Average interval score: {avg_interval_score:.4f}")
    print(f"    Average interval width: ${interval_width.mean().item():.2f}")
    
    # Final precog score for USD prices
    precog_score = 0.5 * mape + 0.5 * (1.0 - avg_interval_score)
    print(f"  PRECOG SCORE (USD Prices): {precog_score:.6f}")
    
    # Return comprehensive metrics
    return {
        'total_loss': total_loss / len(data_loader),
        'point_mae': avg_point_error,
        'point_rmse': rmse,
        'point_mape': mape,
        'width_factor': width_factor.mean().item(),
        'inclusion_factor': inclusion_factor.mean().item(),
        'interval_score': avg_interval_score,
        'interval_width': interval_width.mean().item(),
        'precog_score': precog_score,
        'num_samples': len(all_point_preds_usd)
    } 
#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np

def challenge_loss(point_pred, interval_pred, targets, scaler=None):
    """
    Loss function that exactly matches the precog subnet scoring system from reward.py.
    
    Targets structure: [close_0, low_0, high_0, close_1, low_1, high_1, ..., close_11, low_11, high_11]
    
    The loss encourages:
    1. Accurate point predictions for the exact 1-hour ahead price
    2. Well-calibrated intervals that maximize inclusion_factor * width_factor
    """
    # Extract targets from the new structure
    # targets has shape [batch_size, 36] where 36 = 12 time steps * 3 values (close, low, high)
    # We need to derive the base targets from these detailed targets
    
    # Debug: Check target shape and values
    if torch.isnan(targets).any():
        print(f"Warning: NaN detected in targets in challenge_loss")
        print(f"  targets shape: {targets.shape}")
        print(f"  NaN count: {torch.isnan(targets).sum().item()}")
        return torch.tensor(float('inf'), device=point_pred.device)
    
    # Reshape targets to [batch_size, 12, 3] for easier processing
    targets_reshaped = targets.view(-1, 12, 3)  # [batch_size, 12, 3] for [close, low, high]
    
    # Extract base targets:
    # 1. Exact price 1 hour ahead = close price at the last time step (index 11)
    exact_price_target = targets_reshaped[:, 11, 0].clamp(min=1e-4)  # close at time step 11
    
    # 2. Min price during 1-hour period = minimum of all low prices
    min_price_target = targets_reshaped[:, :, 1].min(dim=1)[0]  # min of all low prices
    
    # 3. Max price during 1-hour period = maximum of all high prices  
    max_price_target = targets_reshaped[:, :, 2].max(dim=1)[0]  # max of all high prices
    
    # 4. All close prices for inclusion factor calculation
    hour_prices = targets_reshaped[:, :, 0]  # All close prices [batch_size, 12]
    
    # 1. Point Prediction Loss (MAPE for exact 1-hour ahead price)
    point_mape = torch.abs(point_pred - exact_price_target) / exact_price_target.clamp(min=1e-4)
    point_loss = point_mape.mean()

    # 2. Interval Loss Components (exactly matching reward.py logic)
    interval_min = interval_pred[:, 0]
    interval_max = interval_pred[:, 1]
    
    # Ensure valid intervals (min < max)
    interval_min = torch.minimum(interval_min, interval_max - 1e-4)
    interval_max = torch.maximum(interval_max, interval_min + 1e-4)
    
    # Calculate width factor (f_w) exactly as in reward.py
    effective_top = torch.minimum(interval_max, max_price_target)
    effective_bottom = torch.maximum(interval_min, min_price_target)
    
    # Handle case where pred_max == pred_min (invalid interval)
    width_factor = torch.where(
        interval_max == interval_min,
        torch.zeros_like(interval_max),
        (effective_top - effective_bottom) / (interval_max - interval_min)
    )
    
    # Calculate inclusion factor (f_i) exactly as in reward.py
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
    
    return total_loss

def test_target_structure():
    """Test the target structure and loss function"""
    print("Testing target structure and loss function...")
    
    # Create sample targets with the new structure
    # [close_0, low_0, high_0, close_1, low_1, high_1, ..., close_11, low_11, high_11]
    batch_size = 4
    horizon = 12
    
    # Create realistic price data
    base_price = 50000.0
    targets = []
    
    for batch in range(batch_size):
        batch_targets = []
        for step in range(horizon):
            # Simulate price movement
            close_price = base_price + np.random.normal(0, 100)
            low_price = close_price - np.random.uniform(50, 200)
            high_price = close_price + np.random.uniform(50, 200)
            
            batch_targets.extend([close_price, low_price, high_price])
        targets.append(batch_targets)
    
    targets = torch.FloatTensor(targets)
    print(f"Targets shape: {targets.shape}")
    print(f"Targets sample: {targets[0, :6]}")  # First 6 values (close_0, low_0, high_0, close_1, low_1, high_1)
    
    # Create sample predictions
    point_pred = torch.FloatTensor([base_price + np.random.normal(0, 50) for _ in range(batch_size)])
    interval_pred = torch.FloatTensor([
        [base_price - 100, base_price + 100] for _ in range(batch_size)
    ])
    
    print(f"Point pred shape: {point_pred.shape}")
    print(f"Interval pred shape: {interval_pred.shape}")
    
    # Test the loss function
    try:
        loss = challenge_loss(point_pred, interval_pred, targets)
        print(f"Loss computed successfully: {loss.item():.4f}")
        print("✅ Target structure and loss function working correctly!")
    except Exception as e:
        print(f"❌ Error in loss function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_target_structure() 
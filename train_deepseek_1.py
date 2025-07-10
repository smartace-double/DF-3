# ! pip install optuna
"""
Precog Subnet Training Script

This script trains a model to predict BTC prices for the precog subnet.
The model predicts:
1. Point prediction: Exact BTC price 1 hour ahead
2. Interval prediction: [min, max] range for the entire 1-hour period

Target structure: [exact_price_1h_ahead, min_price_1h_period, max_price_1h_period, 
                  close_0, low_0, high_0, close_1, low_1, high_1, ..., close_11, low_11, high_11]

Note: Detailed predictions (close, low, high for every 5 minutes) are available as a separate 
DetailedBitcoinPredictor class that can be used later for additional analysis.

This matches the precog subnet scoring system where:
- Miners predict at time T for time T+1 hour
- Validators evaluate at time T+1 hour using actual price data
- Point accuracy: How close prediction is to actual price at T+1 hour
- Interval accuracy: How well interval covers all price movements during T to T+1 hour
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from collections import deque
import random
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import os
import pickle
import json
from typing import Tuple, List, Optional
import warnings
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

EPSILON = 1e-4

def create_enhanced_study():
    # Smart sampling with warm-start capabilities
    sampler = TPESampler(
        n_startup_trials=20,  # Random search for first 20 trials
        multivariate=True,    # Consider parameter interactions
        group=True            # Group related parameters
    )
    
    # Aggressive pruning for faster convergence
    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=10,  # Set explicit max_resource instead of 'auto'
        reduction_factor=3,
        bootstrap_count=5
    )
    
    return optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name='btc_prediction_opt'
    )

# Enhanced parameter space with fixed batch size
def get_enhanced_params(trial):
    return {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),  # Reduced options for faster trials
        'num_layers': trial.suggest_int('num_layers', 1, 2),  # Reduced max layers
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),  # Narrowed range
        'batch_size': 1024,  # Fixed optimal batch size for RTX A4000
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),  # Narrowed range
        'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.1),  # Reduced options
        'epochs': trial.suggest_int('epochs', 3, 8),  # Reduced max epochs
        'grad_clip': trial.suggest_float('grad_clip', 0.5, 1.0),  # Narrowed range
        'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),
        'activation': trial.suggest_categorical('activation', ['SiLU', 'GELU'])  # Removed Mish for speed
    }

# Enhanced callback with more metrics
def enhanced_progress_callback(study, trial):
    print(f"\n{'#'*80}")
    print(f"Trial {trial.number} completed")
    
    # Handle None values safely
    if trial.value is not None:
        print(f"Current value: {trial.value:.4f}")
    else:
        print(f"Current value: None (trial failed)")
    
    if study.best_value is not None:
        print(f"Best value: {study.best_value:.4f} (Trial {study.best_trial.number})")
    else:
        print(f"Best value: None (no successful trials yet)")
    
    print("Current params:")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
    
    # Show parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nParameter importance:")
        for k, v in importance.items():
            print(f"  {k}: {v:.3f}")
    except:
        pass
    
    # Show progress
    total_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nProgress: {completed_trials}/{total_trials} trials completed")
    print(f"{'#'*80}")

# Database integration for resumable studies
storage = optuna.storages.RDBStorage(
    url="sqlite:///btc_optuna.db",
    heartbeat_interval=60,
    grace_period=120
)

# Main optimization loop
def run_optimization(train_loader, val_loader, scaler=None):
    study = create_enhanced_study()
    
    # Load previous study if exists
    try:
        study = optuna.load_study(
            study_name='btc_prediction_opt',
            storage=storage
        )
        print(f"Resuming study with {len(study.trials)} existing trials")
    except:
        pass
    
    # Progress bar for optimization
    print(f"\nStarting hyperparameter optimization...")
    print(f"Target: 50 trials with 24-hour timeout")
    
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, scaler),
        n_trials=50,  # Reduced trial count for faster optimization
        timeout=24*3600,  # 24 hours
        callbacks=[enhanced_progress_callback],
        gc_after_trial=True
    )
    
    return study

class BTCDataset:
    def __init__(self, dataset_path='datasets/structured_dataset.csv', lookback=6*12, horizon=12):
        self.dataset_path = dataset_path
        self.lookback = lookback
        self.horizon = horizon  # 12 steps = 1 hour (5-minute intervals)
        self.scaler = StandardScaler()
        self.feature_cols = []  # Will be set during preprocessing
        self.target_cols = []  # Will be set during preprocessing
        self.scaler_fitted = False

    def load_dataset(self):
        """Load and preprocess the dataset with proper train/val/test split"""
        print(f"Loading dataset from {self.dataset_path}")
        df = pd.read_csv(self.dataset_path, parse_dates=['timestamp'])

        bias = 15000
        df = df.iloc[bias:]
        
        # Debug: Check data time range
        # print(f"DEBUG: Dataset time range:")
        # print(f"  Start: {df['timestamp'].min()}")
        # print(f"  End: {df['timestamp'].max()}")
        # print(f"  Total rows: {len(df)}")
        # print(f"  Close price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")
        
        # 1. Clean the data first
        print(f"Cleaning data...")
        df = self.clean_data(df)
        
        # 2. Add features
        print(f"Adding features...")
        df = self.add_features(df)
        print(f"Features added: {df.columns.tolist()}")
        
        # 3. Split data into train/val/test BEFORE creating targets
        print(f"Splitting data into train/val/test...")
        train_df, val_df, test_df = self.split_data(df)
        
        # create targets before scaling
        train_df = self.create_targets(train_df)
        val_df = self.create_targets(val_df)
        test_df = self.create_targets(test_df)

        # 4. Fit scaler on training data only
        print(f"Fitting scaler on training data...")
        self.fit_scaler(train_df)
        # apply scaler transform to train, val and test data
        transformed_train_df = self.transform_data(train_df, self.scaler)
        transformed_val_df = self.transform_data(val_df, self.scaler)
        transformed_test_df = self.transform_data(test_df, self.scaler)
        
        
        # 5. Fit scaler on training data only (excluding target columns)
        
        # 6. Transform all datasets (excluding target columns)
        
        # 7. Create DataLoaders directly (memory efficient)
        print(f"Creating DataLoaders...")
        # Use fixed optimal batch size
        batch_size = 512
        train_loader = self.create_dataloader(transformed_train_df, batch_size=batch_size, shuffle=False)
        val_loader = self.create_dataloader(transformed_val_df, batch_size=batch_size, shuffle=False)
        test_loader = self.create_dataloader(transformed_test_df, batch_size=batch_size, shuffle=False)
        
        print(f"DataLoaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def split_data(self, df):
        """Split data into train/val/test sets"""
        total_rows = len(df)
        
        # Use 85% train, 10% val, 5% test
        train_end = int(0.85 * total_rows)
        val_end = int(0.95 * total_rows)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    def create_targets(self, df):
        """Create targets that match precog subnet requirements with detailed price information"""
        
        for i in range(self.horizon):
            # Close price at time step i+1 (1 hour ahead)
            df[f'target_close_{i}'] = df['close'].shift(-i-1)
            # Low price at time step i+1
            df[f'target_low_{i}'] = df['low'].shift(-i-1)
            # High price at time step i+1
            df[f'target_high_{i}'] = df['high'].shift(-i-1)
        
        # Drop rows with NaN targets (end of dataset)
        self.target_cols = [col for i in range(self.horizon) 
                   for col in (f'target_close_{i}', f'target_low_{i}', f'target_high_{i}')]
        
        print(f"Created {len(self.target_cols)} target columns")
        print(f"Target columns: {self.target_cols[:6]}...")  # Show first 6 columns
        
        df = df.dropna(subset=self.target_cols)
        print(f"After dropping NaN targets: {len(df)} rows remaining")
        
        return df
    
    def create_dataloader(self, df, batch_size=64, shuffle=False):
        """Create a memory-efficient DataLoader that generates sequences on-the-fly"""
        class SequenceDataset(torch.utils.data.Dataset):
            def __init__(self, df, lookback, horizon, feature_cols, target_cols):
                self.df = df
                self.lookback = lookback
                self.horizon = horizon
                self.feature_cols = feature_cols
                self.target_cols = target_cols
                self.max_i = len(df) - lookback - horizon

            
            def __len__(self):
                return max(0, self.max_i - self.lookback)
            
            def __getitem__(self, idx):
                i = idx + self.lookback
                
                # Features (lookback window) - exclude target columns
                if isinstance(self.df, pd.DataFrame):
                    feature_data = self.df[self.feature_cols].iloc[i-self.lookback:i].values
                else:
                    feature_data = self.df[i-self.lookback:i, self.feature_cols]
                
                if isinstance(self.df, pd.DataFrame):
                    target_data = self.df[self.target_cols].iloc[i].values
                else:
                    target_data = self.df[i, self.target_cols]
                
                # Debug: Check for NaN in target data
                if np.isnan(target_data).any():
                    print(f"Warning: NaN detected in target data at index {i}")
                    print(f"  Target data shape: {target_data.shape}")
                    print(f"  NaN count: {np.isnan(target_data).sum()}")
                    print(f"  Target columns: {self.target_cols[:6]}...")
                
                return torch.FloatTensor(feature_data), torch.FloatTensor(target_data)
        
        # Set feature columns if not already set
        if not self.feature_cols:
            # Handle both DataFrame and numpy array cases
            if hasattr(df, 'columns'):
                self.feature_cols = [col for col in df.columns if not col.startswith('target_')]
            else:
                # If df is a numpy array, assume all columns are features
                self.feature_cols = list(range(df.shape[1]))

        print(f"Feature columns: {self.feature_cols}")
        
        dataset = SequenceDataset(df, self.lookback, self.horizon, self.feature_cols, self.target_cols)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, persistent_workers=True)
    
        
    def clean_data(self, df):
        """Handles missing values, outliers, and normalization"""
        # 1. Handle missing values
        for col in df.columns:
            # Forward fill for technical indicators
            if col in ['rsi_14', 'MACD_12_26_9', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'obv']:
                df[col] = df[col].ffill().bfill()
            # Fill with 0 for whale features
            elif col.startswith('whale_'):
                df[col] = df[col].fillna(0)
            # Fill with mean for other numeric features
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
        
        # 2. Remove extreme outliers (99.9th percentile) - but NOT for targets
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:  # Only clip features, not targets
            upper = df[col].quantile(0.999)
            lower = df[col].quantile(0.001)
            df[col] = np.clip(df[col], lower, upper)

        # Check for missing features
        missing_features = [col for col in numeric_cols if df[col].isna().any()]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            df[missing_features] = df[missing_features].fillna(method='bfill').fillna(method='ffill')
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        if df.isna().any().any():
            raise ValueError("NaNs still present after cleaning")
            
        return df

    def fit_scaler(self, df):
        """Fit scaler on training data to avoid data leakage"""
        if not self.scaler_fitted:
            # Get feature columns (exclude target columns)
            feature_cols = [col for col in df.columns if not col.startswith('target_')]
            feature_data = df[feature_cols].values
            
            self.scaler.fit(feature_data)
            self.scaler_fitted = True
            print(f"Scaler fitted on {len(feature_data)} training samples with {len(feature_cols)} features")
            print(f"Feature columns: {feature_cols}")
    
    def transform_data(self, df, scaler=None):
        """Transform data using the fitted scaler"""
        if scaler is None:
            raise ValueError("Scaler not fitted. Please fit the scaler first.")
        
        # Get feature columns (exclude target columns)
        feature_cols = [col for col in df.columns if not col.startswith('target_')]
        feature_data = df[feature_cols].values
        
        # Transform features
        feature_data_scaled = scaler.transform(feature_data)
        
        # Create new dataframe with scaled features (keep target columns unchanged)
        df_scaled = df.copy()
        for i, col in enumerate(feature_cols):
            df_scaled.loc[:, col] = feature_data_scaled[:, i]
        
        return df_scaled

    def add_features(self, df):
        """Adds additional predictive features to the dataset"""
        
        # Extract time components
        df['date'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Time-based Features
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['month_sin'] = np.sin(df['date'].dt.month * (2 * np.pi / 12))
        df['day_of_month_sin'] = np.sin(df['date'].dt.day * (2 * np.pi / 30))
        df['year_sin'] = np.sin(df['date'].dt.year * (2 * np.pi / 4))

        # Define expected features (excluding target columns)
        ordered_column = [
            'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume',
            'whale_tx_count', 'whale_btc_volume',
            'rsi_14', 'MACD_12_26_9',
            'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'obv', 'vwap',
            'hour_sin', 'day_of_week_sin', 'month_sin', 'day_of_month_sin', 
            'year_sin'
        ]
        
        # Check if all features are present
        missing_features = [col for col in ordered_column if col not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with default values
            for col in missing_features:
                df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
        # Select only the expected features that exist in the dataset
        available_features = [col for col in ordered_column if col in df.columns]
        
        df = df[available_features]

        return df

class EnhancedBitcoinPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.1, 
                 use_layer_norm=True, activation='SiLU'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.input_size = input_size
        
        # Choose activation function
        if activation == 'SiLU':
            self.act_fn = nn.SiLU()
        elif activation == 'GELU':
            self.act_fn = nn.GELU()
        elif activation == 'Mish':
            self.act_fn = nn.Mish()
        else:
            self.act_fn = nn.SiLU()
        
        # Single LSTM encoder for all features (more flexible)
        self.feature_encoder = nn.LSTM(input_size, hidden_size, num_layers, 
                                      batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Layer normalization
        if use_layer_norm:
            self.feature_norm = nn.LayerNorm(hidden_size)

        # Prediction heads
        self.point_head = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # self.act_fn,
            # nn.Dropout(dropout),
            # nn.Linear(hidden_size, hidden_size//2),
            # self.act_fn,
            nn.Linear(hidden_size, 1)
        )
        
        self.interval_head = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # self.act_fn,
            # nn.Dropout(dropout),
            #   nn.Linear(hidden_size, hidden_size//2),
            # self.act_fn,
            nn.Linear(hidden_size, 2),  # min and max
        )
        
    def forward(self, x):
        # Encode all features together
        encoded, _ = self.feature_encoder(x)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            encoded = self.feature_norm(encoded)
        
        # Use the last timestep for prediction
        final_encoding = encoded[:, -1]
        
        # Predictions
        point_pred = self.point_head(final_encoding)
        interval_pred = self.interval_head(final_encoding)
        
        # Ensure min < max in interval by sorting
        interval_pred = torch.sort(interval_pred, dim=-1)[0]
        
        return point_pred.squeeze(), interval_pred

class DetailedBitcoinPredictor(nn.Module):
    """
    Separate predictor class for detailed price predictions.
    This can be used later for predicting close, low, high at each time step.
    """
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.1, 
                 use_layer_norm=True, activation='SiLU'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.input_size = input_size
        
        # Choose activation function
        if activation == 'SiLU':
            self.act_fn = nn.SiLU()
        elif activation == 'GELU':
            self.act_fn = nn.GELU()
        elif activation == 'Mish':
            self.act_fn = nn.Mish()
        else:
            self.act_fn = nn.SiLU()
        
        # LSTM encoder for all features
        self.feature_encoder = nn.LSTM(input_size, hidden_size, num_layers, 
                                      batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Layer normalization
        if use_layer_norm:
            self.feature_norm = nn.LayerNorm(hidden_size)

        # Detailed prediction head for close, low, high at each time step
        self.detailed_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            self.act_fn,
            nn.Linear(hidden_size//2, 3 * 12)  # 3 values (close, low, high) * 12 time steps
        )
        
    def forward(self, x):
        # Encode all features together
        encoded, _ = self.feature_encoder(x)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            encoded = self.feature_norm(encoded)
        
        # Use the last timestep for prediction
        final_encoding = encoded[:, -1]
        
        # Detailed predictions
        detailed_pred = self.detailed_head(final_encoding)
        
        # Reshape detailed predictions to [batch_size, 12, 3] (12 time steps, 3 values each)
        detailed_pred = detailed_pred.view(-1, 12, 3)  # [batch_size, 12, 3] for [close, low, high]
        
        return detailed_pred

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
    exact_price_target = targets_reshaped[:, 11, 0].clamp(min=EPSILON)  # close at time step 11
    
    # 2. Min price during 1-hour period = minimum of all low prices
    min_price_target = targets_reshaped[:, :, 1].min(dim=1)[0]  # min of all low prices
    
    # 3. Max price during 1-hour period = maximum of all high prices  
    max_price_target = targets_reshaped[:, :, 2].max(dim=1)[0]  # max of all high prices
    
    # 4. All close prices for inclusion factor calculation
    hour_prices = targets_reshaped[:, :, :].view(-1, 36)  # All high, low, close prices [batch_size, 36]
    
    # Convert scaled predictions back to original scale for price-related features
    if scaler is not None:
        # Cache scaler parameters on GPU to avoid repeated CPU-GPU transfers
        if not hasattr(challenge_loss, '_scaler_cache'):
            challenge_loss._scaler_cache = {
                'mean': torch.FloatTensor(scaler.mean_).to(point_pred.device),
                'scale': torch.FloatTensor(scaler.scale_).to(point_pred.device),
                'price_indices': torch.tensor([3, 1, 2], device=point_pred.device)  # [close, high, low]
            }
        
        cache = challenge_loss._scaler_cache
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

    # print(f"Interval min: {interval_min}")
    # print(f"Interval max: {interval_max}")
    # print(f"Interval max - interval min: {interval_max - interval_min}")
    # print(f"Effective top: {effective_top}")
    # print(f"Effective bottom: {effective_bottom}")
    # print(f"Effective top - effective bottom: {effective_top - effective_bottom}")


    # Debug prints removed to prevent script from stopping
    
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

    # TODO: Add a penalty for invalid prediction outside the interval
    total_loss = total_loss + 10 * (point_pred < min_price_target).float().mean() + 10 * (point_pred > max_price_target).float().mean()
    
    return total_loss

def detailed_loss(detailed_pred, targets):
    """
    Loss function for detailed predictions.
    
    Args:
        detailed_pred: [batch_size, 12, 3] predictions for close, low, high at each time step
        targets: [batch_size, 36] targets with detailed close, low, high for each time step
    
    Returns:
        MSE loss for detailed predictions
    """
    # Reshape targets to [batch_size, 12, 3] to match detailed_pred
    detailed_targets = targets.view(-1, 12, 3)  # [batch_size, 12, 3] for [close, low, high]
    
    # Calculate MSE loss for detailed predictions
    return F.mse_loss(detailed_pred, detailed_targets)

def train_detailed_predictor(train_loader, val_loader, params, save_dir='models/detailed', scaler=None):
    """
    Example function showing how to train the DetailedBitcoinPredictor.
    This can be used later when you need detailed predictions.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        params: Model parameters
        save_dir: Directory to save the model
        scaler: Scaler for inverse transformation
    
    Returns:
        Trained DetailedBitcoinPredictor model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get input size from first batch
    for batch_X, batch_y in train_loader:
        input_size = batch_X.shape[2]
        break
    
    # Initialize detailed predictor
    model = DetailedBitcoinPredictor(
        input_size=input_size,
        hidden_size=params.get('hidden_size', 256),
        num_layers=params.get('num_layers', 3),
        dropout=params.get('dropout', 0.1),
        use_layer_norm=params.get('use_layer_norm', True),
        activation=params.get('activation', 'SiLU')
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=params.get('lr', 1e-4), 
                            weight_decay=params.get('weight_decay', 1e-5))
    
    # Training loop
    best_val_loss = float('inf')
    epochs = params.get('epochs', 10)
    
    print(f"Training detailed predictor for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training with progress bar
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Train)', leave=False, ncols=100)
        
        for batch_X, batch_y in train_pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            detailed_pred = model(batch_X)
            loss = detailed_loss(detailed_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation with progress bar
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Val)', leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                detailed_pred = model(batch_X)
                loss = detailed_loss(detailed_pred, batch_y)
                val_loss += loss.item()
                
                # Update progress bar
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'detailed_model.pth'))
    
    return model

def train_with_cv(train_loader, val_loader, params, save_dir='models', trial_name=None, scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enable GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    else:
        print("Using CPU")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get input size from first batch
    input_size = None
    for batch_X, batch_y in train_loader:
        input_size = batch_X.shape[2]
        break
    
    if input_size is None:
        print("Error: No valid batches found in train_loader")
        return float('inf')
    
    # Initialize model
    model = EnhancedBitcoinPredictor(
        input_size=input_size,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params.get('dropout', 0.1),
        use_layer_norm=params.get('use_layer_norm', True),
        activation=params.get('activation', 'SiLU')
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], 
                            weight_decay=params['weight_decay'])
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=params['lr'],
        steps_per_epoch=len(train_loader),
        epochs=params['epochs']
    )
    
    # Mixed precision training for better GPU efficiency
    grad_scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    best_val_score = float('inf')
    early_stopping = EarlyStopping(patience=3)  # More aggressive early stopping for speed
    train_losses = []
    val_scores = []
    
    print(f"Starting training for {params['epochs']} epochs...")
    print(f"Batch size: {params['batch_size']}, Learning rate: {params['lr']:.6f}")
    
    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0
        batch_count = 0
        
        # Training phase with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{params["epochs"]} (Train)', 
                         leave=False, ncols=100)
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_pbar):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Debug: Check input data
            if torch.isnan(batch_X).any():
                print(f"Warning: NaN detected in input features")
                print(f"  batch_X shape: {batch_X.shape}")
                print(f"  NaN count: {torch.isnan(batch_X).sum().item()}")
                continue
            
            # Debug: Check target data
            if torch.isnan(batch_y).any():
                print(f"Warning: NaN detected in target data")
                print(f"  batch_y shape: {batch_y.shape}")
                print(f"  NaN count: {torch.isnan(batch_y).sum().item()}")
                continue
            
            # Mixed precision training
            if grad_scaler is not None:
                with autocast():
                    point_pred, interval_pred = model(batch_X)
                    
                    # Debug: Check model output
                    if torch.isnan(point_pred).any() or torch.isnan(interval_pred).any():
                        print(f"Warning: NaN detected in model predictions")
                        print(f"  point_pred shape: {point_pred.shape}")
                        print(f"  interval_pred shape: {interval_pred.shape}")
                        print(f"  point_pred NaN count: {torch.isnan(point_pred).sum().item()}")
                        print(f"  interval_pred NaN count: {torch.isnan(interval_pred).sum().item()}")
                        print(f"  batch_X range: [{batch_X.min().item():.4f}, {batch_X.max().item():.4f}]")
                        print(f"  batch_y range: [{batch_y.min().item():.4f}, {batch_y.max().item():.4f}]")
                        continue
                    
                    # Challenge-specific loss with scaler for real price evaluation
                    loss = challenge_loss(point_pred, interval_pred, batch_y, scaler)
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss detected: {loss.item()}")
                        print(f"  Point pred range: [{point_pred.min().item():.4f}, {point_pred.max().item():.4f}]")
                        print(f"  Interval pred range: [{interval_pred.min().item():.4f}, {interval_pred.max().item():.4f}]")
                        print(f"  Targets range: [{batch_y.min().item():.4f}, {batch_y.max().item():.4f}]")
                        continue
                
                # Scale loss and backward pass
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.get('grad_clip', 1.0))
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                point_pred, interval_pred = model(batch_X)
                
                # Debug: Check model output
                if torch.isnan(point_pred).any() or torch.isnan(interval_pred).any():
                    print(f"Warning: NaN detected in model predictions")
                    print(f"  point_pred shape: {point_pred.shape}")
                    print(f"  interval_pred shape: {interval_pred.shape}")
                    print(f"  point_pred NaN count: {torch.isnan(point_pred).sum().item()}")
                    print(f"  interval_pred NaN count: {torch.isnan(interval_pred).sum().item()}")
                    print(f"  batch_X range: [{batch_X.min().item():.4f}, {batch_X.max().item():.4f}]")
                    print(f"  batch_y range: [{batch_y.min().item():.4f}, {batch_y.max().item():.4f}]")
                    continue
                
                # Challenge-specific loss with scaler for real price evaluation
                loss = challenge_loss(point_pred, interval_pred, batch_y, scaler)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected: {loss.item()}")
                    print(f"  Point pred range: [{point_pred.min().item():.4f}, {point_pred.max().item():.4f}]")
                    print(f"  Interval pred range: [{interval_pred.min().item():.4f}, {interval_pred.max().item():.4f}]")
                    print(f"  Targets range: [{batch_y.min().item():.4f}, {batch_y.max().item():.4f}]")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.get('grad_clip', 1.0))
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{train_loss/batch_count:.4f}'
            })
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        # Validation with progress bar
        val_score = evaluate_with_progress(model, val_loader, device, scaler)
        avg_train_loss = train_loss / batch_count
        
        # Track losses for monitoring
        train_losses.append(avg_train_loss)
        val_scores.append(val_score)
        
        # Print epoch progress
        print(f"  Epoch {epoch+1}/{params['epochs']}: "
              f"Train Loss = {avg_train_loss:.4f}, Val Score = {val_score:.4f}")
        
        # Check for overfitting (val score increasing while train loss decreasing)
        if len(val_scores) > 3:
            recent_val_trend = val_scores[-3:]
            recent_train_trend = train_losses[-3:]
            if (recent_val_trend[-1] > recent_val_trend[0] and 
                recent_train_trend[-1] < recent_train_trend[0]):
                print(f"  Warning: Potential overfitting detected at epoch {epoch+1}")
        
        # Early stopping
        early_stopping(val_score)
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        if val_score < best_val_score:
            best_val_score = val_score
            print(f"  New best validation score: {best_val_score:.4f}")
            # Save best model
            fold_dir = os.path.join(save_dir, 'best_model')
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pth'))
    
    # Save training history
    fold_dir = os.path.join(save_dir, 'best_model')
    history = {
        'train_losses': train_losses,
        'val_scores': val_scores,
        'best_epoch': len(val_scores) - early_stopping.counter if early_stopping.early_stop else len(val_scores)
    }
    with open(os.path.join(fold_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    mean_score = np.mean(val_scores)
    std_score = np.std(val_scores)
    print(f"\n{'='*60}")
    print(f"TRIAL COMPLETED: {trial_name}")
    print(f"Mean CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Individual fold scores: {[f'{score:.4f}' for score in val_scores]}")
    print(f"{'='*60}\n")
    
    return mean_score

def evaluate_with_progress(model, data_loader, device, scaler=None):
    """
    Comprehensive evaluation system with progress bar that exactly matches the precog subnet scoring system from reward.py.
    
    Targets structure: [close_0, low_0, high_0, close_1, low_1, high_1, ..., close_11, low_11, high_11]
    """
    model.eval()
    total_loss = 0
    all_point_preds = []
    all_interval_preds = []
    all_targets = []
    
    # Progress bar for evaluation
    eval_pbar = tqdm(data_loader, desc='Evaluation', leave=False, ncols=100)
    
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
            
            # Challenge-specific loss with scaler for real price evaluation
            loss = challenge_loss(point_pred, interval_pred, batch_y, scaler)
            
            # Collect predictions for detailed analysis (even if loss is NaN)
            all_point_preds.append(point_pred.cpu())
            all_interval_preds.append(interval_pred.cpu())
            all_targets.append(batch_y.cpu())
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss in evaluation: {loss.item()}")
                print(f"  Point pred range: [{point_pred.min().item():.4f}, {point_pred.max().item():.4f}]")
                print(f"  Interval pred range: [{interval_pred.min().item():.4f}, {interval_pred.max().item():.4f}]")
                print(f"  Targets range: [{batch_y.min().item():.4f}, {batch_y.max().item():.4f}]")
                continue
                
            total_loss += loss.item()
            
            # Update progress bar
            eval_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Check if we have any valid predictions
    if len(all_point_preds) == 0:
        print("Warning: No valid predictions collected during evaluation")
        return float('inf')
    
    # Concatenate all predictions (efficient GPU operation)
    all_point_preds = torch.cat(all_point_preds, dim=0)
    all_interval_preds = torch.cat(all_interval_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Extract targets exactly as in reward.py
    # targets has shape [batch_size, 36] where 36 = 12 time steps * 3 values (close, low, high)
    # We need to derive the base targets from these detailed targets
    
    # Reshape targets to [batch_size, 12, 3] for easier processing
    targets_reshaped = all_targets.view(-1, 12, 3)  # [batch_size, 12, 3] for [close, low, high]
    
    # Extract base targets:
    # 1. Exact price 1 hour ahead = close price at the last time step (index 11)
    exact_price_target = targets_reshaped[:, 11, 0]  # close at time step 11
    
    # 2. Min price during 1-hour period = minimum of all low prices
    min_price_target = targets_reshaped[:, :, 1].min(dim=1)[0]  # min of all low prices
    
    # 3. Max price during 1-hour period = maximum of all high prices  
    max_price_target = targets_reshaped[:, :, 2].max(dim=1)[0]  # max of all high prices
    
    # 4. All close prices for inclusion factor calculation
    hour_prices = targets_reshaped[:, :, :].view(-1, 36)  # All high, low, close prices [batch_size, 36]
    
    # Basic data validation
    print(f"  DATA VALIDATION:")
    print(f"    Predictions shape: {all_point_preds.shape}, {all_interval_preds.shape}")
    print(f"    Targets shape: {all_targets.shape}")
    print(f"    Hour prices shape: {hour_prices.shape}")
    print(f"    Point pred range: [{all_point_preds.min().item():.2f}, {all_point_preds.max().item():.2f}]")
    print(f"    Interval pred range: [{all_interval_preds.min().item():.2f}, {all_interval_preds.max().item():.2f}]")
    print(f"    Exact target range: [{exact_price_target.min().item():.2f}, {exact_price_target.max().item():.2f}]")
    
    # 1. Point Prediction Analysis (exactly as in reward.py)
    # current_point_error = abs(prediction_value - actual_price) / actual_price
    point_errors = torch.abs(all_point_preds - exact_price_target) / exact_price_target.clamp(min=EPSILON)
    avg_point_error = point_errors.mean().item()
    
    # Additional point metrics
    mae = torch.abs(all_point_preds - exact_price_target).mean().item()
    rmse = torch.sqrt(torch.mean((all_point_preds - exact_price_target) ** 2)).item()
    
    # 2. Interval Analysis (exactly as in reward.py)
    interval_min = all_interval_preds[:, 0]
    interval_max = all_interval_preds[:, 1]
    
    # Calculate width factor (f_w) exactly as in reward.py with numerical stability
    effective_top = torch.minimum(interval_max, max_price_target)
    effective_bottom = torch.maximum(interval_min, min_price_target)
    
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
    prices_in_bounds = torch.sum(
        (interval_min.unsqueeze(1) <= hour_prices) & (hour_prices <= interval_max.unsqueeze(1)),
        dim=1
    ).float()
    
    inclusion_factor = prices_in_bounds / hour_prices.shape[1]
    
    # Final interval score is the product (exactly as in reward.py)
    interval_scores = inclusion_factor * width_factor
    avg_interval_score = interval_scores.mean().item()
    
    # 3. Diagnostic Information
    print(f"  POINT PREDICTION (1-hour ahead):")
    print(f"    Average Error: {avg_point_error:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"    Target mean: {exact_price_target.mean().item():.2f}, Pred mean: {all_point_preds.mean().item():.2f}")
    
    print(f"  INTERVAL ANALYSIS (1-hour period):")
    print(f"    Average width factor: {width_factor.mean().item():.4f}")
    print(f"    Average inclusion factor: {inclusion_factor.mean().item():.4f}")
    print(f"    Average interval score: {avg_interval_score:.4f}")
    
    point_error = avg_point_error
    interval_score = avg_interval_score
    
    subnet_score = 0.5 * point_error + 0.5 * (1.0 - interval_score)
    print(f"  PRECOG SUBNET SCORE: {subnet_score:.4f}")
    
    issues = []
    if avg_point_error > 0.5:
        issues.append("High point error - predictions are poor")
    if width_factor.mean().item() < 0.1:
        issues.append("Low width factor - intervals too narrow or don't overlap with observed range")
    if inclusion_factor.mean().item() < 0.3:
        issues.append("Low inclusion factor - intervals don't cover many price points")
    if avg_interval_score < 0.1:
        issues.append("Very low interval score")
    
    if issues:
        print(f"  ISSUES DETECTED:")
        for issue in issues:
            print(f"    ⚠️  {issue}")
    else:
        print(f"  ✅ No major issues detected")
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device, scaler=None):
    """
    Wrapper function that calls evaluate_with_progress for consistency.
    """
    return evaluate_with_progress(model, data_loader, device, scaler)

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

class ContinuousLearner:
    def __init__(self, model_path=None, feature_cols=None, scaler=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if feature_cols is None:
            # Default feature count if not provided
            input_size = 24  # Based on expected features (excluding targets)
        else:
            input_size = len(feature_cols)
            
        self.model = EnhancedBitcoinPredictor(input_size=input_size)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        self.buffer = deque(maxlen=10000)
        self.steps = 0
        self.feature_cols = feature_cols
        self.scaler = scaler
        
    def update(self, new_data: pd.DataFrame):
        """Update model with new data"""
        # Preprocess new data
        dataset = BTCDataset()
        train_loader, val_loader, test_loader = dataset.load_dataset()
        
        # Add to buffer (get a few samples from train_loader)
        sample_count = 0
        for batch_X, batch_y in train_loader:
            if sample_count >= 100:  # Limit samples to prevent memory issues
                break
            for i in range(batch_X.shape[0]):
                self.buffer.append((
                    batch_X[i:i+1],
                    batch_y[i:i+1]
                ))
                sample_count += 1
        
        # Online training step
        if len(self.buffer) >= 32:  # Minimum batch size
            self.online_train()
            
    def online_train(self):
        self.model.train()
        
        # Sample batch
        batch = random.sample(self.buffer, min(32, len(self.buffer)))
        batch_X = torch.cat([x for x, _ in batch]).to(self.device)
        batch_y = torch.cat([y for _, y in batch]).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        point_pred, interval_pred = self.model(batch_X)
        
        # Loss calculation with scaler for real price evaluation
        loss = challenge_loss(point_pred, interval_pred, batch_y, self.scaler)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        # Periodic validation
        if self.steps % 100 == 0:
            self.validate()
    
    def validate(self):
        """Validate on recent data"""
        if len(self.buffer) < 10:
            return
            
        # Use recent data for validation
        val_data = list(self.buffer)[-100:]  # Last 100 samples
        val_X = torch.cat([x for x, _ in val_data]).to(self.device)
        val_y = torch.cat([y for _, y in val_data]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            point_pred, interval_pred = self.model(val_X)
            
            # Calculate metrics
            loss = challenge_loss(point_pred, interval_pred, val_y, self.scaler)

            print(f"Validation loss: {loss.item()}")
            
        self.model.train()
    
    def predict(self, X):
        """Make predictions on new data"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            point_pred, interval_pred = self.model(X_tensor)
            return point_pred.cpu().numpy(), interval_pred.cpu().numpy()

def objective(trial, train_loader, val_loader, scaler=None) -> float:
    params = get_enhanced_params(trial)
    
    trial_name = f"Trial_{trial.number}"
    
    try:
        print(f"\n{'*'*80}")
        print(f"STARTING {trial_name}")
        print(f"Parameters: {params}")
        print(f"{'*'*80}")
        
        # Train with CV
        cv_score = train_with_cv(train_loader, val_loader, params, save_dir='models', trial_name=trial_name, scaler=scaler)
        
        print(f"{trial_name} completed successfully with score: {cv_score:.4f}")
        return float(cv_score)
    except Exception as e:
        print(f"{trial_name} FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

if __name__ == "__main__":
    # Run test first
    print("\nStarting hyperparameter optimization...")
    print(f"Target: 50 trials with 24-hour timeout")
    print(f"Each trial: Variable epochs with early stopping")
    
    # Load dataset ONCE with proper train/val/test split
    print("Loading dataset...")
    dataset = BTCDataset(dataset_path='datasets/structured_dataset.csv')
    train_loader, val_loader, test_loader = dataset.load_dataset()
    
    print(f"DataLoaders created successfully")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check if DataLoaders have data
    if len(train_loader) == 0 or len(val_loader) == 0:
        print("Error: Empty DataLoaders detected")
        exit(1)

    # Clear some memory
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Use enhanced optimization with scaler for real price evaluation
    try:
        study = run_optimization(train_loader, val_loader, dataset.scaler)
        
        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print("="*80)
        
        if study.best_trial is None:
            print("No successful trials completed")
            exit(1)
            
        print("Best trial:")
        trial = study.best_trial
        print(f"  Trial Number: {trial.number}")
        print(f"  CV Score: {trial.value}")
        print("  Best Parameters: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    except Exception as e:
        print(f"Optimization failed: {e}")
        exit(1)
    
    # Save best parameters
    os.makedirs('models', exist_ok=True)
    with open('models/best_params.json', 'w') as f:
        json.dump(trial.params, f, indent=2)
    
    print("\nTraining final model with best parameters...")
    # Train final model with best parameters
    final_params = trial.params.copy()
    # Use the same epochs as optimization to maintain consistency
    print(f"Final training epochs: {final_params['epochs']} (same as optimization, with early stopping)")
    
    final_score = train_with_cv(train_loader, val_loader, final_params, save_dir='models', trial_name="FINAL_MODEL", scaler=dataset.scaler)
    print(f"Final CV Score: {final_score}")
    
    # Evaluate on validation set
    print(f"\nEvaluating on validation set...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get input size from first batch
    for batch_X, batch_y in train_loader:
        input_size = batch_X.shape[2]
        break
    
    # Load best model
    model = EnhancedBitcoinPredictor(
        input_size=input_size,
        hidden_size=final_params['hidden_size'],
        num_layers=final_params['num_layers'],
        dropout=final_params.get('dropout', 0.1),
        use_layer_norm=final_params.get('use_layer_norm', True),
        activation=final_params.get('activation', 'SiLU')
    ).to(device)
    
    # Load the best model
    model_path = 'models/best_model/model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        val_score = evaluate(model, val_loader, device, dataset.scaler)
        print(f"Validation Score: {val_score:.4f}")
    else:
        print("No saved model found for validation evaluation")
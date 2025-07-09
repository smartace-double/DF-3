# ! pip install optuna
"""
Precog Subnet Training Script

This script trains a model to predict BTC prices for the precog subnet.
The model predicts:
1. Point prediction: Exact BTC price 1 hour ahead
2. Interval prediction: [min, max] range for the entire 1-hour period

Target structure: [exact_price_1h_ahead, min_price_1h_period, max_price_1h_period]

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
warnings.filterwarnings('ignore')

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

# Enhanced parameter space
def get_enhanced_params(trial):
    return {
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512, 768]),
        'num_layers': trial.suggest_int('num_layers', 1, 6),
        'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
        'epochs': trial.suggest_int('epochs', 2, 5),  # Variable epochs
        'grad_clip': trial.suggest_float('grad_clip', 0.1, 1.0),
        'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),
        'activation': trial.suggest_categorical('activation', ['SiLU', 'GELU', 'Mish'])
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
    
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, scaler),
        n_trials=1,  # Increased trial count
        timeout=48*3600,  # 48 hours
        callbacks=[enhanced_progress_callback],
        gc_after_trial=True
    )
    
    return study

class BTCDataset:
    def __init__(self, dataset_path='datasets/structured_dataset.csv', lookback=2*12, horizon=12):
        self.dataset_path = dataset_path
        self.lookback = lookback
        self.horizon = horizon  # 12 steps = 1 hour (5-minute intervals)
        self.scaler = StandardScaler()
        self.feature_cols = []  # Will be set during preprocessing
        self.scaler_fitted = False

    def load_dataset(self):
        """Load and preprocess the dataset with proper train/val/test split"""
        print(f"Loading dataset from {self.dataset_path}")
        df = pd.read_csv(self.dataset_path, parse_dates=['timestamp'])
        
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
        
        # 4. Fit scaler on training data only
        print(f"Fitting scaler on training data...")
        self.fit_scaler(train_df)
        
        # 5. Transform all datasets
        print(f"Transforming data...")
        train_df_scaled = self.transform_data(train_df)
        val_df_scaled = self.transform_data(val_df)
        test_df_scaled = self.transform_data(test_df)
        
        # 6. Create targets AFTER scaling (to avoid data leakage)
        print(f"Creating targets...")
        train_df_with_targets = self.create_targets(train_df_scaled)
        val_df_with_targets = self.create_targets(val_df_scaled)
        test_df_with_targets = self.create_targets(test_df_scaled)
        
        # 7. Create DataLoaders directly (memory efficient)
        print(f"Creating DataLoaders...")
        train_loader = self.create_dataloader(train_df_with_targets, batch_size=32, shuffle=True)  # Reduced batch size
        val_loader = self.create_dataloader(val_df_with_targets, batch_size=32, shuffle=False)    # Reduced batch size
        test_loader = self.create_dataloader(test_df_with_targets, batch_size=32, shuffle=False)  # Reduced batch size
        
        print(f"DataLoaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def split_data(self, df):
        """Split data into train/val/test sets"""
        total_rows = len(df)
        
        # Use 70% train, 15% val, 15% test
        train_end = int(0.85 * total_rows)
        val_end = int(0.95 * total_rows)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    def create_targets(self, df):
        """Create targets that match precog subnet requirements"""
        # 1. Point target: exact price 1 hour ahead (horizon steps)
        df['target_close'] = df['close'].shift(-self.horizon)
        
        # 2. Interval targets: min and max during the 1-hour period
        # Calculate rolling min/max over the next hour (horizon steps)
        df['target_low'] = df['low'].rolling(window=self.horizon, min_periods=1).min().shift(-self.horizon)
        df['target_high'] = df['high'].rolling(window=self.horizon, min_periods=1).max().shift(-self.horizon)
        
        # 3. Add all price points within the 1-hour period for inclusion factor calculation
        # We need to collect all close prices within the next hour for each timestep
        for i in range(self.horizon):
            df[f'target_price_{i}'] = df['close'].shift(-i-1)
        
        # Drop rows with NaN targets (end of dataset)
        target_cols = ['target_close', 'target_low', 'target_high'] + [f'target_price_{i}' for i in range(self.horizon)]
        df = df.dropna(subset=target_cols)
        
        return df
    
    def create_dataloader(self, df, batch_size=64, shuffle=True):
        """Create a memory-efficient DataLoader that generates sequences on-the-fly"""
        class SequenceDataset(torch.utils.data.Dataset):
            def __init__(self, df, lookback, horizon, feature_cols):
                self.df = df
                self.lookback = lookback
                self.horizon = horizon
                self.feature_cols = feature_cols
                self.max_i = len(df) - lookback - horizon
                # Limit to first 1000 samples to prevent memory issues
                self.max_i = min(1000, self.max_i)
            
            def __len__(self):
                # Limit to 1000 samples to prevent memory issues
                return min(1000, max(0, self.max_i - self.lookback))
            
            def __getitem__(self, idx):
                i = idx + self.lookback
                
                # Features (lookback window) - exclude target columns
                feature_data = self.df[self.feature_cols].iloc[i-self.lookback:i].values
                
                # Targets: [exact_price_1h_ahead, min_price_1h_period, max_price_1h_period, all_prices_1h_period...]
                target_data = self.df[['target_close', 'target_low', 'target_high'] + [f'target_price_{j}' for j in range(self.horizon)]].iloc[i].values
                
                return torch.FloatTensor(feature_data), torch.FloatTensor(target_data)
        
        # Set feature columns if not already set
        if not self.feature_cols:
            self.feature_cols = [col for col in df.columns if not col.startswith('target_')]
        
        dataset = SequenceDataset(df, self.lookback, self.horizon, self.feature_cols)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    def create_sequences(self, df):
        """Create time series sequences (kept for backward compatibility)"""
        X, y = [], []
        
        # Ensure we don't go out of bounds
        max_i = len(df) - self.lookback - self.horizon
        
        for i in range(self.lookback, max_i): 
            # Features (lookback window) - exclude target columns
            feature_cols = [col for col in df.columns if not col.startswith('target_')]
            if not self.feature_cols:  # Set once
                self.feature_cols = feature_cols
            
            feature_data = df[self.feature_cols].iloc[i-self.lookback:i].values
            X.append(feature_data)
            
            # Targets: [exact_price_1h_ahead, min_price_1h_period, max_price_1h_period, all_prices_1h_period...]
            # Get targets at position i (current time), which were created for i+horizon
            target_data = df[['target_close', 'target_low', 'target_high'] + [f'target_price_{j}' for j in range(self.horizon)]].iloc[i].values
            y.append(target_data)
        # convert to numpy arrays and tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        return X, y
        
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
    
    def transform_data(self, df):
        """Transform data using fitted scaler"""
        if not self.scaler_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        
        # Get feature columns (exclude target columns)
        feature_cols = [col for col in df.columns if not col.startswith('target_')]
        feature_data = df[feature_cols].values
        
        # Transform features
        feature_data_scaled = self.scaler.transform(feature_data)
        
        # Create new dataframe with scaled features
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
            nn.Linear(hidden_size, hidden_size),
            self.act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            self.act_fn,
            nn.Linear(hidden_size//2, 1)
        )
        
        self.interval_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            self.act_fn,
            nn.Linear(hidden_size//2, 2),  # min and max
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

def challenge_loss(point_pred, interval_pred, targets, scaler=None):
    """
    Loss function that exactly matches the precog subnet scoring system from reward.py.
    
    Targets structure: [exact_price_1h_ahead, min_price_1h_period, max_price_1h_period, price_1, price_2, ..., price_12]
    
    The loss encourages:
    1. Accurate point predictions for the exact 1-hour ahead price
    2. Well-calibrated intervals that maximize inclusion_factor * width_factor
    """
    # Extract targets
    exact_price_target = targets[:, 0].clamp(min=1e-4)  # Price at exactly 1 hour ahead
    min_price_target = targets[:, 1]  # Minimum price during the 1-hour period
    max_price_target = targets[:, 2]  # Maximum price during the 1-hour period
    
    # Extract all price points within the 1-hour period (for inclusion factor calculation)
    hour_prices = targets[:, 3:]  # All price points within the hour
    
    # Convert scaled predictions back to original scale for price-related features
    if scaler is not None:
        # Get scaler parameters for manual inverse transform
        mean_ = torch.FloatTensor(scaler.mean_).to(point_pred.device)
        scale_ = torch.FloatTensor(scaler.scale_).to(point_pred.device)
        
        # Get the indices of price-related features (close, high, low)
        # We know the order of features from the dataset: ['open', 'high', 'low', 'close', ...]
        # So close=3, high=1, low=2 (0-indexed)
        price_indices = [3, 1, 2]  # [close, high, low] indices
        
        # Debug: Print scaler info
        print(f"DEBUG: Scaler mean shape: {mean_.shape}, scale shape: {scale_.shape}")
        print(f"DEBUG: Price indices: {price_indices}")
        print(f"DEBUG: Before inverse transform - point_pred range: [{point_pred.min().item():.4f}, {point_pred.max().item():.4f}]")
        print(f"DEBUG: Before inverse transform - exact_price_target range: [{exact_price_target.min().item():.4f}, {exact_price_target.max().item():.4f}]")
        
        # Manual inverse transform for price predictions (maintains gradients)
        point_pred_unscaled = point_pred * scale_[price_indices[0]] + mean_[price_indices[0]]
        interval_min_unscaled = interval_pred[:, 0] * scale_[price_indices[1]] + mean_[price_indices[1]]
        interval_max_unscaled = interval_pred[:, 1] * scale_[price_indices[2]] + mean_[price_indices[2]]
        
        # Manual inverse transform for targets
        exact_price_target_unscaled = exact_price_target * scale_[price_indices[0]] + mean_[price_indices[0]]
        min_price_target_unscaled = min_price_target * scale_[price_indices[1]] + mean_[price_indices[1]]
        max_price_target_unscaled = max_price_target * scale_[price_indices[2]] + mean_[price_indices[2]]
        
        # Manual inverse transform for hour prices
        hour_prices_unscaled = hour_prices * scale_[price_indices[0]].unsqueeze(0) + mean_[price_indices[0]].unsqueeze(0)
        
        # Debug: Print after inverse transform
        print(f"DEBUG: After inverse transform - point_pred range: [{point_pred_unscaled.min().item():.4f}, {point_pred_unscaled.max().item():.4f}]")
        print(f"DEBUG: After inverse transform - exact_price_target range: [{exact_price_target_unscaled.min().item():.4f}, {exact_price_target_unscaled.max().item():.4f}]")
        
        # Use unscaled values for calculations
        point_pred = point_pred_unscaled
        interval_pred = torch.stack([interval_min_unscaled, interval_max_unscaled], dim=1)
        exact_price_target = exact_price_target_unscaled
        min_price_target = min_price_target_unscaled
        max_price_target = max_price_target_unscaled
        hour_prices = hour_prices_unscaled
    
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
    
    # Handle case where pred_max == pred_min (invalid interval)
    width_factor = torch.where(
        interval_max == interval_min,
        torch.zeros_like(interval_max),
        (effective_top - effective_bottom) / (interval_max - interval_min)
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
    
    return total_loss


def train_with_cv(train_loader, val_loader, params, save_dir='models', trial_name=None, scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    
    fold_scores = []
    
    # print(f"\n{'='*60}")
    # print(f"TRIAL: {trial_name}")
    # print(f"Parameters: {params}")
    # print(f"{'='*60}")
    
    # For now, we'll use a single fold since we're working with DataLoaders
    # In a full implementation, you'd create multiple DataLoaders for different folds
    
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
    
    # Training loop
    best_val_score = float('inf')
    early_stopping = EarlyStopping(patience=5)  # More aggressive early stopping
    train_losses = []
    val_scores = []
    
    # print(f"Starting training for {params['epochs']} epochs...")
    # print(f"Batch size: {params['batch_size']}, Learning rate: {params['lr']:.6f}")
    
    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0
        batch_count = 0
        
        # Training phase
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            point_pred, interval_pred = model(batch_X)
            
            # Use challenge loss function with scaler for real price evaluation
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
            
            # Print batch progress every 10 batches
            if batch_idx % 10 == 0:
                # print(f"  Epoch {epoch+1}/{params['epochs']}, Batch {batch_idx+1}/{len(train_loader)}, "
                #       f"Loss: {loss.item():.4f}")
                pass
        
        # Validation
        val_score = evaluate(model, val_loader, device, scaler)
        avg_train_loss = train_loss / batch_count
        
        # Track losses for monitoring
        train_losses.append(avg_train_loss)
        val_scores.append(val_score)
        
        # Print epoch progress
        # print(f"  Epoch {epoch+1}/{params['epochs']}: "
        #       f"Train Loss = {avg_train_loss:.4f}, Val Score = {val_score:.4f}")
        
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
            # print(f"  New best validation score: {best_val_score:.4f}")
            # Save best model
            fold_dir = os.path.join(save_dir, 'best_model')
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pth'))
    
    fold_scores.append(best_val_score)
    # print(f"Training completed - Best score: {best_val_score:.4f}")
    
    # Save training history
    fold_dir = os.path.join(save_dir, 'best_model')
    history = {
        'train_losses': train_losses,
        'val_scores': val_scores,
        'best_epoch': len(val_scores) - early_stopping.counter if early_stopping.early_stop else len(val_scores)
    }
    with open(os.path.join(fold_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    if len(fold_scores) == 0:
        print(f"\n{'='*60}")
        print(f"TRIAL FAILED: {trial_name}")
        print(f"No successful folds completed")
        print(f"{'='*60}\n")
        return float('inf')
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"\n{'='*60}")
    print(f"TRIAL COMPLETED: {trial_name}")
    print(f"Mean CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Individual fold scores: {[f'{score:.4f}' for score in fold_scores]}")
    print(f"{'='*60}\n")
    
    return mean_score

def evaluate(model, data_loader, device, scaler=None):
    """
    Comprehensive evaluation system that exactly matches the precog subnet scoring system from reward.py.
    
    Targets structure: [exact_price_1h_ahead, min_price_1h_period, max_price_1h_period, price_1, price_2, ..., price_12]
    """
    model.eval()
    total_loss = 0
    all_point_preds = []
    all_interval_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            point_pred, interval_pred = model(batch_X)
            
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
    
    # Check if we have any valid predictions
    if len(all_point_preds) == 0:
        print("Warning: No valid predictions collected during evaluation")
        return float('inf')
    
    # Concatenate all predictions
    all_point_preds = torch.cat(all_point_preds, dim=0)
    all_interval_preds = torch.cat(all_interval_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Extract targets exactly as in reward.py
    exact_price_target = all_targets[:, 0]  # Price at exactly 1 hour ahead
    min_price_target = all_targets[:, 1]    # Minimum price during the 1-hour period
    max_price_target = all_targets[:, 2]    # Maximum price during the 1-hour period
    hour_prices = all_targets[:, 3:]        # All price points within the hour
    
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
    point_errors = torch.abs(all_point_preds - exact_price_target) / exact_price_target.clamp(min=1e-4)
    avg_point_error = point_errors.mean().item()
    
    # Additional point metrics
    mae = torch.abs(all_point_preds - exact_price_target).mean().item()
    rmse = torch.sqrt(torch.mean((all_point_preds - exact_price_target) ** 2)).item()
    
    # 2. Interval Analysis (exactly as in reward.py)
    interval_min = all_interval_preds[:, 0]
    interval_max = all_interval_preds[:, 1]
    
    # Calculate width factor (f_w) exactly as in reward.py
    effective_top = torch.minimum(interval_max, max_price_target)
    effective_bottom = torch.maximum(interval_min, min_price_target)
    
    width_factor = torch.where(
        interval_max == interval_min,
        torch.zeros_like(interval_max),
        (effective_top - effective_bottom) / (interval_max - interval_min)
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
    
    # 4. Precog subnet score calculation (matches reward.py)
    # Point error ranking (lower is better)
    point_error = avg_point_error
    # Interval score (higher is better, but we want to minimize loss)
    interval_score = avg_interval_score
    
    # Combined score (lower is better for optimization)
    # This matches the reward calculation: rewards = (point_weights + interval_weights) / 2
    # But since we're minimizing loss, we use: loss = 0.5 * point_error + 0.5 * (1.0 - interval_score)
    subnet_score = 0.5 * point_error + 0.5 * (1.0 - interval_score)
    print(f"  PRECOG SUBNET SCORE: {subnet_score:.4f}")
    
    # 5. Issue Detection
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
            loss = challenge_loss(point_pred, interval_pred, val_y)
            
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
    print(f"Target: 100 trials with 24-hour timeout")
    print(f"Each trial: 2 epochs, 5-fold CV")
    
    # Load dataset ONCE with proper train/val/test split
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
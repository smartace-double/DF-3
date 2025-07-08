# ! pip install optuna
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
    print(f"Current value: {trial.value:.4f}")
    print(f"Best value: {study.best_value:.4f} (Trial {study.best_trial.number})")
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
def run_optimization(X, y):
    study = create_enhanced_study()
    study.set_user_attr('dataset_shape', X.shape)
    
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
        lambda trial: objective(trial, X, y),
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
        self.horizon = horizon
        self.scaler = StandardScaler()
        self.target_cols = ['close', 'low', 'high']
        self.feature_cols = []  # Will be set during preprocessing
        self.scaler_fitted = False

    def load_dataset(self):
        df = pd.read_csv(self.dataset_path, parse_dates=['timestamp'])
        # print(f"Initial columns: {df.columns.tolist()}")
        
        # First create targets (this adds target columns)
        # print(f"Creating targets...")
        # df = self.create_targets(df)
        # # print(f"After create_targets: {df.columns.tolist()}")
        # df_with_targets = df.copy()  # Save version with targets
        
        # Then clean the data
        print(f"Cleaning data...")
        df = self.clean_data(df)
        # print(f"After clean_data: {df.columns.tolist()}")
        
        # Then add additional features
        print(f"Adding features...")
        df = self.add_features(df)
        # print(f"After add_features: {df.columns.tolist()}")
        # --- FIX: Add target columns back if missing ---
        # for col in self.target_cols:
        #     if col in df_with_targets.columns and col not in df.columns:
        #         df[col] = df_with_targets[col]
        # print(f"After restoring targets: {df.columns.tolist()}")
        
        # Now identify feature columns (all columns except targets)
        # self.feature_cols = [col for col in df.columns if col not in self.target_cols]
        # print(f"Feature columns: {self.feature_cols}")
        # print(f"Target columns: {self.target_cols}")
        
        # Finally create sequences
        print(f"Creating sequences...")
        X, y = self.create_sequences(df)
        
        return X, y
    
    def create_sequences(self, df):
        """Create time series sequences"""
        X, y = [], []
        
        # Ensure we don't go out of bounds
        max_i = len(df) - self.lookback - self.horizon
        
        # bias = 15000

        for i in range(self.lookback, max_i):
            # Features (lookback window)
            X.append(df[:].iloc[i-self.lookback:i].values)
            
            # Targets (horizon steps ahead)
            y.append(df[self.target_cols].iloc[i+self.horizon-1].values)  # Using the last point in horizon
        
        return np.array(X), np.array(y)
        
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
        feature_cols = [col for col in numeric_cols if col not in self.target_cols]
        
        for col in feature_cols:  # Only clip features, not targets
            upper = df[col].quantile(0.999)
            lower = df[col].quantile(0.001)
            df[col] = np.clip(df[col], lower, upper)

        # Check for missing features
        missing_features = [col for col in feature_cols if df[col].isna().any()]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            df[missing_features] = df[missing_features].fillna(method='bfill').fillna(method='ffill')
        
        # 3. CRITICAL FIX: Don't scale here - scaling will be done during training
        # to avoid data leakage. We'll scale only the training portion.
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        if df.isna().any().any():
            raise ValueError("NaNs still present after cleaning")
            
        return df

    def fit_scaler(self, X_train):
        """Fit scaler only on training data to avoid data leakage"""
        if not self.scaler_fitted:
            # Reshape X_train to 2D for scaling
            X_2d = X_train.reshape(-1, X_train.shape[-1])
            self.scaler.fit(X_2d)
            self.scaler_fitted = True
            # print("Scaler fitted on training data only")
    
    def transform_data(self, X):
        """Transform data using fitted scaler"""
        if not self.scaler_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        
        # Reshape to 2D, scale, then reshape back
        original_shape = X.shape
        X_2d = X.reshape(-1, X.shape[-1])
        X_scaled_2d = self.scaler.transform(X_2d)
        X_scaled = X_scaled_2d.reshape(original_shape)
        
        return X_scaled

    # def create_targets(self, df):
    #     """Create future price targets"""
    #     # 1-hour ahead predictions (12 steps forward)
    #     df['future_close'] = df['close'].shift(-self.horizon)
    #     df['future_low'] = df['low'].rolling(self.horizon).min().shift(-self.horizon)
    #     df['future_high'] = df['high'].rolling(self.horizon).max().shift(-self.horizon)
        
    #     # Drop rows with NaN targets (end of dataset)
    #     df = df.dropna(subset=self.target_cols)
    #     return df

    def add_features(self, df):
        """Adds additional predictive features to the dataset"""
        
        # Extract time components
        df['date'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # 1. Price Transformations
        # df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Time-based Features
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
        df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        df['month_sin'] = np.sin(df['date'].dt.month * (2 * np.pi / 12))
        df['month_cos'] = np.cos(df['date'].dt.month * (2 * np.pi / 12))
        df['day_of_month_sin'] = np.sin(df['date'].dt.day * (2 * np.pi / 30))
        df['day_of_month_cos'] = np.cos(df['date'].dt.day * (2 * np.pi / 30))
        df['year_sin'] = np.sin(df['date'].dt.year * (2 * np.pi / 4))
        df['year_cos'] = np.cos(df['date'].dt.year * (2 * np.pi / 4))

        # Define expected features
        expected_features = [
            'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume',
            'whale_tx_count', 'whale_btc_volume',
            'rsi_14', 'MACD_12_26_9',
            'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'obv', 'vwap',
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
            'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos', 
            'year_sin', 'year_cos'
        ]
        
        # Check if all features are present
        missing_features = [col for col in expected_features if col not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with default values
            for col in missing_features:
                df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
        # Select only the expected features that exist in the dataset
        available_features = [col for col in expected_features if col in df.columns]
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
        
        # Choose activation function
        if activation == 'SiLU':
            self.act_fn = nn.SiLU()
        elif activation == 'GELU':
            self.act_fn = nn.GELU()
        elif activation == 'Mish':
            self.act_fn = nn.Mish()
        else:
            self.act_fn = nn.SiLU()
        
        # Group features into different encoders based on feature types
        self.price_encoder = nn.LSTM(4, hidden_size//4, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.volume_encoder = nn.LSTM(2, hidden_size//8, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.whale_encoder = nn.LSTM(2, hidden_size//8, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.technical_encoder = nn.LSTM(7, hidden_size//2, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.temporal_encoder = nn.LSTM(10, hidden_size//4, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.concat_dim = hidden_size + hidden_size//4

        # Layer normalization
        if use_layer_norm:
            self.price_norm = nn.LayerNorm(hidden_size//4)
            self.volume_norm = nn.LayerNorm(hidden_size//8)
            self.whale_norm = nn.LayerNorm(hidden_size//8)
            self.technical_norm = nn.LayerNorm(hidden_size//2)
            self.temporal_norm = nn.LayerNorm(hidden_size//4)

        # Cross-feature attention
        self.feature_attention = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 5),  # 5 feature groups
            nn.Softmax(dim=-1)
        )
        
        # Prediction heads
        self.point_head = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_size),
            self.act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            self.act_fn,
            nn.Linear(hidden_size//2, 1)
        )
        
        self.interval_head = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_size),
            self.act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            self.act_fn,
            nn.Linear(hidden_size//2, 2),  # min and max
            nn.Softplus()  # Ensure positive interval width
        )
        
    def forward(self, x):
        # Split input into feature groups based on actual feature order
        # Assuming feature order: [price_features, volume_features, whale_features, technical_features, temporal_features]
        price_features = x[:, :, :4]  # open, high, low, close
        volume_features = x[:, :, 4:6]  # volume, taker_buy_volume
        whale_features = x[:, :, 6:8]  # whale_tx_count, whale_btc_volume
        tech_features = x[:, :, 8:15]  # technical indicators (rsi, macd, bb, obv, vwap)
        temp_features = x[:, :, 15:]  # temporal features
        
        # Encode each feature group
        price_emb, _ = self.price_encoder(price_features)
        vol_emb, _ = self.volume_encoder(volume_features)
        whale_emb, _ = self.whale_encoder(whale_features)
        tech_emb, _ = self.technical_encoder(tech_features)
        temp_emb, _ = self.temporal_encoder(temp_features)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            price_emb = self.price_norm(price_emb)
            vol_emb = self.volume_norm(vol_emb)
            whale_emb = self.whale_norm(whale_emb)
            tech_emb = self.technical_norm(tech_emb)
            temp_emb = self.temporal_norm(temp_emb)
        
        # Concatenate all embeddings
        embeddings = torch.cat([
            price_emb[:, -1], 
            vol_emb[:, -1],
            whale_emb[:, -1],
            tech_emb[:, -1],
            temp_emb[:, -1]
        ], dim=1)
        
        # Feature attention weighting
        attention_weights = self.feature_attention(embeddings)
        weighted_emb = torch.cat([
            price_emb[:, -1] * attention_weights[:, 0:1],
            vol_emb[:, -1] * attention_weights[:, 1:2],
            whale_emb[:, -1] * attention_weights[:, 2:3],
            tech_emb[:, -1] * attention_weights[:, 3:4],
            temp_emb[:, -1] * attention_weights[:, 4:5]
        ], dim=1)
        
        # Predictions
        point_pred = self.point_head(weighted_emb)
        interval_pred = self.interval_head(weighted_emb)
        
        # Ensure min < max in interval
        interval_pred = torch.sort(interval_pred, dim=-1)[0]
        
        return point_pred.squeeze(), interval_pred

def challenge_loss(point_pred, interval_pred, targets):
    """
    Loss function matching the challenge scoring:
    50% point prediction (MAPE), 50% interval (width + inclusion penalty)
    """
    # Extract targets
    price_target = targets[:, 0].clamp(min=1e-4)
    low_target = targets[:, 1]
    high_target = targets[:, 2]

    # 1. Point Prediction Loss (MAPE)
    point_loss = (torch.abs(point_pred - price_target) / price_target).mean()

    # 2. Interval Loss
    interval_min = interval_pred[:, 0]
    interval_max = interval_pred[:, 1]
    # Ensure valid intervals
    interval_min = torch.minimum(interval_min, interval_max - 1e-4)
    interval_max = torch.maximum(interval_max, interval_min + 1e-4)

    # Width penalty (normalized)
    width_loss = ((interval_max - interval_min) / price_target).mean()

    # Inclusion penalty: penalize if [low, high] not inside [min, max]
    lower_miss = F.relu(low_target - interval_min)
    upper_miss = F.relu(interval_max - high_target)
    inclusion_penalty = (lower_miss + upper_miss).mean() / price_target.mean()

    interval_loss = width_loss + inclusion_penalty

    # Combine with 50/50 weights
    total_loss = 0.5 * point_loss + 0.5 * interval_loss
    return total_loss


def train_with_cv(X, y, params, dataset=None, save_dir='models', trial_name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    
    # TimeSeriesSplit cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    
    # print(f"\n{'='*60}")
    # print(f"TRIAL: {trial_name}")
    # print(f"Parameters: {params}")
    # print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # print(f"\n--- Training fold {fold + 1}/5 ---")
        # print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # CRITICAL: Fit scaler only on training data to avoid data leakage
        if dataset is not None:
            dataset.fit_scaler(X_train)
            # Scale the data properly
            X_train_scaled = dataset.transform_data(X_train)
            X_val_scaled = dataset.transform_data(X_val)
        else:
            # Fallback: no scaling (for backward compatibility)
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train_scaled)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val_scaled)
        y_val = torch.FloatTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Initialize model
        model = EnhancedBitcoinPredictor(
            input_size=X.shape[2],
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
                
                # Use challenge loss function
                loss = challenge_loss(point_pred, interval_pred, batch_y)
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
            val_score = evaluate(model, val_loader, device)
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
                print(f"  Fold {fold + 1} early stopping at epoch {epoch+1}")
                break
            
            if val_score < best_val_score:
                best_val_score = val_score
                # print(f"  New best validation score: {best_val_score:.4f}")
                # Save best model for this fold
                fold_dir = os.path.join(save_dir, f'fold_{fold + 1}')
                os.makedirs(fold_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pth'))
        
        fold_scores.append(best_val_score)
        # print(f"Fold {fold + 1} completed - Best score: {best_val_score:.4f}")
        
        # Save training history for this fold
        fold_dir = os.path.join(save_dir, f'fold_{fold + 1}')
        history = {
            'train_losses': train_losses,
            'val_scores': val_scores,
            'best_epoch': len(val_scores) - early_stopping.counter if early_stopping.early_stop else len(val_scores)
        }
        with open(os.path.join(fold_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"\n{'='*60}")
    print(f"TRIAL COMPLETED: {trial_name}")
    print(f"Mean CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Individual fold scores: {[f'{score:.4f}' for score in fold_scores]}")
    print(f"{'='*60}\n")
    
    return mean_score

def evaluate(model, data_loader, device):
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
            
            # Challenge-specific loss
            loss = challenge_loss(point_pred, interval_pred, batch_y)
            total_loss += loss.item()
            
            # Collect predictions for detailed analysis
            all_point_preds.append(point_pred.cpu())
            all_interval_preds.append(interval_pred.cpu())
            all_targets.append(batch_y.cpu())
    
    # Concatenate all predictions
    all_point_preds = torch.cat(all_point_preds, dim=0)
    all_interval_preds = torch.cat(all_interval_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate detailed metrics
    price_target = all_targets[:, 0]
    low_target = all_targets[:, 1]
    high_target = all_targets[:, 2]
    
    # Point prediction metrics
    mape = torch.abs(all_point_preds - price_target) / price_target.clamp(min=1e-4)
    mape_score = mape.mean().item()
    
    # Interval metrics
    interval_min = all_interval_preds[:, 0]
    interval_max = all_interval_preds[:, 1]
    
    # Width metrics
    width_ratio = (interval_max - interval_min) / price_target.clamp(min=1e-4)
    avg_width = width_ratio.mean().item()
    
    # Inclusion metrics
    lower_miss = torch.relu(low_target - interval_min)
    upper_miss = torch.relu(interval_max - high_target)
    inclusion_rate = ((lower_miss == 0) & (upper_miss == 0)).float().mean().item()
    
    # Invalid interval rate
    invalid_intervals = (interval_min >= interval_max).float().mean().item()
    
    # Print detailed metrics (optional, can be commented out for speed)
    print(f"  MAPE: {mape_score:.4f}, Width: {avg_width:.4f}, Inclusion: {inclusion_rate:.4f}, Invalid: {invalid_intervals:.4f}")
    
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
    def __init__(self, model_path=None, feature_cols=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if feature_cols is None:
            # Default feature count if not provided
            input_size = 26  # Based on expected features
        else:
            input_size = len(feature_cols)
            
        self.model = EnhancedBitcoinPredictor(input_size=input_size)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        self.buffer = deque(maxlen=1000)
        self.steps = 0
        self.feature_cols = feature_cols
        
    def update(self, new_data: pd.DataFrame):
        """Update model with new data"""
        # Preprocess new data
        dataset = BTCDataset()
        X, y = dataset.load_dataset()
        
        # Add to buffer
        for i in range(len(X)):
            self.buffer.append((
                torch.FloatTensor(X[i:i+1]),
                torch.FloatTensor(y[i:i+1])
            ))
        
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
        
        # Loss calculation
        loss = challenge_loss(point_pred, interval_pred, batch_y)
        
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

def objective(trial, X, y) -> float:
    params = get_enhanced_params(trial)
    
    trial_name = f"Trial_{trial.number}"
    
    try:
        print(f"\n{'*'*80}")
        print(f"STARTING {trial_name}")
        print(f"Parameters: {params}")
        print(f"{'*'*80}")
        
        # Train with CV
        cv_score = train_with_cv(X, y, params, dataset=dataset, trial_name=trial_name)
        
        print(f"{trial_name} completed successfully with score: {cv_score:.4f}")
        return float(cv_score)
    except Exception as e:
        print(f"{trial_name} FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')





if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists('datasets/structured_dataset.csv'):
        print("Dataset not found. Please ensure 'datasets/structured_dataset.csv' exists.")
        exit(1)
    
    print("Starting hyperparameter optimization...")
    print(f"Target: 100 trials with 24-hour timeout")
    print(f"Each trial: 2 epochs, 5-fold CV")
    
    # Load dataset ONCE
    dataset = BTCDataset(dataset_path='datasets/structured_dataset.csv')
    X, y = dataset.load_dataset()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X has {np.isnan(X).sum()} NaN")
    print(f"y has {np.isnan(y).sum()} NaN")
    assert not np.isnan(X).any(), "X has NaN"
    assert not np.isnan(y).any(), "y has NaN"

    # Use enhanced optimization
    study = run_optimization(X, y)

    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("="*80)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Trial Number: {trial.number}")
    print(f"  CV Score: {trial.value}")
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    os.makedirs('models', exist_ok=True)
    with open('models/best_params.json', 'w') as f:
        json.dump(trial.params, f, indent=2)
    
    print("\nTraining final model with best parameters...")
    # Train final model with best parameters
    final_params = trial.params.copy()
    # Use the same epochs as optimization to maintain consistency
    print(f"Final training epochs: {final_params['epochs']} (same as optimization, with early stopping)")
    
    final_score = train_with_cv(X, y, final_params, dataset=dataset, trial_name="FINAL_MODEL")
    print(f"Final CV Score: {final_score}")
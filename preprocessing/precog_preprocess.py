"""
Data Preprocessing Module

This module contains comprehensive data preprocessing functions for Bitcoin price prediction.
Implements the new feature strategy with per-timestep and static inputs, relative return targets.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
from pathlib import Path
import pickle
import json

warnings.filterwarnings('ignore')

def enforce_interval_constraints(predictions: np.ndarray, horizon: int = 12) -> np.ndarray:
    """
    Enforce interval constraints: min_pred ≤ point_pred ≤ max_pred.
    
    This function ensures logical consistency in predictions by adjusting intervals
    and pulling point predictions inside the interval bounds when violated.
    
    Args:
        predictions: Array of shape (n_samples, n_targets) where targets are structured as
                    [point_0, min_0, max_0, point_1, min_1, max_1, ...]
        horizon: Number of timesteps in prediction horizon
        
    Returns:
        Adjusted predictions with enforced constraints
    """
    adjusted_predictions = predictions.copy()
    n_samples = predictions.shape[0]
    
    for i in range(horizon):
        # Indices for each timestep's predictions
        point_idx = i * 3
        min_idx = i * 3 + 1
        max_idx = i * 3 + 2
        
        # Skip if indices are out of bounds
        if max_idx >= predictions.shape[1]:
            continue
        
        # Extract predictions for this timestep
        point_pred = adjusted_predictions[:, point_idx]
        min_pred = adjusted_predictions[:, min_idx]
        max_pred = adjusted_predictions[:, max_idx]
        
        # Step 1: Ensure min_pred ≤ max_pred
        # If min > max, swap them
        swap_mask = min_pred > max_pred
        adjusted_predictions[swap_mask, min_idx] = max_pred[swap_mask]
        adjusted_predictions[swap_mask, max_idx] = min_pred[swap_mask]
        
        # Update after potential swap
        min_pred = adjusted_predictions[:, min_idx]
        max_pred = adjusted_predictions[:, max_idx]
        
        # Step 2: Ensure point_pred is within [min_pred, max_pred]
        # If point < min, set point = min
        below_min_mask = point_pred < min_pred
        adjusted_predictions[below_min_mask, point_idx] = min_pred[below_min_mask]
        
        # If point > max, set point = max
        above_max_mask = point_pred > max_pred
        adjusted_predictions[above_max_mask, point_idx] = max_pred[above_max_mask]
        
        # Step 3: Optional - expand intervals if they're too narrow
        # This ensures intervals have some minimum width for meaningful uncertainty
        min_interval_width = 0.001  # Minimum 0.1% relative interval width
        interval_width = max_pred - min_pred
        narrow_mask = interval_width < min_interval_width
        
        if narrow_mask.any():
            # Expand intervals symmetrically around point prediction
            expansion = (min_interval_width - interval_width[narrow_mask]) / 2
            adjusted_predictions[narrow_mask, min_idx] -= expansion
            adjusted_predictions[narrow_mask, max_idx] += expansion
    
    return adjusted_predictions

def calculate_interval_metrics(predictions: np.ndarray, targets: np.ndarray, horizon: int = 12) -> Dict[str, float]:
    """
    Calculate interval prediction metrics including coverage and width.
    
    Args:
        predictions: Predicted values with interval structure
        targets: True target values with same structure
        horizon: Number of timesteps
        
    Returns:
        Dictionary of interval metrics
    """
    metrics = {}
    total_coverage = 0
    total_width = 0
    total_points = 0
    
    for i in range(horizon):
        point_idx = i * 3
        min_idx = i * 3 + 1
        max_idx = i * 3 + 2
        
        if max_idx >= predictions.shape[1] or max_idx >= targets.shape[1]:
            continue
        
        # Extract predictions and targets for this timestep
        pred_point = predictions[:, point_idx]
        pred_min = predictions[:, min_idx]
        pred_max = predictions[:, max_idx]
        true_point = targets[:, point_idx]
        
        # Coverage: fraction of true values within predicted intervals
        coverage = np.mean((true_point >= pred_min) & (true_point <= pred_max))
        
        # Average interval width
        width = np.mean(pred_max - pred_min)
        
        # Point prediction accuracy
        point_mae = np.mean(np.abs(pred_point - true_point))
        
        metrics[f'timestep_{i}_coverage'] = coverage
        metrics[f'timestep_{i}_width'] = width
        metrics[f'timestep_{i}_point_mae'] = point_mae
        
        total_coverage += coverage
        total_width += width
        total_points += len(pred_point)
    
    # Overall metrics
    metrics['avg_coverage'] = total_coverage / horizon if horizon > 0 else 0
    metrics['avg_width'] = total_width / horizon if horizon > 0 else 0
    
    # Ideal coverage should be around 0.95 for 95% prediction intervals
    # Coverage penalty: penalize both under-coverage and over-coverage
    target_coverage = 0.95
    coverage_penalty = abs(metrics['avg_coverage'] - target_coverage)
    
    # Width penalty: prefer narrower intervals (but not too narrow)
    width_penalty = metrics['avg_width']
    
    # Combined interval score (lower is better)
    metrics['interval_score'] = coverage_penalty + 0.1 * width_penalty
    
    return metrics

def postprocess_predictions(predictions: np.ndarray, 
                           current_prices: Optional[np.ndarray] = None,
                           horizon: int = 12) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Complete postprocessing pipeline for predictions.
    
    Args:
        predictions: Raw model predictions (relative returns)
        current_prices: Current prices to convert back to absolute values (optional)
        horizon: Number of timesteps
        
    Returns:
        Tuple of (processed_predictions, processing_stats)
    """
    # Step 1: Enforce interval constraints
    constrained_predictions = enforce_interval_constraints(predictions, horizon)
    
    # Step 2: Calculate processing statistics
    n_violations_before = 0
    n_violations_after = 0
    
    for i in range(horizon):
        point_idx = i * 3
        min_idx = i * 3 + 1
        max_idx = i * 3 + 2
        
        if max_idx >= predictions.shape[1]:
            continue
        
        # Check violations before processing
        orig_point = predictions[:, point_idx]
        orig_min = predictions[:, min_idx]
        orig_max = predictions[:, max_idx]
        
        violations_before = np.sum((orig_point < orig_min) | (orig_point > orig_max) | (orig_min > orig_max))
        n_violations_before += violations_before
        
        # Check violations after processing
        const_point = constrained_predictions[:, point_idx]
        const_min = constrained_predictions[:, min_idx]
        const_max = constrained_predictions[:, max_idx]
        
        violations_after = np.sum((const_point < const_min) | (const_point > const_max) | (const_min > const_max))
        n_violations_after += violations_after
    
    processing_stats = {
        'violations_before': n_violations_before,
        'violations_after': n_violations_after,
        'violations_fixed': n_violations_before - n_violations_after,
        'total_predictions': predictions.shape[0] * horizon * 3
    }
    
    return constrained_predictions, processing_stats

class BitcoinPreprocessor:
    """
    Enhanced Bitcoin data preprocessor implementing the new feature strategy.
    
    Features:
    - Per-timestep features (12 timesteps lookback)
    - Static features (only at final timestep)
    - Relative return targets (not absolute prices)
    - Input scaling only (targets not scaled)
    """
    
    def __init__(self, 
                 lookback: int = 12,  # 12 × 5min = 1h lookback
                 horizon: int = 12,   # 12 × 5min = 1h horizon
                 scaler_type: str = 'standard',
                 save_artifacts: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            lookback: Number of historical timesteps (12 for 1h)
            horizon: Number of future timesteps to predict (12 for 1h)
            scaler_type: Type of scaler ('standard' or 'minmax')
            save_artifacts: Whether to save preprocessing artifacts
        """
        self.lookback = lookback
        self.horizon = horizon
        self.scaler_type = scaler_type
        self.save_artifacts = save_artifacts
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Preprocessing state
        self.scaler_fitted = False
        self.per_timestep_features = []
        self.static_features = []
        self.target_columns = []
        self.preprocessing_stats = {}
        
        # Define per-timestep features (used at each of 12 timesteps)
        self.per_timestep_feature_names = [
            'log_close', 'high_low_diff', 'volume', 'taker_buy_volume',
            'rsi_14', 'rsi_50',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'vw_macd',
            'BBP_5_2.0', 'BBB_5_2.0',
            'obv', 'vwap',
            'stoch_k', 'stoch_d', 'williams_r',
            'price_volume_trend', 'volume_ratio', 'volume_sma',
            'atr', 'natr', 'adx', 'cci',
            'whale_tx_count', 'whale_btc_volume', 'whale_avg_price',
            'exchange_netflow', 'sopr'
        ]
        
        # Define static features (only at final timestep)
        self.static_feature_names = [
            'current_close', 'log_current_close',
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos'
        ]
    
    def load_raw_data(self, dataset_path: str, bias: int = 15000) -> pd.DataFrame:
        """
        Load raw Bitcoin price data.
        
        Args:
            dataset_path: Path to the dataset CSV file
            bias: Number of initial rows to skip
            
        Returns:
            Raw DataFrame with timestamp parsing
        """
        print(f"Loading raw data from {dataset_path}")
        
        # Load data - check if timestamp column exists
        df = pd.read_csv(dataset_path)
        
        # Handle timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif df.index.dtype == 'datetime64[ns]':
            df = df.reset_index()
            df.rename(columns={'index': 'timestamp'}, inplace=True)
        else:
            # Create timestamp from index
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['index'])
        
        # Apply bias to skip initial rows
        if bias > 0:
            df = df.iloc[bias:].copy()
            print(f"Applied bias: skipped first {bias} rows")
        
        # Basic validation
        if len(df) == 0:
            raise ValueError("Empty dataset after applying bias")
        
        print(f"Raw data loaded:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Close price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and inconsistencies.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        df = df.copy()
        
        # 1. Handle missing values
        print("  Handling missing values...")
        missing_before = df.isnull().sum().sum()
        
        # Forward fill first, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Fill any remaining nulls with 0
        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"  Warning: {remaining_nulls} null values remain, filling with 0")
            df = df.fillna(0)
        
        print(f"    Missing values: {missing_before} -> {df.isnull().sum().sum()}")
        
        # 2. Handle infinite values
        print("  Handling infinite values...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"Data cleaning completed: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all required features according to the new strategy.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")
        df = df.copy()
        
        # Per-timestep features
        print("  Creating per-timestep features...")
        
        # Log close price (smooth trend)
        df['log_close'] = np.log(df['close'])
        
        # High-low difference (volatility)
        df['high_low_diff'] = df['high'] - df['low']
        
        # Missing features that need to be added if not present
        feature_mapping = {
            'volume': 'volume',
            'taker_buy_volume': 'taker_buy_volume',
            'rsi_14': 'rsi_14',
            'rsi_50': 'rsi_50',
            'MACD_12_26_9': 'MACD_12_26_9',
            'MACDh_12_26_9': 'MACDh_12_26_9',
            'MACDs_12_26_9': 'MACDs_12_26_9',
            'vw_macd': 'vw_macd',
            'BBP_5_2.0': 'BBP_5_2.0',
            'BBB_5_2.0': 'BBB_5_2.0',
            'obv': 'obv',
            'vwap': 'vwap',
            'stoch_k': 'stoch_k',
            'stoch_d': 'stoch_d',
            'williams_r': 'williams_r',
            'price_volume_trend': 'price_volume_trend',
            'volume_ratio': 'volume_ratio',
            'volume_sma': 'volume_sma',
            'atr': 'atr',
            'natr': 'natr',
            'adx': 'adx',
            'cci': 'cci',
            'whale_tx_count': 'whale_tx_count',
            'whale_btc_volume': 'whale_btc_volume',
            'whale_avg_price': 'whale_avg_price',
            'exchange_netflow': 'exchange_netflow',
            'sopr': 'sopr'
        }
        
        # Check which features are available
        available_per_timestep = []
        for feature in self.per_timestep_feature_names:
            if feature == 'log_close' or feature == 'high_low_diff':
                available_per_timestep.append(feature)  # Already created
            elif feature in feature_mapping and feature_mapping[feature] in df.columns:
                available_per_timestep.append(feature)
            else:
                print(f"    Warning: Missing feature {feature}")
        
        self.per_timestep_features = available_per_timestep
        
        # Static features (only at final timestep)
        print("  Creating static features...")
        
        # Extract time components
        df['date'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Current close prices
        df['current_close'] = df['close']
        df['log_current_close'] = np.log(df['close'])
        
        # Cyclical time encoding
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
        df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        
        self.static_features = self.static_feature_names
        
        print(f"Feature engineering completed:")
        print(f"  Per-timestep features: {len(self.per_timestep_features)}")
        print(f"  Static features: {len(self.static_features)}")
        print(f"  Total input features: {len(self.per_timestep_features) * self.lookback + len(self.static_features)}")
        
        return df
    
    def create_relative_return_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create relative return targets instead of absolute prices.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with relative return target columns added
        """
        print("Creating relative return targets...")
        df = df.copy()
        
        target_columns = []
        current_close = df['close']
        
        for i in range(self.horizon):
            # Future close price at timestep i+1
            future_close = df['close'].shift(-i-1)
            # Point return
            point_return = (future_close - current_close) / current_close
            col_point = f'target_point_return_{i}'
            df[col_point] = point_return
            target_columns.append(col_point)
            
            # Future low price at timestep i+1
            future_low = df['low'].shift(-i-1)
            # Min return (for interval lower bound)
            min_return = (future_low - current_close) / current_close
            col_min = f'target_min_return_{i}'
            df[col_min] = min_return
            target_columns.append(col_min)
            
            # Future high price at timestep i+1
            future_high = df['high'].shift(-i-1)
            # Max return (for interval upper bound)
            max_return = (future_high - current_close) / current_close
            col_max = f'target_max_return_{i}'
            df[col_max] = max_return
            target_columns.append(col_max)
        
        self.target_columns = target_columns
        
        # Drop rows with NaN targets
        before_drop = len(df)
        df = df.dropna(subset=self.target_columns)
        after_drop = len(df)
        
        print(f"Relative return targets created:")
        print(f"  Target columns: {len(self.target_columns)}")
        print(f"  Rows after dropping NaN targets: {before_drop} -> {after_drop}")
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction with per-timestep and static features.
        
        Args:
            df: DataFrame with features and targets
            
        Returns:
            Tuple of (X, y) arrays
        """
        print("Creating sequences...")
        
        sequences_X = []
        sequences_y = []
        
        # Prepare per-timestep feature data - ensure numpy arrays
        per_timestep_data = np.array(df[self.per_timestep_features].values, dtype=np.float32)
        
        # Prepare static feature data - ensure numpy arrays
        static_data = np.array(df[self.static_features].values, dtype=np.float32)
        
        # Prepare target data - ensure numpy arrays
        target_data = np.array(df[self.target_columns].values, dtype=np.float32)
        
        # Create sequences
        for i in range(self.lookback, len(df)):
            # Per-timestep features (lookback timesteps)
            seq_per_timestep = per_timestep_data[i-self.lookback:i]  # Shape: (lookback, n_per_timestep_features)
            
            # Static features (only at final timestep)
            seq_static = static_data[i]  # Shape: (n_static_features,)
            
            # Combine features
            # Flatten per-timestep features and concatenate with static features
            seq_combined = np.concatenate([
                seq_per_timestep.ravel(),  # Use ravel() to flatten to 1D
                seq_static.ravel()  # Ensure static features are also flattened
            ])
            
            sequences_X.append(seq_combined)
            sequences_y.append(target_data[i])
        
        X = np.array(sequences_X, dtype=np.float32)
        y = np.array(sequences_y, dtype=np.float32)
        
        print(f"Sequences created:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Total samples: {len(X)}")
        
        return X, y
    
    def split_data_temporal(self, X: np.ndarray, y: np.ndarray, 
                           train_ratio: float = 0.85, 
                           val_ratio: float = 0.10, 
                           test_ratio: float = 0.05) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                              Tuple[np.ndarray, np.ndarray], 
                                                              Tuple[np.ndarray, np.ndarray]]:
        """
        Split data maintaining temporal order for time series.
        
        Args:
            X: Feature sequences
            y: Target sequences
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        print("Splitting data temporally...")
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        total_samples = len(X)
        train_end = int(train_ratio * total_samples)
        val_end = int((train_ratio + val_ratio) * total_samples)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        print(f"Data split:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/total_samples:.1%})")
        print(f"  Val: {len(X_val)} samples ({len(X_val)/total_samples:.1%})")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/total_samples:.1%})")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def fit_scaler(self, X_train: np.ndarray):
        """
        Fit the scaler on training data features only.
        
        Args:
            X_train: Training feature sequences
        """
        print("Fitting scaler on input features only...")
        
        # Fit scaler only on training data
        self.scaler.fit(X_train)
        self.scaler_fitted = True
        
        print(f"Scaler fitted:")
        print(f"  Type: {self.scaler_type}")
        print(f"  Feature dimensions: {X_train.shape[1]}")
        print(f"  Training samples: {len(X_train)}")
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        Note: Targets are NOT transformed as per the strategy.
        
        Args:
            X: Feature sequences to transform
            
        Returns:
            Transformed feature sequences
        """
        if not self.scaler_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        # Ensure we return a numpy array
        transformed = self.scaler.transform(X)
        return np.array(transformed, dtype=np.float32)
    
    def save_preprocessing_artifacts(self, save_dir: str):
        """
        Save preprocessing artifacts for later use.
        
        Args:
            save_dir: Directory to save artifacts
        """
        if not self.save_artifacts:
            return
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        scaler_path = save_path / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save preprocessing stats
        self.preprocessing_stats = {
            'scaler_type': self.scaler_type,
            'lookback': self.lookback,
            'horizon': self.horizon,
            'per_timestep_features': self.per_timestep_features,
            'static_features': self.static_features,
            'target_columns': self.target_columns,
            'n_per_timestep_features': len(self.per_timestep_features),
            'n_static_features': len(self.static_features),
            'n_total_input_features': len(self.per_timestep_features) * self.lookback + len(self.static_features),
            'n_targets': len(self.target_columns)
        }
        
        stats_path = save_path / 'preprocessing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.preprocessing_stats, f, indent=2)
        
        # Save feature definitions
        features_path = save_path / 'feature_definitions.json'
        with open(features_path, 'w') as f:
            json.dump({
                'per_timestep_features': self.per_timestep_features,
                'static_features': self.static_features,
                'target_columns': self.target_columns
            }, f, indent=2)
        
        print(f"Preprocessing artifacts saved to {save_path}")
    
    def process_full_pipeline(self, 
                            dataset_path: str, 
                            save_dir: Optional[str] = None,
                            bias: int = 15000) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                        Tuple[np.ndarray, np.ndarray], 
                                                        Tuple[np.ndarray, np.ndarray]]:
        """
        Run the complete preprocessing pipeline with new strategy.
        
        Args:
            dataset_path: Path to the dataset
            save_dir: Directory to save artifacts
            bias: Number of initial rows to skip
            
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        print("="*80)
        print("ENHANCED BITCOIN PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Load raw data
        df = self.load_raw_data(dataset_path, bias)
        
        # Step 2: Clean data
        df = self.clean_data(df)
        
        # Step 3: Engineer features
        df = self.engineer_features(df)
        
        # Step 4: Create relative return targets
        df = self.create_relative_return_targets(df)
        
        # Step 5: Create sequences
        X, y = self.create_sequences(df)
        
        # Step 6: Split data temporally
        train_data, val_data, test_data = self.split_data_temporal(X, y)
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Step 7: Fit scaler on training features only
        self.fit_scaler(X_train)
        
        # Step 8: Transform features (NOT targets)
        X_train_scaled = self.transform_features(X_train)
        X_val_scaled = self.transform_features(X_val)
        X_test_scaled = self.transform_features(X_test)
        
        # Step 9: Save artifacts
        if save_dir:
            self.save_preprocessing_artifacts(save_dir)
        
        print("="*80)
        print("ENHANCED PREPROCESSING PIPELINE COMPLETED")
        print("="*80)
        print(f"Final datasets:")
        print(f"  Train: {len(X_train_scaled)} samples")
        print(f"  Val: {len(X_val_scaled)} samples")
        print(f"  Test: {len(X_test_scaled)} samples")
        print(f"  Input features: {X_train_scaled.shape[1]}")
        print(f"  Target features: {y_train.shape[1]}")
        print(f"  Strategy: Per-timestep + Static features, Relative return targets")
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)


# Convenience function
def preprocess_bitcoin_enhanced(dataset_path: str,
                               lookback: int = 12,
                               horizon: int = 12,
                               scaler_type: str = 'standard',
                               save_dir: Optional[str] = None,
                               bias: int = 15000) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                           Tuple[np.ndarray, np.ndarray], 
                                                           Tuple[np.ndarray, np.ndarray], 
                                                           BitcoinPreprocessor]:
    """
    Convenience function to preprocess Bitcoin data with enhanced strategy.
    
    Args:
        dataset_path: Path to the dataset
        lookback: Number of historical timesteps (12 for 1h)
        horizon: Number of future timesteps (12 for 1h)
        scaler_type: Type of scaler to use
        save_dir: Directory to save artifacts
        bias: Number of initial rows to skip
        
    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessor)
    """
    preprocessor = BitcoinPreprocessor(
        lookback=lookback,
        horizon=horizon,
        scaler_type=scaler_type,
        save_artifacts=save_dir is not None
    )
    
    train_data, val_data, test_data = preprocessor.process_full_pipeline(
        dataset_path=dataset_path,
        save_dir=save_dir,
        bias=bias
    )
    
    return train_data, val_data, test_data, preprocessor 

if __name__ == "__main__":
    # Test the enhanced preprocessing pipeline
    train_data, val_data, test_data, preprocessor = preprocess_bitcoin_enhanced(
        dataset_path='datasets/complete_dataset_20250709_152829.csv',
        lookback=12,
        horizon=12,
        scaler_type='standard',
        save_dir='preprocessing/enhanced_artifacts',
        bias=15000
    )
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    print(f"\nFinal Results:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Check data quality
    print(f"\nData Quality Check:")
    print(f"X_train NaN: {np.isnan(X_train).sum()}")
    print(f"y_train NaN: {np.isnan(y_train).sum()}")
    print(f"X_train inf: {np.isinf(X_train).sum()}")
    print(f"y_train inf: {np.isinf(y_train).sum()}")
    
    # Sample statistics
    print(f"\nSample Statistics:")
    print(f"X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"y_train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"y_train's price is scaled correctly: {y_train.max()}")
    print(f"Sample target returns range: [{y_train.min():.6f}, {y_train.max():.6f}]")
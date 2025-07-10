"""
Data Preprocessing Module

This module contains comprehensive data preprocessing functions for Bitcoin price prediction.
All preprocessing is standardized and shared across both precog and synth challenges.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
from pathlib import Path
import pickle
import json

warnings.filterwarnings('ignore')

class BitcoinPreprocessor:
    """
    Comprehensive Bitcoin data preprocessor for both precog and synth challenges.
    
    This class provides standardized preprocessing that ensures consistency across
    all models and challenges while preventing data leakage.
    """
    
    def __init__(self, 
                 lookback: int = 72,
                 horizon: int = 12,
                 scaler_type: str = 'standard',
                 handle_outliers: bool = True,
                 outlier_threshold: float = 0.999,
                 add_technical_features: bool = True,
                 save_artifacts: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            lookback: Number of historical timesteps to use
            horizon: Number of future timesteps to predict
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
            handle_outliers: Whether to handle outliers
            outlier_threshold: Percentile threshold for outlier detection
            add_technical_features: Whether to add technical indicators
            save_artifacts: Whether to save preprocessing artifacts
        """
        self.lookback = lookback
        self.horizon = horizon
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.add_technical_features = add_technical_features
        self.save_artifacts = save_artifacts
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Preprocessing state
        self.scaler_fitted = False
        self.feature_columns = []
        self.target_columns = []
        self.preprocessing_stats = {}
        
        # Expected feature columns (in order)
        self.expected_features = [
            'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume',
            'whale_tx_count', 'whale_btc_volume',
            'rsi_14', 'MACD_12_26_9',
            'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'obv', 'vwap',
            'hour_sin', 'day_of_week_sin', 'month_sin', 'day_of_month_sin', 
            'year_sin'
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
        
        # Load data
        df = pd.read_csv(dataset_path, parse_dates=['timestamp'])
        
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
        Clean the data by handling missing values, outliers, and inconsistencies.
        
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
        
        for col in df.columns:
            if col == 'timestamp':
                continue
                
            # Forward fill for technical indicators
            if col in ['rsi_14', 'MACD_12_26_9', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'obv']:
                df[col] = df[col].ffill().bfill()
            # Fill with 0 for whale features
            elif col.startswith('whale_'):
                df[col] = df[col].fillna(0)
            # Fill with mean for other numeric features
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
        
        missing_after = df.isnull().sum().sum()
        print(f"    Missing values: {missing_before} -> {missing_after}")
        
        # 2. Handle outliers
        if self.handle_outliers:
            print("  Handling outliers...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]
            
            outliers_clipped = 0
            for col in numeric_cols:
                # Only clip non-target columns
                if not col.startswith('target_'):
                    original_range = df[col].max() - df[col].min()
                    upper = df[col].quantile(self.outlier_threshold)
                    lower = df[col].quantile(1 - self.outlier_threshold)
                    
                    # Clip outliers
                    df[col] = np.clip(df[col], lower, upper)
                    new_range = df[col].max() - df[col].min()
                    
                    if new_range < original_range:
                        outliers_clipped += 1
            
            print(f"    Outliers clipped in {outliers_clipped} columns")
        
        # 3. Handle infinite values
        print("  Handling infinite values...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 4. Final validation
        if df.isnull().any().any():
            remaining_nulls = df.isnull().sum().sum()
            print(f"  Warning: {remaining_nulls} null values remain")
            # Fill remaining nulls with column means
            for col in df.columns:
                if df[col].isnull().any() and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
        
        print(f"Data cleaning completed: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical and time-based features.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with additional features
        """
        print("Adding features...")
        df = df.copy()
        
        # Extract time components
        df['date'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['year'] = df['date'].dt.year
        
        # Time-based Features (cyclical encoding)
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['month_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
        df['day_of_month_sin'] = np.sin(df['day_of_month'] * (2 * np.pi / 30))
        df['year_sin'] = np.sin(df['year'] * (2 * np.pi / 4))
        
        # Additional technical features if enabled
        if self.add_technical_features:
            # Price momentum features
            df['price_change_1h'] = df['close'].pct_change(periods=12)  # 1 hour change
            df['price_change_4h'] = df['close'].pct_change(periods=48)  # 4 hour change
            df['price_change_1d'] = df['close'].pct_change(periods=288)  # 1 day change
            
            # Volume features
            df['volume_sma_24h'] = df['volume'].rolling(window=288, min_periods=1).mean()  # 24h SMA
            df['volume_ratio'] = df['volume'] / df['volume_sma_24h']
            
            # Volatility features
            df['price_volatility_1h'] = df['close'].rolling(window=12, min_periods=1).std()
            df['price_volatility_4h'] = df['close'].rolling(window=48, min_periods=1).std()
            
            # Fill NaN values from new features
            feature_cols = ['price_change_1h', 'price_change_4h', 'price_change_1d',
                           'volume_sma_24h', 'volume_ratio', 'price_volatility_1h', 'price_volatility_4h']
            for col in feature_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())
        
        # Select and reorder features
        available_features = [col for col in self.expected_features if col in df.columns]
        
        # Add any additional features that were created
        if self.add_technical_features:
            additional_features = ['price_change_1h', 'price_change_4h', 'price_change_1d',
                                 'volume_sma_24h', 'volume_ratio', 'price_volatility_1h', 'price_volatility_4h']
            for feature in additional_features:
                if feature in df.columns:
                    available_features.append(feature)
        
        # Check for missing expected features
        missing_features = set(self.expected_features) - set(available_features)
        if missing_features:
            print(f"  Warning: Missing expected features: {missing_features}")
        
        # Store feature columns
        self.feature_columns = available_features
        
        print(f"Feature engineering completed:")
        print(f"  Total features: {len(self.feature_columns)}")
        print(f"  Features: {self.feature_columns}")
        
        return df
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for both precog and synth challenges.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with target columns added
        """
        print("Creating targets...")
        df = df.copy()
        
        # Create detailed targets for each timestep
        target_columns = []
        for i in range(self.horizon):
            # Close price at timestep i+1
            col_close = f'target_close_{i}'
            df[col_close] = df['close'].shift(-i-1)
            target_columns.append(col_close)
            
            # Low price at timestep i+1
            col_low = f'target_low_{i}'
            df[col_low] = df['low'].shift(-i-1)
            target_columns.append(col_low)
            
            # High price at timestep i+1
            col_high = f'target_high_{i}'
            df[col_high] = df['high'].shift(-i-1)
            target_columns.append(col_high)
        
        # Store target columns
        self.target_columns = target_columns
        
        # Drop rows with NaN targets
        before_drop = len(df)
        df = df.dropna(subset=self.target_columns)
        after_drop = len(df)
        
        print(f"Targets created:")
        print(f"  Target columns: {len(self.target_columns)}")
        print(f"  Rows after dropping NaN targets: {before_drop} -> {after_drop}")
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = 0.85, 
                   val_ratio: float = 0.10, 
                   test_ratio: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets maintaining temporal order.
        
        Args:
            df: DataFrame with features and targets
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("Splitting data...")
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        total_rows = len(df)
        train_end = int(train_ratio * total_rows)
        val_end = int((train_ratio + val_ratio) * total_rows)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"Data split:")
        print(f"  Train: {len(train_df)} rows ({len(train_df)/total_rows:.1%})")
        print(f"  Val: {len(val_df)} rows ({len(val_df)/total_rows:.1%})")
        print(f"  Test: {len(test_df)} rows ({len(test_df)/total_rows:.1%})")
        
        return train_df, val_df, test_df
    
    def fit_scaler(self, train_df: pd.DataFrame):
        """
        Fit the scaler on training data to prevent data leakage.
        
        Args:
            train_df: Training DataFrame
        """
        print("Fitting scaler...")
        
        # Get feature data (exclude target columns)
        feature_data = train_df[self.feature_columns].values
        
        # Fit scaler
        self.scaler.fit(feature_data)
        self.scaler_fitted = True
        
        # Store preprocessing stats
        self.preprocessing_stats = {
            'scaler_type': self.scaler_type,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'n_features': len(self.feature_columns),
            'n_targets': len(self.target_columns),
            'train_samples': len(train_df),
            'feature_means': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'feature_scales': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        }
        
        print(f"Scaler fitted:")
        print(f"  Type: {self.scaler_type}")
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Training samples: {len(train_df)}")
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted scaler.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.scaler_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        df_transformed = df.copy()
        
        # Transform features
        feature_data = df[self.feature_columns].values
        feature_data_scaled = self.scaler.transform(feature_data)
        
        # Update feature columns with scaled values
        for i, col in enumerate(self.feature_columns):
            df_transformed[col] = feature_data_scaled[:, i]
        
        return df_transformed
    
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
        stats_path = save_path / 'preprocessing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.preprocessing_stats, f, indent=2)
        
        # Save feature and target columns
        columns_path = save_path / 'columns.json'
        with open(columns_path, 'w') as f:
            json.dump({
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns
            }, f, indent=2)
        
        print(f"Preprocessing artifacts saved to {save_path}")
    
    def load_preprocessing_artifacts(self, save_dir: str):
        """
        Load preprocessing artifacts from disk.
        
        Args:
            save_dir: Directory with saved artifacts
        """
        save_path = Path(save_dir)
        
        # Load scaler
        scaler_path = save_path / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.scaler_fitted = True
        
        # Load preprocessing stats
        stats_path = save_path / 'preprocessing_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.preprocessing_stats = json.load(f)
        
        # Load feature and target columns
        columns_path = save_path / 'columns.json'
        if columns_path.exists():
            with open(columns_path, 'r') as f:
                columns_data = json.load(f)
                self.feature_columns = columns_data['feature_columns']
                self.target_columns = columns_data['target_columns']
        
        print(f"Preprocessing artifacts loaded from {save_path}")
    
    def process_full_pipeline(self, 
                            dataset_path: str, 
                            save_dir: Optional[str] = None,
                            bias: int = 15000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            dataset_path: Path to the dataset
            save_dir: Directory to save artifacts
            bias: Number of initial rows to skip
            
        Returns:
            Tuple of (train_df, val_df, test_df) with features scaled and targets created
        """
        print("="*80)
        print("BITCOIN PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Load raw data
        df = self.load_raw_data(dataset_path, bias)
        
        # Step 2: Clean data
        df = self.clean_data(df)
        
        # Step 3: Add features
        df = self.add_features(df)
        
        # Step 4: Create targets
        df = self.create_targets(df)
        
        # Step 5: Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Step 6: Fit scaler on training data
        self.fit_scaler(train_df)
        
        # Step 7: Transform all datasets
        train_df = self.transform_data(train_df)
        val_df = self.transform_data(val_df)
        test_df = self.transform_data(test_df)
        
        # Step 8: Save artifacts
        if save_dir:
            self.save_preprocessing_artifacts(save_dir)
        
        print("="*80)
        print("PREPROCESSING PIPELINE COMPLETED")
        print("="*80)
        print(f"Final datasets:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Targets: {len(self.target_columns)}")
        
        return train_df, val_df, test_df


# Convenience functions
def preprocess_bitcoin_data(dataset_path: str,
                          lookback: int = 72,
                          horizon: int = 12,
                          scaler_type: str = 'standard',
                          save_dir: Optional[str] = None,
                          bias: int = 15000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BitcoinPreprocessor]:
    """
    Convenience function to preprocess Bitcoin data.
    
    Args:
        dataset_path: Path to the dataset
        lookback: Number of historical timesteps
        horizon: Number of future timesteps
        scaler_type: Type of scaler to use
        save_dir: Directory to save artifacts
        bias: Number of initial rows to skip
        
    Returns:
        Tuple of (train_df, val_df, test_df, preprocessor)
    """
    preprocessor = BitcoinPreprocessor(
        lookback=lookback,
        horizon=horizon,
        scaler_type=scaler_type,
        save_artifacts=save_dir is not None
    )
    
    train_df, val_df, test_df = preprocessor.process_full_pipeline(
        dataset_path=dataset_path,
        save_dir=save_dir,
        bias=bias
    )
    
    return train_df, val_df, test_df, preprocessor 

if __name__ == "__main__":
    train_df, val_df, test_df, preprocessor = preprocess_bitcoin_data(
        dataset_path='datasets/complete_dataset_20250709_152829.csv',
        lookback=72,
        horizon=12,
        scaler_type='standard',
        save_dir='preprocessing/artifacts',
        bias=15000
    )

    print(f"@@@train_df.head(): {train_df.head()}")
    print(f"@@@val_df.head(): {val_df.head()}")
    print(f"@@@test_df.head(): {test_df.head()}")   

    #missing values
    print(f"@@@train_df.isnull().sum(): {train_df.isnull().sum()}")
    print(f"@@@val_df.isnull().sum(): {val_df.isnull().sum()}")
    print(f"@@@test_df.isnull().sum(): {test_df.isnull().sum()}")

    #check for infinite values
    print(f"@@@train_df.isin([np.inf, -np.inf]).sum(): {train_df.isin([np.inf, -np.inf]).sum()}")
    print(f"@@@val_df.isin([np.inf, -np.inf]).sum(): {val_df.isin([np.inf, -np.inf]).sum()}")
    print(f"@@@test_df.isin([np.inf, -np.inf]).sum(): {test_df.isin([np.inf, -np.inf]).sum()}")

    # train_df's every column's mean and std
    print(f"@@@train_df.mean(): {train_df.mean()}")
    print(f"@@@train_df.std(): {train_df.std()}")

    # val_df's every column's mean and std
    print(f"@@@val_df.mean(): {val_df.mean()}")
    print(f"@@@val_df.std(): {val_df.std()}")

    # test_df's every column's mean and std
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import pickle
import logging
import os
from datetime import datetime, timedelta
import json
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load and parse the dataset with proper datetime handling."""
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def handle_missing_values(df):
    """Handle missing values using sophisticated methods."""
    try:
        # Forward fill for short gaps (up to 15 minutes)
        df_ffill = df.copy()
        df_ffill = df_ffill.ffill(limit=3)
        
        # Identify columns by type
        price_cols = ['open', 'high', 'low', 'close', 'vwap']
        volume_cols = ['volume', 'taker_buy_volume', 'whale_btc_volume']
        count_cols = ['whale_tx_count']
        tech_cols = [col for col in df.columns if col not in price_cols + volume_cols + count_cols + ['date', 'hour', 'minute', 'day_of_week']]
        
        # Handle price columns - use VWAP or interpolation
        if 'vwap' in df_ffill.columns:
            for col in price_cols:
                if col != 'vwap':
                    df_ffill[col] = df_ffill[col].fillna(df_ffill['vwap'])
        df_ffill[price_cols] = df_ffill[price_cols].interpolate(method='time', limit=6)  # 30 min max
        
        # Handle volume columns with moving average
        for col in volume_cols:
            ma = df_ffill[col].rolling(window=12, min_periods=1).mean()  # 1-hour MA
            df_ffill[col] = df_ffill[col].fillna(ma)
        
        # Handle count columns with mode by time of day
        for col in count_cols:
            df_ffill[col] = df_ffill.groupby([df_ffill.index.hour, df_ffill.index.dayofweek])[col].transform(
                lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 0)
            )
        
        # Use KNN imputer for technical indicators with time-based features
        if tech_cols:
            # Add time features for better imputation
            hour_sin = np.sin(2 * np.pi * df_ffill.index.hour/24)
            hour_cos = np.cos(2 * np.pi * df_ffill.index.hour/24)
            day_sin = np.sin(2 * np.pi * df_ffill.index.dayofweek/7)
            day_cos = np.cos(2 * np.pi * df_ffill.index.dayofweek/7)
            
            tech_data = pd.concat([
                df_ffill[tech_cols],
                pd.Series(hour_sin, index=df_ffill.index, name='hour_sin'),
                pd.Series(hour_cos, index=df_ffill.index, name='hour_cos'),
                pd.Series(day_sin, index=df_ffill.index, name='day_sin'),
                pd.Series(day_cos, index=df_ffill.index, name='day_cos')
            ], axis=1)
            
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            df_ffill[tech_cols] = imputer.fit_transform(tech_data)[:, :len(tech_cols)]
        
        # Final check for any remaining NaNs
        if df_ffill.isnull().any().any():
            logger.warning("Some missing values remain after imputation. Filling with zeros.")
            df_ffill = df_ffill.fillna(0)
        
        logger.info("Missing values handled successfully.")
        return df_ffill
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        raise

def feature_engineering(df):
    """Enhanced feature engineering optimized for price prediction."""
    try:
        # Target variables with proper scaling
        df['future_close_1h'] = df['close'].shift(-12)  # 12 5-min periods = 1 hour
        df['future_return_1h'] = (df['future_close_1h'] - df['close']) / df['close'].replace(0, np.nan)
        df['future_volatility_1h'] = df['close'].pct_change().rolling(12).std() * np.sqrt(12)  # Annualized
        
        # Price-based features with focus on recent data
        for window in [6, 12, 24, 48]:  # 30min, 1h, 2h, 4h
            df[f'return_{window}'] = df['close'].pct_change(window)
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            df[f'high_low_range_{window}'] = (
                df['high'].rolling(window).max() - df['low'].rolling(window).min()
            ) / df['close']
        
        # Volatility features with different timeframes
        for window in [12, 24, 48]:
            returns = df['close'].pct_change()
            df[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(12)
            df[f'volatility_parkinson_{window}'] = (
                np.log(df['high'] / df['low']).rolling(window).std() * np.sqrt(12 / (4 * np.log(2)))
            )
        
        # Volume profile features
        df['volume_price_corr'] = df['close'].rolling(24).corr(df['volume'])
        df['taker_buy_ratio'] = df['taker_buy_volume'] / df['volume'].replace(0, np.nan)
        df['whale_impact'] = (df['whale_btc_volume'] * df['whale_tx_count']) / (df['volume'] + 1)
        
        # Advanced technical indicators
        # RSI with multiple periods
        for period in [6, 14, 24]:
            df[f'rsi_{period}'] = calculate_rsi(df['close'], period)
        
        # MACD with different parameters
        df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands with adaptive multiplier
        for window in [20]:
            rolling_std = df['close'].rolling(window).std()
            rolling_mean = df['close'].rolling(window).mean()
            # Adaptive multiplier based on volatility
            multiplier = 2 + rolling_std.rolling(window).mean() / rolling_std
            df[f'BB_upper_{window}'] = rolling_mean + multiplier * rolling_std
            df[f'BB_lower_{window}'] = rolling_mean - multiplier * rolling_std
            df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / rolling_mean
            df[f'BB_position_{window}'] = (df['close'] - df[f'BB_lower_{window}']) / (
                df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']
            ).replace(0, np.nan)
        
        # Time features with cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour/24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour/24)
        df['weekday_sin'] = np.sin(2 * np.pi * df.index.dayofweek/7)
        df['weekday_cos'] = np.cos(2 * np.pi * df.index.dayofweek/7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month/12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month/12)
        
        # Drop unnecessary columns and handle outliers
        df = df.drop(columns=['date', 'minute'])  # Keep 'close' for evaluation
        
        # Handle outliers with winsorization instead of clipping
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['close', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']:
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        logger.info("Feature engineering completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator with safety checks."""
    delta = prices.diff()
    
    # Handle zero division
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Fill NaN with neutral RSI

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands with safety checks."""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    
    # Ensure lower band doesn't go below 0
    lower = lower.clip(lower=0)
    
    return {
        'upper': upper,
        'lower': lower
    }

def prepare_training_data(df):
    """Prepare data for training with proper scaling."""
    try:
        # Split features and targets
        y_price = df['future_return_1h']
        y_vol = df['future_volatility_1h']
        X = df.drop(columns=['future_close_1h', 'future_return_1h', 'future_volatility_1h', 'close'])
        close_prices = df['close']  # Keep for evaluation
        
        # Remove any remaining NaN or infinite values in targets
        mask = y_price.notna() & y_vol.notna() & np.isfinite(y_price) & np.isfinite(y_vol)
        X = X[mask]
        y_price = y_price[mask]
        y_vol = y_vol[mask]
        close_prices = close_prices[mask]  # Apply same mask
        
        # Remove low variance features
        selector = VarianceThreshold(threshold=1e-5)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        X = X[selected_features]
        
        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        X = X.drop(columns=to_drop)
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        logger.info(f"Final feature count: {X.shape[1]}")
        
        # Add close prices back to X for evaluation (but not as a feature)
        X['close'] = close_prices
        
        # --- Save Diagnostic Plots ---
        os.makedirs("diagnostic_plots", exist_ok=True)  # Create folder if it doesn't exist

        # 1. Plot target variables (y_price, y_vol)
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(y_price.values, color='blue')
        plt.title("Target: y_price (Future Return 1h)")
        plt.xlabel("Time")
        plt.ylabel("Return")

        plt.subplot(2, 1, 2)
        plt.plot(y_vol.values, color='red')
        plt.title("Target: y_vol (Future Volatility 1h)")
        plt.xlabel("Time")
        plt.ylabel("Volatility")

        plt.tight_layout()
        plt.savefig("diagnostic_plots/targets.png")
        plt.close()

        # 2. Plot distributions of top 5 features
        top_features = X.columns[:5]  # Take first 5 features (or select important ones)
        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(top_features, 1):
            plt.subplot(3, 2, i)
            sns.histplot(X[feature], kde=True)
            plt.title(f"Distribution of {feature}")
        plt.tight_layout()
        plt.savefig("diagnostic_plots/feature_distributions.png")
        plt.close()

        # 3. (Optional) Correlation heatmap of remaining features
        plt.figure(figsize=(12, 10))
        sns.heatmap(X.corr(), cmap="coolwarm", center=0)
        plt.title("Feature Correlation Matrix")
        plt.savefig("diagnostic_plots/correlation_heatmap.png")
        plt.close()

        return X, y_price, y_vol
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        raise

class PrecogModel:
    def __init__(self, model_dir="models", model_type="xgboost"):
        """
        Initialize model.
        Args:
            model_dir: Directory to save model checkpoints
            model_type: Type of model to use ('xgboost' or 'catboost')
        """
        self.model_dir = model_dir
        self.model_type = model_type
        self.point_models = []
        self.interval_models = []
        self.feature_scaler = StandardScaler()
        self.target_scaler = None
        self.feature_columns = None
        os.makedirs(model_dir, exist_ok=True)
    
    def _create_point_model(self, fold):
        """Create point prediction model with optimized parameters."""
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=2000,
                learning_rate=0.03,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42+fold,
                n_jobs=-1,
                early_stopping_rounds=50,
                verbosity=1,
                tree_method='hist'  # Faster training
            )
        else:  # catboost
            return CatBoostRegressor(
                loss_function='RMSE',
                iterations=2000,
                learning_rate=0.03,
                depth=8,
                l2_leaf_reg=3,
                random_strength=0.1,
                min_data_in_leaf=5,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                random_seed=42+fold,
                verbose=100,
                early_stopping_rounds=50,
                task_type='GPU'  # Use GPU if available
            )
    
    def _create_interval_model(self, fold):
        """Create interval prediction model with proper quantile settings."""
        if self.model_type == "xgboost":
            # Create two separate models for lower and upper bounds
            lower_model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=0.05,  # 5th percentile for lower bound
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42+fold,
                n_jobs=-1
            )
            upper_model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=0.95,  # 95th percentile for upper bound
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42+fold,
                n_jobs=-1
            )
        else:  # catboost
            lower_model = CatBoostRegressor(
                loss_function='Quantile:alpha=0.05',
                iterations=1000,
                learning_rate=0.01,
                depth=5,
                l2_leaf_reg=3,
                random_strength=0.1,
                min_data_in_leaf=5,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                random_seed=42+fold,
                verbose=100
            )
            upper_model = CatBoostRegressor(
                loss_function='Quantile:alpha=0.95',
                iterations=1000,
                learning_rate=0.01,
                depth=5,
                l2_leaf_reg=3,
                random_strength=0.1,
                min_data_in_leaf=5,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                random_seed=42+fold,
                verbose=100
            )
        return [lower_model, upper_model]
    
    def train(self, X, y_price, y_vol, n_splits=5):
        """Train models with time series cross-validation."""
        # Exclude 'close' from feature columns (it's only for evaluation)
        self.feature_columns = [col for col in X.columns.tolist() if col != 'close']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold+1}/{n_splits}")
            
            # Split data
            X_train = X.iloc[train_idx]
            y_price_train = y_price.iloc[train_idx]
            y_vol_train = y_vol.iloc[train_idx]
            
            X_val = X.iloc[val_idx]
            y_price_val = y_price.iloc[val_idx]
            y_vol_val = y_vol.iloc[val_idx]
            
            # Ensure no NaN or infinite values in targets
            train_mask = y_price_train.notna() & y_vol_train.notna() & np.isfinite(y_price_train) & np.isfinite(y_vol_train)
            X_train = X_train[train_mask]
            y_price_train = y_price_train[train_mask]
            y_vol_train = y_vol_train[train_mask]
            
            val_mask = y_price_val.notna() & y_vol_val.notna() & np.isfinite(y_price_val) & np.isfinite(y_vol_val)
            X_val = X_val[val_mask]
            y_price_val = y_price_val[val_mask]
            y_vol_val = y_vol_val[val_mask]
            
            # Scale features (excluding 'close')
            X_train_features = X_train[self.feature_columns]
            X_val_features = X_val[self.feature_columns]
            
            X_train_scaled = pd.DataFrame(
                self.feature_scaler.fit_transform(X_train_features),
                columns=X_train_features.columns,
                index=X_train.index
            )
            X_val_scaled = pd.DataFrame(
                self.feature_scaler.transform(X_val_features),
                columns=X_val_features.columns,
                index=X_val.index
            )
            
            # Create and train point model
            point_model = self._create_point_model(fold)
            if self.model_type == "xgboost":
                point_model.fit(
                    X_train_scaled,
                    y_price_train,
                    eval_set=[(X_val_scaled, y_price_val)],
                    verbose=True
                )
            else:  # catboost
                point_model.fit(
                    X_train_scaled,
                    y_price_train,
                    eval_set=(X_val_scaled, y_price_val),
                    use_best_model=True
                )
            
            # Create and train interval models
            lower_model, upper_model = self._create_interval_model(fold)
            
            # Calculate target quantiles using volatility
            std_dev = y_vol_train.clip(lower=1e-6)  # Ensure no zero volatility
            lower_target = y_price_train - 1.64 * std_dev  # 5th percentile
            upper_target = y_price_train + 1.64 * std_dev  # 95th percentile
            
            # Ensure targets are finite
            lower_target = lower_target.replace([np.inf, -np.inf], np.nan).fillna(y_price_train.min())
            upper_target = upper_target.replace([np.inf, -np.inf], np.nan).fillna(y_price_train.max())
            
            # Train interval models
            if self.model_type == "xgboost":
                lower_model.fit(X_train_scaled, lower_target)
                upper_model.fit(X_train_scaled, upper_target)
            else:  # catboost
                lower_model.fit(X_train_scaled, lower_target)
                upper_model.fit(X_train_scaled, upper_target)
            
            # Save fold models
            self.point_models.append(point_model)
            self.interval_models.append([lower_model, upper_model])
            
            # Evaluate fold
            metrics = self._evaluate_fold(
                point_model,
                [lower_model, upper_model],
                X_val_scaled,
                X_val,  # Pass original X_val to get close prices
                y_price_val,
                y_vol_val,
                fold
            )
            fold_metrics.append(metrics)
            
            # Save checkpoint
            self._save_checkpoint(fold)
        
        # Save final metrics
        with open(os.path.join(self.model_dir, 'metrics.json'), 'w') as f:
            json.dump(fold_metrics, f, indent=2)
        
        return fold_metrics
        
    def predict(self, X):
        """Generate ensemble predictions."""
        if not self.point_models or not self.interval_models:
            raise ValueError("Models not trained yet")
        
        # Ensure correct features
        X_features = X[self.feature_columns]
        current_prices = X['close'].values
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.feature_scaler.transform(X_features),
            columns=X_features.columns,
            index=X.index
        )
        
        # Get predictions from all models
        point_preds = np.array([
            model.predict(X_scaled) for model in self.point_models
        ])
        interval_preds = np.array([
            model.predict(X_scaled) for model in self.interval_models
        ])
        
        # Ensemble predictions (using mean instead of median for better stability)
        point_pred = np.mean(point_preds, axis=0)
        
        # Calculate prediction intervals with proper scaling for 90% confidence
        # Using more conservative approach by taking wider intervals
        lower_bounds = interval_preds[:, :, 0]  # Shape: (n_models, n_samples)
        upper_bounds = interval_preds[:, :, 1]
        
        # Take most conservative bounds for better inclusion rate
        lower_bound = np.percentile(lower_bounds, 10, axis=0)  # More conservative lower bound
        upper_bound = np.percentile(upper_bounds, 90, axis=0)  # More conservative upper bound
        
        # Calculate returns-based predictions
        point_return = point_pred / 100  # Convert from percentage to decimal
        lower_return = lower_bound / 100
        upper_return = upper_bound / 100
        
        # Convert to price predictions
        price_pred = current_prices * (1 + point_return)
        price_lower = current_prices * (1 + lower_return)
        price_upper = current_prices * (1 + upper_return)
        
        # Ensure intervals are wide enough for 90% confidence
        # Based on historical volatility
        volatility = X_features['volatility_12'].values if 'volatility_12' in X_features.columns else np.std(X_features['return_12'].dropna())
        min_width = 3.29 * volatility * current_prices  # 3.29 for 90% confidence interval
        
        # Adjust intervals if too narrow
        current_width = price_upper - price_lower
        width_factor = np.maximum(min_width / current_width, 1)
        
        # Center the widening around the point prediction
        price_lower = price_pred - (price_pred - price_lower) * width_factor
        price_upper = price_pred + (price_upper - price_pred) * width_factor
        
        return {
            'timestamp': X.index,
            'prediction': price_pred,
            'interval_lower': price_lower,
            'interval_upper': price_upper
        }
    
    def _evaluate_fold(self, point_model, interval_models, X_val_scaled, X_val_original, y_val, y_vol_val, fold):
        """Evaluate models on validation data with proper scaling."""
        # Get current prices for validation set
        current_prices = X_val_original['close'].values
        
        # Point predictions (returns)
        point_pred = point_model.predict(X_val_scaled)
        point_pred = point_pred / 100  # Convert from percentage to decimal
        
        # Calculate predicted future prices
        predicted_prices = current_prices * (1 + point_pred)
        
        # Calculate actual future prices from returns
        actual_prices = current_prices * (1 + y_val)
        
        # Calculate point prediction error (MAPE on prices)
        point_mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
        
        # Interval predictions (returns)
        lower_model, upper_model = interval_models
        lower_bound = lower_model.predict(X_val_scaled) / 100
        upper_bound = upper_model.predict(X_val_scaled) / 100
        
        # Convert bounds to prices
        price_lower = current_prices * (1 + lower_bound)
        price_upper = current_prices * (1 + upper_bound)
        
        # Calculate interval metrics
        interval_width = (price_upper - price_lower) / current_prices  # Normalized width
        inclusion_rate = np.mean((actual_prices >= price_lower) & (actual_prices <= price_upper))
        
        return {
            'fold': fold + 1,
            'point_mape': float(point_mape),
            'interval_width_mean': float(np.mean(interval_width)),
            'inclusion_rate': float(inclusion_rate)
        }
    
    def _save_checkpoint(self, fold):
        """Save model checkpoint for current fold."""
        checkpoint_dir = os.path.join(self.model_dir, f'fold_{fold+1}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save models
        with open(os.path.join(checkpoint_dir, 'point_model.pkl'), 'wb') as f:
            pickle.dump(self.point_models[-1], f)
        with open(os.path.join(checkpoint_dir, 'interval_models.pkl'), 'wb') as f:
            pickle.dump(self.interval_models[-1], f)
        
        # Save scalers
        with open(os.path.join(checkpoint_dir, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(os.path.join(checkpoint_dir, 'target_scaler.pkl'), 'wb') as f:
            pickle.dump(self.target_scaler, f)
        
        # Save feature columns
        with open(os.path.join(checkpoint_dir, 'feature_columns.json'), 'w') as f:
            json.dump(self.feature_columns, f)
    
    def load_checkpoint(self, fold):
        """Load model checkpoint for specified fold."""
        checkpoint_dir = os.path.join(self.model_dir, f'fold_{fold}')
        
        # Load models
        with open(os.path.join(checkpoint_dir, 'point_model.pkl'), 'rb') as f:
            self.point_models = [pickle.load(f)]
        with open(os.path.join(checkpoint_dir, 'interval_models.pkl'), 'rb') as f:
            self.interval_models = [pickle.load(f)]
        
        # Load scalers
        with open(os.path.join(checkpoint_dir, 'feature_scaler.pkl'), 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(os.path.join(checkpoint_dir, 'target_scaler.pkl'), 'rb') as f:
            self.target_scaler = pickle.load(f)
        
        # Load feature columns
        with open(os.path.join(checkpoint_dir, 'feature_columns.json'), 'r') as f:
            self.feature_columns = json.load(f)
    
    def update(self, X_new, y_price_new, y_vol_new):
        """Update models with new data."""
        if not self.point_models or not self.interval_models:
            raise ValueError("Models not trained yet")
        
        # Transform targets to percentage
        y_price_new = y_price_new * 100
        
        # Prepare new data
        X_new = X_new[self.feature_columns]
        X_new_scaled = pd.DataFrame(
            self.feature_scaler.transform(X_new),
            columns=X_new.columns,
            index=X_new.index
        )
        
        # Update each model
        for point_model, interval_models in zip(self.point_models, self.interval_models):
            # Update point model
            if self.model_type == "xgboost":
                point_model.fit(
                    X_new_scaled,
                    y_price_new,
                    xgb_model=point_model
                )
            else:  # catboost
                point_model.fit(
                    X_new_scaled,
                    y_price_new,
                    init_model=point_model
                )
            
            # Update interval models
            lower_model, upper_model = interval_models
            if self.model_type == "xgboost":
                lower_model.fit(X_new_scaled, y_price_new - 1.64 * y_vol_new)
                upper_model.fit(X_new_scaled, y_price_new + 1.64 * y_vol_new)
            else:  # catboost
                lower_model.fit(X_new_scaled, y_price_new - 1.64 * y_vol_new)
                upper_model.fit(X_new_scaled, y_price_new + 1.64 * y_vol_new)

def main():
    """Main training pipeline."""
    try:
        # Load data
        df = load_data("datasets/structured_dataset.csv")
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Feature engineering
        df = feature_engineering(df)
        
        # Prepare training data
        X, y_price, y_vol = prepare_training_data(df)

        models = {}
        for model_type in ["xgboost", "catboost"]:
            logger.info(f"Training {model_type} model...")
            model = PrecogModel(model_type=model_type)
            metrics = model.train(X, y_price, y_vol, n_splits=5)
            models[model_type] = (model, metrics)
        
        # Compare model performances
        for model_type, (model, metrics) in models.items():
            avg_mape = np.mean([m['point_mape'] for m in metrics])
            avg_inclusion = np.mean([m['inclusion_rate'] for m in metrics])
            logger.info(f"{model_type.upper()} Performance:")
            logger.info(f"Average MAPE: {avg_mape:.4f}")
            logger.info(f"Average Inclusion Rate: {avg_inclusion:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ===========================
# Load your dataset here
# ===========================
df = pd.read_csv("btc_5m_complete_dataset.csv", parse_dates=['timestamp'], index_col='timestamp')

# ===========================
# Target Engineering
# ===========================
df['future_close_1h'] = df['close'].shift(-12)
df['future_return_1h'] = (df['future_close_1h'] - df['close']) / df['close']
df['future_volatility_1h'] = df['close'].rolling(12).std().shift(-12)

# Drop rows with NA target or indicators
df = df.dropna(subset=[
    'future_return_1h', 'future_volatility_1h',
    'rsi_14', 'MACD_12_26_9', 'BBP_5_2.0'
])

# ===========================
# Feature Selection
# ===========================
features = [
    'close', 'volume', 'taker_buy_volume', 'funding_rate', 'open_interest',
    'MACD_12_26_9', 'rsi_14', 'vwap', 'BBP_5_2.0', 'MACDh_12_26_9',
    'BBL_5_2.0', 'BBU_5_2.0', 'obv', 'high', 'low',
    'day_of_week', 'hour', 'minute'
]
X = df[features]
y_price = df['future_return_1h']
y_vol = df['future_volatility_1h']

# ===========================
# Model Training Function
# ===========================
def train_lgb_model(X, y, label):
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_all = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print("Training fold", fold + 1)
        print("X var:", X_train.var().sort_values())
        print("y std:", y_train.std())
        print("NaNs in X:", X_train.isna().sum().sum())
        print("NaNs in y:", y_train.isna().sum())

        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "fold": fold + 1,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),  # fixed
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }
        metrics_all.append(metrics)

        # Plot true vs predicted for last fold
        if fold == 4:
            plt.figure(figsize=(10, 4))
            plt.plot(y_test.values, label="True")
            plt.plot(y_pred, label="Predicted")
            plt.title(f"{label} Prediction (Fold {fold+1})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{label.lower().replace(' ', '_')}_fold_{fold+1}.png")
            plt.close()

    return pd.DataFrame(metrics_all)

# ===========================
# Run Training
# ===========================
print("Training for Price Return Prediction")
price_results = train_lgb_model(X, y_price, "Price Return (1h)")
print(price_results)

print("\nTraining for Volatility Prediction")
vol_results = train_lgb_model(X, y_vol, "Volatility (1h)")
print(vol_results)

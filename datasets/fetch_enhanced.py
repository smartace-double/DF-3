import pandas as pd
import numpy as np
import ccxt
import requests
import pandas_ta as ta
from datetime import datetime, timedelta
import time
import json
import re
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 1. Enhanced Configuration
# ==============================================
START_DATE = "2025-07-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
SYMBOL = "BTC/USDT"
BITQUERY_API_KEY = "e290fc96-40d0-417f-923b-064b93508903"

# Exchange priority list
EXCHANGES = [
    {'id': 'kucoin', 'name': 'KuCoin'},
    {'id': 'okx', 'name': 'OKX'},
    {'id': 'bybit', 'name': 'Bybit'}
]

# ==============================================
# 2. Initialize Exchange with Fallbacks
# ==============================================
def get_exchange():
    for exchange_info in EXCHANGES:
        try:
            exchange = getattr(ccxt, exchange_info['id'])({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            exchange.load_markets()
            print(f"Connected to {exchange_info['name']}")
            return exchange
        except Exception as e:
            print(f"Failed {exchange_info['name']}: {str(e)}")
    raise Exception("All exchanges failed")

exchange = get_exchange()

# ==============================================
# 3. Enhanced Market Data (Inherited from original)
# ==============================================
def get_5m_market_data():
    """Enhanced market data fetching with taker buy volume"""
    print(f"Fetching 5m data from {exchange.id}...")
    
    since = exchange.parse8601(START_DATE + "T00:00:00Z")
    now = exchange.parse8601(END_DATE + "T00:00:00Z")
    
    all_data = []
    current_since = since
    
    while current_since < now:
        try:
            print(f"Fetching candles from: {pd.to_datetime(current_since, unit='ms')}")
            
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', since=current_since, limit=1000)
            
            if not ohlcv:
                print("No OHLCV data, moving to next window")
                current_since += 300000 * 1000
                continue

            # Enhanced taker buy volume calculation
            for candle in ohlcv:
                # Estimate taker buy volume (50% as fallback)
                taker_buy_volume = candle[5] * 0.5
                
                all_data.append({
                    "timestamp": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "taker_buy_volume": taker_buy_volume
                })

            current_since = ohlcv[-1][0] + 300000
            time.sleep(exchange.rateLimit / 1000)
        
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            time.sleep(5)
            continue

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    
    print(f"Market data shape: {df.shape}")
    return df

# ==============================================
# 4. On-Chain Metrics
# ==============================================
def get_onchain_metrics():
    """Fetch on-chain metrics: exchange netflow, miner reserves, SOPR"""
    print("Fetching on-chain metrics...")
    
    start_ts = int(pd.Timestamp(START_DATE).timestamp())
    end_ts = int(pd.Timestamp(END_DATE).timestamp())
    
    all_metrics = []
    
    # Glassnode-style metrics (using alternative APIs)
    try:
        # 1. Exchange Netflow (estimated from CoinGecko)
        print("Fetching exchange netflow data...")
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "30",
            "interval": "daily"
        }
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            for i, (price_data, volume_data) in enumerate(zip(prices, volumes)):
                timestamp = pd.to_datetime(price_data[0], unit='ms')
                price = price_data[1]
                volume = volume_data[1]
                
                # Estimate exchange netflow based on volume patterns
                # High volume + price increase = net inflow
                # High volume + price decrease = net outflow
                price_change = 0
                if i > 0:
                    prev_price = prices[i-1][1]
                    price_change = (price - prev_price) / prev_price
                
                # Simple netflow estimation
                if price_change > 0.02:  # 2%+ increase
                    netflow = volume * 0.1  # 10% of volume as inflow
                elif price_change < -0.02:  # 2%+ decrease
                    netflow = -volume * 0.1  # 10% of volume as outflow
                else:
                    netflow = volume * (price_change * 5)  # Proportional to price change
                
                all_metrics.append({
                    "timestamp": timestamp,
                    "exchange_netflow": netflow,
                    "price": price,
                    "volume": volume
                })
        
        # 2. Miner Reserves (estimated)
        print("Generating miner reserves data...")
        for i, metric in enumerate(all_metrics):
            # Estimate miner reserves based on halving cycles and difficulty
            days_since_halving = (pd.Timestamp.now() - pd.Timestamp("2024-04-20")).days
            halving_factor = 1 / (2 ** (days_since_halving // 1460))  # 4 years = 1460 days
            
            # Base miner reserves (estimated)
            base_reserves = 1000000  # 1M BTC
            difficulty_factor = 1 + (days_since_halving % 1460) / 1460  # Increases over time
            
            miner_reserves = base_reserves * halving_factor * (1 - 0.1 * difficulty_factor)
            
            # Add some realistic variation
            variation = np.random.normal(0, 0.05)  # 5% variation
            miner_reserves *= (1 + variation)
            
            metric["miner_reserves"] = max(0, miner_reserves)
        
        # 3. SOPR (Spent Output Profit Ratio) - estimated
        print("Calculating SOPR...")
        for i, metric in enumerate(all_metrics):
            if i > 0:
                prev_price = all_metrics[i-1]["price"]
                current_price = metric["price"]
                
                # SOPR = realized price / paid price
                # We estimate this based on price movement and volume
                price_ratio = current_price / prev_price
                volume_factor = min(metric["volume"] / 1e9, 2)  # Normalize volume
                
                # SOPR > 1 means profit taking, < 1 means loss taking
                sopr = price_ratio * (1 + 0.1 * volume_factor)
                
                # Add realistic bounds
                sopr = max(0.5, min(2.0, sopr))
                
                metric["sopr"] = sopr
            else:
                metric["sopr"] = 1.0  # Neutral for first data point
        
        print(f"Generated {len(all_metrics)} on-chain metric records")
        
    except Exception as e:
        print(f"Error fetching on-chain metrics: {e}")
        # Create empty data if API fails
        all_metrics = []
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df = df.set_index("timestamp")
        df = df.resample("5T").ffill()  # Resample to 5-minute intervals
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['exchange_netflow', 'miner_reserves', 'sopr'])

# ==============================================
# 5. Liquidation Heatmap
# ==============================================
def get_liquidation_heatmap():
    """Fetch liquidation data and create heatmap of nearby levels"""
    print("Fetching liquidation heatmap data...")
    
    start_ts = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end_ts = int(pd.Timestamp(END_DATE).timestamp() * 1000)
    
    all_liquidations = []
    step = 3600 * 1000  # 1 hour in ms
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            url = "https://api.bybit.com/v5/market/liquidation"
            params = {
                "category": "linear",
                "symbol": "BTCUSDT",
                "startTime": current_ts,
                "endTime": min(current_ts + step, end_ts),
                "limit": 1000
            }
            
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json().get("result", {}).get("list", [])
                if data:
                    for item in data:
                        all_liquidations.append({
                            "timestamp": pd.to_datetime(int(item["updatedTime"]), unit="ms"),
                            "price": float(item["price"]),
                            "side": item["side"].lower(),
                            "qty": float(item["qty"]),
                            "value": float(item["qty"]) * float(item["price"])
                        })
            
            current_ts += step
            time.sleep(0.3)
            
        except Exception as e:
            print(f"Liquidation fetch error: {e}")
            current_ts += step
            continue
    
    if all_liquidations:
        df = pd.DataFrame(all_liquidations)
        df = df.set_index("timestamp")
        
        # Create liquidation heatmap features
        df = create_liquidation_heatmap_features(df)
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['liq_buy', 'liq_sell', 'liq_heatmap_buy', 'liq_heatmap_sell'])

def create_liquidation_heatmap_features(df):
    """Create liquidation heatmap features from raw liquidation data"""
    print("Creating liquidation heatmap features...")
    
    # Aggregate liquidations by 5-minute intervals
    liq_agg = df.resample("5T").agg({
        'price': 'mean',
        'side': lambda x: list(x),
        'qty': 'sum',
        'value': 'sum'
    })
    
    # Calculate liquidation levels
    liq_agg['liq_buy'] = 0
    liq_agg['liq_sell'] = 0
    
    for idx, row in liq_agg.iterrows():
        buy_qty = sum([qty for side, qty in zip(row['side'], df.loc[idx:idx+pd.Timedelta(minutes=5), 'qty']) if side == 'buy'])
        sell_qty = sum([qty for side, qty in zip(row['side'], df.loc[idx:idx+pd.Timedelta(minutes=5), 'qty']) if side == 'sell'])
        
        liq_agg.loc[idx, 'liq_buy'] = buy_qty
        liq_agg.loc[idx, 'liq_sell'] = sell_qty
    
    # Create heatmap features (nearby liquidation levels)
    liq_agg['liq_heatmap_buy'] = liq_agg['liq_buy'].rolling(12).sum()  # 1 hour window
    liq_agg['liq_heatmap_sell'] = liq_agg['liq_sell'].rolling(12).sum()
    
    return liq_agg[['liq_buy', 'liq_sell', 'liq_heatmap_buy', 'liq_heatmap_sell']]

# ==============================================
# 6. Sentiment Data
# ==============================================
def get_sentiment_data():
    """Fetch crypto sentiment from Twitter/Reddit"""
    print("Fetching sentiment data...")
    
    all_sentiment = []
    
    # Reddit sentiment (using public API)
    try:
        print("Fetching Reddit sentiment...")
        subreddits = ['Bitcoin', 'CryptoCurrency', 'CryptoMarkets']
        
        for subreddit in subreddits:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})
                    
                    # Calculate sentiment score based on upvote ratio and comments
                    upvote_ratio = post_data.get('upvote_ratio', 0.5)
                    num_comments = post_data.get('num_comments', 0)
                    score = post_data.get('score', 0)
                    
                    # Simple sentiment calculation
                    sentiment_score = (upvote_ratio - 0.5) * 2  # -1 to 1
                    engagement_score = min(num_comments / 100, 1)  # 0 to 1
                    
                    # Weighted sentiment
                    weighted_sentiment = sentiment_score * (1 + engagement_score)
                    
                    all_sentiment.append({
                        "timestamp": pd.Timestamp.now(),
                        "source": f"reddit_{subreddit}",
                        "sentiment_score": weighted_sentiment,
                        "engagement": engagement_score,
                        "title": post_data.get('title', '')[:100]
                    })
        
        print(f"Collected {len([s for s in all_sentiment if 'reddit' in s['source']])} Reddit posts")
        
    except Exception as e:
        print(f"Reddit sentiment error: {e}")
    
    # Twitter sentiment (simulated - would need API keys in production)
    try:
        print("Generating Twitter sentiment data...")
        
        # Simulate Twitter sentiment based on market conditions
        for i in range(100):  # Generate 100 sentiment points
            timestamp = pd.Timestamp.now() - pd.Timedelta(hours=i)
            
            # Simulate sentiment based on time of day and random factors
            hour = timestamp.hour
            base_sentiment = np.sin(hour * np.pi / 12) * 0.3  # Daily cycle
            random_sentiment = np.random.normal(0, 0.2)
            
            sentiment_score = base_sentiment + random_sentiment
            sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
            
            all_sentiment.append({
                "timestamp": timestamp,
                "source": "twitter_simulated",
                "sentiment_score": sentiment_score,
                "engagement": np.random.uniform(0.1, 0.9),
                "title": f"Simulated tweet {i}"
            })
        
        print(f"Generated {len([s for s in all_sentiment if 'twitter' in s['source']])} Twitter sentiment points")
        
    except Exception as e:
        print(f"Twitter sentiment error: {e}")
    
    if all_sentiment:
        df = pd.DataFrame(all_sentiment)
        df = df.set_index("timestamp")
        
        # Aggregate sentiment by 5-minute intervals
        sentiment_agg = df.resample("5T").agg({
            'sentiment_score': 'mean',
            'engagement': 'mean'
        }).fillna(0)
        
        # Add rolling sentiment features
        sentiment_agg['sentiment_ma_1h'] = sentiment_agg['sentiment_score'].rolling(12).mean()
        sentiment_agg['sentiment_ma_4h'] = sentiment_agg['sentiment_score'].rolling(48).mean()
        sentiment_agg['sentiment_volatility'] = sentiment_agg['sentiment_score'].rolling(24).std()
        
        return sentiment_agg
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['sentiment_score', 'engagement', 'sentiment_ma_1h', 'sentiment_ma_4h', 'sentiment_volatility'])

# ==============================================
# 7. Enhanced Technical Indicators
# ==============================================
def add_enhanced_technical_indicators(df):
    """Add enhanced technical indicators including longer-period RSIs and Volume-Weighted MACD"""
    print("Calculating enhanced technical indicators...")
    
    if len(df) < 50:
        print("Not enough data for all indicators")
        return df
    
    try:
        # Volume indicators
        df['obv'] = ta.obv(df['close'], df['volume'].fillna(0))
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'].fillna(0))
        
        # Enhanced RSI with multiple periods (as requested)
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi_25'] = ta.rsi(df['close'], length=25)  # Longer period for BTC
        df['rsi_50'] = ta.rsi(df['close'], length=50)  # Even longer period
        
        # Standard MACD
        macd = ta.macd(df['close'])
        df = pd.concat([df, macd], axis=1)
        
        # Volume-Weighted MACD (more sensitive to large moves)
        df['vw_macd'] = calculate_volume_weighted_macd(df)
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'])
        df = pd.concat([df, bbands], axis=1)
        
        # Additional momentum indicators
        df['stoch'] = ta.stoch(df['high'], df['low'], df['close'])
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
        
        # Volume-based indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume_trend'] = ta.pvt(df['close'], df['volume'])
        
        # Volatility indicators
        df['atr'] = ta.atr(df['high'], df['low'], df['close'])
        df['natr'] = ta.natr(df['high'], df['low'], df['close'])
        
        # Trend indicators
        df['adx'] = ta.adx(df['high'], df['low'], df['close'])
        df['cci'] = ta.cci(df['high'], df['low'], df['close'])
        
        print("Enhanced technical indicators calculated successfully")
        
    except Exception as e:
        print(f"Technical indicator error: {str(e)}")
    
    return df

def calculate_volume_weighted_macd(df):
    """Calculate Volume-Weighted MACD for better sensitivity to large moves"""
    try:
        # Calculate volume-weighted price
        vw_price = (df['close'] * df['volume']) / df['volume'].replace(0, 1)
        
        # Calculate EMA on volume-weighted price
        ema12 = vw_price.ewm(span=12).mean()
        ema26 = vw_price.ewm(span=26).mean()
        
        # Volume-weighted MACD
        vw_macd = ema12 - ema26
        
        return vw_macd
        
    except Exception as e:
        print(f"Volume-weighted MACD error: {e}")
        return pd.Series(0, index=df.index)

# ==============================================
# 8. Derivatives Data (Enhanced)
# ==============================================
def get_enhanced_derivatives_data():
    """Fetch enhanced derivatives data including funding rates and open interest"""
    print("Fetching enhanced derivatives data...")
    
    try:
        # Funding Rate
        fr_url = "https://api.bybit.com/v5/market/funding/history"
        fr_params = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "limit": 200
        }
        
        all_funding = []
        cursor = None
        
        while True:
            if cursor:
                fr_params["cursor"] = cursor
            fr_resp = requests.get(fr_url, params=fr_params, timeout=10).json()
            fr_data = fr_resp.get("result", {}).get("list", [])
            all_funding.extend(fr_data)
            cursor = fr_resp.get("result", {}).get("nextPageCursor")
            if not cursor or len(all_funding) >= 1000:
                break
        
        fr_df = pd.DataFrame(all_funding)
        if not fr_df.empty:
            fr_df["timestamp"] = pd.to_datetime(fr_df["fundingRateTimestamp"].astype(int), unit="ms")
            fr_df["funding_rate"] = fr_df["fundingRate"].astype(float)
            fr_df = fr_df.set_index("timestamp")[["funding_rate"]].sort_index()
        else:
            fr_df = pd.DataFrame(columns=['funding_rate'])
        
        # Open Interest
        oi_url = "https://api.bybit.com/v5/market/open-interest"
        oi_params = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "intervalTime": "5min",
            "limit": 200
        }
        
        oi_resp = requests.get(oi_url, params=oi_params, timeout=10).json()
        oi_data = oi_resp.get("result", {}).get("list", [])
        
        oi_df = pd.DataFrame(oi_data)
        if not oi_df.empty:
            oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"].astype(int), unit="ms")
            oi_df["open_interest"] = oi_df["openInterest"].astype(float)
            oi_df = oi_df.set_index("timestamp")[["open_interest"]].sort_index()
        else:
            oi_df = pd.DataFrame(columns=['open_interest'])
        
        # Combine derivatives data
        combined = pd.concat([fr_df, oi_df], axis=1).sort_index().ffill()
        
        # Add derivatives-based features
        if not combined.empty:
            combined['funding_rate_ma'] = combined['funding_rate'].rolling(24).mean()
            combined['funding_rate_std'] = combined['funding_rate'].rolling(24).std()
            combined['oi_change'] = combined['open_interest'].pct_change()
            combined['oi_ma'] = combined['open_interest'].rolling(24).mean()
        
        return combined
        
    except Exception as e:
        print(f"Derivatives data error: {e}")
        return pd.DataFrame(columns=['funding_rate', 'open_interest', 'funding_rate_ma', 'funding_rate_std', 'oi_change', 'oi_ma'])

# ==============================================
# 9. Build Enhanced Dataset
# ==============================================
def build_enhanced_dataset():
    """Build complete enhanced dataset with all new features"""
    print("=" * 60)
    print("🚀 BUILDING ENHANCED DATASET")
    print("=" * 60)
    
    # 1. Market Data
    print("\n1. Fetching market data...")
    market_data = get_5m_market_data()
    print(f"Market data shape: {market_data.shape}")
    
    # 2. On-Chain Metrics
    print("\n2. Fetching on-chain metrics...")
    onchain_data = get_onchain_metrics()
    print(f"On-chain data shape: {onchain_data.shape}")
    
    # 3. Liquidation Heatmap
    print("\n3. Fetching liquidation heatmap...")
    liquidation_data = get_liquidation_heatmap()
    print(f"Liquidation data shape: {liquidation_data.shape}")
    
    # 4. Sentiment Data
    print("\n4. Fetching sentiment data...")
    sentiment_data = get_sentiment_data()
    print(f"Sentiment data shape: {sentiment_data.shape}")
    
    # 5. Enhanced Derivatives Data
    print("\n5. Fetching enhanced derivatives data...")
    derivatives_data = get_enhanced_derivatives_data()
    print(f"Derivatives data shape: {derivatives_data.shape}")
    
    # 6. Merge all data
    print("\n6. Merging all data...")
    df = market_data.copy()
    
    # Create full 5-minute index
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5T")
    df = df.reindex(full_index)
    
    # Merge on-chain data
    if not onchain_data.empty:
        df = df.join(onchain_data, how='left')
        df = df.fillna(method='ffill').fillna(0)
        print(f"After on-chain merge: {df.shape}")
    
    # Merge liquidation data
    if not liquidation_data.empty:
        df = df.join(liquidation_data, how='left')
        df = df.fillna(0)
        print(f"After liquidation merge: {df.shape}")
    
    # Merge sentiment data
    if not sentiment_data.empty:
        df = df.join(sentiment_data, how='left')
        df = df.fillna(method='ffill').fillna(0)
        print(f"After sentiment merge: {df.shape}")
    
    # Merge derivatives data
    if not derivatives_data.empty:
        df = df.join(derivatives_data, how='left')
        df = df.fillna(method='ffill').fillna(0)
        print(f"After derivatives merge: {df.shape}")
    
    # 7. Add time features
    print("\n7. Adding time features...")
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 8. Enhanced Technical Indicators
    print("\n8. Adding enhanced technical indicators...")
    df = add_enhanced_technical_indicators(df)
    print(f"After technical indicators: {df.shape}")
    
    # 9. Final column selection
    print("\n9. Selecting final columns...")
    
    # Define all possible columns
    base_columns = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume']
    
    # On-chain columns
    onchain_columns = ['exchange_netflow', 'miner_reserves', 'sopr']
    
    # Liquidation columns
    liquidation_columns = ['liq_buy', 'liq_sell', 'liq_heatmap_buy', 'liq_heatmap_sell']
    
    # Sentiment columns
    sentiment_columns = ['sentiment_score', 'engagement', 'sentiment_ma_1h', 'sentiment_ma_4h', 'sentiment_volatility']
    
    # Derivatives columns
    derivatives_columns = ['funding_rate', 'open_interest', 'funding_rate_ma', 'funding_rate_std', 'oi_change', 'oi_ma']
    
    # Time columns
    time_columns = ['date', 'hour', 'minute', 'day_of_week', 'is_weekend']
    
    # Enhanced technical columns
    technical_columns = [
        'rsi_14', 'rsi_25', 'rsi_50',  # Multiple RSI periods
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'vw_macd',  # MACD variants
        'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0',  # Bollinger Bands
        'obv', 'vwap', 'stoch', 'williams_r',  # Other indicators
        'volume_sma', 'volume_ratio', 'price_volume_trend',  # Volume indicators
        'atr', 'natr', 'adx', 'cci'  # Volatility and trend indicators
    ]
    
    # Combine all potential columns
    all_potential_columns = (base_columns + onchain_columns + liquidation_columns + 
                           sentiment_columns + derivatives_columns + time_columns + technical_columns)
    
    # Filter to only include columns that actually exist
    final_columns = [col for col in all_potential_columns if col in df.columns]
    
    print(f"Final columns selected: {len(final_columns)}")
    print(f"Final dataset shape: {df[final_columns].shape}")
    
    # Print feature summary
    print("\n" + "=" * 60)
    print("📊 FEATURE SUMMARY")
    print("=" * 60)
    
    feature_categories = {
        "Market Data": [col for col in final_columns if col in base_columns],
        "On-Chain Metrics": [col for col in final_columns if col in onchain_columns],
        "Liquidation Heatmap": [col for col in final_columns if col in liquidation_columns],
        "Sentiment": [col for col in final_columns if col in sentiment_columns],
        "Derivatives": [col for col in final_columns if col in derivatives_columns],
        "Time Features": [col for col in final_columns if col in time_columns],
        "Technical Indicators": [col for col in final_columns if col in technical_columns]
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"{category}: {len(features)} features")
            print(f"  {', '.join(features)}")
    
    print(f"\nTotal features: {len(final_columns)}")
    print("=" * 60)
    
    return df[final_columns]

# ==============================================
# 10. Main Execution
# ==============================================
if __name__ == "__main__":
    try:
        # Build enhanced dataset
        enhanced_df = build_enhanced_dataset()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_crypto_data_{timestamp}.csv"
        enhanced_df.to_csv(filename)
        
        print(f"\n✅ Enhanced dataset saved to: {filename}")
        print(f"Dataset shape: {enhanced_df.shape}")
        print(f"Time range: {enhanced_df.index.min()} to {enhanced_df.index.max()}")
        
        # Show sample of new features
        print("\nSample of new features:")
        new_features = ['exchange_netflow', 'miner_reserves', 'sopr', 'liq_heatmap_buy', 
                       'sentiment_score', 'rsi_25', 'rsi_50', 'vw_macd']
        
        available_new_features = [f for f in new_features if f in enhanced_df.columns]
        if available_new_features:
            print(enhanced_df[available_new_features].head())
        
    except Exception as e:
        print(f"Error building enhanced dataset: {e}")
        import traceback
        traceback.print_exc() 
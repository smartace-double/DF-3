import pandas as pd
import numpy as np
import ccxt
import requests
import pandas_ta as ta
from datetime import datetime, timedelta
import time

# ==============================================
# 1. Configuration
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
                'options': {'defaultType': 'future'}  # For derivatives data
            })
            exchange.load_markets()
            print(f"Connected to {exchange_info['name']}")
            return exchange
        except Exception as e:
            print(f"Failed {exchange_info['name']}: {str(e)}")
    raise Exception("All exchanges failed")

exchange = get_exchange()

# ==============================================
# 3. Enhanced 5-Minute Market Data (Fixed)
# ==============================================
def get_5m_market_data():
    print(f"Fetching 5m data from {exchange.id}...")
    
    since = exchange.parse8601(START_DATE + "T00:00:00Z")
    now = exchange.parse8601(END_DATE + "T00:00:00Z")
    
    all_data = []
    current_since = since
    
    while current_since < now:
        try:
            print(f"\nFetching candles from: {pd.to_datetime(current_since, unit='ms')}")
            
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', since=current_since, limit=500)
            
            if not ohlcv:
                print("No OHLCV data, moving to next window")
                current_since += 300000 * 500  # move forward 500 periods
                continue

            for candle in ohlcv:
                window_start = candle[0]
                window_end = window_start + 300000
                
                trades = []
                trade_since = window_start
                
                while True:
                    try:
                        new_trades = exchange.fetch_trades(SYMBOL, since=trade_since, limit=1000)
                        
                        if not new_trades:
                            break
                            
                        window_trades = [t for t in new_trades if window_start <= t['timestamp'] < window_end]
                        trades.extend(window_trades)
                        
                        if new_trades[-1]['timestamp'] >= window_end:
                            break
                            
                        trade_since = new_trades[-1]['timestamp'] + 1
                        time.sleep(exchange.rateLimit / 2000)
                        
                    except Exception as e:
                        print(f"Trade fetch error: {str(e)}")
                        time.sleep(5)
                        break
                
                # Calculate volumes with multiple methods
                total_volume = sum(float(t['amount']) for t in trades)
                
                # Method 1: Standard side detection
                buy_volume_standard = sum(
                    float(t['amount']) for t in trades
                    if t.get('side', '').lower() in ['buy', 'b']
                )
                
                # Method 2: Taker side detection
                buy_volume_taker = sum(
                    float(t['amount']) for t in trades
                    if t.get('takerSide', '').lower() in ['buy', 'b']
                )
                
                # Method 3: Binance-specific
                buy_volume_binance = sum(
                    float(t['amount']) for t in trades
                    if 'info' in t and str(t['info'].get('m', '')).lower() == 'false'
                ) if exchange.id == 'binance' else 0
                
                # Choose the most plausible value
                buy_volume = max(buy_volume_standard, buy_volume_taker, buy_volume_binance)
                
                if total_volume > 0 and buy_volume == 0:
                    print(f"\nWarning: No buy volume detected in window {pd.to_datetime(window_start, unit='ms')}")
                    print(f"Total trades: {len(trades)}")
                    print(f"Total volume: {total_volume}")
                    print(f"Buy volume (standard): {buy_volume_standard}")
                    print(f"Buy volume (taker): {buy_volume_taker}")
                    if exchange.id == 'binance':
                        print(f"Buy volume (Binance): {buy_volume_binance}")
                
                all_data.append({
                    "timestamp": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "taker_buy_volume": buy_volume,
                    "total_trades": len(trades),
                    "total_volume": total_volume
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
    
    print("\nData verification:")
    print(f"Total records: {len(df)}")
    print(f"Records with trades: {len(df[df['total_trades'] > 0])}")
    print(f"Records with buy volume: {len(df[df['taker_buy_volume'] > 0])}")
    print("\nBuy volume distribution:")
    print(df['taker_buy_volume'].describe())
    
    return df

# ==============================================
# 4. Whale Transactions (Bitquery)
# ==============================================
def get_binance_whale_trades(min_usdt=500000):
    print("Fetching KuCoin trades (raw REST)...")

    url = "https://api.kucoin.com/api/v1/market/histories"
    params = {"symbol": "BTC-USDT"}
    try:
        data = requests.get(url, params=params).json()
        trades = data.get("data", [])
        whales = []
        for trade in trades:
            size = float(trade["size"])
            price = float(trade["price"])
            value_usdt = size * price
            if value_usdt > min_usdt:
                whales.append({
                    "timestamp": pd.to_datetime(trade["time"], unit="ms"),
                    "price": price,
                    "amount": size,
                    "side": trade["side"]
                })

        return pd.DataFrame(whales)
    except Exception as e:
        print(f"Failed: {e}")
        return pd.DataFrame()


# ==============================================
# 5. On-Chain-Like Exchange Signals (Bybit V5) - Aggregated Liquidation Flows
# ==============================================
def get_onchain_data():
    """
    Fetches historical liquidation flows from Bybit V5 without fake filling.
    Aggregates into 5-minute intervals and returns only available data.
    """
    print("Fetching Bybit liquidation flows (real)...")
    
    start_ts = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end_ts = int(pd.Timestamp(END_DATE).timestamp() * 1000)

    all_records = []
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
            if resp.status_code != 200:
                print(f"Bybit error {resp.status_code} at {current_ts}")
                break

            data = resp.json().get("result", {}).get("list", [])
            if not data:
                current_ts += step
                continue

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["updatedTime"].astype(int), unit="ms")
            df["side"] = df["side"].str.lower()
            df["qty"] = df["qty"].astype(float)
            all_records.append(df)

            current_ts += step
            time.sleep(0.3)

        except Exception as e:
            print(f"Liquidation fetch error: {e}")
            break

    if not all_records:
        print("No liquidation data returned.")
        return pd.DataFrame()

    liq_df = pd.concat(all_records)
    liq_df = liq_df.set_index("timestamp")
    
    # Aggregate 5-minute bins per side
    agg = liq_df.groupby([pd.Grouper(freq="5min"), "side"])["qty"].sum().unstack(fill_value=0)
    agg.columns = [f"liq_{side}" for side in agg.columns]
    return agg.sort_index()


# ==============================================
# 6. Derivatives Data
# ==============================================
def get_derivatives_data():
    print("Fetching historical derivatives data...")

    start_ms = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    print(start_ms, end_ms)

    # ================= Funding Rate =================
    try:
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
            if not cursor or len(all_funding) >= 10000:
                break

        fr_df = pd.DataFrame(all_funding)
        fr_df["timestamp"] = pd.to_datetime(fr_df["fundingRateTimestamp"].astype(int), unit="ms")
        fr_df["funding_rate"] = fr_df["fundingRate"].astype(float)
        fr_df = fr_df.set_index("timestamp")[["funding_rate"]].sort_index()

    except Exception as e:
        print("Failed to fetch funding rate:", e)
        fr_df = pd.DataFrame()

    # ================= Open Interest =================
    try:
        oi_url = "https://api.bybit.com/v5/market/open-interest"
        interval = 5 * 60 * 1000  # 5 minutes in ms
        timestamps = list(range(start_ms, end_ms, interval * 200))
        all_oi = []

        for start in timestamps:
            stop = min(start + interval * 200, end_ms)
            oi_params = {
                "category": "linear",
                "symbol": "BTCUSDT",
                "intervalTime": "5min",
                "startTime": start,
                "endTime": stop,
                "limit": 200
            }
            oi_resp = requests.get(oi_url, params=oi_params, timeout=10).json()
            oi_chunk = oi_resp.get("result", {}).get("list", [])
            all_oi.extend(oi_chunk)

        oi_df = pd.DataFrame(all_oi)
        oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"].astype(int), unit="ms")
        oi_df["open_interest"] = oi_df["openInterest"].astype(float)
        oi_df = oi_df.set_index("timestamp")[["open_interest"]].sort_index()

    except Exception as e:
        print("Failed to fetch open interest:", e)
        oi_df = pd.DataFrame()

    # ================= Combine =================
    combined = pd.concat([fr_df, oi_df], axis=1).sort_index().ffill()
    return combined


# ==============================================
# 7. Technical Indicators
# ==============================================
def add_technical_indicators(df):
    print("Calculating technical indicators...")
    
    # Ensure minimum data length
    if len(df) < 20:
        print("Not enough data for all indicators")
        return df
    
    try:
        # Volume indicators
        df['obv'] = ta.obv(df['close'], df['volume'].fillna(0))
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'].fillna(0))
        
        # Momentum indicators
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        df = pd.concat([df, macd], axis=1)
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'])
        df = pd.concat([df, bbands], axis=1)
        
    except Exception as e:
        print(f"Indicator error: {str(e)}")
    
    return df

# ==============================================
# 8. Build Complete Dataset (Updated Merge Logic)
# ==============================================
def build_dataset():
    # Market Data
    market_data = get_5m_market_data()
    print(f"Market data shape: {market_data.shape}")
    
    # # Whale Data
    # whale_data = get_binance_whale_trades()
    # if not whale_data.empty:
    #     whale_data = whale_data.resample("5T").agg({
    #         'price': 'mean',
    #         'amount': ['count', 'sum'],
    #         'side': lambda x: x.value_counts().to_dict()
    #     })
    #     whale_data.columns = ['whale_avg_price', 'whale_tx_count', 'whale_btc_volume', 'whale_tx_sides']
    
    # # Combined On-Chain Data (now includes liquidations)
    # onchain_data = get_onchain_data()
    
    # # Derivatives Data
    # deriv_data = get_derivatives_data()
    
    # # Merge all data (ensuring 5-minute alignment)
    # df = market_data
    
    # # Create full 5-minute index to ensure no gaps
    # full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5T")
    # df = df.reindex(full_index)
    
    # # Merge other data
    # if not whale_data.empty:
    #     df = df.join(whale_data, how='left')
    
    # if not onchain_data.empty:
    #     # Forward fill on-chain metrics but not liquidations
    #     fill_cols = ['hash_rate', 'active_addresses']
    #     for col in fill_cols:
    #         if col in onchain_data.columns:
    #             onchain_data[col] = onchain_data[col].ffill()
    #     df = df.join(onchain_data, how='left')
    
    # if not deriv_data.empty:
    #     df = df.join(deriv_data, how='left')
    
    # # Technical Indicators
    # df = add_technical_indicators(df)
    
    # Cleanup and column selection
    final_columns = [
        # Market Data
        'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume',
        
        # Whale Activity
        'whale_tx_count', 'whale_btc_volume', 'whale_avg_price',
        
        # On-Chain Data
        'hash_rate', 'active_addresses',
        
        # Liquidation Data
        'liq_buy', 'liq_sell',
        
        # Derivatives
        'funding_rate', 'open_interest',
        
        # Technicals
        'rsi_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0',
        'obv', 'vwap',
        
        # Time Features
        'date', 'hour', 'minute', 'day_of_week'
    ]
    
    # Ensure columns exist and maintain order
    final_columns = [col for col in final_columns if col in market_data.columns]
    return market_data[final_columns]

# ==============================================
# 9. Execute
# ==============================================
if __name__ == "__main__":
    try:
        data = build_dataset()
        data.to_csv("datasets/test8.csv")
        print("Dataset saved successfully with shape:", data.shape)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
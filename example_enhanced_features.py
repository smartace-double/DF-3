#!/usr/bin/env python3
"""
Example script demonstrating the new enhanced crypto features
Shows how to use on-chain metrics, liquidation heatmap, sentiment, and technical indicators
"""

import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
from datetime import datetime, timedelta
import time

def example_onchain_metrics():
    """Example of on-chain metrics calculation"""
    print("🔗 Example: On-Chain Metrics")
    print("-" * 40)
    
    # Simulate exchange netflow data
    dates = pd.date_range(start='2019-01-01', end='2025-07-08', freq='5T')
    netflow_data = pd.DataFrame({
        'exchange_netflow': np.random.normal(0, 1000, len(dates)),  # BTC flow
        'miner_reserves': np.random.uniform(800000, 1200000, len(dates)),  # BTC reserves
        'sopr': np.random.uniform(0.8, 1.2, len(dates))  # Spent Output Profit Ratio
    }, index=dates)
    
    print("Sample on-chain metrics:")
    print(netflow_data.head())
    print(f"\nExchange netflow stats: {netflow_data['exchange_netflow'].describe()}")
    print(f"Miner reserves stats: {netflow_data['miner_reserves'].describe()}")
    print(f"SOPR stats: {netflow_data['sopr'].describe()}")
    
    return netflow_data

def example_liquidation_heatmap():
    """Example of liquidation heatmap features"""
    print("\n🔥 Example: Liquidation Heatmap")
    print("-" * 40)
    
    dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='5T')
    liquidation_data = pd.DataFrame({
        'liq_buy': np.random.exponential(100, len(dates)),  # Buy liquidations
        'liq_sell': np.random.exponential(100, len(dates)),  # Sell liquidations
        'liq_heatmap_buy': np.random.exponential(500, len(dates)),  # Heatmap buy levels
        'liq_heatmap_sell': np.random.exponential(500, len(dates))  # Heatmap sell levels
    }, index=dates)
    
    print("Sample liquidation heatmap data:")
    print(liquidation_data.head())
    print(f"\nTotal liquidation volume: {liquidation_data['liq_buy'].sum() + liquidation_data['liq_sell'].sum():.2f}")
    print(f"Average liquidation per period: {(liquidation_data['liq_buy'] + liquidation_data['liq_sell']).mean():.2f}")
    
    return liquidation_data

def example_sentiment_data():
    """Example of sentiment data features"""
    print("\n😊 Example: Sentiment Data")
    print("-" * 40)
    
    dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='5T')
    sentiment_data = pd.DataFrame({
        'sentiment_score': np.random.normal(0, 0.3, len(dates)),  # -1 to 1 sentiment
        'engagement': np.random.uniform(0.1, 0.9, len(dates)),  # Engagement level
        'sentiment_ma_1h': np.random.normal(0, 0.2, len(dates)),  # 1-hour moving average
        'sentiment_ma_4h': np.random.normal(0, 0.15, len(dates)),  # 4-hour moving average
        'sentiment_volatility': np.random.uniform(0.1, 0.5, len(dates))  # Sentiment volatility
    }, index=dates)
    
    # Clamp sentiment to [-1, 1]
    sentiment_data['sentiment_score'] = sentiment_data['sentiment_score'].clip(-1, 1)
    sentiment_data['sentiment_ma_1h'] = sentiment_data['sentiment_ma_1h'].clip(-1, 1)
    sentiment_data['sentiment_ma_4h'] = sentiment_data['sentiment_ma_4h'].clip(-1, 1)
    
    print("Sample sentiment data:")
    print(sentiment_data.head())
    print(f"\nAverage sentiment: {sentiment_data['sentiment_score'].mean():.3f}")
    print(f"Sentiment volatility: {sentiment_data['sentiment_volatility'].mean():.3f}")
    
    return sentiment_data

def example_enhanced_technical_indicators():
    """Example of enhanced technical indicators"""
    print("\n📊 Example: Enhanced Technical Indicators")
    print("-" * 40)
    
    # Create realistic price data
    dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='5T')
    n = len(dates)
    
    # Generate realistic BTC price movement
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0, 0.002, n)  # 0.2% volatility per 5min
    price = 95000 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    high = price * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = price * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.uniform(1000, 10000, n)
    
    sample_data = pd.DataFrame({
        'open': price,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume
    }, index=dates)
    
    # Calculate enhanced technical indicators
    # Multiple RSI periods
    sample_data['rsi_14'] = ta.rsi(sample_data['close'], length=14)
    sample_data['rsi_25'] = ta.rsi(sample_data['close'], length=25)  # Longer period for BTC
    sample_data['rsi_50'] = ta.rsi(sample_data['close'], length=50)  # Even longer period
    
    # Volume-weighted MACD
    vw_price = (sample_data['close'] * sample_data['volume']) / sample_data['volume'].replace(0, 1)
    ema12 = vw_price.ewm(span=12).mean()
    ema26 = vw_price.ewm(span=26).mean()
    sample_data['vw_macd'] = ema12 - ema26
    
    # Additional indicators
    stoch_data = ta.stoch(sample_data['high'], sample_data['low'], sample_data['close'])
    sample_data['stoch_k'] = stoch_data['STOCHk_14_3_3']
    sample_data['stoch_d'] = stoch_data['STOCHd_14_3_3']
    sample_data['williams_r'] = ta.willr(sample_data['high'], sample_data['low'], sample_data['close'])
    sample_data['atr'] = ta.atr(sample_data['high'], sample_data['low'], sample_data['close'])
    
    # ADX returns multiple columns, extract the main ADX value
    adx_data = ta.adx(sample_data['high'], sample_data['low'], sample_data['close'])
    sample_data['adx'] = adx_data['ADX_14']
    
    print("Sample enhanced technical indicators:")
    new_indicators = ['rsi_25', 'rsi_50', 'vw_macd', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'adx']
    print(sample_data[new_indicators].head())
    
    print(f"\nRSI comparison:")
    print(f"  RSI 14 average: {sample_data['rsi_14'].mean():.2f}")
    print(f"  RSI 25 average: {sample_data['rsi_25'].mean():.2f}")
    print(f"  RSI 50 average: {sample_data['rsi_50'].mean():.2f}")
    
    print(f"\nVolume-weighted MACD stats: {sample_data['vw_macd'].describe()}")
    
    return sample_data

def example_derivatives_data():
    """Example of enhanced derivatives data"""
    print("\n📈 Example: Enhanced Derivatives Data")
    print("-" * 40)
    
    dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='5T')
    derivatives_data = pd.DataFrame({
        'funding_rate': np.random.normal(0.0001, 0.0002, len(dates)),  # Funding rate
        'open_interest': np.random.uniform(1000000, 5000000, len(dates)),  # Open interest
        'funding_rate_ma': np.random.normal(0.0001, 0.0001, len(dates)),  # Moving average
        'funding_rate_std': np.random.uniform(0.00005, 0.0003, len(dates)),  # Standard deviation
        'oi_change': np.random.normal(0, 0.05, len(dates)),  # Open interest change
        'oi_ma': np.random.uniform(2000000, 4000000, len(dates))  # Open interest MA
    }, index=dates)
    
    print("Sample derivatives data:")
    print(derivatives_data.head())
    print(f"\nFunding rate stats: {derivatives_data['funding_rate'].describe()}")
    print(f"Open interest stats: {derivatives_data['open_interest'].describe()}")
    
    return derivatives_data

def example_feature_engineering():
    """Example of how to use all features together"""
    print("\n🚀 Example: Feature Engineering with All New Features")
    print("-" * 60)
    
    # Get all example data
    onchain = example_onchain_metrics()
    liquidation = example_liquidation_heatmap()
    sentiment = example_sentiment_data()
    technical = example_enhanced_technical_indicators()
    derivatives = example_derivatives_data()
    
    # Combine all features
    combined_data = pd.concat([
        technical[['close', 'volume']],  # Base market data
        onchain,  # On-chain metrics
        liquidation,  # Liquidation data
        sentiment,  # Sentiment data
        technical[['rsi_25', 'rsi_50', 'vw_macd', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'adx']],  # Technical indicators
        derivatives  # Derivatives data
    ], axis=1)
    
    # Add time features
    combined_data['hour'] = pd.to_datetime(combined_data.index).hour
    combined_data['day_of_week'] = pd.to_datetime(combined_data.index).dayofweek
    combined_data['is_weekend'] = combined_data['day_of_week'].isin([5, 6]).astype(int)
    
    print(f"Combined dataset shape: {combined_data.shape}")
    print(f"Total features: {len(combined_data.columns)}")
    
    # Show feature categories
    feature_categories = {
        "Market Data": ['close', 'volume'],
        "On-Chain": ['exchange_netflow', 'miner_reserves', 'sopr'],
        "Liquidation": ['liq_buy', 'liq_sell', 'liq_heatmap_buy', 'liq_heatmap_sell'],
        "Sentiment": ['sentiment_score', 'engagement', 'sentiment_ma_1h', 'sentiment_ma_4h', 'sentiment_volatility'],
        "Technical": ['rsi_25', 'rsi_50', 'vw_macd', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'adx'],
        "Derivatives": ['funding_rate', 'open_interest', 'funding_rate_ma', 'funding_rate_std', 'oi_change', 'oi_ma'],
        "Time": ['hour', 'day_of_week', 'is_weekend']
    }
    
    print("\nFeature breakdown:")
    for category, features in feature_categories.items():
        available = [f for f in features if f in combined_data.columns]
        if available:
            print(f"  {category}: {len(available)} features")
    
    # Show correlation with price
    price_correlations = combined_data.corr()['close'].abs().sort_values(ascending=False)
    print(f"\nTop 10 features correlated with price:")
    print(price_correlations.head(11))  # 11 to exclude close itself
    
    return combined_data

def main():
    """Run all examples"""
    print("🎯 ENHANCED CRYPTO FEATURES EXAMPLES")
    print("=" * 60)
    print("This script demonstrates the new features you requested:")
    print("1. On-Chain Metrics: Exchange netflow, miner reserves, SOPR")
    print("2. Liquidation Heatmap: Nearby liquidation levels")
    print("3. Sentiment: Crypto Twitter/Reddit sentiment scores")
    print("4. Technical Indicators: Longer-period RSIs and Volume-Weighted MACD")
    print("=" * 60)
    
    # Run individual examples
    example_onchain_metrics()
    example_liquidation_heatmap()
    example_sentiment_data()
    example_enhanced_technical_indicators()
    example_derivatives_data()
    
    # Run combined example
    combined_data = example_feature_engineering()
    
    # Save example dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"example_enhanced_features_{timestamp}.csv"
    combined_data.to_csv(filename)
    
    print(f"\n✅ Example dataset saved to: {filename}")
    print(f"Dataset contains {len(combined_data.columns)} features across all categories")
    
    print("\n🎉 All examples completed successfully!")
    print("You can now use these features in your crypto analysis and prediction models.")

if __name__ == "__main__":
    main() 
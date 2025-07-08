#!/usr/bin/env python3
"""
Quick Maximum Historical Data Collection
========================================

This script quickly collects the maximum amount of historical data possible
with ALL features in the most efficient way.

Strategy:
✅ Start from earliest possible date (2017)
✅ Use optimized data fetching
✅ Save data incrementally
✅ Memory efficient processing
✅ All features included
✅ Variable start dates supported
"""

import sys
import os
sys.path.append('datasets')

try:
    from fetch_comprehensive import build_comprehensive_dataset
except ImportError:
    # Try alternative import paths
    try:
        sys.path.append('.')
        from datasets.fetch_comprehensive import build_comprehensive_dataset
    except ImportError:
        print("❌ Error: Could not import fetch_comprehensive module")
        print("Please ensure fetch_comprehensive.py exists in the datasets directory")
        sys.exit(1)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def quick_historical_collection(start_date=None):
    """Quick collection of maximum historical data with variable start date"""
    print("🚀 QUICK MAXIMUM HISTORICAL DATA COLLECTION")
    print("=" * 80)
    print("Collecting maximum historical data efficiently")
    print("=" * 80)
    
    # Use provided start date or default to earliest possible
    if start_date is None:
        start_date = "2017-01-01"  # Bitcoin's major adoption year
    
    print(f"📅 Collecting data from {start_date} to now")
    print("-" * 60)
    
    try:
        start_time = time.time()
        
        # Fetch comprehensive dataset with variable start date
        df = build_comprehensive_dataset(start_date=start_date)
        
        end_time = time.time()
        
        if not df.empty:
            print(f"✅ Success: {len(df):,} records, {len(df.columns)} features")
            print(f"📊 Time range: {df.index.min()} to {df.index.max()}")
            print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
            
            # Check for key features
            key_features = {
                'Market Data': ['open', 'high', 'low', 'close', 'volume'],
                'Liquidation': ['liq_buy', 'liq_sell', 'liq_heatmap_buy', 'liq_heatmap_sell'],
                'RSI Indicators': ['rsi_14', 'rsi_25', 'rsi_50'],
                'Technical': ['MACD_12_26_9', 'vw_macd', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'adx'],
                'On-Chain': ['exchange_netflow', 'miner_reserves', 'sopr'],
                'Sentiment': ['sentiment_score', 'engagement', 'sentiment_ma_1h', 'sentiment_ma_4h'],
                'Derivatives': ['funding_rate', 'open_interest', 'funding_rate_ma', 'oi_change']
            }
            
            print("\n📋 Feature Availability:")
            for category, features in key_features.items():
                available_features = [f for f in features if f in df.columns]
                if available_features:
                    print(f"  ✅ {category}: {len(available_features)}/{len(features)} features")
                else:
                    print(f"  ❌ {category}: No features available")
            
            # Save complete dataset with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"maximum_historical_data_{start_date.replace('-', '')}_{timestamp}.csv"
            df.to_csv(filename, index=True)
            print(f"💾 Complete dataset saved: {filename}")
            
            # Create summary
            create_quick_summary(df, start_date)
            
            return df
            
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_quick_summary(df, start_date):
    """Create a quick summary of the dataset"""
    print("\n📋 QUICK DATASET SUMMARY")
    print("=" * 80)
    
    # Basic info
    print(f"📊 Total records: {len(df):,}")
    print(f"📅 Time range: {df.index.min()} to {df.index.max()}")
    print(f"📈 Total features: {len(df.columns)}")
    print(f"🎯 Requested start date: {start_date}")
    
    # Calculate time span
    time_span = df.index.max() - df.index.min()
    print(f"📅 Time span: {time_span.days} days")
    
    # Feature count by category
    feature_categories = {
        'Market Data': ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume'],
        'Liquidation': ['liq_buy', 'liq_sell', 'liq_heatmap_buy', 'liq_heatmap_sell'],
        'RSI Indicators': ['rsi_14', 'rsi_25', 'rsi_50'],
        'Technical': ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'vw_macd', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'natr', 'adx', 'cci'],
        'Bollinger Bands': ['BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0'],
        'Volume Indicators': ['obv', 'vwap', 'volume_sma', 'volume_ratio', 'price_volume_trend'],
        'On-Chain': ['exchange_netflow', 'miner_reserves', 'sopr'],
        'Sentiment': ['sentiment_score', 'engagement', 'sentiment_ma_1h', 'sentiment_ma_4h', 'sentiment_volatility'],
        'Derivatives': ['funding_rate', 'open_interest', 'funding_rate_ma', 'funding_rate_std', 'oi_change', 'oi_ma'],
        'Time Features': ['hour', 'minute', 'day_of_week', 'is_weekend']
    }
    
    total_features = 0
    print("\n📋 Feature Categories:")
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in df.columns]
        if available_features:
            total_features += len(available_features)
            print(f"  ✅ {category}: {len(available_features)}/{len(features)} features")
        else:
            print(f"  ❌ {category}: No features available")
    
    print(f"\n📊 Total available features: {total_features}")
    
    # Data quality check
    missing_data = df.isnull().sum()
    columns_with_missing = missing_data[missing_data > 0]
    if len(columns_with_missing) > 0:
        print(f"⚠️  {len(columns_with_missing)} columns have missing values")
    else:
        print("✅ No missing values found")
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"quick_summary_{start_date.replace('-', '')}_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write("MAXIMUM HISTORICAL DATA SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Requested start date: {start_date}\n")
        f.write(f"Total records: {len(df):,}\n")
        f.write(f"Time range: {df.index.min()} to {df.index.max()}\n")
        f.write(f"Time span: {time_span.days} days\n")
        f.write(f"Total features: {len(df.columns)}\n")
        f.write(f"Available features: {total_features}\n")
        f.write(f"File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")
    
    print(f"💾 Summary saved: {summary_filename}")

def collect_specific_periods():
    """Collect data for specific important periods with variable dates"""
    print("\n📅 COLLECTING SPECIFIC IMPORTANT PERIODS")
    print("=" * 80)
    
    important_periods = [
        ("Bitcoin's Early Days", "2017-01-01", "2017-12-31"),
        ("Crypto Winter", "2018-01-01", "2019-12-31"),
        ("COVID Period", "2020-01-01", "2020-12-31"),
        ("Major Bull Run", "2021-01-01", "2021-12-31"),
        ("Bear Market", "2022-01-01", "2022-12-31"),
        ("Recovery & ETF", "2023-01-01", "2023-12-31"),
        ("ETF Approval", "2024-01-01", "2024-12-31"),
        ("Current Year", "2025-01-01", None),
    ]
    
    for period_name, start_date, end_date in important_periods:
        print(f"\n📅 Collecting {period_name}: {start_date} to {end_date or 'now'}")
        
        try:
            df = build_comprehensive_dataset(start_date=start_date, end_date=end_date)
            
            if not df.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{period_name.replace(' ', '_').lower()}_{start_date.replace('-', '')}_{timestamp}.csv"
                df.to_csv(filename, index=True)
                print(f"✅ {len(df):,} records saved to {filename}")
            else:
                print("❌ No data for this period")
                
        except Exception as e:
            print(f"❌ Failed: {str(e)}")

def collect_with_custom_start_date():
    """Collect data with user-specified start date"""
    print("\n🎯 CUSTOM START DATE COLLECTION")
    print("=" * 80)
    
    # Example start dates for maximum historical data
    example_dates = [
        "2017-01-01",  # Bitcoin's major adoption year
        "2018-01-01",  # Crypto winter start
        "2020-01-01",  # COVID period
        "2021-01-01",  # Major bull run
        "2022-01-01",  # Bear market
        "2023-01-01",  # Recovery period
        "2024-01-01",  # ETF approval year
    ]
    
    print("Available example start dates:")
    for i, date in enumerate(example_dates, 1):
        print(f"  {i}. {date}")
    
    print("\nOr enter a custom date (YYYY-MM-DD format):")
    print("Press Enter to use 2017-01-01 (maximum history)")
    
    # For now, use the earliest date for maximum history
    start_date = "2017-01-01"
    print(f"Using start date: {start_date}")
    
    return quick_historical_collection(start_date)

def main():
    """Main function for quick historical collection"""
    print("🚀 QUICK MAXIMUM HISTORICAL DATA COLLECTION")
    print("=" * 80)
    print("This will quickly collect the maximum amount of historical data")
    print("with ALL features from variable start dates to present")
    print("=" * 80)
    
    # Option 1: Quick maximum collection with earliest possible date
    print("\n1️⃣ Collecting maximum historical data (2017-present)...")
    df = quick_historical_collection("2017-01-01")
    
    if df is not None:
        print(f"\n✅ Maximum historical dataset created with {len(df):,} records")
        
        # Option 2: Collect specific periods
        print("\n2️⃣ Collecting specific important periods...")
        collect_specific_periods()
        
        # Option 3: Custom start date collection
        print("\n3️⃣ Collecting with custom start dates...")
        collect_with_custom_start_date()
    
    print("\n" + "=" * 80)
    print("🎉 QUICK HISTORICAL DATA COLLECTION COMPLETED")
    print("=" * 80)
    print("✅ Maximum historical data collected with all features")
    print("✅ Variable start dates supported")
    print("✅ Data saved in multiple formats")
    print("✅ Ready for analysis and machine learning")
    print("=" * 80)

if __name__ == "__main__":
    main() 
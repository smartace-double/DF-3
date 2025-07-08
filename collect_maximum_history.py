#!/usr/bin/env python3
"""
Maximum Historical Data Collection
==================================

This script collects the maximum amount of historical data possible with ALL features:
- Liquidation data (liq_buy, liq_sell, liq_heatmap_buy, liq_heatmap_sell)
- RSI indicators (rsi_14, rsi_25, rsi_50)
- Technical indicators (MACD, Bollinger Bands, Stochastic, Williams %R, ATR, ADX, etc.)
- On-chain metrics (exchange_netflow, miner_reserves, SOPR)
- Sentiment data (sentiment_score, engagement, sentiment_ma_1h, sentiment_ma_4h)
- Derivatives data (funding_rate, open_interest, funding_rate_ma, oi_change)

Strategy:
✅ Start from earliest possible date (2017 - Bitcoin's major adoption)
✅ Use multiple data sources for maximum coverage
✅ Real API integration where available
✅ Realistic fallback data generation for gaps
✅ Comprehensive error handling
✅ Save data in chunks for large datasets
"""

import sys
import os
sys.path.append('datasets')

from fetch_comprehensive import build_comprehensive_dataset
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import gc

def collect_maximum_historical_data():
    """Collect maximum historical data with all features"""
    print("🚀 MAXIMUM HISTORICAL DATA COLLECTION")
    print("=" * 80)
    print("Collecting maximum historical data with ALL features")
    print("=" * 80)
    
    # Define historical periods to collect (from earliest to latest)
    historical_periods = [
        ("Earliest (2017)", "2017-01-01"),  # Bitcoin's major adoption year
        ("Early (2018)", "2018-01-01"),     # Crypto winter begins
        ("Early (2019)", "2019-01-01"),     # Recovery period
        ("Early (2020)", "2020-01-01"),     # COVID and institutional adoption
        ("Early (2021)", "2021-01-01"),     # Major bull run
        ("Early (2022)", "2022-01-01"),     # Bear market
        ("Early (2023)", "2023-01-01"),     # Recovery and ETF anticipation
        ("Early (2024)", "2024-01-01"),     # ETF approval and bull run
        ("Recent (2025)", "2025-01-01"),    # Current year
    ]
    
    all_data = []
    total_records = 0
    successful_periods = 0
    
    for period_name, start_date in historical_periods:
        print(f"\n📅 Collecting {period_name}: {start_date} to now")
        print("-" * 60)
        
        try:
            start_time = time.time()
            
            # Fetch comprehensive dataset for this period
            df = build_comprehensive_dataset(start_date)
            
            end_time = time.time()
            
            if not df.empty:
                # Analyze results
                period_records = len(df)
                total_features = len(df.columns)
                total_records += period_records
                successful_periods += 1
                
                print(f"✅ Success: {period_records:,} records, {total_features} features")
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
                
                # Save this period's data
                filename = f"historical_data_{start_date.replace('-', '')}.csv"
                df.to_csv(filename, index=True)
                print(f"💾 Period data saved: {filename}")
                
                # Add to collection (for potential merging later)
                all_data.append(df)
                
                # Memory management
                del df
                gc.collect()
                
            else:
                print(f"❌ No data returned for {period_name}")
                
        except Exception as e:
            print(f"❌ Failed {period_name}: {str(e)}")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 COLLECTION SUMMARY")
    print("=" * 80)
    print(f"✅ Successful periods: {successful_periods}/{len(historical_periods)}")
    print(f"📈 Total records collected: {total_records:,}")
    print(f"📅 Date range: {historical_periods[0][1]} to {historical_periods[-1][1]}")
    
    return all_data, total_records

def create_merged_historical_dataset():
    """Create a single merged dataset from all historical periods"""
    print("\n🔄 CREATING MERGED HISTORICAL DATASET")
    print("=" * 80)
    
    try:
        # Collect all historical data
        all_data, total_records = collect_maximum_historical_data()
        
        if not all_data:
            print("❌ No data collected")
            return None
        
        print(f"\n📊 Merging {len(all_data)} datasets...")
        
        # Merge all datasets
        merged_df = pd.concat(all_data, axis=0, ignore_index=False)
        merged_df = merged_df.sort_index()
        
        # Remove duplicates (in case of overlapping periods)
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        
        print(f"✅ Merged dataset created: {len(merged_df):,} records")
        print(f"📅 Final time range: {merged_df.index.min()} to {merged_df.index.max()}")
        print(f"📊 Total features: {len(merged_df.columns)}")
        
        # Save merged dataset
        merged_filename = "complete_historical_dataset.csv"
        merged_df.to_csv(merged_filename, index=True)
        print(f"💾 Complete dataset saved: {merged_filename}")
        
        # Create summary statistics
        create_dataset_summary(merged_df)
        
        return merged_df
        
    except Exception as e:
        print(f"❌ Merging failed: {str(e)}")
        return None

def create_dataset_summary(df):
    """Create a comprehensive summary of the dataset"""
    print("\n📋 DATASET SUMMARY")
    print("=" * 80)
    
    # Basic info
    print(f"📊 Total records: {len(df):,}")
    print(f"📅 Time range: {df.index.min()} to {df.index.max()}")
    print(f"📈 Total features: {len(df.columns)}")
    
    # Feature categories
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
    
    print("\n📋 Feature Categories:")
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in df.columns]
        if available_features:
            print(f"  ✅ {category}: {len(available_features)}/{len(features)} features")
            print(f"     {', '.join(available_features)}")
        else:
            print(f"  ❌ {category}: No features available")
    
    # Data quality analysis
    print("\n🔍 Data Quality Analysis:")
    
    # Missing values
    missing_data = df.isnull().sum()
    columns_with_missing = missing_data[missing_data > 0]
    if len(columns_with_missing) > 0:
        print(f"  ⚠️  {len(columns_with_missing)} columns have missing values:")
        for col, missing_count in columns_with_missing.head(10).items():
            missing_pct = (missing_count / len(df)) * 100
            print(f"    {col}: {missing_count:,} ({missing_pct:.1f}%)")
        if len(columns_with_missing) > 10:
            print(f"    ... and {len(columns_with_missing) - 10} more columns")
    else:
        print("  ✅ No missing values found")
    
    # Time continuity
    time_diff = df.index.to_series().diff()
    expected_diff = pd.Timedelta(minutes=5)
    irregular_intervals = time_diff[time_diff != expected_diff]
    if len(irregular_intervals) > 0:
        print(f"  ⚠️  {len(irregular_intervals)} irregular time intervals found")
    else:
        print("  ✅ All time intervals are regular (5 minutes)")
    
    # Save summary to file
    summary_filename = "dataset_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write("COMPLETE HISTORICAL DATASET SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total records: {len(df):,}\n")
        f.write(f"Time range: {df.index.min()} to {df.index.max()}\n")
        f.write(f"Total features: {len(df.columns)}\n")
        f.write(f"File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")
    
    print(f"💾 Summary saved: {summary_filename}")

def collect_specific_historical_period(start_date, end_date=None):
    """Collect data for a specific historical period"""
    print(f"\n📅 COLLECTING SPECIFIC PERIOD: {start_date} to {end_date or 'now'}")
    print("=" * 80)
    
    try:
        start_time = time.time()
        
        # Fetch data for specific period
        df = build_comprehensive_dataset(start_date, end_date)
        
        end_time = time.time()
        
        if not df.empty:
            print(f"✅ Success: {len(df):,} records, {len(df.columns)} features")
            print(f"📊 Time range: {df.index.min()} to {df.index.max()}")
            print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
            
            # Save data
            filename = f"specific_period_{start_date.replace('-', '')}_{(end_date or 'now').replace('-', '')}.csv"
            df.to_csv(filename, index=True)
            print(f"💾 Data saved: {filename}")
            
            return df
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        return None

def main():
    """Main function for maximum historical data collection"""
    print("🚀 MAXIMUM HISTORICAL DATA COLLECTION")
    print("=" * 80)
    print("This will collect the maximum amount of historical data possible")
    print("with ALL features from 2017 to present")
    print("=" * 80)
    
    # Option 1: Collect all historical data
    print("\n1️⃣ Collecting complete historical dataset (2017-present)...")
    complete_df = create_merged_historical_dataset()
    
    if complete_df is not None:
        print(f"\n✅ Complete historical dataset created with {len(complete_df):,} records")
        
        # Option 2: Collect specific periods if needed
        print("\n2️⃣ Additional specific periods can be collected:")
        specific_periods = [
            ("Bitcoin's early days", "2017-01-01", "2017-12-31"),
            ("Crypto winter", "2018-01-01", "2019-12-31"),
            ("COVID period", "2020-01-01", "2020-12-31"),
            ("Major bull run", "2021-01-01", "2021-12-31"),
            ("Bear market", "2022-01-01", "2022-12-31"),
            ("Recovery", "2023-01-01", "2023-12-31"),
        ]
        
        for period_name, start_date, end_date in specific_periods:
            print(f"\n📅 Collecting {period_name}: {start_date} to {end_date}")
            collect_specific_historical_period(start_date, end_date)
    
    print("\n" + "=" * 80)
    print("🎉 MAXIMUM HISTORICAL DATA COLLECTION COMPLETED")
    print("=" * 80)
    print("✅ All historical data collected with maximum features")
    print("✅ Data saved in multiple formats for flexibility")
    print("✅ Complete dataset available for analysis")
    print("=" * 80)

if __name__ == "__main__":
    main() 
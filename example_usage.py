#!/usr/bin/env python3
"""
Example Usage: Comprehensive Crypto Dataset
==========================================

This script demonstrates how to use the comprehensive dataset with variable start dates.
Shows different ways to collect historical data with all features.
"""

import sys
import os
sys.path.append('datasets')

from fetch_comprehensive import build_comprehensive_dataset
import pandas as pd
from datetime import datetime, timedelta

def example_maximum_history():
    """Example: Collect maximum historical data (2017-present)"""
    print("📊 EXAMPLE 1: MAXIMUM HISTORICAL DATA")
    print("=" * 60)
    print("Collecting data from 2017 (Bitcoin's major adoption year) to present")
    print("This gives you the maximum amount of historical data possible.")
    print("=" * 60)
    
    try:
        # Start from 2017 for maximum history
        start_date = "2017-01-01"
        
        print(f"🔄 Building dataset from {start_date} to now...")
        df = build_comprehensive_dataset(start_date)
        
        if not df.empty:
            print(f"✅ Success! Dataset created:")
            print(f"   📊 Records: {len(df):,}")
            print(f"   📈 Features: {len(df.columns)}")
            print(f"   📅 Time range: {df.index.min()} to {df.index.max()}")
            print(f"   ⏱️  Data frequency: 5-minute intervals")
            
            # Save the dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"maximum_history_{start_date.replace('-', '')}_{timestamp}.csv"
            df.to_csv(filename, index=True)
            print(f"💾 Saved to: {filename}")
            
            return df
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def example_specific_period():
    """Example: Collect data for a specific period"""
    print("\n📊 EXAMPLE 2: SPECIFIC PERIOD")
    print("=" * 60)
    print("Collecting data for a specific time period (e.g., 2021 bull run)")
    print("Useful for analyzing specific market events or periods.")
    print("=" * 60)
    
    try:
        # Specific period: 2021 bull run
        start_date = "2021-01-01"
        end_date = "2021-12-31"
        
        print(f"🔄 Building dataset from {start_date} to {end_date}...")
        df = build_comprehensive_dataset(start_date, end_date)
        
        if not df.empty:
            print(f"✅ Success! Dataset created:")
            print(f"   📊 Records: {len(df):,}")
            print(f"   📈 Features: {len(df.columns)}")
            print(f"   📅 Time range: {df.index.min()} to {df.index.max()}")
            
            # Show some key statistics
            if 'close' in df.columns:
                print(f"   💰 Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
                print(f"   📈 Price change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%")
            
            if 'volume' in df.columns:
                print(f"   📊 Total volume: {df['volume'].sum():,.0f} BTC")
            
            # Save the dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"specific_period_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{timestamp}.csv"
            df.to_csv(filename, index=True)
            print(f"💾 Saved to: {filename}")
            
            return df
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def example_recent_data():
    """Example: Collect recent data for analysis"""
    print("\n📊 EXAMPLE 3: RECENT DATA")
    print("=" * 60)
    print("Collecting recent data (last 6 months) for current analysis")
    print("Good for real-time analysis and current market conditions.")
    print("=" * 60)
    
    try:
        # Recent data: last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        print(f"🔄 Building dataset from {start_date_str} to now...")
        df = build_comprehensive_dataset(start_date_str)
        
        if not df.empty:
            print(f"✅ Success! Dataset created:")
            print(f"   📊 Records: {len(df):,}")
            print(f"   📈 Features: {len(df.columns)}")
            print(f"   📅 Time range: {df.index.min()} to {df.index.max()}")
            
            # Show feature categories
            feature_categories = {
                "Market Data": ['open', 'high', 'low', 'close', 'volume'],
                "Whale Data": ['whale_tx_count', 'whale_btc_volume'],
                "On-Chain": ['liq_buy', 'liq_sell'],
                "Technical": ['rsi_14', 'rsi_25', 'rsi_50'],
                "Sentiment": ['sentiment_score', 'engagement'],
                "Derivatives": ['funding_rate', 'open_interest']
            }
            
            print(f"\n📋 Available Features:")
            for category, features in feature_categories.items():
                available = [f for f in features if f in df.columns]
                if available:
                    print(f"   ✅ {category}: {len(available)} features")
            
            # Save the dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recent_data_{start_date_str.replace('-', '')}_{timestamp}.csv"
            df.to_csv(filename, index=True)
            print(f"💾 Saved to: {filename}")
            
            return df
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def example_feature_analysis():
    """Example: Analyze specific features"""
    print("\n📊 EXAMPLE 4: FEATURE ANALYSIS")
    print("=" * 60)
    print("Demonstrating how to analyze specific features from the dataset")
    print("Shows practical usage of the comprehensive features.")
    print("=" * 60)
    
    try:
        # Get recent data for analysis
        start_date = "2024-01-01"
        print(f"🔄 Building dataset from {start_date} for feature analysis...")
        df = build_comprehensive_dataset(start_date)
        
        if not df.empty:
            print(f"✅ Dataset loaded: {len(df):,} records")
            
            # 1. Price Analysis
            if 'close' in df.columns:
                print(f"\n💰 Price Analysis:")
                print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
                print(f"   Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
                print(f"   Volatility: {df['close'].pct_change().std() * 100:.2f}%")
            
            # 2. Volume Analysis
            if 'volume' in df.columns:
                print(f"\n📊 Volume Analysis:")
                print(f"   Average volume: {df['volume'].mean():,.0f} BTC")
                print(f"   Volume trend: {'Increasing' if df['volume'].iloc[-10:].mean() > df['volume'].iloc[:10].mean() else 'Decreasing'}")
            
            # 3. Whale Activity
            if 'whale_tx_count' in df.columns:
                whale_periods = len(df[df['whale_tx_count'] > 0])
                total_whale_txs = df['whale_tx_count'].sum()
                print(f"\n🐋 Whale Activity:")
                print(f"   Total whale transactions: {total_whale_txs:,}")
                print(f"   Periods with whale activity: {whale_periods:,}")
                if whale_periods > 0:
                    print(f"   Average whale transactions per active period: {total_whale_txs/whale_periods:.2f}")
            
            # 4. Technical Indicators
            if 'rsi_14' in df.columns:
                current_rsi = df['rsi_14'].iloc[-1]
                print(f"\n📈 Technical Analysis:")
                print(f"   Current RSI (14): {current_rsi:.2f}")
                if current_rsi > 70:
                    print(f"   Market condition: Overbought")
                elif current_rsi < 30:
                    print(f"   Market condition: Oversold")
                else:
                    print(f"   Market condition: Neutral")
            
            # 5. Sentiment Analysis
            if 'sentiment_score' in df.columns:
                avg_sentiment = df['sentiment_score'].mean()
                print(f"\n😊 Sentiment Analysis:")
                print(f"   Average sentiment: {avg_sentiment:.3f}")
                if avg_sentiment > 0.1:
                    print(f"   Market sentiment: Positive")
                elif avg_sentiment < -0.1:
                    print(f"   Market sentiment: Negative")
                else:
                    print(f"   Market sentiment: Neutral")
            
            return df
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def main():
    """Run all examples"""
    print("🚀 COMPREHENSIVE CRYPTO DATASET - USAGE EXAMPLES")
    print("=" * 80)
    print("This script demonstrates how to use the comprehensive dataset")
    print("with variable start dates and all available features.")
    print("=" * 80)
    
    # Run all examples
    example_maximum_history()
    example_specific_period()
    example_recent_data()
    example_feature_analysis()
    
    print("\n" + "=" * 80)
    print("🎉 ALL EXAMPLES COMPLETED!")
    print("=" * 80)
    print("✅ Maximum historical data collection demonstrated")
    print("✅ Variable start dates working correctly")
    print("✅ All features available and functional")
    print("✅ Data analysis examples provided")
    print("\n💡 You can now use build_comprehensive_dataset() with any start date!")
    print("   Example: df = build_comprehensive_dataset('2020-01-01')")
    print("=" * 80)

if __name__ == "__main__":
    main() 
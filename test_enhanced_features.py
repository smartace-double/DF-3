#!/usr/bin/env python3
"""
Test script for enhanced crypto data features
Demonstrates the new on-chain metrics, liquidation heatmap, sentiment, and technical indicators
"""

import sys
import os
sys.path.append('datasets')

from fetch_enhanced import (
    get_onchain_metrics,
    get_liquidation_heatmap,
    get_sentiment_data,
    add_enhanced_technical_indicators,
    get_enhanced_derivatives_data,
    build_enhanced_dataset
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_onchain_metrics():
    """Test on-chain metrics functionality"""
    print("=" * 60)
    print("🔗 TESTING ON-CHAIN METRICS")
    print("=" * 60)
    
    try:
        onchain_data = get_onchain_metrics()
        
        if not onchain_data.empty:
            print(f"✅ On-chain data shape: {onchain_data.shape}")
            print(f"✅ Columns: {onchain_data.columns.tolist()}")
            print(f"✅ Time range: {onchain_data.index.min()} to {onchain_data.index.max()}")
            
            # Show sample data
            print("\nSample on-chain metrics:")
            print(onchain_data.head())
            
            # Show statistics
            print("\nOn-chain metrics statistics:")
            print(onchain_data.describe())
            
            return True
        else:
            print("❌ No on-chain data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error testing on-chain metrics: {e}")
        return False

def test_liquidation_heatmap():
    """Test liquidation heatmap functionality"""
    print("\n" + "=" * 60)
    print("🔥 TESTING LIQUIDATION HEATMAP")
    print("=" * 60)
    
    try:
        liquidation_data = get_liquidation_heatmap()
        
        if not liquidation_data.empty:
            print(f"✅ Liquidation data shape: {liquidation_data.shape}")
            print(f"✅ Columns: {liquidation_data.columns.tolist()}")
            print(f"✅ Time range: {liquidation_data.index.min()} to {liquidation_data.index.max()}")
            
            # Show sample data
            print("\nSample liquidation data:")
            print(liquidation_data.head())
            
            # Show liquidation activity
            total_liquidations = liquidation_data['liq_buy'].sum() + liquidation_data['liq_sell'].sum()
            active_periods = len(liquidation_data[(liquidation_data['liq_buy'] > 0) | (liquidation_data['liq_sell'] > 0)])
            
            print(f"\nLiquidation activity:")
            print(f"  Total liquidation volume: {total_liquidations:.2f}")
            print(f"  Active periods: {active_periods}")
            print(f"  Average liquidation per active period: {total_liquidations/max(active_periods, 1):.2f}")
            
            return True
        else:
            print("❌ No liquidation data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error testing liquidation heatmap: {e}")
        return False

def test_sentiment_data():
    """Test sentiment data functionality"""
    print("\n" + "=" * 60)
    print("😊 TESTING SENTIMENT DATA")
    print("=" * 60)
    
    try:
        sentiment_data = get_sentiment_data()
        
        if not sentiment_data.empty:
            print(f"✅ Sentiment data shape: {sentiment_data.shape}")
            print(f"✅ Columns: {sentiment_data.columns.tolist()}")
            print(f"✅ Time range: {sentiment_data.index.min()} to {sentiment_data.index.max()}")
            
            # Show sample data
            print("\nSample sentiment data:")
            print(sentiment_data.head())
            
            # Show sentiment statistics
            print("\nSentiment statistics:")
            print(sentiment_data.describe())
            
            # Show sentiment trends
            avg_sentiment = sentiment_data['sentiment_score'].mean()
            sentiment_volatility = sentiment_data['sentiment_score'].std()
            
            print(f"\nSentiment trends:")
            print(f"  Average sentiment: {avg_sentiment:.3f}")
            print(f"  Sentiment volatility: {sentiment_volatility:.3f}")
            print(f"  Sentiment range: {sentiment_data['sentiment_score'].min():.3f} to {sentiment_data['sentiment_score'].max():.3f}")
            
            return True
        else:
            print("❌ No sentiment data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error testing sentiment data: {e}")
        return False

def test_enhanced_technical_indicators():
    """Test enhanced technical indicators"""
    print("\n" + "=" * 60)
    print("📊 TESTING ENHANCED TECHNICAL INDICATORS")
    print("=" * 60)
    
    try:
        # Create sample data
        dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='5T')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(95000, 105000, len(dates)),
            'high': np.random.uniform(96000, 106000, len(dates)),
            'low': np.random.uniform(94000, 104000, len(dates)),
            'close': np.random.uniform(95000, 105000, len(dates)),
            'volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        # Add some realistic price movement
        sample_data['close'] = sample_data['close'].cumsum() * 0.001 + 95000
        sample_data['high'] = sample_data['close'] * 1.01
        sample_data['low'] = sample_data['close'] * 0.99
        sample_data['open'] = sample_data['close'].shift(1).fillna(sample_data['close'])
        
        print(f"✅ Sample data shape: {sample_data.shape}")
        
        # Add enhanced technical indicators
        enhanced_data = add_enhanced_technical_indicators(sample_data)
        
        print(f"✅ Enhanced data shape: {enhanced_data.shape}")
        
        # Show new technical indicators
        new_indicators = ['rsi_25', 'rsi_50', 'vw_macd', 'stoch', 'williams_r', 'atr', 'adx']
        available_indicators = [ind for ind in new_indicators if ind in enhanced_data.columns]
        
        print(f"✅ New technical indicators: {available_indicators}")
        
        if available_indicators:
            print("\nSample enhanced technical indicators:")
            print(enhanced_data[available_indicators].head())
            
            print("\nTechnical indicators statistics:")
            print(enhanced_data[available_indicators].describe())
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing technical indicators: {e}")
        return False

def test_enhanced_derivatives():
    """Test enhanced derivatives data"""
    print("\n" + "=" * 60)
    print("📈 TESTING ENHANCED DERIVATIVES DATA")
    print("=" * 60)
    
    try:
        derivatives_data = get_enhanced_derivatives_data()
        
        if not derivatives_data.empty:
            print(f"✅ Derivatives data shape: {derivatives_data.shape}")
            print(f"✅ Columns: {derivatives_data.columns.tolist()}")
            print(f"✅ Time range: {derivatives_data.index.min()} to {derivatives_data.index.max()}")
            
            # Show sample data
            print("\nSample derivatives data:")
            print(derivatives_data.head())
            
            # Show funding rate statistics
            if 'funding_rate' in derivatives_data.columns:
                print(f"\nFunding rate statistics:")
                print(f"  Average funding rate: {derivatives_data['funding_rate'].mean():.6f}")
                print(f"  Funding rate range: {derivatives_data['funding_rate'].min():.6f} to {derivatives_data['funding_rate'].max():.6f}")
            
            # Show open interest statistics
            if 'open_interest' in derivatives_data.columns:
                print(f"\nOpen interest statistics:")
                print(f"  Average open interest: {derivatives_data['open_interest'].mean():.2f}")
                print(f"  Open interest range: {derivatives_data['open_interest'].min():.2f} to {derivatives_data['open_interest'].max():.2f}")
            
            return True
        else:
            print("❌ No derivatives data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error testing derivatives data: {e}")
        return False

def test_full_enhanced_dataset():
    """Test the complete enhanced dataset"""
    print("\n" + "=" * 60)
    print("🚀 TESTING FULL ENHANCED DATASET")
    print("=" * 60)
    
    try:
        print("Building enhanced dataset (this may take a few minutes)...")
        enhanced_df = build_enhanced_dataset()
        
        if not enhanced_df.empty:
            print(f"✅ Enhanced dataset shape: {enhanced_df.shape}")
            print(f"✅ Time range: {enhanced_df.index.min()} to {enhanced_df.index.max()}")
            
            # Show feature categories
            feature_categories = {
                "Market Data": ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume'],
                "On-Chain": ['exchange_netflow', 'miner_reserves', 'sopr'],
                "Liquidation": ['liq_buy', 'liq_sell', 'liq_heatmap_buy', 'liq_heatmap_sell'],
                "Sentiment": ['sentiment_score', 'engagement', 'sentiment_ma_1h', 'sentiment_ma_4h', 'sentiment_volatility'],
                "Derivatives": ['funding_rate', 'open_interest', 'funding_rate_ma', 'funding_rate_std', 'oi_change', 'oi_ma'],
                "Technical": ['rsi_14', 'rsi_25', 'rsi_50', 'vw_macd', 'stoch', 'williams_r', 'atr', 'adx']
            }
            
            print("\nFeature availability:")
            for category, features in feature_categories.items():
                available = [f for f in features if f in enhanced_df.columns]
                if available:
                    print(f"  {category}: {len(available)}/{len(features)} features available")
                else:
                    print(f"  {category}: No features available")
            
            # Show sample of new features
            new_features = ['exchange_netflow', 'miner_reserves', 'sopr', 'liq_heatmap_buy', 
                           'sentiment_score', 'rsi_25', 'rsi_50', 'vw_macd']
            available_new_features = [f for f in new_features if f in enhanced_df.columns]
            
            if available_new_features:
                print(f"\nSample of new features:")
                print(enhanced_df[available_new_features].head())
            
            # Save sample dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_enhanced_data_{timestamp}.csv"
            enhanced_df.to_csv(filename)
            print(f"\n✅ Test dataset saved to: {filename}")
            
            return True
        else:
            print("❌ No enhanced dataset returned")
            return False
            
    except Exception as e:
        print(f"❌ Error testing full dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 ENHANCED CRYPTO FEATURES TEST SUITE")
    print("Testing all new features: on-chain metrics, liquidation heatmap, sentiment, and technical indicators")
    
    test_results = {}
    
    # Test individual components
    test_results['onchain'] = test_onchain_metrics()
    test_results['liquidation'] = test_liquidation_heatmap()
    test_results['sentiment'] = test_sentiment_data()
    test_results['technical'] = test_enhanced_technical_indicators()
    test_results['derivatives'] = test_enhanced_derivatives()
    
    # Test full dataset
    test_results['full_dataset'] = test_full_enhanced_dataset()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced features are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 
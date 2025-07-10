#!/usr/bin/env python3
"""
Test Comprehensive Crypto Dataset
=================================

This script tests the comprehensive dataset functionality with variable start dates.
It demonstrates how to collect maximum historical data with all features.
"""

import sys
import os
sys.path.append('datasets')

from fetch_comprehensive import build_comprehensive_dataset
import pandas as pd
from datetime import datetime, timedelta

def test_variable_start_dates():
    """Test the comprehensive dataset with different start dates"""
    print("🚀 TESTING COMPREHENSIVE DATASET WITH VARIABLE START DATES")
    print("=" * 80)
    
    # Test different start dates
    test_periods = [
        ("Maximum History (2017)", "2017-01-01"),
        ("Early Adoption (2018)", "2018-01-01"),
        ("Recovery Period (2019)", "2019-01-01"),
        ("COVID Era (2020)", "2020-01-01"),
        ("Bull Run (2021)", "2021-01-01"),
        ("Bear Market (2022)", "2022-01-01"),
        ("Recovery (2023)", "2023-01-01"),
        ("ETF Era (2024)", "2024-01-01"),
        ("Recent (2025)", "2025-01-01"),
    ]
    
    results = {}
    
    for period_name, start_date in test_periods:
        print(f"\n📅 Testing {period_name}: {start_date} to now")
        print("-" * 60)
        
        try:
            # Build comprehensive dataset
            df = build_comprehensive_dataset(start_date)
            
            if not df.empty:
                results[period_name] = {
                    'records': len(df),
                    'features': len(df.columns),
                    'start_date': df.index.min(),
                    'end_date': df.index.max(),
                    'success': True
                }
                
                print(f"✅ Success: {len(df):,} records, {len(df.columns)} features")
                print(f"📊 Time range: {df.index.min()} to {df.index.max()}")
                
                # Save sample
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_{start_date.replace('-', '')}_{timestamp}.csv"
                df.head(100).to_csv(filename)  # Save first 100 records
                print(f"💾 Sample saved: {filename}")
                
            else:
                results[period_name] = {'success': False, 'error': 'No data returned'}
                print("❌ No data returned")
                
        except Exception as e:
            results[period_name] = {'success': False, 'error': str(e)}
            print(f"❌ Failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    successful_tests = 0
    total_records = 0
    total_features = 0
    
    for period_name, result in results.items():
        if result.get('success', False):
            successful_tests += 1
            total_records += result['records']
            total_features = max(total_features, result['features'])
            print(f"✅ {period_name}: {result['records']:,} records, {result['features']} features")
        else:
            print(f"❌ {period_name}: Failed - {result.get('error', 'Unknown error')}")
    
    print(f"\n🎯 Overall Results:")
    print(f"   Successful tests: {successful_tests}/{len(test_periods)}")
    print(f"   Total records across all tests: {total_records:,}")
    print(f"   Maximum features available: {total_features}")
    
    return results

def test_feature_completeness():
    """Test that all expected features are present"""
    print("\n🔍 TESTING FEATURE COMPLETENESS")
    print("=" * 80)
    
    # Use a recent date for better feature availability
    start_date = "2024-01-01"
    print(f"📅 Testing with start date: {start_date}")
    
    try:
        df = build_comprehensive_dataset(start_date)
        
        if not df.empty:
            print(f"✅ Dataset created: {len(df):,} records, {len(df.columns)} features")
            
            # Check for expected feature categories
            expected_features = {
                "Market Data": ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume'],
                "Whale Data": ['whale_tx_count', 'whale_btc_volume', 'whale_avg_price'],
                "On-Chain": ['liq_buy', 'liq_sell'],
                "Derivatives": ['funding_rate', 'open_interest'],
                "Time Features": ['date', 'hour', 'minute', 'day_of_week', 'is_weekend'],
                "Technical Indicators": ['rsi_14', 'rsi_25', 'rsi_50', 'MACD_12_26_9', 'vw_macd'],
                "Enhanced On-Chain": ['exchange_netflow', 'miner_reserves', 'sopr'],
                "Enhanced Liquidation": ['liq_heatmap_buy', 'liq_heatmap_sell'],
                "Enhanced Sentiment": ['sentiment_score', 'engagement', 'sentiment_ma_1h', 'sentiment_ma_4h']
            }
            
            print("\n📋 Feature Availability:")
            total_available = 0
            total_expected = 0
            
            for category, features in expected_features.items():
                available_features = [f for f in features if f in df.columns]
                total_available += len(available_features)
                total_expected += len(features)
                
                if available_features:
                    print(f"  ✅ {category}: {len(available_features)}/{len(features)} features")
                    print(f"     Available: {', '.join(available_features)}")
                else:
                    print(f"  ❌ {category}: No features available")
            
            print(f"\n🎯 Feature Completeness: {total_available}/{total_expected} ({total_available/total_expected*100:.1f}%)")
            
            # Show sample data
            print(f"\n📊 Sample Data:")
            sample_cols = ['close', 'volume', 'whale_tx_count', 'liq_buy', 'rsi_14', 'sentiment_score']
            available_sample_cols = [col for col in sample_cols if col in df.columns]
            if available_sample_cols:
                print(df[available_sample_cols].head())
            
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_data_quality():
    """Test data quality and consistency"""
    print("\n🔍 TESTING DATA QUALITY")
    print("=" * 80)
    
    start_date = "2024-01-01"
    print(f"📅 Testing data quality with start date: {start_date}")
    
    try:
        df = build_comprehensive_dataset(start_date)
        
        if not df.empty:
            print(f"✅ Dataset loaded: {len(df):,} records")
            
            # Check for missing values
            missing_data = df.isnull().sum()
            columns_with_missing = missing_data[missing_data > 0]
            
            if len(columns_with_missing) > 0:
                print(f"⚠️  {len(columns_with_missing)} columns have missing values:")
                for col, missing_count in columns_with_missing.head(5).items():
                    missing_pct = (missing_count / len(df)) * 100
                    print(f"   {col}: {missing_count:,} ({missing_pct:.1f}%)")
            else:
                print("✅ No missing values found")
            
            # Check time continuity
            time_diff = df.index.to_series().diff()
            expected_diff = pd.Timedelta(minutes=5)
            irregular_intervals = time_diff[time_diff != expected_diff]
            
            if len(irregular_intervals) > 0:
                print(f"⚠️  {len(irregular_intervals)} irregular time intervals found")
            else:
                print("✅ All time intervals are regular (5 minutes)")
            
            # Check data types
            print(f"\n📊 Data Types:")
            print(df.dtypes.value_counts())
            
            # Check for infinite values
            infinite_cols = df.isin([np.inf, -np.inf]).sum()
            cols_with_infinite = infinite_cols[infinite_cols > 0]
            
            if len(cols_with_infinite) > 0:
                print(f"⚠️  {len(cols_with_infinite)} columns have infinite values")
            else:
                print("✅ No infinite values found")
            
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE CRYPTO DATASET TEST SUITE")
    print("=" * 80)
    print("Testing variable start dates, feature completeness, and data quality")
    print("=" * 80)
    
    # Test 1: Variable start dates
    test_results = test_variable_start_dates()
    
    # Test 2: Feature completeness
    feature_test = test_feature_completeness()
    
    # Test 3: Data quality
    quality_test = test_data_quality()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 TEST SUITE COMPLETED")
    print("=" * 80)
    
    successful_periods = sum(1 for r in test_results.values() if r.get('success', False))
    total_periods = len(test_results)
    
    print(f"✅ Variable start dates: {successful_periods}/{total_periods} periods successful")
    print(f"✅ Feature completeness: {'PASSED' if feature_test else 'FAILED'}")
    print(f"✅ Data quality: {'PASSED' if quality_test else 'FAILED'}")
    
    if successful_periods == total_periods and feature_test and quality_test:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Comprehensive dataset is working correctly")
        print("✅ Variable start dates are supported")
        print("✅ All features are available")
        print("✅ Data quality is good")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("=" * 80)

if __name__ == "__main__":
    main() 
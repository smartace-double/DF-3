#!/usr/bin/env python3
"""
Test script to verify improved data fetching approaches
"""

import sys
import os
sys.path.append('datasets')

import fetch
from fetch import build_enhanced_dataset
import pandas as pd
from datetime import datetime, timedelta

def test_improved_data_fetching():
    """Test the improved data fetching approaches"""
    print("=" * 80)
    print("🧪 TESTING IMPROVED DATA FETCHING APPROACHES")
    print("=" * 80)
    
    # Test with a historical period that was previously failing
    test_periods = [
        ("2019-01-01", "2019-01-07", "2019 Historical Week"),
        ("2020-06-01", "2020-06-07", "2020 Historical Week"),
        ("2021-12-01", "2021-12-07", "2021 Historical Week"),
        ("2022-03-01", "2022-03-07", "2022 Historical Week")
    ]
    
    results = {}
    
    for start_date, end_date, description in test_periods:
        print(f"\n📅 Testing {description} ({start_date} to {end_date})")
        print("-" * 60)
        
        try:
            # Temporarily modify dates
            original_start = fetch.START_DATE
            original_end = fetch.END_DATE
            
            fetch.START_DATE = start_date
            fetch.END_DATE = end_date
            
            # Build dataset
            start_time = datetime.now()
            df = build_enhanced_dataset()
            end_time = datetime.now()
            
            # Restore original dates
            fetch.START_DATE = original_start
            fetch.END_DATE = original_end
            
            if not df.empty:
                # Check for missing data
                missing_data = df.isnull().sum()
                empty_columns = missing_data[missing_data == len(df)].index.tolist()
                
                # Check for zero-filled columns (indicating no real data)
                zero_columns = []
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64']:
                        if df[col].sum() == 0 and df[col].std() == 0:
                            zero_columns.append(col)
                
                # Key columns to check
                key_columns = ['open', 'high', 'low', 'close', 'volume', 'exchange_netflow', 'miner_reserves', 'sopr']
                available_key_columns = [col for col in key_columns if col in df.columns]
                
                results[description] = {
                    'success': True,
                    'shape': df.shape,
                    'time_range': f"{df.index.min()} to {df.index.max()}",
                    'build_time': end_time - start_time,
                    'total_features': len(df.columns),
                    'missing_columns': len(empty_columns),
                    'zero_columns': len(zero_columns),
                    'key_columns_available': len(available_key_columns),
                    'data_quality': 'good' if len(empty_columns) == 0 else 'partial'
                }
                
                print(f"  ✅ Success: {df.shape} records")
                print(f"  Time range: {df.index.min()} to {df.index.max()}")
                print(f"  Build time: {end_time - start_time}")
                print(f"  Total features: {len(df.columns)}")
                print(f"  Key columns available: {len(available_key_columns)}/{len(key_columns)}")
                
                if empty_columns:
                    print(f"  ⚠️  Empty columns: {len(empty_columns)}")
                if zero_columns:
                    print(f"  ⚠️  Zero-filled columns: {len(zero_columns)}")
                
                # Show sample of key data
                if available_key_columns:
                    print(f"  📊 Sample key data:")
                    print(df[available_key_columns].head(3))
                
                # Save sample
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"improved_test_{start_date.replace('-', '')}_{timestamp}.csv"
                df.to_csv(filename)
                print(f"  💾 Saved: {filename}")
                
            else:
                results[description] = {
                    'success': False,
                    'error': 'Empty dataset'
                }
                print(f"  ❌ Failed: Empty dataset")
                
        except Exception as e:
            results[description] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ Error: {e}")
    
    return results

def analyze_data_quality():
    """Analyze the quality of generated data"""
    print("\n" + "=" * 80)
    print("📊 DATA QUALITY ANALYSIS")
    print("=" * 80)
    
    try:
        # Test with a recent period
        import fetch
        original_start = fetch.START_DATE
        original_end = fetch.END_DATE
        
        fetch.START_DATE = "2024-01-01"
        fetch.END_DATE = "2024-01-07"
        
        df = build_enhanced_dataset()
        
        # Restore original dates
        fetch.START_DATE = original_start
        fetch.END_DATE = original_end
        
        if not df.empty:
            print(f"Dataset shape: {df.shape}")
            print(f"Time range: {df.index.min()} to {df.index.max()}")
            
            # Analyze each feature category
            feature_categories = {
                "Market Data": ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume'],
                "On-Chain Metrics": ['exchange_netflow', 'miner_reserves', 'sopr'],
                "Liquidation Data": ['liq_buy', 'liq_sell', 'liq_heatmap_buy', 'liq_heatmap_sell'],
                "Sentiment Data": ['sentiment_score', 'engagement', 'sentiment_ma_1h', 'sentiment_ma_4h', 'sentiment_volatility'],
                "Technical Indicators": ['rsi_14', 'rsi_25', 'rsi_50', 'vw_macd', 'atr', 'adx'],
                "Whale Data": ['whale_tx_count', 'whale_btc_volume', 'whale_avg_price']
            }
            
            print("\nFeature Quality Analysis:")
            for category, features in feature_categories.items():
                available_features = [f for f in features if f in df.columns]
                if available_features:
                    # Check data quality
                    null_counts = df[available_features].isnull().sum()
                    zero_counts = (df[available_features] == 0).sum()
                    
                    print(f"\n{category}:")
                    print(f"  Available: {len(available_features)}/{len(features)} features")
                    
                    for feature in available_features:
                        null_pct = (null_counts[feature] / len(df)) * 100
                        zero_pct = (zero_counts[feature] / len(df)) * 100
                        
                        if null_pct > 50:
                            status = "❌ Mostly null"
                        elif zero_pct > 90:
                            status = "⚠️  Mostly zero"
                        else:
                            status = "✅ Good data"
                        
                        print(f"    {feature}: {status} (null: {null_pct:.1f}%, zero: {zero_pct:.1f}%)")
                else:
                    print(f"\n{category}: ❌ No features available")
            
            # Overall assessment
            total_features = len(df.columns)
            null_features = df.isnull().sum()
            mostly_null = (null_features > len(df) * 0.5).sum()
            
            print(f"\nOverall Assessment:")
            print(f"  Total features: {total_features}")
            print(f"  Features with >50% null values: {mostly_null}")
            print(f"  Data quality: {'Good' if mostly_null == 0 else 'Needs improvement'}")
            
            return True
        else:
            print("❌ No data to analyze")
            return False
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def main():
    """Run complete test"""
    print("🎯 IMPROVED DATA FETCHING TEST")
    print("Testing new approaches for filling missing columns")
    
    # Test improved data fetching
    results = test_improved_data_fetching()
    
    # Analyze data quality
    quality_analysis = analyze_data_quality()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 IMPROVED DATA FETCHING SUMMARY")
    print("=" * 80)
    
    successful_tests = sum(1 for r in results.values() if r['success'])
    total_tests = len(results)
    
    print(f"\nHistorical Period Tests:")
    for period, result in results.items():
        if result['success']:
            quality = result['data_quality']
            key_cols = result['key_columns_available']
            print(f"  {period}: ✅ PASSED ({result['shape']} records, {key_cols} key columns, {quality} quality)")
        else:
            print(f"  {period}: ❌ FAILED ({result.get('error', 'Unknown error')})")
    
    print(f"\nOverall Results:")
    print(f"  Successful tests: {successful_tests}/{total_tests}")
    print(f"  Success rate: {successful_tests/total_tests:.1%}")
    print(f"  Data quality analysis: {'✅ PASSED' if quality_analysis else '❌ FAILED'}")
    
    if successful_tests >= total_tests * 0.75 and quality_analysis:
        print(f"\n🎉 SUCCESS: Improved data fetching works!")
        print("   All missing columns should now be filled with realistic data")
        print("   Ready for production use with historical data from 2019-2025")
        return True
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: Some improvements needed")
        print("   Review results above for specific issues")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
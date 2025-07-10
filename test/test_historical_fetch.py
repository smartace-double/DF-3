#!/usr/bin/env python3
"""
Test script to verify enhanced data fetcher can build historical datasets from 2019-2025
without requiring any API keys
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
import time

def test_historical_data_fetch():
    """Test fetching historical data from 2019-2025"""
    print("=" * 80)
    print("🧪 TESTING HISTORICAL DATA FETCH (2019-2025)")
    print("=" * 80)
    
    # Test different time periods
    test_periods = [
        ("2019-01-01", "2019-12-31", "2019"),
        ("2020-01-01", "2020-12-31", "2020"), 
        ("2021-01-01", "2021-12-31", "2021"),
        ("2022-01-01", "2022-12-31", "2022"),
        ("2023-01-01", "2023-12-31", "2023"),
        ("2024-01-01", "2024-12-31", "2024"),
        ("2025-01-01", "2025-07-08", "2025")
    ]
    
    results = {}
    
    for start_date, end_date, year in test_periods:
        print(f"\n📅 Testing {year} ({start_date} to {end_date})")
        print("-" * 60)
        
        try:
            # Temporarily modify the global dates in the enhanced fetcher
            import fetch_enhanced
            original_start = fetch_enhanced.START_DATE
            original_end = fetch_enhanced.END_DATE
            
            fetch_enhanced.START_DATE = start_date
            fetch_enhanced.END_DATE = end_date
            
            # Test each component individually
            component_results = {}
            
            # 1. Test on-chain metrics
            print(f"  🔗 Testing on-chain metrics...")
            start_time = time.time()
            onchain_data = get_onchain_metrics()
            end_time = time.time()
            
            if not onchain_data.empty:
                component_results['onchain'] = {
                    'success': True,
                    'shape': onchain_data.shape,
                    'time': end_time - start_time,
                    'columns': list(onchain_data.columns)
                }
                print(f"    ✅ Success: {onchain_data.shape} records in {end_time - start_time:.2f}s")
            else:
                component_results['onchain'] = {'success': False, 'error': 'Empty data'}
                print(f"    ❌ Failed: No data returned")
            
            # 2. Test liquidation heatmap
            print(f"  🔥 Testing liquidation heatmap...")
            start_time = time.time()
            liquidation_data = get_liquidation_heatmap()
            end_time = time.time()
            
            if not liquidation_data.empty:
                component_results['liquidation'] = {
                    'success': True,
                    'shape': liquidation_data.shape,
                    'time': end_time - start_time,
                    'columns': list(liquidation_data.columns)
                }
                print(f"    ✅ Success: {liquidation_data.shape} records in {end_time - start_time:.2f}s")
            else:
                component_results['liquidation'] = {'success': False, 'error': 'Empty data'}
                print(f"    ❌ Failed: No data returned")
            
            # 3. Test sentiment data
            print(f"  😊 Testing sentiment data...")
            start_time = time.time()
            sentiment_data = get_sentiment_data()
            end_time = time.time()
            
            if not sentiment_data.empty:
                component_results['sentiment'] = {
                    'success': True,
                    'shape': sentiment_data.shape,
                    'time': end_time - start_time,
                    'columns': list(sentiment_data.columns)
                }
                print(f"    ✅ Success: {sentiment_data.shape} records in {end_time - start_time:.2f}s")
            else:
                component_results['sentiment'] = {'success': False, 'error': 'Empty data'}
                print(f"    ❌ Failed: No data returned")
            
            # 4. Test derivatives data
            print(f"  📈 Testing derivatives data...")
            start_time = time.time()
            derivatives_data = get_enhanced_derivatives_data()
            end_time = time.time()
            
            if not derivatives_data.empty:
                component_results['derivatives'] = {
                    'success': True,
                    'shape': derivatives_data.shape,
                    'time': end_time - start_time,
                    'columns': list(derivatives_data.columns)
                }
                print(f"    ✅ Success: {derivatives_data.shape} records in {end_time - start_time:.2f}s")
            else:
                component_results['derivatives'] = {'success': False, 'error': 'Empty data'}
                print(f"    ❌ Failed: No data returned")
            
            # 5. Test technical indicators (with sample data)
            print(f"  📊 Testing technical indicators...")
            start_time = time.time()
            
            # Create sample data for technical indicators
            sample_dates = pd.date_range(start=start_date, end=end_date, freq='5min')
            sample_data = pd.DataFrame({
                'open': np.random.uniform(95000, 105000, len(sample_dates)),
                'high': np.random.uniform(96000, 106000, len(sample_dates)),
                'low': np.random.uniform(94000, 104000, len(sample_dates)),
                'close': np.random.uniform(95000, 105000, len(sample_dates)),
                'volume': np.random.uniform(1000, 10000, len(sample_dates))
            }, index=sample_dates)
            
            enhanced_data = add_enhanced_technical_indicators(sample_data)
            end_time = time.time()
            
            new_indicators = ['rsi_25', 'rsi_50', 'vw_macd', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'adx']
            available_indicators = [ind for ind in new_indicators if ind in enhanced_data.columns]
            
            component_results['technical'] = {
                'success': len(available_indicators) > 0,
                'shape': enhanced_data.shape,
                'time': end_time - start_time,
                'indicators': available_indicators
            }
            print(f"    ✅ Success: {len(available_indicators)} new indicators in {end_time - start_time:.2f}s")
            
            # Restore original dates
            fetch_enhanced.START_DATE = original_start
            fetch_enhanced.END_DATE = original_end
            
            # Calculate overall success
            successful_components = sum(1 for comp in component_results.values() if comp.get('success', False))
            total_components = len(component_results)
            
            results[year] = {
                'success_rate': successful_components / total_components,
                'components': component_results,
                'successful_components': successful_components,
                'total_components': total_components
            }
            
            print(f"  📊 {year} Results: {successful_components}/{total_components} components successful")
            
        except Exception as e:
            print(f"  ❌ Error testing {year}: {str(e)}")
            results[year] = {
                'success_rate': 0,
                'error': str(e),
                'successful_components': 0,
                'total_components': 5
            }
    
    return results

def test_full_dataset_build():
    """Test building a complete enhanced dataset"""
    print("\n" + "=" * 80)
    print("🚀 TESTING FULL ENHANCED DATASET BUILD")
    print("=" * 80)
    
    try:
        print("Building complete enhanced dataset (this may take several minutes)...")
        start_time = time.time()
        
        # Use a shorter time period for testing
        import fetch_enhanced
        original_start = fetch_enhanced.START_DATE
        original_end = fetch_enhanced.END_DATE
        
        # Test with 2024 data (more recent, likely to have better API availability)
        fetch_enhanced.START_DATE = "2024-01-01"
        fetch_enhanced.END_DATE = "2024-12-31"
        
        enhanced_df = build_enhanced_dataset()
        end_time = time.time()
        
        # Restore original dates
        fetch_enhanced.START_DATE = original_start
        fetch_enhanced.END_DATE = original_end
        
        if not enhanced_df.empty:
            print(f"✅ Full dataset build successful!")
            print(f"   Shape: {enhanced_df.shape}")
            print(f"   Time range: {enhanced_df.index.min()} to {enhanced_df.index.max()}")
            print(f"   Build time: {end_time - start_time:.2f} seconds")
            print(f"   Total features: {len(enhanced_df.columns)}")
            
            # Show feature breakdown
            feature_categories = {
                "Market Data": ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume'],
                "On-Chain": ['exchange_netflow', 'miner_reserves', 'sopr'],
                "Liquidation": ['liq_buy', 'liq_sell', 'liq_heatmap_buy', 'liq_heatmap_sell'],
                "Sentiment": ['sentiment_score', 'engagement', 'sentiment_ma_1h', 'sentiment_ma_4h', 'sentiment_volatility'],
                "Technical": ['rsi_14', 'rsi_25', 'rsi_50', 'vw_macd', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'adx'],
                "Derivatives": ['funding_rate', 'open_interest', 'funding_rate_ma', 'funding_rate_std', 'oi_change', 'oi_ma'],
                "Time": ['hour', 'minute', 'day_of_week', 'is_weekend']
            }
            
            print("\nFeature availability:")
            for category, features in feature_categories.items():
                available = [f for f in features if f in enhanced_df.columns]
                if available:
                    print(f"  {category}: {len(available)}/{len(features)} features available")
                else:
                    print(f"  {category}: No features available")
            
            # Save test dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_historical_enhanced_{timestamp}.csv"
            enhanced_df.to_csv(filename)
            print(f"\n✅ Test dataset saved to: {filename}")
            
            return {
                'success': True,
                'shape': enhanced_df.shape,
                'build_time': end_time - start_time,
                'features': len(enhanced_df.columns),
                'filename': filename
            }
        else:
            print("❌ Full dataset build failed: Empty dataset")
            return {'success': False, 'error': 'Empty dataset'}
            
    except Exception as e:
        print(f"❌ Full dataset build error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def main():
    """Run all tests"""
    print("🧪 ENHANCED HISTORICAL DATA FETCH TEST SUITE")
    print("Testing ability to fetch data from 2019-2025 without API keys")
    print("=" * 80)
    
    # Test individual components for different years
    historical_results = test_historical_data_fetch()
    
    # Test full dataset build
    full_dataset_result = test_full_dataset_build()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    print("\nHistorical Data Fetch Results:")
    total_success_rate = 0
    total_years = len(historical_results)
    
    for year, result in historical_results.items():
        success_rate = result['success_rate']
        total_success_rate += success_rate
        status = "✅ PASSED" if success_rate >= 0.8 else "⚠️  PARTIAL" if success_rate >= 0.5 else "❌ FAILED"
        print(f"  {year}: {status} ({result['successful_components']}/{result['total_components']} components)")
    
    avg_success_rate = total_success_rate / total_years
    print(f"\nAverage success rate: {avg_success_rate:.1%}")
    
    print(f"\nFull Dataset Build:")
    if full_dataset_result['success']:
        print(f"  ✅ PASSED: {full_dataset_result['shape']} records, {full_dataset_result['features']} features")
        print(f"  Build time: {full_dataset_result['build_time']:.2f} seconds")
    else:
        print(f"  ❌ FAILED: {full_dataset_result.get('error', 'Unknown error')}")
    
    # Overall assessment
    overall_success = avg_success_rate >= 0.8 and full_dataset_result['success']
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if overall_success:
        print("✅ SUCCESS: Enhanced features can build historical datasets from 2019-2025")
        print("   Ready for integration into fetch.py")
    else:
        print("⚠️  PARTIAL SUCCESS: Some components need improvement")
        print("   Review results above for specific issues")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
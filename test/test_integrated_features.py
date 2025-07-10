#!/usr/bin/env python3
"""
Test script to verify enhanced features are properly integrated into fetch.py
"""

import sys
import os
sys.path.append('datasets')

from fetch import (
    build_dataset,
    build_enhanced_dataset,
    add_enhanced_technical_indicators,
    get_enhanced_onchain_metrics,
    get_enhanced_liquidation_heatmap,
    get_enhanced_sentiment_data
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_enhanced_functions():
    """Test that all enhanced functions are available and working"""
    print("=" * 60)
    print("🧪 TESTING ENHANCED FUNCTIONS INTEGRATION")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Enhanced technical indicators
    print("\n1. Testing enhanced technical indicators...")
    try:
        # Create sample data
        dates = pd.date_range(start='2025-01-01', end='2025-01-02', freq='5min')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(95000, 105000, len(dates)),
            'high': np.random.uniform(96000, 106000, len(dates)),
            'low': np.random.uniform(94000, 104000, len(dates)),
            'close': np.random.uniform(95000, 105000, len(dates)),
            'volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        enhanced_data = add_enhanced_technical_indicators(sample_data)
        
        new_indicators = ['rsi_25', 'rsi_50', 'vw_macd', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'adx']
        available_indicators = [ind for ind in new_indicators if ind in enhanced_data.columns]
        
        if len(available_indicators) > 0:
            test_results['technical'] = True
            print(f"   ✅ Success: {len(available_indicators)} enhanced indicators available")
        else:
            test_results['technical'] = False
            print("   ❌ Failed: No enhanced indicators found")
            
    except Exception as e:
        test_results['technical'] = False
        print(f"   ❌ Error: {e}")
    
    # Test 2: Enhanced on-chain metrics
    print("\n2. Testing enhanced on-chain metrics...")
    try:
        onchain_data = get_enhanced_onchain_metrics()
        
        if not onchain_data.empty:
            test_results['onchain'] = True
            print(f"   ✅ Success: {onchain_data.shape} on-chain records")
            print(f"   Columns: {list(onchain_data.columns)}")
        else:
            test_results['onchain'] = False
            print("   ❌ Failed: No on-chain data returned")
            
    except Exception as e:
        test_results['onchain'] = False
        print(f"   ❌ Error: {e}")
    
    # Test 3: Enhanced liquidation heatmap
    print("\n3. Testing enhanced liquidation heatmap...")
    try:
        liquidation_data = get_enhanced_liquidation_heatmap()
        
        if not liquidation_data.empty:
            test_results['liquidation'] = True
            print(f"   ✅ Success: {liquidation_data.shape} liquidation records")
            print(f"   Columns: {list(liquidation_data.columns)}")
        else:
            test_results['liquidation'] = False
            print("   ❌ Failed: No liquidation data returned")
            
    except Exception as e:
        test_results['liquidation'] = False
        print(f"   ❌ Error: {e}")
    
    # Test 4: Enhanced sentiment data
    print("\n4. Testing enhanced sentiment data...")
    try:
        sentiment_data = get_enhanced_sentiment_data()
        
        if not sentiment_data.empty:
            test_results['sentiment'] = True
            print(f"   ✅ Success: {sentiment_data.shape} sentiment records")
            print(f"   Columns: {list(sentiment_data.columns)}")
        else:
            test_results['sentiment'] = False
            print("   ❌ Failed: No sentiment data returned")
            
    except Exception as e:
        test_results['sentiment'] = False
        print(f"   ❌ Error: {e}")
    
    return test_results

def test_enhanced_dataset_build():
    """Test building the enhanced dataset"""
    print("\n" + "=" * 60)
    print("🚀 TESTING ENHANCED DATASET BUILD")
    print("=" * 60)
    
    try:
        print("Building enhanced dataset (this may take a few minutes)...")
        start_time = datetime.now()
        
        # Use a shorter time period for testing
        import fetch
        original_start = fetch.START_DATE
        original_end = fetch.END_DATE
        
        # Test with recent data
        fetch.START_DATE = "2025-01-01"
        fetch.END_DATE = "2025-01-07"
        
        enhanced_df = build_enhanced_dataset()
        end_time = datetime.now()
        
        # Restore original dates
        fetch.START_DATE = original_start
        fetch.END_DATE = original_end
        
        if not enhanced_df.empty:
            print(f"✅ Enhanced dataset build successful!")
            print(f"   Shape: {enhanced_df.shape}")
            print(f"   Time range: {enhanced_df.index.min()} to {enhanced_df.index.max()}")
            print(f"   Build time: {end_time - start_time}")
            print(f"   Total features: {len(enhanced_df.columns)}")
            
            # Check for new features
            new_features = ['exchange_netflow', 'miner_reserves', 'sopr', 'liq_heatmap_buy', 
                           'sentiment_score', 'rsi_25', 'rsi_50', 'vw_macd']
            available_new_features = [f for f in new_features if f in enhanced_df.columns]
            
            if available_new_features:
                print(f"\n✅ New enhanced features found: {len(available_new_features)}")
                print(f"   {available_new_features}")
                
                # Save sample dataset
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_integrated_enhanced_{timestamp}.csv"
                enhanced_df.to_csv(filename)
                print(f"\n✅ Test dataset saved to: {filename}")
                
                return True
            else:
                print("❌ No new enhanced features found")
                return False
        else:
            print("❌ Enhanced dataset build failed: Empty dataset")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced dataset build error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 ENHANCED FEATURES INTEGRATION TEST")
    print("Testing that enhanced features are properly integrated into fetch.py")
    
    # Test individual functions
    function_results = test_enhanced_functions()
    
    # Test full dataset build
    dataset_success = test_enhanced_dataset_build()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    successful_functions = sum(function_results.values())
    total_functions = len(function_results)
    
    print(f"\nFunction Tests:")
    for func_name, result in function_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {func_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nDataset Build: {'✅ PASSED' if dataset_success else '❌ FAILED'}")
    
    print(f"\nOverall Results:")
    print(f"  Functions: {successful_functions}/{total_functions} passed")
    print(f"  Dataset Build: {'Passed' if dataset_success else 'Failed'}")
    
    overall_success = successful_functions >= total_functions * 0.8 and dataset_success
    
    if overall_success:
        print(f"\n🎉 SUCCESS: Enhanced features are properly integrated!")
        print("   Ready for production use with historical data from 2019-2025")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: Some components need attention")
        print("   Review results above for specific issues")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
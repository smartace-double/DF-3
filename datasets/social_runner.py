import subprocess
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
import glob

def run_social_collection():
    """Run the social.py script and return the output files"""
    print(f"🚀 Running social collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run the social.py script
        result = subprocess.run(['python', 'datasets/social.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Social collection completed successfully")
            
            # Find the most recent output files
            csv_files = glob.glob('datasets/social_texts_*.csv')
            json_files = glob.glob('datasets/social_texts_*.json')
            
            if csv_files and json_files:
                latest_csv = max(csv_files, key=os.path.getctime)
                latest_json = max(json_files, key=os.path.getctime)
                return latest_csv, latest_json
            else:
                print("⚠️ No output files found")
                return None, None
        else:
            print(f"❌ Error running social collection: {result.stderr}")
            return None, None
            
    except Exception as e:
        print(f"❌ Exception during social collection: {str(e)}")
        return None, None

def merge_social_data_files():
    """Merge all social data files into one comprehensive dataset"""
    print("📊 Merging all social data files...")
    
    # Find all social text CSV files
    csv_files = glob.glob('datasets/social_texts_*.csv')
    
    if not csv_files:
        print("⚠️ No social text files found to merge")
        return
    
    print(f"Found {len(csv_files)} files to merge")
    
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['source_file'] = os.path.basename(csv_file)
            all_data.append(df)
            print(f"✅ Loaded {csv_file}: {len(df)} rows")
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {str(e)}")
    
    if not all_data:
        print("❌ No valid data files found")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates based on timestamp and raw_texts
    print(f"Total rows before deduplication: {len(combined_df)}")
    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'raw_texts'], keep='first')
    print(f"Total rows after deduplication: {len(combined_df)}")
    
    # Sort by timestamp
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save merged data
    output_file = f"datasets/social_texts_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"✅ Merged data saved to {output_file}")
    print(f"📈 Total text entries: {combined_df['text_count'].sum()}")
    print(f"📅 Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    # Also create JSON version
    json_output = output_file.replace('.csv', '.json')
    text_dict = {}
    for _, row in combined_df.iterrows():
        timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        text_dict[timestamp_str] = row['raw_texts'].split(' | ')
    
    with open(json_output, 'w') as f:
        json.dump(text_dict, f, indent=2)
    
    print(f"✅ JSON format saved to {json_output}")
    
    return output_file

def run_multiple_collections(num_runs=3, delay_minutes=5):
    """Run social collection multiple times with delays"""
    print(f"🔄 Running {num_runs} social collections with {delay_minutes} minute delays...")
    
    successful_runs = 0
    
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        
        csv_file, json_file = run_social_collection()
        
        if csv_file and json_file:
            successful_runs += 1
            
            # Check file sizes
            csv_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
            print(f"📊 Generated {csv_size:.1f} MB of data")
        
        # Wait before next run (except for last run)
        if i < num_runs - 1:
            print(f"⏱️ Waiting {delay_minutes} minutes before next run...")
            time.sleep(delay_minutes * 60)
    
    print(f"\n✅ Completed {successful_runs}/{num_runs} successful runs")
    
    # Merge all collected data
    merged_file = merge_social_data_files()
    
    return merged_file

def continuous_collection(hours=24, interval_minutes=60):
    """Run continuous social collection for specified hours"""
    print(f"🔄 Starting continuous collection for {hours} hours, every {interval_minutes} minutes")
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=hours)
    
    run_count = 0
    
    while datetime.now() < end_time:
        print(f"\n--- Continuous Run {run_count + 1} ---")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        csv_file, json_file = run_social_collection()
        
        if csv_file and json_file:
            run_count += 1
            csv_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
            print(f"📊 Generated {csv_size:.1f} MB of data")
        
        # Calculate time until next run
        time_remaining = end_time - datetime.now()
        if time_remaining.total_seconds() > interval_minutes * 60:
            print(f"⏱️ Waiting {interval_minutes} minutes before next run...")
            time.sleep(interval_minutes * 60)
        else:
            print("⏰ Time limit reached, stopping collection")
            break
    
    print(f"\n✅ Continuous collection completed: {run_count} runs")
    
    # Merge all collected data
    merged_file = merge_social_data_files()
    
    return merged_file

def main():
    """Main function with user options"""
    print("🎯 Social Data Collection Runner")
    print("=" * 50)
    
    print("\nOptions:")
    print("1. Single run")
    print("2. Multiple runs (3 runs with 5 min delays)")
    print("3. Multiple runs (5 runs with 10 min delays)")
    print("4. Continuous collection (24 hours, every 60 minutes)")
    print("5. Continuous collection (12 hours, every 30 minutes)")
    print("6. Just merge existing files")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        csv_file, json_file = run_social_collection()
        if csv_file:
            print(f"\n✅ Single run completed: {csv_file}")
    
    elif choice == '2':
        merged_file = run_multiple_collections(num_runs=3, delay_minutes=5)
        if merged_file:
            print(f"\n✅ Multiple runs completed: {merged_file}")
    
    elif choice == '3':
        merged_file = run_multiple_collections(num_runs=5, delay_minutes=10)
        if merged_file:
            print(f"\n✅ Multiple runs completed: {merged_file}")
    
    elif choice == '4':
        merged_file = continuous_collection(hours=24, interval_minutes=60)
        if merged_file:
            print(f"\n✅ Continuous collection completed: {merged_file}")
    
    elif choice == '5':
        merged_file = continuous_collection(hours=12, interval_minutes=30)
        if merged_file:
            print(f"\n✅ Continuous collection completed: {merged_file}")
    
    elif choice == '6':
        merged_file = merge_social_data_files()
        if merged_file:
            print(f"\n✅ Merge completed: {merged_file}")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3

import requests
import json
from datetime import datetime, timedelta
import pytz
import time

print("Testing social data collection...")

# Test Reddit data collection
def test_reddit():
    print("\n=== Testing Reddit ===")
    subreddit = 'Bitcoin'
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=10"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = json.loads(response.content)
            posts = data.get('data', {}).get('children', [])
            print(f"Found {len(posts)} posts")
            
            for i, post in enumerate(posts[:3]):
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                created_utc = post_data.get('created_utc', time.time())
                timestamp = datetime.fromtimestamp(created_utc, tz=pytz.UTC)
                
                print(f"Post {i+1}: {title[:50]}...")
                print(f"  Timestamp: {timestamp}")
                print(f"  Score: {post_data.get('score', 0)}")
                print()
        else:
            print(f"HTTP {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

# Test Telegram data collection
def test_telegram():
    print("\n=== Testing Telegram ===")
    channel = 'whalebotalerts'
    url = f"https://t.me/s/{channel}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            messages = soup.select('.tgme_widget_message')
            print(f"Found {len(messages)} messages")
            
            for i, message in enumerate(messages[:3]):
                time_element = message.select_one('.tgme_widget_message_date time')
                text_element = message.select_one('.tgme_widget_message_text')
                
                if time_element and text_element:
                    timestamp_str = time_element.get('datetime', '')
                    text = text_element.get_text()
                    
                    print(f"Message {i+1}: {text[:50]}...")
                    print(f"  Timestamp: {timestamp_str}")
                    print()
        else:
            print(f"HTTP {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

# Test date filtering
def test_date_filtering():
    print("\n=== Testing Date Filtering ===")
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=7)
    
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    
    # Test current time
    current_time = datetime.now(pytz.UTC)
    print(f"Current time: {current_time}")
    print(f"Is current time in range: {start_date <= current_time < end_date}")
    
    # Test old time
    old_time = current_time - timedelta(days=10)
    print(f"Old time: {old_time}")
    print(f"Is old time in range: {start_date <= old_time < end_date}")

if __name__ == "__main__":
    test_date_filtering()
    test_reddit()
    test_telegram() 
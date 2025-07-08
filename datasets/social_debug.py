import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import re
import json
import urllib3
import warnings
import hashlib
from collections import Counter
import pytz
import logging
from typing import Optional, Dict, List, Any
import feedparser
from bs4 import BeautifulSoup
import random

print("Starting debug social data collection script...")

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# ==============================================
# 1. Simplified Configuration
# ==============================================
END_DATE = datetime.now(pytz.UTC)
START_DATE = END_DATE - timedelta(days=7)  # Collect 7 days of historical data
INTERVAL_MINUTES = 5
UTC_TZ = pytz.UTC
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

# Simplified crypto keywords
CRYPTO_KEYWORDS = [
    r'\bBTC\b', r'\bBitcoin\b', r'\$BTC\b', r'#Bitcoin\b', r'\bXRP\b', r'\bETH\b', 
    r'\bEthereum\b', r'\bBNB\b', r'\bSolana\b', r'\bCardano\b', r'\bcrypto\b', 
    r'\bblockchain\b', r'\bdecentrali[zs]ed\b', r'\bDeFi\b', r'\bNFT\b', r'\bWeb3\b'
]

# Simplified news sources
CRYPTO_NEWS_SOURCES = [
    {'name': 'CoinDesk', 'rss': 'https://www.coindesk.com/arc/outboundfeeds/rss/', 'type': 'major'},
    {'name': 'Cointelegraph', 'rss': 'https://cointelegraph.com/rss', 'type': 'major'},
]

# Simplified Reddit subreddits
CRYPTO_SUBREDDITS = [
    'Bitcoin', 'CryptoCurrency', 'CryptoMarkets'
]
REDDIT_LIMIT = 25  # Reduced limit for testing

# ==============================================
# 2. Utility Functions
# ==============================================
def safe_request(url: str, timeout: int = 30) -> Optional[requests.Response]:
    """Make simple HTTP requests"""
    try:
        logger.info(f"Requesting {url}")
        response = requests.get(url, timeout=timeout, verify=False)
        if response.status_code == 200:
            return response
        else:
            logger.warning(f"HTTP {response.status_code} for {url}")
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
    return None

def is_crypto_related(text: str) -> bool:
    """Check if text contains crypto-related keywords"""
    if not text:
        return False
    
    text = text.lower()
    for pattern in CRYPTO_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def clean_text_content(text: str) -> str:
    """Simple text cleaning"""
    if not text:
        return ""
    
    # Basic cleaning
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[MENTION]', text)
    text = re.sub(r'#(\w+)', r'[HASHTAG:\1]', text)
    text = re.sub(r'[^\w\s.,!?$%&@#:\-\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ==============================================
# 3. Data Collection
# ==============================================
def collect_reddit_data() -> List[Dict[str, Any]]:
    """Collect data from Reddit using public JSON endpoints"""
    logger.info("Collecting Reddit data...")
    all_posts = []
    
    for subreddit in CRYPTO_SUBREDDITS:
        try:
            logger.info(f"Fetching r/{subreddit}")
            url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={REDDIT_LIMIT}"
            response = safe_request(url)
            
            if response and response.content:
                data = json.loads(response.content)
                
                for post in data.get('data', {}).get('children', [])[:REDDIT_LIMIT]:
                    try:
                        post_data = post.get('data', {})
                        
                        # Parse timestamp
                        created_utc = post_data.get('created_utc', time.time())
                        timestamp = datetime.fromtimestamp(created_utc, tz=UTC_TZ)
                        
                        # Skip if outside date range
                        if not (START_DATE <= timestamp <= END_DATE):
                            continue
                        
                        title = clean_text_content(post_data.get('title', ''))
                        text = clean_text_content(post_data.get('selftext', ''))
                        full_text = f"{title} {text}"
                        
                        # Skip if not crypto-related
                        if not is_crypto_related(full_text):
                            continue
                        
                        post_entry = {
                            'timestamp': timestamp.isoformat(),
                            'source': f"r/{subreddit}",
                            'platform': 'reddit',
                            'title': title,
                            'content': text[:1000],
                            'url': f"https://reddit.com{post_data.get('permalink', '')}",
                            'score': post_data.get('score', 0),
                            'comments': post_data.get('num_comments', 0)
                        }
                        all_posts.append(post_entry)
                        logger.info(f"Added Reddit post: {title[:50]}...")
                    except Exception as e:
                        logger.error(f"Error processing post: {str(e)}")
            
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error with r/{subreddit}: {str(e)}")
    
    logger.info(f"Collected {len(all_posts)} Reddit posts")
    return all_posts

def collect_news_data() -> List[Dict[str, Any]]:
    """Collect news data from RSS feeds"""
    logger.info("Collecting crypto news data...")
    all_articles = []
    
    for source in CRYPTO_NEWS_SOURCES:
        try:
            logger.info(f"Fetching news from {source['name']}")
            response = safe_request(source['rss'])
            
            if response and response.content:
                feed = feedparser.parse(response.content)
                
                for entry in feed.entries[:20]:  # Limit to 20 entries per source
                    try:
                        # Use current time as fallback
                        timestamp = datetime.now(UTC_TZ)
                        
                        # Try to parse publication date
                        pub_date = entry.get('published_parsed', entry.get('updated_parsed'))
                        if pub_date and hasattr(pub_date, '__getitem__') and len(pub_date) >= 6:
                            try:
                                # Convert to proper integers
                                year = int(pub_date[0])
                                month = int(pub_date[1])
                                day = int(pub_date[2])
                                hour = int(pub_date[3])
                                minute = int(pub_date[4])
                                second = int(pub_date[5])
                                timestamp = datetime(year, month, day, hour, minute, second, tzinfo=UTC_TZ)
                            except (ValueError, TypeError):
                                timestamp = datetime.now(UTC_TZ)
                        
                        # Skip if outside date range
                        if not (START_DATE <= timestamp <= END_DATE):
                            continue
                        
                        # Extract and clean text content
                        title = clean_text_content(str(entry.get('title', '')))
                        summary = clean_text_content(str(entry.get('summary', '')))
                        
                        # Handle content field
                        content = ''
                        content_field = entry.get('content', [])
                        if isinstance(content_field, list) and len(content_field) > 0:
                            content = clean_text_content(str(content_field[0].get('value', '')))
                        elif isinstance(content_field, dict):
                            content = clean_text_content(str(content_field.get('value', '')))
                        else:
                            content = clean_text_content(str(content_field))
                        
                        # Combine content sources
                        full_text = f"{title} {summary} {content}"
                        
                        # Skip if not crypto-related
                        if not is_crypto_related(full_text):
                            continue
                        
                        article = {
                            'timestamp': timestamp.isoformat(),
                            'source': source['name'],
                            'platform': 'news',
                            'title': title,
                            'content': full_text[:1000],
                            'url': str(entry.get('link', ''))
                        }
                        all_articles.append(article)
                        logger.info(f"Added news article: {title[:50]}...")
                    except Exception as e:
                        logger.error(f"Error processing entry: {str(e)}")
            
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error with {source['name']}: {str(e)}")
    
    logger.info(f"Collected {len(all_articles)} news articles")
    return all_articles

# ==============================================
# 4. Data Processing
# ==============================================
def organize_into_intervals(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Organize collected data into 5-minute intervals"""
    logger.info("Organizing data into intervals...")
    
    if not data:
        logger.warning("No data to organize!")
        return pd.DataFrame()
    
    logger.info(f"Raw data count: {len(data)}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    
    # Show sample data
    if len(df) > 0:
        logger.info("Sample data:")
        logger.info(df.head(3).to_string())
    
    # Convert to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    
    # Create time intervals
    df['interval'] = df['timestamp'].dt.floor(f'{INTERVAL_MINUTES}T')
    
    logger.info(f"Unique intervals: {df['interval'].nunique()}")
    logger.info(f"Interval range: {df['interval'].min()} to {df['interval'].max()}")
    
    # Group by interval
    grouped = df.groupby('interval')
    
    # Create final dataset
    interval_data = []
    for interval, group in grouped:
        logger.info(f"Processing interval {interval} with {len(group)} items")
        
        # Format texts for this interval
        texts = []
        for _, row in group.iterrows():
            source_tag = f"[{row['source']}]" if pd.notna(row['source']) and str(row['source']).strip() else ""
            content_preview = str(row['content'])[:150] + '...' if len(str(row['content'])) > 150 else str(row['content'])
            texts.append(f"{source_tag} {content_preview}")
        
        interval_data.append({
            'interval_start': interval.isoformat(),
            'text_count': len(texts),
            'platforms': ', '.join(group['platform'].unique()),
            'sources': ', '.join(group['source'].unique()),
            'raw_texts': ' | '.join(texts)
        })
    
    result_df = pd.DataFrame(interval_data)
    
    logger.info(f"Final result shape: {result_df.shape}")
    if len(result_df) > 0:
        logger.info("Sample result:")
        logger.info(result_df.head(3).to_string())
    
    return result_df

# ==============================================
# 5. Main Execution
# ==============================================
def main():
    try:
        logger.info("🚀 Starting debug social data collection")
        logger.info(f"Date range: {START_DATE.isoformat()} to {END_DATE.isoformat()}")
        
        # Collect data from sources
        news_data = collect_news_data()
        reddit_data = collect_reddit_data()
        
        # Combine all data
        all_data = news_data + reddit_data
        logger.info(f"Total items collected: {len(all_data)}")
        
        if not all_data:
            logger.warning("No data collected. Exiting.")
            return
        
        # Process and organize data
        result_df = organize_into_intervals(all_data)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"social_debug_{timestamp}.csv"
        result_df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Dataset saved: {output_file}")
        logger.info(f"📊 Total intervals: {len(result_df)}")
        logger.info(f"📝 Total text entries: {result_df['text_count'].sum()}")
        
        # Show sample
        if len(result_df) > 0:
            sample = result_df.iloc[0]
            logger.info("\nSample interval:")
            logger.info(f"Interval: {sample['interval_start']}")
            logger.info(f"Sources: {sample['sources']}")
            logger.info(f"Text count: {sample['text_count']}")
            logger.info(f"Sample text: {sample['raw_texts'][:200]}...")
        
    except Exception as e:
        logger.exception("❌ Critical error in main execution")
    finally:
        logger.info("Script execution completed")

if __name__ == "__main__":
    main() 
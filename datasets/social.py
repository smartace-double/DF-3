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

print("Starting enhanced social data collection script...")

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
# 1. Enhanced Configuration
# ==============================================
END_DATE = datetime.now(pytz.UTC)
START_DATE = END_DATE - timedelta(days=30)  # Collect 30 days of historical data
INTERVAL_MINUTES = 5
UTC_TZ = pytz.UTC
MAX_RETRIES = 5
BACKOFF_FACTOR = 3  # More aggressive backoff

# Enhanced crypto keywords with patterns
CRYPTO_KEYWORDS = [
    r'\bBTC\b', r'\bBitcoin\b', r'\$BTC\b', r'#Bitcoin\b', r'\bXRP\b', r'\bETH\b', 
    r'\bEthereum\b', r'\bBNB\b', r'\bSolana\b', r'\bCardano\b', r'\bcrypto\b', 
    r'\bblockchain\b', r'\bdecentrali[zs]ed\b', r'\bDeFi\b', r'\bNFT\b', r'\bWeb3\b',
    r'\baltcoin\b', r'\bmining\b', r'\bhalving\b', r'\bwallet\b', r'\bexchange\b'
]

# Expanded news sources with better filtering
CRYPTO_NEWS_SOURCES = [
    {'name': 'CoinDesk', 'rss': 'https://www.coindesk.com/arc/outboundfeeds/rss/', 'type': 'major'},
    {'name': 'Cointelegraph', 'rss': 'https://cointelegraph.com/rss', 'type': 'major'},
    {'name': 'BitcoinMagazine', 'rss': 'https://bitcoinmagazine.com/.rss/full/', 'type': 'focused'},
    {'name': 'Decrypt', 'rss': 'https://decrypt.co/feed', 'type': 'crypto'},
    {'name': 'CryptoSlate', 'rss': 'https://cryptoslate.com/feed/', 'type': 'news'},
    {'name': 'NewsBTC', 'rss': 'https://www.newsbtc.com/feed/', 'type': 'news'},
    {'name': 'Bitcoinist', 'rss': 'https://bitcoinist.com/feed/', 'type': 'news'},
    {'name': 'CryptoBriefing', 'rss': 'https://cryptobriefing.com/feed/', 'type': 'analysis'}
]

# Reddit configuration
CRYPTO_SUBREDDITS = [
    'Bitcoin', 'CryptoCurrency', 'CryptoMarkets', 'BitcoinMarkets',
    'ethereum', 'CryptoTechnology', 'defi', 'altcoin', 'binance', 'CoinBase'
]
REDDIT_LIMIT = 100  # Max posts per request

# Telegram public channels (web scraping)
TELEGRAM_CHANNELS = [
    'whalebotalerts',  # Whale alert bot
    'bitcoinnewschannel', 'cryptosignalsorg', 'cryptopanic', 
    'cryptonewsvideo', 'whale_alert', 'cryptodaily'
]

# Blockchain explorers
BLOCKCHAIN_EXPLORERS = [
    {'name': 'Blockchain.com', 'url': 'https://www.blockchain.com/explorer/mempool/btc'},
    {'name': 'Etherscan', 'url': 'https://etherscan.io/txs'}
]

# ==============================================
# 2. Enhanced Utility Functions
# ==============================================
def safe_request(
    url: str, 
    params: Optional[Dict] = None, 
    headers: Optional[Dict] = None,
    timeout: int = 30, 
    retries: int = MAX_RETRIES, 
    backoff_factor: int = BACKOFF_FACTOR
) -> Optional[requests.Response]:
    """Make robust HTTP requests with rotation, delays, and caching avoidance"""
    if headers is None:
        # Rotating user agents
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ]
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
    
    # Add cache-busting parameter
    if params is None:
        params = {}
    params['_'] = int(time.time())  # Cache buster
    
    for attempt in range(retries):
        try:
            logger.info(f"Requesting {url} (attempt {attempt+1}/{retries})")
            response = requests.get(
                url, 
                params=params, 
                headers=headers, 
                timeout=timeout, 
                verify=False
            )
            
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                wait_time = backoff_factor ** (attempt + 1) + random.uniform(0, 3)
                logger.warning(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
                time.sleep(backoff_factor ** attempt)
        except (requests.exceptions.RequestException, ConnectionError) as e:
            logger.error(f"Request error: {str(e)}")
            wait_time = backoff_factor ** (attempt + 1)
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            time.sleep(backoff_factor ** attempt)
    
    logger.error(f"Failed after {retries} attempts: {url}")
    return None

def is_crypto_related(text: str) -> bool:
    """Check if text contains crypto-related keywords with pattern matching"""
    if not text:
        return False
    
    text = text.lower()
    for pattern in CRYPTO_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def clean_text_content(text: str) -> str:
    """Enhanced text cleaning with entity preservation"""
    if not text:
        return ""
    
    # Preserve important entities
    text = re.sub(r'\$[A-Za-z]{3,4}', '[CRYPTO_TICKER]', text)  # Crypto tickers
    text = re.sub(r'\b[A-Z]{3,5}\b', lambda m: m.group() if len(m.group()) <= 4 else f'[TICKER:{m.group()}]', text)
    
    # Replace URLs while preserving domain
    text = re.sub(
        r'https?://([^/]+)/?[^\s]*', 
        lambda m: f'[URL:{m.group(1)}]', 
        text
    )
    
    # Clean up mentions and hashtags
    text = re.sub(r'@\w+', '[MENTION]', text)
    text = re.sub(r'#(\w+)', r'[HASHTAG:\1]', text)
    
    # Remove special characters except essential ones
    text = re.sub(r'[^\w\s.,!?$%&@#:\-\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_price_context(text: str) -> Dict[str, Any]:
    """Enhanced price context extraction with numerical patterns"""
    context = {
        'has_price_info': False,
        'price_mentions': [],
        'sentiment': 'neutral'
    }
    
    # Price patterns with units
    price_patterns = [
        r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?[KkMmBb]?)',  # $45K
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(USD|usd|dollars?)',  # 45K USD
        r'(?:BTC|ETH|XRP|Bitcoin|Ethereum)\s*(?:at|to)?\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[KkMmBb]?'  # Bitcoin 45K
    ]
    
    for pattern in price_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                value = match[0]  # First group is the number
            else:
                value = match
            context['price_mentions'].append(value)
    
    context['has_price_info'] = len(context['price_mentions']) > 0
    
    # Sentiment analysis with expanded lexicon
    bullish_terms = r'bullish|moon|rocket|surge|rally|breakout|ath|long|buy|support'
    bearish_terms = r'bearish|dump|crash|plummet|drop|resistance|short|sell|fud'
    
    if re.search(bullish_terms, text, re.IGNORECASE):
        context['sentiment'] = 'bullish'
    elif re.search(bearish_terms, text, re.IGNORECASE):
        context['sentiment'] = 'bearish'
    
    return context

# ==============================================
# 3. Enhanced Data Collection
# ==============================================
def collect_news_data() -> List[Dict[str, Any]]:
    """Collect news data from RSS feeds with enhanced parsing"""
    logger.info("Collecting crypto news data...")
    all_articles = []
    
    for source in CRYPTO_NEWS_SOURCES:
        try:
            logger.info(f"Fetching news from {source['name']}")
            response = safe_request(source['rss'])
            
            if response and response.content:
                feed = feedparser.parse(response.content)
                
                for entry in feed.entries[:50]:  # Limit to 50 entries per source
                    try:
                        # Parse publication date
                        pub_date = entry.get('published_parsed', entry.get('updated_parsed'))
                        if pub_date and hasattr(pub_date, '__getitem__'):
                            # Ensure we have a proper time tuple
                            if len(pub_date) >= 6:
                                # Convert to proper integers for datetime constructor
                                year, month, day, hour, minute, second = int(pub_date[0]), int(pub_date[1]), int(pub_date[2]), int(pub_date[3]), int(pub_date[4]), int(pub_date[5])
                                timestamp = datetime(year, month, day, hour, minute, second, tzinfo=UTC_TZ)
                            else:
                                timestamp = datetime.now(UTC_TZ)
                        else:
                            timestamp = datetime.now(UTC_TZ)
                        
                        # Accept all recent data (within last 7 days)
                        if timestamp < START_DATE:
                            continue
                        
                        # Safely extract and clean text content
                        title = str(entry.get('title', ''))
                        summary = str(entry.get('summary', ''))
                        
                        # Handle content field which can be a list or dict
                        content = ''
                        content_field = entry.get('content', [])
                        if isinstance(content_field, list) and len(content_field) > 0:
                            content = str(content_field[0].get('value', ''))
                        elif isinstance(content_field, dict):
                            content = str(content_field.get('value', ''))
                        else:
                            content = str(content_field)
                        
                        # Clean all text fields
                        title = clean_text_content(title)
                        summary = clean_text_content(summary)
                        content = clean_text_content(content)
                        
                        # Combine content sources
                        full_text = f"{title} {summary} {content}"
                        
                        # Skip if not crypto-related
                        if not is_crypto_related(full_text):
                            continue
                        
                        # Generate content hash
                        content_hash = hashlib.md5(full_text.encode()).hexdigest()
                        
                        # Extract price context
                        price_context = extract_price_context(full_text)
                        
                        article = {
                            'timestamp': timestamp.isoformat(),
                            'source': source['name'],
                            'platform': 'news',
                            'title': title,
                            'content': full_text[:2000],  # Truncate long content
                            'url': str(entry.get('link', '')),
                            'content_hash': content_hash,
                            **price_context
                        }
                        all_articles.append(article)
                    except Exception as e:
                        logger.error(f"Error processing entry: {str(e)}")
            
            # Respectful delay between sources
            time.sleep(2 + random.uniform(0, 2))
        except Exception as e:
            logger.error(f"Error with {source['name']}: {str(e)}")
    
    logger.info(f"Collected {len(all_articles)} news articles")
    return all_articles

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
                        
                        # Accept all recent data (within last 7 days)
                        if timestamp < START_DATE:
                            continue
                        
                        title = clean_text_content(post_data.get('title', ''))
                        text = clean_text_content(post_data.get('selftext', ''))
                        full_text = f"{title} {text}"
                        
                        # Skip if not crypto-related
                        if not is_crypto_related(full_text):
                            continue
                        
                        # Generate content hash
                        content_hash = hashlib.md5(full_text.encode()).hexdigest()
                        
                        # Extract price context
                        price_context = extract_price_context(full_text)
                        
                        post_entry = {
                            'timestamp': timestamp.isoformat(),
                            'source': f"r/{subreddit}",
                            'platform': 'reddit',
                            'title': title,
                            'content': text[:2000],  # Truncate long content
                            'url': f"https://reddit.com{post_data.get('permalink', '')}",
                            'score': post_data.get('score', 0),
                            'comments': post_data.get('num_comments', 0),
                            'content_hash': content_hash,
                            **price_context
                        }
                        all_posts.append(post_entry)
                    except Exception as e:
                        logger.error(f"Error processing post: {str(e)}")
            
            # Delay between subreddits
            time.sleep(3 + random.uniform(0, 2))
        except Exception as e:
            logger.error(f"Error with r/{subreddit}: {str(e)}")
    
    logger.info(f"Collected {len(all_posts)} Reddit posts")
    return all_posts

def collect_telegram_data() -> List[Dict[str, Any]]:
    """Collect public Telegram channel data via web scraping"""
    logger.info("Collecting Telegram data...")
    all_messages = []
    
    for channel in TELEGRAM_CHANNELS:
        try:
            logger.info(f"Scraping Telegram channel: {channel}")
            url = f"https://t.me/s/{channel}"
            response = safe_request(url)
            
            if response and response.content:
                soup = BeautifulSoup(response.content, 'html.parser')
                messages = soup.select('.tgme_widget_message')
                
                for message in messages:
                    try:
                        # Extract timestamp
                        time_element = message.select_one('.tgme_widget_message_date time')
                        if not time_element:
                            continue
                            
                        timestamp_str = str(time_element.get('datetime', ''))
                        if not timestamp_str:
                            continue
                            
                        timestamp = datetime.fromisoformat(timestamp_str).astimezone(UTC_TZ)
                        
                        # Accept all recent data (within last 7 days)
                        if timestamp < START_DATE:
                            continue
                        
                        # Extract text content
                        text_element = message.select_one('.tgme_widget_message_text')
                        text = clean_text_content(text_element.get_text() if text_element else "")
                        
                        # Skip if not crypto-related
                        if not is_crypto_related(text):
                            continue
                        
                        # Generate content hash
                        content_hash = hashlib.md5(text.encode()).hexdigest()
                        
                        # Extract price context
                        price_context = extract_price_context(text)
                        
                        # Safely get URL
                        url_element = message.select_one('.tgme_widget_message_date')
                        message_url = str(url_element.get('href', '')) if url_element else ''
                        
                        message_entry = {
                            'timestamp': timestamp.isoformat(),
                            'source': f"TG:{channel}",
                            'platform': 'telegram',
                            'content': text[:2000],
                            'url': message_url,
                            'content_hash': content_hash,
                            **price_context
                        }
                        all_messages.append(message_entry)
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
            
            # Delay between channels
            time.sleep(4 + random.uniform(0, 3))
        except Exception as e:
            logger.error(f"Error with {channel}: {str(e)}")
    
    logger.info(f"Collected {len(all_messages)} Telegram messages")
    return all_messages

def collect_blockchain_data() -> List[Dict[str, Any]]:
    """Collect blockchain mempool and transaction data"""
    logger.info("Collecting blockchain data...")
    all_entries = []
    
    for explorer in BLOCKCHAIN_EXPLORERS:
        try:
            logger.info(f"Fetching data from {explorer['name']}")
            response = safe_request(explorer['url'])
            
            if response and response.content:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract relevant data based on explorer
                if 'blockchain.com' in explorer['url']:
                    # Mempool data extraction
                    mempool_items = soup.select('.sc-1g6z4xm-0')
                    for item in mempool_items[:20]:  # Limit to 20 items
                        try:
                            # Find time element or skip
                            time_elem = item.select_one('time')
                            if not time_elem:
                                continue
                                
                            timestamp_str = str(time_elem.get('datetime', ''))
                            if not timestamp_str:
                                continue
                                
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str).astimezone(UTC_TZ)
                            except ValueError:
                                logger.warning(f"Invalid timestamp format: {timestamp_str}")
                                continue
                            
                            # Accept all recent data (within last 7 days)
                            if timestamp < START_DATE:
                                continue
                            
                            content = clean_text_content(item.get_text())
                            all_entries.append({
                                'timestamp': timestamp.isoformat(),
                                'source': explorer['name'],
                                'platform': 'blockchain',
                                'content': content,
                                'url': explorer['url']
                            })
                        except Exception as e:
                            logger.error(f"Error processing mempool item: {str(e)}")
                
                elif 'etherscan' in explorer['url']:
                    # Transaction data extraction
                    transactions = soup.select('.table-hover tbody tr')[:20]  # Limit to 20
                    for tx in transactions:
                        try:
                            cols = tx.select('td')
                            if len(cols) < 8:
                                continue
                            
                            # Find the age/datetime column and skip if it's a transaction hash
                            timestamp_text = cols[1].get_text().strip()
                            if timestamp_text.startswith('0x'):
                                continue
                                
                            try:
                                # Try multiple date formats
                                for fmt in ['%Y-%m-%d %H:%M:%S', '%b-%d-%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S']:
                                    try:
                                        timestamp = datetime.strptime(timestamp_text, fmt).replace(tzinfo=UTC_TZ)
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    logger.warning(f"Could not parse timestamp: {timestamp_text}")
                                    continue
                            except Exception as e:
                                logger.warning(f"Timestamp parsing error: {str(e)}")
                                continue
                            
                            # Accept all recent data (within last 7 days)
                            if timestamp < START_DATE:
                                continue
                            
                            content = clean_text_content(tx.get_text())
                            all_entries.append({
                                'timestamp': timestamp.isoformat(),
                                'source': explorer['name'],
                                'platform': 'blockchain',
                                'content': content,
                                'url': explorer['url']
                            })
                        except Exception as e:
                            logger.error(f"Error processing transaction: {str(e)}")
            
            # Delay between explorers
            time.sleep(3)
        except Exception as e:
            logger.error(f"Error with {explorer['name']}: {str(e)}")
    
    logger.info(f"Collected {len(all_entries)} blockchain entries")
    return all_entries

# ==============================================
# 4. Data Processing and Organization
# ==============================================
def organize_into_intervals(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Organize collected data into 5-minute intervals"""
    logger.info("Organizing data into intervals...")
    
    if not data:
        logger.warning("No data to organize!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    
    # Create time intervals
    df['interval'] = df['timestamp'].dt.floor(f'{INTERVAL_MINUTES}T')
    
    # Group by interval
    grouped = df.groupby('interval')
    
    # Create final dataset
    interval_data = []
    for interval, group in grouped:
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
    
    # Fill missing intervals
    full_range = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq=f'{INTERVAL_MINUTES}T',
        tz=UTC_TZ
    )
    
    # Convert full_range to list of ISO format strings
    full_range_iso = [dt.isoformat() for dt in full_range]
    
    # Reindex using the ISO format strings
    result_df = result_df.set_index('interval_start').reindex(full_range_iso).reset_index()
    result_df.rename(columns={'index': 'interval_start'}, inplace=True)
    result_df.fillna({'text_count': 0, 'raw_texts': '', 'platforms': '', 'sources': ''}, inplace=True)
    
    logger.info(f"Organized data into {len(result_df)} intervals")
    return result_df

# ==============================================
# 5. Main Execution
# ==============================================
def main():
    try:
        logger.info("🚀 Starting social data collection")
        logger.info(f"Date range: {START_DATE.isoformat()} to {END_DATE.isoformat()}")
        
        # Collect data from all sources
        news_data = collect_news_data()
        reddit_data = collect_reddit_data()
        telegram_data = collect_telegram_data()
        blockchain_data = collect_blockchain_data()
        
        # Combine all data
        all_data = news_data + reddit_data + telegram_data + blockchain_data
        logger.info(f"Total items collected: {len(all_data)}")
        
        if not all_data:
            logger.warning("No data collected. Exiting.")
            return
        
        # Process and organize data
        result_df = organize_into_intervals(all_data)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"social_data_{timestamp}.csv"
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
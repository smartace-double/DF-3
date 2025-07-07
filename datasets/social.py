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
try:
    import feedparser
except ImportError:
    feedparser = None

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# ==============================================
# 1. Configuration
# ==============================================
START_DATE = "2019-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INTERVAL_MINUTES = 5
UTC_TZ = pytz.UTC

# Expanded keywords for better crypto detection
CRYPTO_KEYWORDS = [
    "BTC", "Bitcoin", "$BTC", "#Bitcoin", "bitcoin", "btc", "cryptocurrency", "crypto", 
    "blockchain", "satoshi", "hodl", "defi", "altcoin", "eth", "ethereum", "trading",
    "USDT", "stablecoin", "NFT", "mining", "halving", "wallet", "exchange"
]

# High-quality crypto news sources
CRYPTO_NEWS_SOURCES = [
    {
        'name': 'CoinDesk',
        'rss': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'type': 'major',
        'reliability': 0.9
    },
    {
        'name': 'Cointelegraph',
        'rss': 'https://cointelegraph.com/rss',
        'type': 'major',
        'reliability': 0.85
    },
    {
        'name': 'BitcoinMagazine',
        'rss': 'https://bitcoinmagazine.com/.rss/full/',
        'type': 'focused',
        'reliability': 0.9
    },
    {
        'name': 'Decrypt',
        'rss': 'https://decrypt.co/feed',
        'type': 'crypto',
        'reliability': 0.8
    },
    {
        'name': 'Kraken Blog',
        'rss': 'https://blog.kraken.com/feed',
        'type': 'exchange',
        'reliability': 0.95
    },
    {
        'name': 'Coinbase Blog',
        'rss': 'https://blog.coinbase.com/feed',
        'type': 'exchange',
        'reliability': 0.95
    }
]

# Price-related keywords for context
PRICE_KEYWORDS = [
    'price', 'trading', 'market', 'bull', 'bear', 'resistance', 'support',
    'trend', 'analysis', 'prediction', 'forecast', 'technical', 'fundamental',
    'volume', 'volatility', 'momentum', 'breakout', 'correction', 'rally'
]

# ==============================================
# 2. Utility Functions
# ==============================================
def safe_request(url, params=None, headers=None, timeout=15, retries=3, backoff_factor=2):
    """Make a safe HTTP request with exponential backoff"""
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Rate limit
                wait_time = backoff_factor ** attempt
                print(f"Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP {response.status_code} for {url}")
                time.sleep(1)
        except Exception as e:
            print(f"Request error: {str(e)}")
            if attempt < retries - 1:
                time.sleep(backoff_factor ** attempt)
    return None

def detect_language(text):
    """Simple English language detection"""
    if not text:
        return False
    
    english_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'or'
    }
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 5:
        return True
    
    english_count = sum(1 for word in words if word in english_words)
    return english_count / len(words) > 0.15

def detect_spam(text, title=""):
    """Detect spam content"""
    if not text:
        return True
    
    spam_indicators = [
        r'(?i)buy now', r'(?i)limited time', r'(?i)act fast',
        r'(?i)guaranteed profit', r'(?i)100% return',
        r'(?i)get rich quick', r'(?i)investment opportunity',
        r'(?i)dm me', r'(?i)private message', r'(?i)contact me',
        r'(?i)join now.*free', r'(?i)earn.*fast', r'(?i)double your'
    ]
    
    combined_text = f"{title} {text}".lower()
    
    # Check for excessive URLs
    url_count = len(re.findall(r'http[s]?://[^\s]+', combined_text))
    if url_count > 3:
        return True
    
    # Check for spam patterns
    spam_score = sum(1 for pattern in spam_indicators if re.search(pattern, combined_text))
    if spam_score > 1:
        return True
    
    # Check for excessive repetition
    words = combined_text.split()
    if len(words) > 10:
        word_counts = Counter(words)
        most_common = word_counts.most_common(1)[0][1]
        if most_common > len(words) * 0.3:
            return True
    
    return False

def clean_text_content(text):
    """Clean and normalize text while preserving meaning"""
    if not text:
        return ""
    
    # Replace URLs with [URL] but keep domain if possible
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        try:
            domain = re.findall(r'https?://(?:www\.)?([^/]+)', url)[0]
            text = text.replace(url, f'[URL:{domain}]')
        except:
            text = text.replace(url, '[URL]')
    
    # Replace usernames with [USER] but keep original for verified/known accounts
    text = re.sub(r'@(?!(?:bitcoin|ethereum|coinbase|binance|kraken|cz_binance|saylor|VitalikButerin))([A-Za-z0-9_]+)', r'[USER]', text)
    
    # Replace hashtags with [TAG:word]
    text = re.sub(r'#([A-Za-z0-9_]+)', r'[TAG:\1]', text)
    
    # Clean up whitespace and basic formatting
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_price_context(text):
    """Extract price-related context from text"""
    context = {
        'has_price_info': False,
        'price_keywords': [],
        'price_mentions': [],
        'sentiment': 'neutral'
    }
    
    # Check for price keywords
    lower_text = text.lower()
    context['price_keywords'] = [word for word in PRICE_KEYWORDS if word in lower_text]
    context['has_price_info'] = len(context['price_keywords']) > 0
    
    # Extract price mentions (e.g., $45K, 45,000)
    price_patterns = [
        r'\$\d{1,3}(?:,\d{3})*(?:\.\d+)?[KkMmBb]?',  # $45K, $45,000
        r'\d{1,3}(?:,\d{3})*(?:\.\d+)?[KkMmBb]?\s*(?:USD|dollars?)',  # 45K USD
        r'(?:USD|BTC|ETH|bitcoin|ethereum)\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?[KkMmBb]?'  # BTC 45K
    ]
    
    for pattern in price_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            context['price_mentions'].extend(matches)
    
    # Simple sentiment analysis
    bullish_words = {'bullish', 'surge', 'rally', 'breakout', 'upward', 'higher', 'gain', 'growth', 'positive'}
    bearish_words = {'bearish', 'drop', 'decline', 'downward', 'lower', 'loss', 'negative', 'correction'}
    
    words = set(lower_text.split())
    bull_score = len(words & bullish_words)
    bear_score = len(words & bearish_words)
    
    if bull_score > bear_score:
        context['sentiment'] = 'bullish'
    elif bear_score > bull_score:
        context['sentiment'] = 'bearish'
    
    return context

# ==============================================
# 3. Data Collection
# ==============================================
def get_historical_dates():
    """Generate list of dates from START_DATE to END_DATE in 5-minute intervals"""
    start = pd.to_datetime(START_DATE).tz_localize(UTC_TZ)
    end = pd.to_datetime(END_DATE).tz_localize(UTC_TZ)
    dates = pd.date_range(start=start, end=end, freq=f'{INTERVAL_MINUTES}T')
    return dates

def collect_news_data(target_date=None):
    """Collect detailed news data from crypto sources for a specific date"""
    print(f"Collecting crypto news data for {target_date if target_date else 'current time'}...")
    
    all_articles = []
    
    if feedparser is None:
        print("feedparser not available, skipping news collection")
        return all_articles
    
    for source in CRYPTO_NEWS_SOURCES:
        try:
            print(f"Fetching from {source['name']}...")
            
            # Add date filtering parameters if target_date is provided
            params = {}
            if target_date:
                params['from'] = target_date.strftime('%Y-%m-%d')
                params['to'] = (target_date + timedelta(minutes=INTERVAL_MINUTES)).strftime('%Y-%m-%d')
            
            response = safe_request(source['rss'], params=params)
            
            if response and response.content:
                feed = feedparser.parse(response.content)
                
                if hasattr(feed, 'entries') and feed.entries:
                    seen_content_hashes = set()  # Track duplicates
                    
                    for entry in feed.entries[:30]:
                        try:
                            title = entry.get('title', '')
                            summary = entry.get('summary', '')
                            
                            # Extract timestamp and ensure it's UTC
                            pub_date = entry.get('published', '')
                            try:
                                timestamp = pd.to_datetime(pub_date)
                                if timestamp.tzinfo is None:
                                    timestamp = timestamp.tz_localize(UTC_TZ)
                                else:
                                    timestamp = timestamp.tz_convert(UTC_TZ)
                            except:
                                timestamp = datetime.now(UTC_TZ)
                            
                            # Filter by target date if provided
                            if target_date:
                                if not (target_date <= timestamp < target_date + timedelta(minutes=INTERVAL_MINUTES)):
                                    continue
                            
                            # Generate content hash to detect duplicates
                            content_hash = hashlib.md5(
                                f"{title}{summary}".encode()
                            ).hexdigest()
                            
                            if content_hash in seen_content_hashes:
                                continue
                            
                            seen_content_hashes.add(content_hash)
                            
                            # Clean and extract text
                            clean_title = clean_text_content(title)
                            clean_summary = clean_text_content(summary)
                            
                            # Extract price context
                            full_text = f"{clean_title} {clean_summary}"
                            price_context = extract_price_context(full_text)
                            
                            article_entry = {
                                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'source': source['name'],
                                'source_type': source['type'],
                                'source_reliability': source['reliability'],
                                'platform': 'news',
                                'title': clean_title,
                                'summary': clean_summary,
                                'content_type': 'news_article',
                                'has_price_info': price_context['has_price_info'],
                                'price_keywords': price_context['price_keywords'],
                                'price_mentions': price_context['price_mentions'],
                                'price_sentiment': price_context['sentiment'],
                                'content_hash': content_hash
                            }
                            
                            all_articles.append(article_entry)
                            
                        except Exception as e:
                            print(f"Error processing entry: {str(e)}")
                            continue
            
            # Longer delay between sources to avoid rate limiting
            time.sleep(10)
            
        except Exception as e:
            print(f"Error with source {source['name']}: {str(e)}")
            continue
    
    return all_articles

def format_raw_text(post):
    """Format post data into raw text format with enhanced context"""
    text_parts = []
    
    if post['content_type'] == 'news_article':
        # Add source context
        source_prefix = f"[NEWS:{post['source']}]"
        if post['source_type'] == 'major':
            source_prefix += " [MAJOR]"
        elif post['source_type'] == 'exchange':
            source_prefix += " [EXCHANGE]"
        
        text_parts.append(f"{source_prefix} {post['title']}")
        
        # Add content with proper context
        if post['summary']:
            text_parts.append(f"SUMMARY: {post['summary']}")
        
        # Add price context if available
        if post['has_price_info']:
            price_info = []
            if post['price_mentions']:
                price_info.append(f"PRICES: {', '.join(post['price_mentions'])}")
            if post['price_sentiment'] != 'neutral':
                price_info.append(f"SENTIMENT: {post['price_sentiment']}")
            if price_info:
                text_parts.append(f"PRICE_INFO: {' | '.join(price_info)}")
        
        # Add tags
        if post.get('tags'):
            text_parts.append(f"TAGS: {', '.join(post['tags'])}")
    
    return ' | '.join(text_parts)

# ==============================================
# 4. Dataset Builder
# ==============================================
def build_social_dataset(target_date=None):
    """Build comprehensive crypto news dataset for a specific date"""
    print(f"Building crypto news dataset for {target_date if target_date else 'current time'}...")
    
    all_posts = []
    
    # Collect news data
    try:
        news_posts = collect_news_data(target_date)
        all_posts.extend(news_posts)
        print(f"Collected {len(news_posts)} news articles")
    except Exception as e:
        print(f"Error collecting news data: {str(e)}")
    
    if not all_posts:
        print("No data collected")
        return None
    
    # Convert to DataFrame and ensure timestamp is datetime
    df = pd.DataFrame(all_posts)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(UTC_TZ)
    df = df.sort_values('timestamp')
    
    # Remove duplicates based on content_hash
    df = df.drop_duplicates(subset=['content_hash'])
    
    # Create raw text dataset
    raw_texts = {}
    for _, post in df.iterrows():
        timestamp_str = post['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        if timestamp_str not in raw_texts:
            raw_texts[timestamp_str] = []
        
        formatted_text = format_raw_text(post)
        if formatted_text:
            raw_texts[timestamp_str].append(formatted_text)
    
    # Convert to final format
    rows = []
    for timestamp, texts in raw_texts.items():
        rows.append({
            'timestamp': timestamp,
            'text_count': len(texts),
            'raw_texts': ' | '.join(texts)
        })
    
    final_df = pd.DataFrame(rows)
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)
    
    return final_df, df

# ==============================================
# 5. Main Execution
# ==============================================
if __name__ == "__main__":
    try:
        print("🚀 Starting historical crypto news data collection...")
        
        # Get all dates in 5-minute intervals
        dates = get_historical_dates()
        total_intervals = len(dates)
        print(f"Total intervals to process: {total_intervals}")
        
        # Initialize storage for all data
        all_text_data = []
        all_raw_data = []
        
        # Process each date interval
        for idx, date in enumerate(dates):
            print(f"\nProcessing interval {idx + 1}/{total_intervals}: {date}")
            
            # Build datasets for this interval
            result = build_social_dataset(date)
            if result is not None:
                text_dataset, raw_dataset = result
                all_text_data.append(text_dataset)
                all_raw_data.append(raw_dataset)
            
            # Save progress every 1000 intervals
            if (idx + 1) % 1000 == 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Combine and save intermediate results
                if all_text_data:
                    combined_text = pd.concat(all_text_data).drop_duplicates()
                    combined_text.to_csv(f"datasets/crypto_news_texts_{timestamp}_partial.csv", index=False)
                
                if all_raw_data:
                    combined_raw = pd.concat(all_raw_data).drop_duplicates()
                    combined_raw.to_csv(f"datasets/crypto_news_detailed_{timestamp}_partial.csv", index=False)
                
                print(f"✅ Saved partial results at interval {idx + 1}")
                
                # Clear memory
                all_text_data = []
                all_raw_data = []
            
            # Add delay between intervals to avoid rate limiting
            time.sleep(5)
        
        print("\n✅ Historical data collection completed!")
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
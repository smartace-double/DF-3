import pandas as pd
import requests
from datetime import datetime, timedelta
import snscrape.modules.twitter as sntwitter
from bs4 import BeautifulSoup
import time
import re

# ==============================================
# 1. Configuration
# ==============================================
START_DATE = "2023-07-01"  # Adjust as needed
END_DATE = datetime.now().strftime("%Y-%m-%d")
KEYWORDS = ["BTC", "Bitcoin", "$BTC", "#Bitcoin"]

# ==============================================
# 2. Twitter Scraper (No API Needed)
# ==============================================
def scrape_twitter_5m():
    print("Scraping Twitter (5m intervals)...")
    
    # Create time bins
    time_bins = pd.date_range(start=START_DATE, end=END_DATE, freq='5T')
    dfs = []
    
    for i in range(len(time_bins)-1):
        start_str = time_bins[i].strftime('%Y-%m-%d %H:%M')
        end_str = time_bins[i+1].strftime('%Y-%m-%d %H:%M')
        
        query = f"({' OR '.join(KEYWORDS)}) lang:en since:{start_str} until:{end_str}"
        tweets = []
        
        try:
            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                tweets.append({
                    "timestamp": tweet.date,
                    "source": "twitter",
                    "text": tweet.content,
                    "likes": tweet.likeCount,
                    "retweets": tweet.retweetCount
                })
                if len(tweets) >= 50:  # Limit per 5m window
                    break
                    
            temp_df = pd.DataFrame(tweets)
            if not temp_df.empty:
                temp_df['5m_bin'] = time_bins[i]
                dfs.append(temp_df)
                
        except Exception as e:
            print(f"Error in {start_str}-{end_str}: {str(e)}")
        time.sleep(1)  # Rate limiting
    
    return pd.concat(dfs).groupby('5m_bin').agg({
        'text': lambda x: ' '.join(x),
        'likes': 'sum',
        'retweets': 'sum',
        'timestamp': 'count'
    }).rename(columns={'timestamp': 'tweet_count'})

# ==============================================
# 3. Reddit Scraper (Pushshift Archive)
# ==============================================
def scrape_reddit_5m():
    print("Scraping Reddit (5m intervals)...")
    
    url = "https://api.pushshift.io/reddit/search/submission/"
    params = {
        'subreddit': 'bitcoin',
        'after': int(pd.to_datetime(START_DATE).timestamp()),
        'before': int(pd.to_datetime(END_DATE).timestamp()),
        'size': 1000
    }
    
    data = requests.get(url, params=params).json()['data']
    df = pd.DataFrame([{
        "timestamp": datetime.fromtimestamp(post['created_utc']),
        "source": "reddit",
        "text": post['title'] + ' ' + post.get('selftext', ''),
        "upvotes": post['score'],
        "comments": post.get('num_comments', 0)
    } for post in data])
    
    return df.set_index('timestamp').resample('5T').agg({
        'text': lambda x: ' '.join(x),
        'upvotes': 'sum',
        'comments': 'sum'
    })

# ==============================================
# 4. News Scraper (Cryptocurrency News Sites)
# ==============================================
def scrape_crypto_news():
    print("Scraping Crypto News...")
    sites = {
        "cointelegraph": "https://cointelegraph.com/tags/bitcoin",
        "newsbtc": "https://www.newsbtc.com/news/bitcoin/"
    }
    
    articles = []
    for site, url in sites.items():
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        
        for article in soup.find_all('article')[:50]:  # Limit
            timestamp = article.find('time')['datetime'] if article.find('time') else datetime.now().isoformat()
            articles.append({
                "timestamp": pd.to_datetime(timestamp),
                "source": site,
                "text": article.find('h3').text if article.find('h3') else "",
                "url": article.find('a')['href'] if article.find('a') else ""
            })
    
    return pd.DataFrame(articles).set_index('timestamp').resample('5T').agg({
        'text': lambda x: ' '.join(x),
        'source': 'count'
    }).rename(columns={'source': 'news_count'})

# ==============================================
# 5. Merge All Data
# ==============================================
def build_dataset():
    # Get social data
    twitter = scrape_twitter_5m()
    reddit = scrape_reddit_5m()
    news = scrape_crypto_news()
    
    # Merge (outer join to preserve all 5m intervals)
    social_df = pd.concat([twitter, reddit, news], axis=1)
    
    # Load market data (from previous script)
    market_df = pd.read_csv("btc_5m_complete_dataset.csv", index_col='timestamp', parse_dates=True)
    
    # Final merge
    return market_df.join(social_df, how='left')

# ==============================================
# 6. Run and Export
# ==============================================
if __name__ == "__main__":
    df = build_dataset()
    
    # Clean text data
    df['social_text'] = df['text'].fillna('') + ' ' + df['text'].fillna('')
    df['social_text'] = df['social_text'].apply(lambda x: re.sub(r'http\S+|@\w+|#\w+', '', x))
    
    # Save
    df.to_csv("datasets/btc_5m_with_social_no_fill.csv")
    print(f"Dataset saved with shape: {df.shape}")
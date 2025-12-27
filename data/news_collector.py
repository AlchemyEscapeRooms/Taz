"""News collection from various sources."""

import requests
from typing import List, Dict, Any
from datetime import datetime, timedelta

from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class NewsCollector:
    """Collects news from multiple sources."""

    def __init__(self):
        self.news_api_key = config.get('api_keys.news_api.key', '')

    def get_stock_news(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get news articles for a specific stock."""

        articles = []

        # NewsAPI (if key is available)
        if self.news_api_key:
            try:
                articles.extend(self._get_newsapi_articles(symbol, lookback_hours))
            except Exception as e:
                logger.warning(f"NewsAPI error: {e}")

        # You can add more news sources here (Alpaca, Finnhub, etc.)

        logger.info(f"Collected {len(articles)} news articles for {symbol}")

        return articles

    def _get_newsapi_articles(
        self,
        symbol: str,
        lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Get articles from NewsAPI."""

        if not self.news_api_key:
            return []

        url = "https://newsapi.org/v2/everything"

        from_date = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()

        params = {
            'q': symbol,
            'from': from_date,
            'sortBy': 'relevancy',
            'apiKey': self.news_api_key,
            'language': 'en'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'symbol': symbol,
                    'headline': article.get('title', ''),
                    'summary': article.get('description', ''),
                    'content': article.get('content', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'timestamp': datetime.now()
                })

            return articles

        except Exception as e:
            logger.error(f"Error fetching NewsAPI articles: {e}")
            return []

    def get_market_news(self, lookback_hours: int = 24) -> List[Dict[str, Any]]:
        """Get general market news."""

        if not self.news_api_key:
            return []

        url = "https://newsapi.org/v2/top-headlines"

        params = {
            'category': 'business',
            'country': 'us',
            'apiKey': self.news_api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'headline': article.get('title', ''),
                    'summary': article.get('description', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'timestamp': datetime.now()
                })

            logger.info(f"Collected {len(articles)} market news articles")

            return articles

        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

    def filter_relevant_news(
        self,
        articles: List[Dict[str, Any]],
        keywords: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter news articles by relevance."""

        if not keywords:
            keywords = config.get('news.keywords.positive', []) + config.get('news.keywords.negative', [])

        relevant = []

        for article in articles:
            text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()

            relevance_score = sum(1 for keyword in keywords if keyword.lower() in text)

            if relevance_score > 0:
                article['relevance_score'] = relevance_score
                relevant.append(article)

        # Sort by relevance
        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)

        return relevant

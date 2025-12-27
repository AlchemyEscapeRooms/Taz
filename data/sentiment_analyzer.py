"""Sentiment analysis for news and text data."""

import numpy as np
from typing import Dict, List, Any, Tuple
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.logger import get_logger
from utils.database import Database

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment of news and text data."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.db = Database()

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text."""

        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)

        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity

        # Combined score (weighted average)
        combined_score = (vader_scores['compound'] + textblob_polarity) / 2

        # Determine label
        if combined_score >= 0.05:
            label = 'positive'
        elif combined_score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'combined_score': combined_score,
            'label': label
        }

    def analyze_news_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of a news article."""

        # Combine headline and summary for analysis
        text = f"{article.get('headline', '')} {article.get('summary', '')}"

        sentiment = self.analyze_text(text)

        # Store in database
        self.db.store_news_sentiment(
            symbol=article.get('symbol', 'MARKET'),
            headline=article.get('headline', ''),
            summary=article.get('summary', ''),
            source=article.get('source', 'Unknown'),
            sentiment_score=sentiment['combined_score'],
            sentiment_label=sentiment['label'],
            relevance_score=article.get('relevance_score', 1.0),
            url=article.get('url', '')
        )

        return sentiment

    def analyze_multiple_articles(
        self,
        articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment of multiple articles and aggregate."""

        if not articles:
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': 'neutral',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'confidence': 0.0
            }

        sentiments = []
        labels = []

        for article in articles:
            sentiment = self.analyze_news_article(article)
            sentiments.append(sentiment['combined_score'])
            labels.append(sentiment['label'])

        # Calculate aggregate metrics
        overall_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)

        # Determine overall label
        if overall_sentiment >= 0.05:
            sentiment_label = 'positive'
        elif overall_sentiment <= -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        # Count labels
        positive_count = labels.count('positive')
        negative_count = labels.count('negative')
        neutral_count = labels.count('neutral')

        # Confidence based on agreement
        max_count = max(positive_count, negative_count, neutral_count)
        confidence = max_count / len(labels) if labels else 0

        result = {
            'overall_sentiment': overall_sentiment,
            'sentiment_std': sentiment_std,
            'sentiment_label': sentiment_label,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': len(articles),
            'confidence': confidence
        }

        logger.info(f"Analyzed {len(articles)} articles: {sentiment_label} ({overall_sentiment:.3f})")

        return result

    def get_symbol_sentiment(
        self,
        symbol: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get aggregated sentiment for a symbol from database."""

        df = self.db.get_recent_sentiment(symbol=symbol, hours=hours)

        if df.empty:
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'article_count': 0
            }

        avg_sentiment = df['sentiment_score'].mean()

        if avg_sentiment >= 0.05:
            label = 'positive'
        elif avg_sentiment <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'symbol': symbol,
            'sentiment_score': avg_sentiment,
            'sentiment_label': label,
            'article_count': len(df),
            'lookback_hours': hours
        }

    def get_market_sentiment(self, hours: int = 24) -> Dict[str, Any]:
        """Get overall market sentiment."""

        df = self.db.get_recent_sentiment(hours=hours)

        if df.empty:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'article_count': 0
            }

        avg_sentiment = df['sentiment_score'].mean()

        if avg_sentiment >= 0.05:
            label = 'positive'
        elif avg_sentiment <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'sentiment_score': avg_sentiment,
            'sentiment_label': label,
            'article_count': len(df),
            'lookback_hours': hours
        }

    def sentiment_to_trading_signal(
        self,
        sentiment_score: float,
        confidence: float,
        threshold: float = 0.3
    ) -> Tuple[str, float]:
        """Convert sentiment to trading signal."""

        # Only act on strong sentiment with high confidence
        if confidence < 0.6:
            return 'neutral', 0.0

        if sentiment_score > threshold:
            strength = min(sentiment_score, 1.0) * confidence
            return 'bullish', strength
        elif sentiment_score < -threshold:
            strength = min(abs(sentiment_score), 1.0) * confidence
            return 'bearish', strength
        else:
            return 'neutral', 0.0

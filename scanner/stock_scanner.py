"""
Stock Discovery Scanner
=======================

Scans S&P 500 stocks for high-potential opportunities by combining:
- News sentiment analysis
- Technical indicators
- Volume/momentum signals

Auto-adds high-scoring stocks to SimpleTrader.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
import pandas as pd

from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


@dataclass
class ScanResult:
    """Result from scanning a single stock."""
    symbol: str
    composite_score: float  # 0-100
    sentiment_score: float  # -1 to 1
    technical_score: float  # 0-100
    volume_score: float  # 0-100
    momentum_score: float  # 0-100
    news_count: int
    news_headlines: List[str] = field(default_factory=list)
    recommended_strategy: Optional[str] = None
    scan_timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'composite_score': round(self.composite_score, 2),
            'sentiment_score': round(self.sentiment_score, 3),
            'technical_score': round(self.technical_score, 2),
            'volume_score': round(self.volume_score, 2),
            'momentum_score': round(self.momentum_score, 2),
            'news_count': self.news_count,
            'news_headlines': self.news_headlines[:3],
            'recommended_strategy': self.recommended_strategy,
            'scan_timestamp': self.scan_timestamp.isoformat(),
            'error': self.error
        }


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0.0

    def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class NewsBudgetTracker:
    """Tracks NewsAPI daily request budget."""

    def __init__(self, daily_budget: int = 100, cache_file: str = "data/news_budget.json"):
        self.daily_budget = daily_budget
        self.cache_file = Path(cache_file)
        self._load_state()

    def _load_state(self):
        """Load state from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    saved_date = data.get('date', '')
                    if saved_date == datetime.now().strftime('%Y-%m-%d'):
                        self.requests_today = data.get('requests', 0)
                        return
        except Exception:
            pass
        self.requests_today = 0

    def _save_state(self):
        """Save state to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'requests': self.requests_today
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save budget state: {e}")

    def use_request(self) -> bool:
        """Use one request. Returns False if budget exhausted."""
        self._load_state()  # Refresh in case date changed
        if self.requests_today >= self.daily_budget:
            return False
        self.requests_today += 1
        self._save_state()
        return True

    def remaining(self) -> int:
        """Get remaining requests for today."""
        self._load_state()
        return max(0, self.daily_budget - self.requests_today)


class StockScanner:
    """
    Discovers high-potential stocks by combining:
    - News sentiment analysis
    - Technical indicators
    - Volume/momentum signals
    """

    def __init__(
        self,
        news_collector=None,
        sentiment_analyzer=None,
        market_data=None,
        simple_trader=None,
        daily_news_budget: int = 100,
        scan_batch_size: int = 50
    ):
        # Import here to avoid circular imports
        if news_collector is None:
            from data.news_collector import NewsCollector
            news_collector = NewsCollector()
        if sentiment_analyzer is None:
            from data.sentiment_analyzer import SentimentAnalyzer
            sentiment_analyzer = SentimentAnalyzer()
        if market_data is None:
            from data.market_data import MarketDataCollector
            market_data = MarketDataCollector()

        self.news_collector = news_collector
        self.sentiment_analyzer = sentiment_analyzer
        self.market_data = market_data
        self.simple_trader = simple_trader

        self.scan_batch_size = scan_batch_size
        self.budget_tracker = NewsBudgetTracker(daily_budget=daily_news_budget)
        self.rate_limiter = RateLimiter(requests_per_minute=10)

        # Score weights
        self.weights = {
            'sentiment': config.get('scanner.weights.sentiment', 0.30),
            'technical': config.get('scanner.weights.technical', 0.30),
            'volume': config.get('scanner.weights.volume', 0.20),
            'momentum': config.get('scanner.weights.momentum', 0.20)
        }

        # Results storage
        self.latest_results: List[ScanResult] = []
        self.results_file = Path("data/scanner_results.json")

        logger.info(f"StockScanner initialized (news budget: {daily_news_budget}/day)")

    def scan_symbol(self, symbol: str, fetch_news: bool = True) -> ScanResult:
        """
        Scan a single symbol and compute its score.

        Args:
            symbol: Stock ticker
            fetch_news: Whether to fetch fresh news (uses budget)

        Returns:
            ScanResult with scores
        """
        try:
            # Get technical data
            tech_data = self._get_technical_data(symbol)
            technical_score = tech_data.get('score', 50)
            volume_score = tech_data.get('volume_score', 50)
            momentum_score = tech_data.get('momentum_score', 50)

            # Get news and sentiment
            sentiment_score = 0.0
            news_count = 0
            headlines = []

            if fetch_news and self.budget_tracker.remaining() > 0:
                self.rate_limiter.wait()
                if self.budget_tracker.use_request():
                    articles = self.news_collector.get_stock_news(symbol, lookback_hours=48)
                    if articles:
                        sentiment_result = self.sentiment_analyzer.analyze_multiple_articles(articles)
                        sentiment_score = sentiment_result.get('overall_sentiment', 0.0)
                        news_count = len(articles)
                        headlines = [a.get('headline', '')[:100] for a in articles[:5]]

            # Compute composite score
            # Normalize sentiment from [-1, 1] to [0, 100]
            sentiment_normalized = (sentiment_score + 1) / 2 * 100

            composite_score = (
                sentiment_normalized * self.weights['sentiment'] +
                technical_score * self.weights['technical'] +
                volume_score * self.weights['volume'] +
                momentum_score * self.weights['momentum']
            )

            return ScanResult(
                symbol=symbol,
                composite_score=composite_score,
                sentiment_score=sentiment_score,
                technical_score=technical_score,
                volume_score=volume_score,
                momentum_score=momentum_score,
                news_count=news_count,
                news_headlines=headlines,
                recommended_strategy=tech_data.get('recommended_strategy'),
                scan_timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return ScanResult(
                symbol=symbol,
                composite_score=0,
                sentiment_score=0,
                technical_score=0,
                volume_score=0,
                momentum_score=0,
                news_count=0,
                error=str(e)
            )

    def _get_technical_data(self, symbol: str) -> Dict[str, Any]:
        """Calculate technical indicators and scores for a symbol."""
        try:
            # Get historical data (last 30 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            df = self.market_data.get_historical_data(symbol, start_date=start_date, end_date=end_date, interval='1d')

            if df is None or len(df) < 20:
                return {'score': 50, 'volume_score': 50, 'momentum_score': 50}

            close = df['close']
            volume = df['volume']

            # RSI
            rsi = self._calculate_rsi(close)
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50

            # RSI Score: Lower RSI = higher opportunity (oversold)
            if current_rsi < 30:
                rsi_score = 90  # Very oversold - high opportunity
            elif current_rsi < 40:
                rsi_score = 75
            elif current_rsi < 50:
                rsi_score = 60
            elif current_rsi < 60:
                rsi_score = 50
            elif current_rsi < 70:
                rsi_score = 40
            else:
                rsi_score = 20  # Overbought - low opportunity

            # MACD
            macd, signal, diff = self._calculate_macd(close)
            current_diff = diff.iloc[-1] if not diff.empty else 0
            prev_diff = diff.iloc[-2] if len(diff) > 1 else 0

            # MACD Score: Bullish crossover = high score
            if current_diff > 0 and prev_diff <= 0:
                macd_score = 95  # Just crossed bullish
            elif current_diff > 0:
                macd_score = 70  # Already bullish
            elif current_diff < 0 and prev_diff >= 0:
                macd_score = 20  # Just crossed bearish
            else:
                macd_score = 40  # Already bearish

            # Bollinger Bands
            _, _, bb_position = self._calculate_bollinger(close)
            current_bb = bb_position.iloc[-1] if not bb_position.empty else 0.5

            # BB Score: Near lower band = higher opportunity
            if current_bb < 0.2:
                bb_score = 90  # Very near lower band
            elif current_bb < 0.4:
                bb_score = 70
            elif current_bb < 0.6:
                bb_score = 50
            elif current_bb < 0.8:
                bb_score = 35
            else:
                bb_score = 15  # Near upper band

            # Technical composite
            technical_score = (rsi_score + macd_score + bb_score) / 3

            # Volume score
            if len(volume) >= 20:
                avg_volume = volume.rolling(20).mean().iloc[-1]
                current_volume = volume.iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                # Higher volume = higher score (but capped)
                volume_score = min(100, volume_ratio * 50)
            else:
                volume_score = 50

            # Momentum score (5-day price change)
            if len(close) >= 5:
                price_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
                # Positive momentum = higher score
                if price_change > 5:
                    momentum_score = 85
                elif price_change > 2:
                    momentum_score = 70
                elif price_change > 0:
                    momentum_score = 55
                elif price_change > -2:
                    momentum_score = 40
                else:
                    momentum_score = 25
            else:
                momentum_score = 50

            # Determine recommended strategy based on indicators
            if current_rsi < 35:
                recommended = "RSI 30/70"
            elif current_diff > 0 and prev_diff <= 0:
                recommended = "MACD Cross"
            elif current_bb < 0.3:
                recommended = "Bollinger Mean Rev"
            else:
                recommended = "Momentum Simple"

            return {
                'score': technical_score,
                'volume_score': volume_score,
                'momentum_score': momentum_score,
                'rsi': current_rsi,
                'macd_diff': current_diff,
                'bb_position': current_bb,
                'recommended_strategy': recommended
            }

        except Exception as e:
            logger.warning(f"Error getting technical data for {symbol}: {e}")
            return {'score': 50, 'volume_score': 50, 'momentum_score': 50}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series):
        """Calculate MACD."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        diff = macd - signal
        return macd, signal, diff

    def _calculate_bollinger(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        position = (prices - lower) / (upper - lower)
        return upper, lower, position

    def scan_batch(
        self,
        symbols: List[str],
        fetch_news: bool = True,
        progress_callback: Callable[[int, int, str], None] = None
    ) -> List[ScanResult]:
        """
        Scan a batch of symbols.

        Args:
            symbols: List of stock tickers
            fetch_news: Whether to fetch news (uses API budget)
            progress_callback: Optional callback(current, total, symbol)

        Returns:
            List of ScanResults sorted by composite score
        """
        results = []
        total = len(symbols)

        # Smart news fetching: pre-filter by technicals first
        if fetch_news:
            remaining_budget = self.budget_tracker.remaining()
            logger.info(f"News budget remaining: {remaining_budget}")

            if remaining_budget < total:
                # Pre-scan without news, then get news for top candidates
                logger.info(f"Budget limited: pre-filtering {total} stocks by technicals first")

                # First pass: technicals only
                pre_results = []
                for i, symbol in enumerate(symbols):
                    if progress_callback:
                        progress_callback(i + 1, total, f"Pre-filtering {symbol}")
                    result = self.scan_symbol(symbol, fetch_news=False)
                    pre_results.append(result)

                # Sort by technical score
                pre_results.sort(key=lambda x: x.technical_score, reverse=True)

                # Get news for top candidates
                news_count = min(remaining_budget, len(pre_results))
                top_symbols = [r.symbol for r in pre_results[:news_count]]

                logger.info(f"Fetching news for top {news_count} candidates")

                # Re-scan top candidates with news
                for symbol in top_symbols:
                    result = self.scan_symbol(symbol, fetch_news=True)
                    # Update in pre_results
                    for i, r in enumerate(pre_results):
                        if r.symbol == symbol:
                            pre_results[i] = result
                            break

                results = pre_results
            else:
                # Full scan with news for all
                for i, symbol in enumerate(symbols):
                    if progress_callback:
                        progress_callback(i + 1, total, symbol)
                    result = self.scan_symbol(symbol, fetch_news=True)
                    results.append(result)
        else:
            # No news, technical only
            for i, symbol in enumerate(symbols):
                if progress_callback:
                    progress_callback(i + 1, total, symbol)
                result = self.scan_symbol(symbol, fetch_news=False)
                results.append(result)

        # Sort by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)

        # Store results
        self.latest_results = results
        self._save_results()

        return results

    def get_top_discoveries(self, limit: int = 10) -> List[ScanResult]:
        """Get top N highest-scoring stocks from latest scan."""
        if not self.latest_results:
            self._load_results()
        return self.latest_results[:limit]

    def auto_add_to_trader(
        self,
        min_score: float = 60.0,
        max_additions: int = 10
    ) -> List[str]:
        """
        Automatically add high-scoring stocks to SimpleTrader.

        Args:
            min_score: Minimum composite score (0-100)
            max_additions: Maximum number of stocks to add

        Returns:
            List of symbols that were added
        """
        if self.simple_trader is None:
            logger.warning("No SimpleTrader instance provided")
            return []

        if not self.latest_results:
            logger.warning("No scan results available. Run a scan first.")
            return []

        added = []
        existing_symbols = set(self.simple_trader.symbols)

        for result in self.latest_results:
            if len(added) >= max_additions:
                break

            if result.composite_score < min_score:
                continue

            if result.symbol in existing_symbols:
                continue

            if result.error:
                continue

            # Try to add to trader
            try:
                add_result = self.simple_trader.add_symbol(result.symbol)
                if add_result.get('success'):
                    added.append(result.symbol)
                    logger.info(f"Auto-added {result.symbol} (score: {result.composite_score:.1f})")
                else:
                    logger.warning(f"Failed to add {result.symbol}: {add_result.get('message')}")
            except Exception as e:
                logger.error(f"Error adding {result.symbol}: {e}")

        return added

    def _save_results(self):
        """Save latest results to file."""
        try:
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_file, 'w') as f:
                json.dump({
                    'scan_time': datetime.now().isoformat(),
                    'count': len(self.latest_results),
                    'results': [r.to_dict() for r in self.latest_results]
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")

    def _load_results(self):
        """Load results from file."""
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    self.latest_results = [
                        ScanResult(
                            symbol=r['symbol'],
                            composite_score=r['composite_score'],
                            sentiment_score=r['sentiment_score'],
                            technical_score=r['technical_score'],
                            volume_score=r['volume_score'],
                            momentum_score=r['momentum_score'],
                            news_count=r['news_count'],
                            news_headlines=r.get('news_headlines', []),
                            recommended_strategy=r.get('recommended_strategy'),
                            scan_timestamp=datetime.fromisoformat(r['scan_timestamp']),
                            error=r.get('error')
                        )
                        for r in data.get('results', [])
                    ]
        except Exception as e:
            logger.warning(f"Failed to load results: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get scanner status."""
        return {
            'news_budget_remaining': self.budget_tracker.remaining(),
            'news_budget_total': self.budget_tracker.daily_budget,
            'last_scan_results': len(self.latest_results),
            'last_scan_time': self.latest_results[0].scan_timestamp.isoformat() if self.latest_results else None
        }

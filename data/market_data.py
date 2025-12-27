"""Market data collection from Alpaca (primary) and yfinance (backup)."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import os
import time

# Data sources
import yfinance as yf

# Alpaca SDK
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class MarketDataCollector:
    """Collects market data from Alpaca (primary) with yfinance fallback."""

    def __init__(self):
        self.primary_source = config.get('data.sources.market_data.primary', 'alpaca')
        self.backup_source = config.get('data.sources.market_data.backup', 'yfinance')
        self.cache = {}
        self.max_retries = 3
        self.retry_delay = 1  # seconds

        # Initialize Alpaca client
        self.alpaca_client = None
        self._init_alpaca()

        logger.info(f"MarketDataCollector initialized - Primary: {self.primary_source}, Backup: {self.backup_source}")

    def _init_alpaca(self):
        """Initialize Alpaca client with API credentials."""
        if not ALPACA_AVAILABLE:
            logger.warning("Alpaca SDK not installed. Install with: pip install alpaca-py")
            return

        try:
            api_key = os.getenv('ALPACA_API_KEY') or config.get('api_keys.alpaca.api_key', '')
            secret_key = os.getenv('ALPACA_SECRET_KEY') or config.get('api_keys.alpaca.secret_key', '')

            # Remove ${} wrapper if present (from config file)
            if api_key.startswith('${'):
                api_key = os.getenv(api_key[2:-1], '')
            if secret_key.startswith('${'):
                secret_key = os.getenv(secret_key[2:-1], '')

            if api_key and secret_key:
                self.alpaca_client = StockHistoricalDataClient(api_key, secret_key)
                logger.info("Alpaca client initialized successfully")
            else:
                logger.warning("Alpaca API keys not found - will use yfinance as primary")

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            self.alpaca_client = None

    def _get_alpaca_timeframe(self, interval: str):
        """Convert interval string to Alpaca TimeFrame."""
        if not ALPACA_AVAILABLE:
            return None

        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        interval_map = {
            '1m': TimeFrame(1, TimeFrameUnit.Minute),
            '5m': TimeFrame(5, TimeFrameUnit.Minute),
            '15m': TimeFrame(15, TimeFrameUnit.Minute),
            '30m': TimeFrame(30, TimeFrameUnit.Minute),
            '1h': TimeFrame(1, TimeFrameUnit.Hour),
            '1d': TimeFrame(1, TimeFrameUnit.Day),
            '1wk': TimeFrame(1, TimeFrameUnit.Week),
            '1mo': TimeFrame(1, TimeFrameUnit.Month),
        }
        return interval_map.get(interval, TimeFrame(1, TimeFrameUnit.Day))

    def _fetch_from_alpaca(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch data from Alpaca API."""
        if not self.alpaca_client:
            raise Exception("Alpaca client not initialized")

        try:
            timeframe = self._get_alpaca_timeframe(interval)

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=datetime.strptime(start_date, '%Y-%m-%d'),
                end=datetime.strptime(end_date, '%Y-%m-%d')
            )

            bars = self.alpaca_client.get_stock_bars(request_params)

            if symbol not in bars.data or not bars.data[symbol]:
                logger.warning(f"No Alpaca data for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for bar in bars.data[symbol]:
                data.append({
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'timestamp': bar.timestamp
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            df['symbol'] = symbol

            logger.info(f"Alpaca: Retrieved {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Alpaca error for {symbol}: {e}")
            raise

    def _fetch_from_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch data from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                logger.warning(f"No yfinance data for {symbol}")
                return pd.DataFrame()

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            df['symbol'] = symbol

            logger.info(f"yfinance: Retrieved {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            raise

    def get_historical_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data with automatic fallback.

        Tries Alpaca first, falls back to yfinance if Alpaca fails.
        If both fail, retries Alpaca again before giving up.
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

        # Smart source selection: skip Alpaca if it's been failing
        if not hasattr(self, '_alpaca_fail_count'):
            self._alpaca_fail_count = 0
            self._alpaca_disabled_until = None
        
        # Re-enable Alpaca after 5 minutes
        if self._alpaca_disabled_until and datetime.now() > self._alpaca_disabled_until:
            self._alpaca_fail_count = 0
            self._alpaca_disabled_until = None
            logger.info("Re-enabling Alpaca after timeout")
        
        # If Alpaca has failed 3+ times, skip it temporarily
        skip_alpaca = self._alpaca_fail_count >= 3
        
        if skip_alpaca:
            sources = ['yfinance']
        else:
            sources = ['alpaca', 'yfinance', 'alpaca']  # Try alpaca, yfinance, then alpaca again
            
        last_error = None

        for attempt, source in enumerate(sources):
            try:
                if source == 'alpaca' and self.alpaca_client and not skip_alpaca:
                    df = self._fetch_from_alpaca(symbol, start_date, end_date, interval)
                    if not df.empty:
                        self._alpaca_fail_count = 0  # Reset on success
                        return df
                elif source == 'yfinance':
                    df = self._fetch_from_yfinance(symbol, start_date, end_date, interval)
                    if not df.empty:
                        return df
                elif source == 'alpaca' and not self.alpaca_client:
                    # Skip alpaca if client not available
                    continue

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} ({source}) failed for {symbol}: {e}")
                
                # Track Alpaca failures
                if source == 'alpaca':
                    self._alpaca_fail_count += 1
                    if self._alpaca_fail_count >= 3:
                        self._alpaca_disabled_until = datetime.now() + timedelta(minutes=5)
                        logger.warning(f"Alpaca disabled for 5 minutes after {self._alpaca_fail_count} failures")

                if attempt < len(sources) - 1:
                    time.sleep(self.retry_delay)
                continue

        logger.error(f"All data sources failed for {symbol}. Last error: {last_error}")
        return pd.DataFrame()

    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote with fallback.

        Tries Alpaca first, falls back to yfinance.
        """
        # Try Alpaca first
        if self.alpaca_client:
            try:
                request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote_data = self.alpaca_client.get_stock_latest_quote(request_params)

                if symbol in quote_data:
                    q = quote_data[symbol]
                    return {
                        'symbol': symbol,
                        'price': (q.ask_price + q.bid_price) / 2 if q.ask_price and q.bid_price else q.ask_price or q.bid_price,
                        'bid': q.bid_price,
                        'ask': q.ask_price,
                        'bid_size': q.bid_size,
                        'ask_size': q.ask_size,
                        'timestamp': q.timestamp,
                        'source': 'alpaca'
                    }
            except Exception as e:
                logger.warning(f"Alpaca quote failed for {symbol}: {e}, trying yfinance")

        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'volume': info.get('volume', 0),
                'timestamp': datetime.now(),
                'source': 'yfinance'
            }

        except Exception as e:
            logger.error(f"All quote sources failed for {symbol}: {e}")
            return {}

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols."""
        logger.info(f"Fetching data for {len(symbols)} symbols")

        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, start_date, end_date, interval)
            if not df.empty:
                data[symbol] = df

        logger.info(f"Successfully retrieved data for {len(data)}/{len(symbols)} symbols")
        return data

    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a symbol (uses yfinance - Alpaca doesn't provide this)."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            fundamentals = {
                'symbol': symbol,
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'ps_ratio': info.get('priceToSalesTrailing12Months', None),
                'peg_ratio': info.get('pegRatio', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'profit_margin': info.get('profitMargins', None),
                'market_cap': info.get('marketCap', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'dividend_yield': info.get('dividendYield', None),
                'sector': info.get('sector', None),
                'industry': info.get('industry', None)
            }

            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}

    def screen_stocks(
        self,
        min_price: float = 5.0,
        min_volume: int = 1000000,
        min_market_cap: float = 1e9
    ) -> List[str]:
        """Screen for stocks meeting criteria."""
        logger.info("Screening stocks...")

        # Common liquid stocks
        candidates = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
            'SPY', 'QQQ', 'IWM', 'DIA',
            'JPM', 'BAC', 'WFC', 'GS',
            'XOM', 'CVX', 'COP',
            'JNJ', 'UNH', 'PFE',
            'WMT', 'HD', 'COST',
            'DIS', 'NFLX', 'CMCSA'
        ]

        filtered = []

        for symbol in candidates:
            try:
                quote = self.get_real_time_quote(symbol)
                fundamentals = self.get_fundamentals(symbol)

                if (quote.get('price', 0) >= min_price and
                    quote.get('volume', 0) >= min_volume and
                    fundamentals.get('market_cap', 0) >= min_market_cap):
                    filtered.append(symbol)

            except Exception as e:
                logger.warning(f"Error screening {symbol}: {e}")
                continue

        logger.info(f"Found {len(filtered)} stocks meeting criteria")
        return filtered

    def calculate_correlations(self, symbols: List[str], period: int = 60) -> pd.DataFrame:
        """Calculate correlation matrix for symbols."""
        logger.info(f"Calculating correlations for {len(symbols)} symbols")

        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol)
            if not df.empty and len(df) >= period:
                data[symbol] = df['close'].tail(period)

        prices_df = pd.DataFrame(data)
        correlations = prices_df.pct_change().corr()

        return correlations

    def get_market_regime(self, symbol: str = 'SPY') -> str:
        """Determine current market regime."""
        df = self.get_historical_data(symbol)

        if df.empty or len(df) < 50:
            return 'unknown'

        returns = df['close'].pct_change()
        volatility = returns.std()
        trend = (df['close'].iloc[-1] / df['close'].iloc[-20]) - 1

        if trend > 0.05 and volatility < returns.mean():
            return 'bull_trending'
        elif trend < -0.05 and volatility < returns.mean():
            return 'bear_trending'
        elif volatility > returns.std() * 1.5:
            return 'high_volatility'
        else:
            return 'ranging'

    def get_data_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources."""
        return {
            'alpaca': {
                'available': ALPACA_AVAILABLE,
                'initialized': self.alpaca_client is not None,
                'is_primary': self.primary_source == 'alpaca'
            },
            'yfinance': {
                'available': True,
                'initialized': True,
                'is_backup': self.backup_source == 'yfinance'
            },
            'fallback_order': ['alpaca', 'yfinance', 'alpaca']
        }

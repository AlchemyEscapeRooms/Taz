"""S&P 500 symbol list provider with caching."""

import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Hardcoded fallback list of major S&P 500 stocks
FALLBACK_SP500 = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC',
    'CRM', 'ORCL', 'ADBE', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'IBM', 'NOW', 'INTU',
    'MU', 'AMAT', 'LRCX', 'ADI', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'NXPI', 'MCHP',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
    'PNC', 'TFC', 'COF', 'BK', 'STT', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'ISRG', 'MDT', 'SYK', 'REGN', 'VRTX', 'BSX', 'ZBH', 'EW',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    'LOW', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CL', 'EL', 'KMB',
    # Industrial
    'CAT', 'DE', 'BA', 'HON', 'UNP', 'RTX', 'LMT', 'GE', 'MMM', 'UPS',
    'FDX', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'CMI', 'PCAR', 'GD', 'NOC',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
    'HAL', 'DVN', 'HES', 'FANG', 'BKR',
    # Utilities & REITs
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED',
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SPG', 'WELL', 'DLR', 'AVB',
    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'PPG',
    # ETFs (common trading vehicles)
    'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI',
]


class SP500Provider:
    """Provides S&P 500 symbol list with caching."""

    def __init__(self, cache_dir: str = "data"):
        self.cache_file = Path(cache_dir) / "sp500_symbols.json"
        self.cache_max_age_days = 7
        self._symbols: Optional[List[str]] = None

    def get_symbols(self, refresh: bool = False) -> List[str]:
        """
        Get S&P 500 symbols.

        Args:
            refresh: Force refresh from source

        Returns:
            List of stock symbols
        """
        if self._symbols and not refresh:
            return self._symbols

        # Try loading from cache
        if not refresh and self._is_cache_valid():
            cached = self._load_cache()
            if cached:
                self._symbols = cached
                logger.info(f"Loaded {len(cached)} symbols from cache")
                return self._symbols

        # Try fetching from Wikipedia
        fetched = self._fetch_from_wikipedia()
        if fetched:
            self._symbols = fetched
            self._save_cache(fetched)
            logger.info(f"Fetched {len(fetched)} symbols from Wikipedia")
            return self._symbols

        # Fallback to hardcoded list
        self._symbols = FALLBACK_SP500.copy()
        logger.warning(f"Using fallback list of {len(self._symbols)} symbols")
        return self._symbols

    def get_symbols_batch(self, batch_size: int, offset: int = 0) -> List[str]:
        """Get a batch of symbols for paginated scanning."""
        symbols = self.get_symbols()
        return symbols[offset:offset + batch_size]

    def refresh_list(self) -> List[str]:
        """Force refresh the symbol list."""
        return self.get_symbols(refresh=True)

    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is not too old."""
        if not self.cache_file.exists():
            return False

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                cached_at = datetime.fromisoformat(data.get('cached_at', ''))
                age = datetime.now() - cached_at
                return age < timedelta(days=self.cache_max_age_days)
        except Exception:
            return False

    def _load_cache(self) -> Optional[List[str]]:
        """Load symbols from cache file."""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                return data.get('symbols', [])
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_cache(self, symbols: List[str]) -> None:
        """Save symbols to cache file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'cached_at': datetime.now().isoformat(),
                    'count': len(symbols),
                    'symbols': symbols
                }, f, indent=2)
            logger.info(f"Cached {len(symbols)} symbols")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _fetch_from_wikipedia(self) -> Optional[List[str]]:
        """Fetch S&P 500 constituents from Wikipedia."""
        try:
            import pandas as pd

            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)

            if tables:
                df = tables[0]
                # The symbol column is usually named 'Symbol'
                if 'Symbol' in df.columns:
                    symbols = df['Symbol'].tolist()
                    # Clean up symbols (remove periods, spaces)
                    symbols = [s.replace('.', '-').strip() for s in symbols if isinstance(s, str)]
                    return symbols

        except ImportError:
            logger.warning("pandas not available for Wikipedia scraping")
        except Exception as e:
            logger.warning(f"Failed to fetch from Wikipedia: {e}")

        return None

    def get_by_sector(self, sector: str) -> List[str]:
        """
        Get symbols filtered by sector.

        Note: This is a simplified version using the hardcoded list.
        For full sector data, would need to fetch from data provider.
        """
        sector_map = {
            'tech': FALLBACK_SP500[:30],
            'finance': FALLBACK_SP500[30:50],
            'healthcare': FALLBACK_SP500[50:70],
            'consumer': FALLBACK_SP500[70:90],
            'industrial': FALLBACK_SP500[90:110],
            'energy': FALLBACK_SP500[110:125],
            'utilities': FALLBACK_SP500[125:145],
            'materials': FALLBACK_SP500[145:155],
        }
        return sector_map.get(sector.lower(), [])

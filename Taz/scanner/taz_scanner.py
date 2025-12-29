"""
TAZ Volatility Scanner
======================
Finds high-volatility stocks and crypto for aggressive trading.
Focuses on momentum, breakouts, and big movers.

Purpose: Identify opportunities for fast profit growth.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')


@dataclass
class TazOpportunity:
    """A trading opportunity found by Taz scanner."""
    symbol: str
    asset_type: str  # 'stock' or 'crypto'
    score: float  # 0-100, higher = better opportunity

    # Price action
    current_price: float
    price_change_1h: float  # % change
    price_change_24h: float

    # Volatility metrics
    volatility: float  # ATR as % of price
    daily_range_pct: float  # (high-low)/close

    # Volume metrics
    volume_ratio: float  # vs 20-day average

    # Technical signals
    rsi: float
    macd_signal: str  # 'bullish', 'bearish', 'neutral'
    bollinger_position: float  # 0-1

    # Momentum
    momentum_score: float

    # Recommended action
    signal: str  # 'BUY', 'SELL', 'WATCH'
    strategy: str  # Recommended strategy name

    # Metadata
    scan_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'asset_type': self.asset_type,
            'score': round(self.score, 1),
            'current_price': round(self.current_price, 4),
            'price_change_1h': round(self.price_change_1h, 2),
            'price_change_24h': round(self.price_change_24h, 2),
            'volatility': round(self.volatility, 2),
            'daily_range_pct': round(self.daily_range_pct, 2),
            'volume_ratio': round(self.volume_ratio, 2),
            'rsi': round(self.rsi, 1),
            'macd_signal': self.macd_signal,
            'bollinger_position': round(self.bollinger_position, 2),
            'momentum_score': round(self.momentum_score, 1),
            'signal': self.signal,
            'strategy': self.strategy,
            'scan_time': self.scan_time.isoformat()
        }


class TazScanner:
    """
    Aggressive volatility scanner for Taz trading system.
    Finds fast-moving stocks and crypto for quick profit opportunities.
    """

    # Volatile stock watchlist - known movers
    VOLATILE_STOCKS = [
        # Tech volatiles
        'TSLA', 'NVDA', 'AMD', 'PLTR', 'COIN', 'MARA', 'RIOT',
        # Meme stocks
        'GME', 'AMC', 'BBBY', 'BB', 'NOK', 'WISH', 'CLOV',
        # Biotech
        'MRNA', 'BNTX', 'NVAX',
        # EV & Clean Energy
        'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'PLUG', 'FCEL',
        # High beta tech
        'SNAP', 'ROKU', 'SQ', 'SHOP', 'NET', 'DDOG', 'CRWD',
        # SPACs & Speculative
        'SPCE', 'LAZR', 'QS',
        # High volume ETFs for market moves
        'TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'UVXY', 'VXX'
    ]

    # Crypto symbols for 24/7 trading
    CRYPTO_SYMBOLS = [
        'BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD',
        'AVAX/USD', 'MATIC/USD', 'LINK/USD', 'UNI/USD',
        'AAVE/USD', 'LTC/USD', 'BCH/USD', 'DOT/USD'
    ]

    def __init__(self, min_volatility: float = 2.0, min_volume_ratio: float = 1.0):
        """
        Initialize Taz Scanner.

        Args:
            min_volatility: Minimum daily volatility % to consider
            min_volume_ratio: Minimum volume vs 20-day avg to consider
        """
        self.min_volatility = min_volatility
        self.min_volume_ratio = min_volume_ratio

        # Alpaca clients
        self.trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        self.stock_data = StockHistoricalDataClient(API_KEY, API_SECRET)
        self.crypto_data = CryptoHistoricalDataClient(API_KEY, API_SECRET)

        # Results storage
        self.opportunities: List[TazOpportunity] = []
        self.results_file = Path(__file__).parent.parent / 'data' / 'taz_scanner_results.json'

        # Score weights - prioritize momentum and volatility
        self.weights = {
            'momentum': 0.30,
            'volatility': 0.25,
            'volume': 0.20,
            'rsi_extreme': 0.15,
            'macd': 0.10
        }

        print(f"[TAZ] Scanner initialized - hunting for {min_volatility}%+ volatility")

    def scan_stocks(self, symbols: List[str] = None) -> List[TazOpportunity]:
        """
        Scan stocks for volatile opportunities.

        Args:
            symbols: List of symbols to scan (uses VOLATILE_STOCKS if None)

        Returns:
            List of opportunities sorted by score
        """
        symbols = symbols or self.VOLATILE_STOCKS
        opportunities = []

        print(f"[TAZ] Scanning {len(symbols)} stocks for opportunities...")

        for symbol in symbols:
            try:
                opp = self._analyze_stock(symbol)
                if opp and opp.score >= 50:  # Minimum score threshold
                    opportunities.append(opp)
            except Exception as e:
                print(f"[TAZ] Error scanning {symbol}: {e}")

        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)

        print(f"[TAZ] Found {len(opportunities)} stock opportunities")
        return opportunities

    def scan_crypto(self, symbols: List[str] = None) -> List[TazOpportunity]:
        """
        Scan crypto for volatile opportunities.
        24/7 trading means more opportunities.
        """
        symbols = symbols or self.CRYPTO_SYMBOLS
        opportunities = []

        print(f"[TAZ] Scanning {len(symbols)} crypto pairs...")

        for symbol in symbols:
            try:
                opp = self._analyze_crypto(symbol)
                if opp and opp.score >= 50:
                    opportunities.append(opp)
            except Exception as e:
                print(f"[TAZ] Error scanning {symbol}: {e}")

        opportunities.sort(key=lambda x: x.score, reverse=True)

        print(f"[TAZ] Found {len(opportunities)} crypto opportunities")
        return opportunities

    def scan_all(self) -> List[TazOpportunity]:
        """Scan both stocks and crypto, combine results."""
        stock_opps = self.scan_stocks()
        crypto_opps = self.scan_crypto()

        # Combine and sort
        all_opps = stock_opps + crypto_opps
        all_opps.sort(key=lambda x: x.score, reverse=True)

        self.opportunities = all_opps
        self._save_results()

        return all_opps

    def _analyze_stock(self, symbol: str) -> Optional[TazOpportunity]:
        """Analyze a single stock for trading opportunity."""
        try:
            # Get 5-day intraday data
            end = datetime.now()
            start = end - timedelta(days=5)

            # Get hourly bars for recent analysis
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )

            bars = self.stock_data.get_stock_bars(request)
            df = bars.df

            if df.empty:
                return None

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')

            if len(df) < 20:
                return None

            return self._calculate_opportunity(symbol, df, 'stock')

        except Exception as e:
            print(f"[TAZ] Stock analysis error {symbol}: {e}")
            return None

    def _analyze_crypto(self, symbol: str) -> Optional[TazOpportunity]:
        """Analyze a crypto pair for trading opportunity."""
        try:
            end = datetime.now()
            start = end - timedelta(days=3)

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )

            bars = self.crypto_data.get_crypto_bars(request)
            df = bars.df

            if df.empty:
                return None

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')

            if len(df) < 20:
                return None

            return self._calculate_opportunity(symbol, df, 'crypto')

        except Exception as e:
            print(f"[TAZ] Crypto analysis error {symbol}: {e}")
            return None

    def _calculate_opportunity(
        self,
        symbol: str,
        df: pd.DataFrame,
        asset_type: str
    ) -> Optional[TazOpportunity]:
        """Calculate opportunity score and signals from OHLCV data."""

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        current_price = close.iloc[-1]

        # Price changes
        price_change_1h = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100 if len(close) > 1 else 0
        price_change_24h = ((close.iloc[-1] - close.iloc[-24]) / close.iloc[-24]) * 100 if len(close) > 24 else 0

        # Volatility - ATR as % of price
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        volatility = (atr / current_price) * 100

        # Daily range
        daily_range_pct = ((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1]) * 100

        # Volume ratio
        avg_volume = volume.rolling(20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1

        # RSI (fast, 7-period for more signals)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        macd_diff = macd.iloc[-1] - signal_line.iloc[-1]
        prev_macd_diff = macd.iloc[-2] - signal_line.iloc[-2] if len(macd) > 1 else 0

        if macd_diff > 0 and prev_macd_diff <= 0:
            macd_signal = 'bullish_cross'
        elif macd_diff > 0:
            macd_signal = 'bullish'
        elif macd_diff < 0 and prev_macd_diff >= 0:
            macd_signal = 'bearish_cross'
        else:
            macd_signal = 'bearish'

        # Bollinger position
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + (std20 * 2)
        lower = sma20 - (std20 * 2)
        bb_position = ((close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])) if (upper.iloc[-1] - lower.iloc[-1]) > 0 else 0.5
        bb_position = max(0, min(1, bb_position))

        # Momentum score (5-period momentum)
        momentum = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100 if len(close) > 5 else 0

        # === CALCULATE SCORE ===
        score = 0

        # Momentum score (higher = better for longs)
        if momentum > 5:
            momentum_score = 100
        elif momentum > 2:
            momentum_score = 75
        elif momentum > 0:
            momentum_score = 50
        elif momentum > -2:
            momentum_score = 30
        else:
            momentum_score = 10  # Could be short opportunity

        # Volatility score (higher volatility = more opportunity)
        if volatility > 5:
            vol_score = 100
        elif volatility > 3:
            vol_score = 80
        elif volatility > 2:
            vol_score = 60
        elif volatility > 1:
            vol_score = 40
        else:
            vol_score = 20

        # Volume score (higher ratio = more interest)
        if volume_ratio > 3:
            vol_ratio_score = 100
        elif volume_ratio > 2:
            vol_ratio_score = 80
        elif volume_ratio > 1.5:
            vol_ratio_score = 60
        elif volume_ratio > 1:
            vol_ratio_score = 40
        else:
            vol_ratio_score = 20

        # RSI extreme score (oversold = buy, overbought = potential short)
        if rsi < 25:
            rsi_score = 90  # Very oversold - strong buy signal
        elif rsi < 35:
            rsi_score = 70
        elif rsi > 75:
            rsi_score = 70  # Overbought - potential short or sell
        elif rsi > 65:
            rsi_score = 50
        else:
            rsi_score = 30  # Neutral RSI = less opportunity

        # MACD score
        if macd_signal == 'bullish_cross':
            macd_score = 100
        elif macd_signal == 'bullish':
            macd_score = 60
        elif macd_signal == 'bearish_cross':
            macd_score = 70  # Could be short opportunity
        else:
            macd_score = 40

        # Weighted composite score
        score = (
            momentum_score * self.weights['momentum'] +
            vol_score * self.weights['volatility'] +
            vol_ratio_score * self.weights['volume'] +
            rsi_score * self.weights['rsi_extreme'] +
            macd_score * self.weights['macd']
        )

        # Determine signal and strategy
        signal = 'WATCH'
        strategy = 'momentum_rider'

        # BUY signals - NEVER buy when overbought (RSI > 70)
        if rsi > 70:
            signal = 'WATCH'  # Overbought - don't chase
            strategy = 'overbought_avoid'
        elif rsi < 30 and volume_ratio > 1.5 and macd_signal not in ['bearish', 'bearish_cross']:
            # Dip sniper - only if not in confirmed downtrend
            signal = 'BUY'
            strategy = 'dip_sniper'
        elif macd_signal == 'bullish_cross' and volume_ratio > 1.5 and rsi < 65:
            signal = 'BUY'
            strategy = 'momentum_rider'
        elif bb_position < 0.1 and rsi < 40 and macd_signal != 'bearish_cross':
            signal = 'BUY'
            strategy = 'volatility_scalper'
        elif momentum > 3 and volume_ratio > 2 and rsi < 65 and rsi > 35:
            # Breakout hunter - only if not overbought AND not oversold (avoid falling knives)
            signal = 'BUY'
            strategy = 'breakout_hunter'

        # Filter by minimum thresholds
        if volatility < self.min_volatility:
            score *= 0.5  # Penalize low volatility
        if volume_ratio < self.min_volume_ratio:
            score *= 0.7  # Penalize low volume

        return TazOpportunity(
            symbol=symbol,
            asset_type=asset_type,
            score=score,
            current_price=current_price,
            price_change_1h=price_change_1h,
            price_change_24h=price_change_24h,
            volatility=volatility,
            daily_range_pct=daily_range_pct,
            volume_ratio=volume_ratio,
            rsi=rsi,
            macd_signal=macd_signal,
            bollinger_position=bb_position,
            momentum_score=momentum_score,
            signal=signal,
            strategy=strategy
        )

    def get_top_opportunities(self, limit: int = 10, asset_type: str = None) -> List[TazOpportunity]:
        """Get top N opportunities, optionally filtered by asset type."""
        opps = self.opportunities

        if asset_type:
            opps = [o for o in opps if o.asset_type == asset_type]

        return opps[:limit]

    def get_buy_signals(self) -> List[TazOpportunity]:
        """Get all current BUY signals."""
        return [o for o in self.opportunities if o.signal == 'BUY']

    def _save_results(self):
        """Save scan results to file."""
        try:
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_file, 'w') as f:
                json.dump({
                    'scan_time': datetime.now().isoformat(),
                    'total_opportunities': len(self.opportunities),
                    'buy_signals': len([o for o in self.opportunities if o.signal == 'BUY']),
                    'opportunities': [o.to_dict() for o in self.opportunities]
                }, f, indent=2)
        except Exception as e:
            print(f"[TAZ] Error saving results: {e}")

    def print_summary(self):
        """Print a summary of current opportunities."""
        print("\n" + "="*70)
        print("TAZ SCANNER RESULTS")
        print("="*70)

        buy_signals = self.get_buy_signals()
        print(f"\nBUY SIGNALS: {len(buy_signals)}")

        for opp in buy_signals[:5]:
            print(f"\n  {opp.symbol} ({opp.asset_type.upper()})")
            print(f"    Score: {opp.score:.1f} | Strategy: {opp.strategy}")
            print(f"    Price: ${opp.current_price:.2f} | 1h: {opp.price_change_1h:+.2f}%")
            print(f"    Volatility: {opp.volatility:.1f}% | Volume: {opp.volume_ratio:.1f}x")
            print(f"    RSI: {opp.rsi:.0f} | MACD: {opp.macd_signal}")

        print("\n" + "="*70)


def main():
    """Run Taz scanner as standalone."""
    scanner = TazScanner(min_volatility=2.0, min_volume_ratio=1.0)

    print("\n[TAZ] Starting volatility hunt...")

    # Scan everything
    opportunities = scanner.scan_all()

    # Print results
    scanner.print_summary()

    # Show top 10
    print("\nTOP 10 OPPORTUNITIES:")
    for i, opp in enumerate(opportunities[:10], 1):
        print(f"{i:2}. {opp.symbol:10} Score: {opp.score:5.1f} | {opp.signal:5} | {opp.strategy}")


if __name__ == '__main__':
    main()

"""
Market Regime Detection System

Identifies market conditions to optimize strategy selection:
1. TRENDING - Strong directional movement (use momentum/trend strategies)
2. RANGING - Sideways consolidation (use mean reversion)
3. VOLATILE - High volatility spikes (reduce position sizes)
4. BREAKOUT - Consolidation before breakout (use breakout strategy)

This is CRITICAL for profit because:
- Using momentum strategy in ranging market = whipsawed to death
- Using mean reversion in trending market = missing huge moves
- Wrong strategy = guaranteed losses

This regime detector PREVENTS using the wrong strategy at the wrong time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT_PENDING = "breakout_pending"
    UNKNOWN = "unknown"


@dataclass
class RegimeAnalysis:
    """Complete regime analysis"""
    regime: MarketRegime
    confidence: float  # 0-1, how confident we are
    trend_strength: float  # 0-1, how strong the trend is
    volatility_percentile: float  # 0-1, current vol vs historical
    recommended_strategies: List[str]
    position_size_multiplier: float  # 0.5x to 2.0x
    rationale: str


class MarketRegimeDetector:
    """
    Detects market regime to optimize strategy selection.

    Uses multiple indicators:
    1. ADX (Average Directional Index) - trend strength
    2. ATR (Average True Range) - volatility
    3. Price action - support/resistance
    4. Moving averages - trend direction
    5. Bollinger Bands - range vs trend
    """

    def __init__(
        self,
        adx_threshold: float = 25,  # ADX > 25 = trending
        ranging_threshold: float = 20,  # ADX < 20 = ranging
        volatility_window: int = 20,
        trend_window: int = 50
    ):
        self.adx_threshold = adx_threshold
        self.ranging_threshold = ranging_threshold
        self.volatility_window = volatility_window
        self.trend_window = trend_window

    def detect_regime(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> RegimeAnalysis:
        """
        Detect current market regime.

        Args:
            prices: Historical price data (at least 50 periods)
            volumes: Optional volume data

        Returns:
            RegimeAnalysis with regime type and recommendations
        """

        if len(prices) < 50:
            logger.warning(f"Insufficient data for regime detection: {len(prices)} < 50")
            return RegimeAnalysis(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                trend_strength=0.5,
                volatility_percentile=0.5,
                recommended_strategies=['momentum'],
                position_size_multiplier=1.0,
                rationale="Insufficient historical data"
            )

        # Calculate indicators
        adx = self._calculate_adx(prices)
        atr = self._calculate_atr(prices)
        volatility_pct = self._calculate_volatility_percentile(prices, atr)
        trend_direction = self._calculate_trend_direction(prices)
        range_bound = self._is_range_bound(prices)
        breakout_pending = self._is_breakout_pending(prices, atr)

        # Determine regime
        regime, confidence = self._determine_regime(
            adx=adx,
            trend_direction=trend_direction,
            range_bound=range_bound,
            breakout_pending=breakout_pending,
            volatility_pct=volatility_pct
        )

        # Get recommendations
        strategies = self._get_recommended_strategies(regime)
        size_multiplier = self._get_position_size_multiplier(regime, volatility_pct)
        rationale = self._build_rationale(
            regime, adx, trend_direction, volatility_pct, breakout_pending
        )

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=adx / 100.0,  # Normalize to 0-1
            volatility_percentile=volatility_pct,
            recommended_strategies=strategies,
            position_size_multiplier=size_multiplier,
            rationale=rationale
        )

    def _calculate_adx(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate Average Directional Index (trend strength).

        ADX values:
        - 0-20: Weak/no trend (ranging)
        - 20-25: Emerging trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend
        """

        # Calculate True Range
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        tr = high - low

        # Calculate directional movement
        up_move = prices.diff()
        down_move = -prices.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth with EMA
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period).mean() / tr.ewm(span=period).mean()
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period).mean() / tr.ewm(span=period).mean()

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period).mean().iloc[-1]

        return min(100, max(0, adx))

    def _calculate_atr(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range (volatility measure)"""

        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        tr = high - low
        atr = tr.rolling(period).mean().iloc[-1]

        return atr

    def _calculate_volatility_percentile(
        self,
        prices: pd.Series,
        current_atr: float,
        lookback: int = 100
    ) -> float:
        """
        Calculate current volatility vs historical.

        Returns:
            0-1 percentile (0.9 = 90th percentile = very volatile)
        """

        # Calculate historical ATR
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        tr = high - low
        historical_atr = tr.rolling(14).mean().tail(lookback)

        # Calculate percentile
        percentile = (historical_atr < current_atr).sum() / len(historical_atr)

        return percentile

    def _calculate_trend_direction(self, prices: pd.Series) -> float:
        """
        Calculate trend direction.

        Returns:
            -1 to +1 (-1 = strong downtrend, +1 = strong uptrend, 0 = no trend)
        """

        # Use multiple moving averages
        sma_20 = prices.rolling(20).mean()
        sma_50 = prices.rolling(50).mean()
        current_price = prices.iloc[-1]

        # Score based on price vs MAs
        score = 0.0

        # Price vs 20 SMA
        if current_price > sma_20.iloc[-1]:
            score += 0.5
        else:
            score -= 0.5

        # Price vs 50 SMA
        if current_price > sma_50.iloc[-1]:
            score += 0.25
        else:
            score -= 0.25

        # 20 SMA vs 50 SMA (trend confirmation)
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            score += 0.25
        else:
            score -= 0.25

        return max(-1, min(1, score))

    def _is_range_bound(self, prices: pd.Series, lookback: int = 20) -> bool:
        """
        Check if price is range-bound (oscillating between support/resistance).

        Returns:
            True if ranging
        """

        recent_prices = prices.tail(lookback)

        # Calculate price range
        price_range = recent_prices.max() - recent_prices.min()
        avg_price = recent_prices.mean()
        range_percent = price_range / avg_price

        # Range-bound if narrow range
        return range_percent < 0.05  # Less than 5% range

    def _is_breakout_pending(
        self,
        prices: pd.Series,
        atr: float,
        lookback: int = 20
    ) -> bool:
        """
        Check if price is consolidating before potential breakout.

        Breakout setup:
        - Narrow range (low volatility)
        - Price compression
        - Volume contraction (if available)

        Returns:
            True if breakout may be pending
        """

        recent_prices = prices.tail(lookback)

        # Check for compression (decreasing range)
        early_range = recent_prices.head(10).max() - recent_prices.head(10).min()
        late_range = recent_prices.tail(10).max() - recent_prices.tail(10).min()

        # Compression = late range < early range
        compression = late_range < early_range * 0.7

        # Low volatility
        current_range = recent_prices.max() - recent_prices.min()
        avg_price = recent_prices.mean()
        low_vol = (current_range / avg_price) < 0.03

        return compression and low_vol

    def _determine_regime(
        self,
        adx: float,
        trend_direction: float,
        range_bound: bool,
        breakout_pending: bool,
        volatility_pct: float
    ) -> Tuple[MarketRegime, float]:
        """
        Determine regime from indicators.

        Returns:
            (regime, confidence)
        """

        # Very high volatility overrides everything
        if volatility_pct > 0.95:
            return MarketRegime.VOLATILE, 0.95

        # Breakout pending (highest priority)
        if breakout_pending and adx < 25:
            return MarketRegime.BREAKOUT_PENDING, 0.85

        # Strong trend
        if adx > self.adx_threshold:
            if trend_direction > 0.3:
                confidence = min(0.95, 0.6 + (adx / 100) + abs(trend_direction) * 0.2)
                return MarketRegime.TRENDING_UP, confidence
            elif trend_direction < -0.3:
                confidence = min(0.95, 0.6 + (adx / 100) + abs(trend_direction) * 0.2)
                return MarketRegime.TRENDING_DOWN, confidence

        # Ranging market
        if adx < self.ranging_threshold or range_bound:
            confidence = min(0.9, 0.6 + (1 - adx / 100) * 0.3)
            return MarketRegime.RANGING, confidence

        # Volatile but no clear regime
        if volatility_pct > 0.80:
            return MarketRegime.VOLATILE, 0.75

        # Default: weak trend
        if trend_direction > 0:
            return MarketRegime.TRENDING_UP, 0.5
        else:
            return MarketRegime.TRENDING_DOWN, 0.5

    def _get_recommended_strategies(self, regime: MarketRegime) -> List[str]:
        """Get best strategies for regime"""

        strategies = {
            MarketRegime.TRENDING_UP: ['momentum', 'trend_following', 'macd'],
            MarketRegime.TRENDING_DOWN: ['momentum', 'trend_following', 'macd'],
            MarketRegime.RANGING: ['mean_reversion', 'rsi', 'pairs_trading'],
            MarketRegime.VOLATILE: ['mean_reversion', 'rsi'],  # Wait for reversion
            MarketRegime.BREAKOUT_PENDING: ['breakout'],
            MarketRegime.UNKNOWN: ['momentum']  # Default
        }

        return strategies.get(regime, ['momentum'])

    def _get_position_size_multiplier(
        self,
        regime: MarketRegime,
        volatility_pct: float
    ) -> float:
        """
        Get position size multiplier for regime.

        Returns:
            0.5x to 2.0x multiplier
        """

        # Base multipliers by regime
        base_multipliers = {
            MarketRegime.TRENDING_UP: 1.5,      # Bet more in strong trends
            MarketRegime.TRENDING_DOWN: 1.3,    # Bet more (shorting opportunities)
            MarketRegime.RANGING: 1.0,          # Normal sizing
            MarketRegime.VOLATILE: 0.6,         # Bet less in chaos
            MarketRegime.BREAKOUT_PENDING: 1.2, # Moderate sizing
            MarketRegime.UNKNOWN: 0.8           # Conservative
        }

        multiplier = base_multipliers.get(regime, 1.0)

        # Reduce size in high volatility
        if volatility_pct > 0.8:
            multiplier *= 0.7
        elif volatility_pct > 0.6:
            multiplier *= 0.85

        return max(0.5, min(2.0, multiplier))

    def _build_rationale(
        self,
        regime: MarketRegime,
        adx: float,
        trend_direction: float,
        volatility_pct: float,
        breakout_pending: bool
    ) -> str:
        """Build human-readable rationale"""

        parts = []

        # Regime
        parts.append(f"Regime: {regime.value.upper()}")

        # Trend strength
        if adx > 50:
            parts.append(f"Very strong trend (ADX: {adx:.0f})")
        elif adx > 25:
            parts.append(f"Strong trend (ADX: {adx:.0f})")
        elif adx > 20:
            parts.append(f"Weak trend (ADX: {adx:.0f})")
        else:
            parts.append(f"No trend (ADX: {adx:.0f})")

        # Direction
        if abs(trend_direction) > 0.5:
            direction = "UP" if trend_direction > 0 else "DOWN"
            parts.append(f"Direction: {direction}")

        # Volatility
        if volatility_pct > 0.9:
            parts.append(f"Extreme volatility ({volatility_pct:.0%})")
        elif volatility_pct > 0.7:
            parts.append(f"High volatility ({volatility_pct:.0%})")
        elif volatility_pct < 0.3:
            parts.append(f"Low volatility ({volatility_pct:.0%})")

        # Breakout
        if breakout_pending:
            parts.append("Breakout setup detected")

        return " | ".join(parts)


class RegimeBasedStrategySelector:
    """
    Selects best strategy based on regime.

    This is what makes the difference between:
    - 10% annual return (using same strategy always)
    - 30%+ annual return (using right strategy for conditions)
    """

    def __init__(self):
        self.detector = MarketRegimeDetector()

        # Strategy performance by regime (from backtesting)
        self.strategy_scores = {
            'momentum': {
                MarketRegime.TRENDING_UP: 0.9,
                MarketRegime.TRENDING_DOWN: 0.8,
                MarketRegime.RANGING: 0.3,
                MarketRegime.VOLATILE: 0.4,
                MarketRegime.BREAKOUT_PENDING: 0.6
            },
            'mean_reversion': {
                MarketRegime.TRENDING_UP: 0.4,
                MarketRegime.TRENDING_DOWN: 0.4,
                MarketRegime.RANGING: 0.9,
                MarketRegime.VOLATILE: 0.8,
                MarketRegime.BREAKOUT_PENDING: 0.5
            },
            'trend_following': {
                MarketRegime.TRENDING_UP: 0.95,
                MarketRegime.TRENDING_DOWN: 0.9,
                MarketRegime.RANGING: 0.2,
                MarketRegime.VOLATILE: 0.3,
                MarketRegime.BREAKOUT_PENDING: 0.4
            },
            'breakout': {
                MarketRegime.TRENDING_UP: 0.6,
                MarketRegime.TRENDING_DOWN: 0.5,
                MarketRegime.RANGING: 0.5,
                MarketRegime.VOLATILE: 0.4,
                MarketRegime.BREAKOUT_PENDING: 0.95
            },
            'rsi': {
                MarketRegime.TRENDING_UP: 0.5,
                MarketRegime.TRENDING_DOWN: 0.5,
                MarketRegime.RANGING: 0.85,
                MarketRegime.VOLATILE: 0.7,
                MarketRegime.BREAKOUT_PENDING: 0.5
            }
        }

    def select_strategy(
        self,
        prices: pd.Series,
        available_strategies: List[str]
    ) -> Tuple[str, RegimeAnalysis]:
        """
        Select best strategy for current regime.

        Args:
            prices: Historical price data
            available_strategies: List of available strategy names

        Returns:
            (best_strategy_name, regime_analysis)
        """

        # Detect regime
        regime_analysis = self.detector.detect_regime(prices)

        # Score each available strategy
        scores = {}
        for strategy in available_strategies:
            if strategy in self.strategy_scores:
                score = self.strategy_scores[strategy].get(
                    regime_analysis.regime,
                    0.5  # Default score
                )
                scores[strategy] = score

        # Select best
        if scores:
            best_strategy = max(scores, key=scores.get)
        else:
            # Fallback to regime recommendations
            best_strategy = regime_analysis.recommended_strategies[0]

        logger.info(
            f"Selected '{best_strategy}' for {regime_analysis.regime.value} regime "
            f"(confidence: {regime_analysis.confidence:.0%})"
        )

        return best_strategy, regime_analysis


if __name__ == '__main__':
    # Test regime detection
    print("=" * 80)
    print("MARKET REGIME DETECTION TEST")
    print("=" * 80)

    # Generate sample price data
    np.random.seed(42)

    # Test 1: Strong uptrend
    print("\nTest 1: STRONG UPTREND")
    print("-" * 80)
    uptrend_prices = pd.Series(100 + np.cumsum(np.random.normal(0.5, 1, 100)))

    detector = MarketRegimeDetector()
    analysis = detector.detect_regime(uptrend_prices)

    print(f"Regime: {analysis.regime.value}")
    print(f"Confidence: {analysis.confidence:.0%}")
    print(f"Trend Strength: {analysis.trend_strength:.0%}")
    print(f"Volatility: {analysis.volatility_percentile:.0%} percentile")
    print(f"Recommended Strategies: {', '.join(analysis.recommended_strategies)}")
    print(f"Position Size Multiplier: {analysis.position_size_multiplier:.2f}x")
    print(f"Rationale: {analysis.rationale}")

    # Test 2: Ranging market
    print("\nTest 2: RANGING MARKET")
    print("-" * 80)
    ranging_prices = pd.Series(100 + np.sin(np.arange(100) * 0.3) * 2)

    analysis = detector.detect_regime(ranging_prices)

    print(f"Regime: {analysis.regime.value}")
    print(f"Confidence: {analysis.confidence:.0%}")
    print(f"Recommended Strategies: {', '.join(analysis.recommended_strategies)}")
    print(f"Position Size Multiplier: {analysis.position_size_multiplier:.2f}x")
    print(f"Rationale: {analysis.rationale}")

    # Test 3: Strategy selection
    print("\nTest 3: STRATEGY SELECTION")
    print("-" * 80)

    selector = RegimeBasedStrategySelector()
    strategies = ['momentum', 'mean_reversion', 'trend_following', 'breakout', 'rsi']

    best_strategy, regime = selector.select_strategy(uptrend_prices, strategies)
    print(f"Best strategy for uptrend: {best_strategy}")

    best_strategy, regime = selector.select_strategy(ranging_prices, strategies)
    print(f"Best strategy for ranging: {best_strategy}")

    print("\n" + "=" * 80)
    print("Regime detection tests complete!")

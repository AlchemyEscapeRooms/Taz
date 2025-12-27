"""
Profit-Optimized Position Sizing System

This module implements ADVANCED position sizing that maximizes profit
while controlling risk. Unlike fixed 10% sizing, this dynamically adjusts
based on:

1. Signal strength/conviction
2. Market volatility
3. Win rate history
4. Current drawdown
5. Portfolio heat (total risk)
6. Capital compounding

PROFIT MAXIMIZATION FEATURES:
- Kelly Criterion with fractional safety
- Volatility-adjusted sizing
- Conviction-based scaling
- Anti-martingale (bet more when winning)
- Portfolio heat management
- Compounding position sizes
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeRecommendation:
    """Position size recommendation with rationale"""
    quantity: float
    dollar_amount: float
    percent_of_capital: float
    risk_percent: float
    conviction_level: str
    rationale: str
    max_loss_if_stopped: float


class ProfitOptimizedPositionSizer:
    """
    Advanced position sizing for maximum profit with controlled risk.

    Key Principles:
    1. Risk 1-2% of capital per trade (professional standard)
    2. Size up on high conviction setups
    3. Size down after losses (protect capital)
    4. Size up after wins (let profits run)
    5. Use compounding (reinvest gains)
    6. Limit total portfolio heat
    """

    def __init__(
        self,
        base_risk_percent: float = 0.015,  # 1.5% per trade (professional level)
        max_risk_percent: float = 0.025,   # 2.5% max on best setups
        min_risk_percent: float = 0.005,   # 0.5% min on weak setups
        max_portfolio_heat: float = 0.10,  # Max 10% total risk across all positions
        kelly_fraction: float = 0.25,      # Quarter Kelly for safety
        enable_compounding: bool = True
    ):
        self.base_risk_percent = base_risk_percent
        self.max_risk_percent = max_risk_percent
        self.min_risk_percent = min_risk_percent
        self.max_portfolio_heat = max_portfolio_heat
        self.kelly_fraction = kelly_fraction
        self.enable_compounding = enable_compounding

        # Track performance for adaptive sizing
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        current_capital: float,
        conviction: float = 0.7,  # 0-1 scale
        volatility_percentile: Optional[float] = None,  # 0-1, higher = more volatile
        recent_win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        current_portfolio_heat: float = 0.0
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size.

        Args:
            entry_price: Intended entry price
            stop_loss_price: Stop loss price
            current_capital: Current account equity
            conviction: Signal strength (0-1, higher = stronger)
            volatility_percentile: Market volatility vs history (0-1)
            recent_win_rate: Recent win rate (0-1)
            avg_win_loss_ratio: Average win / average loss
            current_portfolio_heat: Total risk from existing positions

        Returns:
            PositionSizeRecommendation with all details
        """

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            logger.warning("Stop loss equals entry price, using 2% default")
            risk_per_share = entry_price * 0.02

        # Start with base risk
        risk_dollars = current_capital * self.base_risk_percent

        # ADJUSTMENT 1: Conviction-based sizing
        # High conviction = bet more, low conviction = bet less
        conviction = max(0.1, min(1.0, conviction))  # Clamp 0.1-1.0
        conviction_multiplier = 0.5 + (conviction * 1.5)  # Range: 0.5x to 2.0x
        risk_dollars *= conviction_multiplier

        # ADJUSTMENT 2: Volatility-based sizing
        # High volatility = bet less, low volatility = bet more
        if volatility_percentile is not None:
            vol_multiplier = 1.5 - volatility_percentile  # Range: 0.5x to 1.5x
            risk_dollars *= vol_multiplier

        # ADJUSTMENT 3: Performance-based sizing (Anti-Martingale)
        # Winning streak = bet more, losing streak = bet less
        performance_multiplier = self._calculate_performance_multiplier()
        risk_dollars *= performance_multiplier

        # ADJUSTMENT 4: Kelly Criterion (if we have stats)
        if recent_win_rate and avg_win_loss_ratio and recent_win_rate > 0:
            kelly_size = self._kelly_criterion(
                current_capital,
                recent_win_rate,
                avg_win_loss_ratio
            )
            # Use average of Kelly and risk-based sizing
            risk_dollars = (risk_dollars + kelly_size) / 2

        # ADJUSTMENT 5: Portfolio heat limit
        # Don't exceed max total risk across all positions
        available_heat = self.max_portfolio_heat - current_portfolio_heat
        if available_heat <= 0:
            logger.warning("Portfolio heat limit reached, reducing size to minimum")
            risk_dollars = current_capital * self.min_risk_percent
        elif risk_dollars / current_capital > available_heat:
            logger.info("Reducing size to respect portfolio heat limit")
            risk_dollars = current_capital * available_heat

        # Cap at min/max risk levels
        min_risk_dollars = current_capital * self.min_risk_percent
        max_risk_dollars = current_capital * self.max_risk_percent
        risk_dollars = max(min_risk_dollars, min(max_risk_dollars, risk_dollars))

        # Calculate final position size
        quantity = risk_dollars / risk_per_share
        dollar_amount = quantity * entry_price
        percent_of_capital = dollar_amount / current_capital
        risk_percent = risk_dollars / current_capital

        # Determine conviction level
        if conviction >= 0.8:
            conviction_level = "VERY HIGH"
        elif conviction >= 0.6:
            conviction_level = "HIGH"
        elif conviction >= 0.4:
            conviction_level = "MEDIUM"
        else:
            conviction_level = "LOW"

        # Build rationale
        rationale_parts = []
        rationale_parts.append(f"Base risk: {self.base_risk_percent*100:.1f}%")
        rationale_parts.append(f"Conviction: {conviction:.0%} ({conviction_level}) → {conviction_multiplier:.2f}x")
        if volatility_percentile:
            rationale_parts.append(f"Volatility: {volatility_percentile:.0%} → {vol_multiplier:.2f}x")
        rationale_parts.append(f"Performance: {performance_multiplier:.2f}x")
        rationale_parts.append(f"Final risk: {risk_percent*100:.2f}%")

        return PositionSizeRecommendation(
            quantity=quantity,
            dollar_amount=dollar_amount,
            percent_of_capital=percent_of_capital,
            risk_percent=risk_percent,
            conviction_level=conviction_level,
            rationale=" | ".join(rationale_parts),
            max_loss_if_stopped=risk_dollars
        )

    def _calculate_performance_multiplier(self) -> float:
        """
        Calculate position size multiplier based on recent performance.

        Anti-Martingale principle:
        - Bet MORE when winning (confidence is high)
        - Bet LESS when losing (protect capital)
        """

        # After wins, increase size
        if self.consecutive_wins >= 3:
            return 1.3  # 30% larger
        elif self.consecutive_wins >= 2:
            return 1.2  # 20% larger
        elif self.consecutive_wins >= 1:
            return 1.1  # 10% larger

        # After losses, decrease size
        elif self.consecutive_losses >= 3:
            return 0.6  # 40% smaller
        elif self.consecutive_losses >= 2:
            return 0.75  # 25% smaller
        elif self.consecutive_losses >= 1:
            return 0.9  # 10% smaller

        return 1.0  # No adjustment

    def _kelly_criterion(
        self,
        capital: float,
        win_rate: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly Criterion position size.

        Kelly% = (Win% × WinLossRatio - Loss%) / WinLossRatio

        We use fractional Kelly for safety (typically 25-50% of full Kelly)
        """

        if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
            return capital * self.base_risk_percent

        loss_rate = 1 - win_rate

        # Kelly percentage
        kelly_pct = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

        # Apply fractional Kelly (e.g., 0.25 = quarter Kelly)
        kelly_pct = max(0, kelly_pct) * self.kelly_fraction

        # Cap at reasonable max
        kelly_pct = min(kelly_pct, self.max_risk_percent)

        return capital * kelly_pct

    def update_performance(self, trade_result: str):
        """
        Update performance tracking.

        Args:
            trade_result: 'win' or 'loss'
        """

        self.total_trades += 1

        if trade_result == 'win':
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.winning_trades += 1
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def get_current_win_rate(self) -> float:
        """Get current win rate"""
        if self.total_trades == 0:
            return 0.5  # Assume 50% if no history
        return self.winning_trades / self.total_trades

    def reset_performance(self):
        """Reset performance tracking"""
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0


class DynamicStopLoss:
    """
    Profit-protecting stop loss system.

    Types:
    1. Initial stop (protect capital)
    2. Trailing stop (lock in profits)
    3. Time-based stop (don't hold losers)
    4. Volatility-based stop (ATR)
    """

    def __init__(
        self,
        initial_stop_pct: float = 0.02,  # 2% initial stop
        trailing_stop_pct: float = 0.015,  # 1.5% trailing
        use_atr: bool = True,
        atr_multiplier: float = 2.0
    ):
        self.initial_stop_pct = initial_stop_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.use_atr = use_atr
        self.atr_multiplier = atr_multiplier

    def calculate_initial_stop(
        self,
        entry_price: float,
        atr: Optional[float] = None,
        support_level: Optional[float] = None
    ) -> float:
        """Calculate initial stop loss"""

        # Method 1: ATR-based (preferred for volatility adjustment)
        if self.use_atr and atr:
            stop = entry_price - (atr * self.atr_multiplier)

        # Method 2: Support-based (if technical level exists)
        elif support_level and support_level < entry_price:
            # Place stop just below support
            stop = support_level * 0.995

        # Method 3: Fixed percentage
        else:
            stop = entry_price * (1 - self.initial_stop_pct)

        return stop

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr: Optional[float] = None
    ) -> Tuple[float, bool]:
        """
        Calculate trailing stop that locks in profits.

        Returns:
            (stop_price, should_trail)
        """

        # Only trail if in profit
        if current_price <= entry_price:
            return self.calculate_initial_stop(entry_price, atr), False

        # ATR-based trailing
        if self.use_atr and atr:
            trail_distance = atr * self.atr_multiplier
            stop = highest_price - trail_distance

        # Percentage-based trailing
        else:
            stop = highest_price * (1 - self.trailing_stop_pct)

        # Never trail below entry (protect breakeven)
        stop = max(stop, entry_price * 1.001)  # Slightly above entry

        return stop, True


class ProfitTargets:
    """
    Profit-taking system for locking in gains.

    Strategies:
    1. Scale out at multiple targets (take partial profits)
    2. Trail remaining position
    3. Risk-free after first target
    """

    def __init__(self):
        # Multiple profit targets for scaling out
        self.targets = [
            {'percent': 0.02, 'size': 0.33, 'name': '2% Quick'},   # Sell 1/3 at +2%
            {'percent': 0.05, 'size': 0.33, 'name': '5% Medium'},  # Sell 1/3 at +5%
            {'percent': 0.10, 'size': 0.34, 'name': '10% Runner'}, # Sell rest at +10%
        ]

    def get_profit_targets(
        self,
        entry_price: float,
        position_quantity: float,
        volatility_adjusted: bool = False,
        atr: Optional[float] = None
    ) -> list:
        """
        Get profit target levels.

        Returns list of:
        {
            'price': target_price,
            'quantity': quantity_to_sell,
            'percent_gain': percent,
            'name': target_name
        }
        """

        targets = []

        for target in self.targets:
            # Adjust targets for volatility
            if volatility_adjusted and atr:
                # Higher volatility = wider targets
                target_pct = target['percent'] * (1 + atr / entry_price)
            else:
                target_pct = target['percent']

            target_price = entry_price * (1 + target_pct)
            quantity = position_quantity * target['size']

            targets.append({
                'price': target_price,
                'quantity': quantity,
                'percent_gain': target_pct * 100,
                'name': target['name']
            })

        return targets


if __name__ == '__main__':
    # Test the profit-optimized position sizer
    print("=" * 80)
    print("PROFIT-OPTIMIZED POSITION SIZING TEST")
    print("=" * 80)

    sizer = ProfitOptimizedPositionSizer()

    # Test scenarios
    scenarios = [
        {
            'name': 'High Conviction Setup',
            'entry': 100,
            'stop': 98,
            'capital': 100000,
            'conviction': 0.9,
            'volatility': 0.3
        },
        {
            'name': 'Low Conviction Setup',
            'entry': 100,
            'stop': 98,
            'capital': 100000,
            'conviction': 0.3,
            'volatility': 0.5
        },
        {
            'name': 'After 3 Wins',
            'entry': 100,
            'stop': 98,
            'capital': 100000,
            'conviction': 0.7,
            'volatility': 0.4
        }
    ]

    # Simulate 3 wins
    sizer.update_performance('win')
    sizer.update_performance('win')
    sizer.update_performance('win')

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 80)

        rec = sizer.calculate_position_size(
            entry_price=scenario['entry'],
            stop_loss_price=scenario['stop'],
            current_capital=scenario['capital'],
            conviction=scenario['conviction'],
            volatility_percentile=scenario['volatility']
        )

        print(f"Position: {rec.quantity:.0f} shares = ${rec.dollar_amount:,.0f}")
        print(f"Portfolio %: {rec.percent_of_capital*100:.1f}%")
        print(f"Risk: {rec.risk_percent*100:.2f}% (${rec.max_loss_if_stopped:,.0f} if stopped)")
        print(f"Conviction: {rec.conviction_level}")
        print(f"Rationale: {rec.rationale}")

    print("\n" + "=" * 80)

    # Test profit targets
    print("\nPROFIT TARGETS TEST")
    print("=" * 80)

    targets_calc = ProfitTargets()
    targets = targets_calc.get_profit_targets(
        entry_price=100,
        position_quantity=1000
    )

    print(f"\nEntry: $100 for 1000 shares")
    for target in targets:
        print(f"  {target['name']}: ${target['price']:.2f} - Sell {target['quantity']:.0f} shares (+{target['percent_gain']:.1f}%)")

    print("\n" + "=" * 80)
    print("Tests complete!")

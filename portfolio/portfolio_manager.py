"""Portfolio management with diversification and rebalancing."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

from portfolio.position_tracker import PositionTracker
from portfolio.risk_manager import RiskManager
from utils.logger import get_logger
from utils.database import Database
from config import config

logger = get_logger(__name__)


class PortfolioManager:
    """Manages portfolio composition, diversification, and rebalancing."""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_tracker = PositionTracker()
        self.risk_manager = RiskManager(initial_capital)
        self.db = Database()

        self.min_positions = config.get('portfolio.diversification.min_positions', 10)
        self.target_positions = config.get('portfolio.diversification.target_positions', 25)
        self.rebalance_threshold = config.get('portfolio.rebalancing.threshold', 0.05)

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            pos['total_quantity'] * current_prices.get(symbol, 0)
            for symbol, pos in self.position_tracker.get_all_positions().items()
        )
        return self.cash + positions_value

    def get_positions_summary(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """Get summary of all positions."""
        positions = []
        for symbol, pos in self.position_tracker.get_all_positions().items():
            if symbol in current_prices:
                value_data = self.position_tracker.get_position_value(symbol, current_prices[symbol])
                positions.append(value_data)

        return pd.DataFrame(positions) if positions else pd.DataFrame()

    def calculate_allocations(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate current allocation percentages."""
        total_value = self.get_portfolio_value(current_prices)
        allocations = {}

        for symbol, pos in self.position_tracker.get_all_positions().items():
            if symbol in current_prices:
                position_value = pos['total_quantity'] * current_prices[symbol]
                allocations[symbol] = position_value / total_value if total_value > 0 else 0

        return allocations

    def needs_rebalancing(self, current_prices: Dict[str, float], target_allocations: Dict[str, float]) -> bool:
        """Check if portfolio needs rebalancing."""
        current = self.calculate_allocations(current_prices)

        for symbol, target in target_allocations.items():
            current_alloc = current.get(symbol, 0)
            if abs(current_alloc - target) > self.rebalance_threshold:
                return True

        return False

    def suggest_rebalancing_trades(
        self,
        current_prices: Dict[str, float],
        target_allocations: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Suggest trades to rebalance portfolio."""
        trades = []
        current_alloc = self.calculate_allocations(current_prices)
        total_value = self.get_portfolio_value(current_prices)

        for symbol, target in target_allocations.items():
            current = current_alloc.get(symbol, 0)
            diff = target - current

            if abs(diff) > self.rebalance_threshold:
                target_value = total_value * target
                current_value = total_value * current
                value_diff = target_value - current_value

                if value_diff > 0:  # Need to buy
                    quantity = value_diff / current_prices.get(symbol, 1)
                    trades.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': quantity,
                        'reason': 'rebalancing'
                    })
                else:  # Need to sell
                    quantity = abs(value_diff) / current_prices.get(symbol, 1)
                    trades.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': quantity,
                        'reason': 'rebalancing'
                    })

        return trades

    def record_snapshot(self, current_prices: Dict[str, float]):
        """Record portfolio snapshot."""
        total_value = self.get_portfolio_value(current_prices)
        positions_value = total_value - self.cash

        returns_data = self.calculate_returns()

        self.db.store_portfolio_snapshot(
            total_value=total_value,
            cash=self.cash,
            positions_value=positions_value,
            daily_return=returns_data['daily_return'],
            cumulative_return=returns_data['cumulative_return'],
            positions=self.position_tracker.get_all_positions(),
            metrics=self.get_portfolio_metrics(current_prices)
        )

    def calculate_returns(self) -> Dict[str, float]:
        """Calculate portfolio returns."""
        portfolio_history = self.db.get_portfolio_history(days=2)

        if len(portfolio_history) < 2:
            return {'daily_return': 0.0, 'cumulative_return': 0.0}

        latest = portfolio_history.iloc[-1]['total_value']
        previous = portfolio_history.iloc[-2]['total_value']

        daily_return = (latest - previous) / previous if previous > 0 else 0
        cumulative_return = (latest - self.initial_capital) / self.initial_capital

        return {
            'daily_return': daily_return,
            'cumulative_return': cumulative_return
        }

    def get_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        total_value = self.get_portfolio_value(current_prices)
        positions = self.get_positions_summary(current_prices)

        metrics = {
            'total_value': total_value,
            'cash': self.cash,
            'cash_pct': self.cash / total_value if total_value > 0 else 1.0,
            'num_positions': len(self.position_tracker.get_all_positions()),
            'total_return': (total_value - self.initial_capital) / self.initial_capital,
            'diversification_score': self._calculate_diversification_score(current_prices)
        }

        if not positions.empty:
            metrics.update({
                'total_unrealized_pl': positions['unrealized_pl'].sum(),
                'largest_position': positions['current_value'].max(),
                'smallest_position': positions['current_value'].min(),
                'avg_position_size': positions['current_value'].mean()
            })

        return metrics

    def _calculate_diversification_score(self, current_prices: Dict[str, float]) -> float:
        """Calculate diversification score (0-1, higher is better)."""
        allocations = list(self.calculate_allocations(current_prices).values())

        if not allocations:
            return 0.0

        # Inverse Herfindahl index
        herfindahl = sum(x**2 for x in allocations)
        diversification = 1 - herfindahl

        return diversification

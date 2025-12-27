"""Position tracking with FIFO (First-In-First-Out) inventory management."""

from collections import deque
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class Position:
    """Represents a single position lot (for FIFO tracking)."""

    def __init__(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        entry_time: datetime,
        order_id: str = None
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.order_id = order_id
        self.cost_basis = quantity * entry_price

    def __repr__(self):
        return f"Position({self.symbol}, {self.quantity}@${self.entry_price:.2f})"


class PositionTracker:
    """Tracks positions with FIFO inventory management."""

    def __init__(self):
        # Store positions as FIFO queues per symbol
        self.positions: Dict[str, deque] = {}
        self.position_history = []

    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime = None,
        order_id: str = None
    ):
        """Add a new position (buy order)."""

        if timestamp is None:
            timestamp = datetime.now()

        position = Position(symbol, quantity, price, timestamp, order_id)

        if symbol not in self.positions:
            self.positions[symbol] = deque()

        self.positions[symbol].append(position)

        logger.debug(f"Added position: {position}")

    def remove_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime = None
    ) -> List[Dict[str, Any]]:
        """Remove positions using FIFO (sell order)."""

        if timestamp is None:
            timestamp = datetime.now()

        if symbol not in self.positions or not self.positions[symbol]:
            logger.warning(f"No positions found for {symbol}")
            return []

        remaining_to_sell = quantity
        closed_lots = []

        while remaining_to_sell > 0 and self.positions[symbol]:
            position = self.positions[symbol][0]  # FIFO: take first position

            if position.quantity <= remaining_to_sell:
                # Close entire position
                closed_quantity = position.quantity
                remaining_to_sell -= closed_quantity

                # Calculate P&L
                proceeds = closed_quantity * price
                cost = closed_quantity * position.entry_price
                profit_loss = proceeds - cost
                profit_loss_pct = (profit_loss / cost) * 100

                closed_lots.append({
                    'symbol': symbol,
                    'quantity': closed_quantity,
                    'entry_price': position.entry_price,
                    'exit_price': price,
                    'entry_time': position.entry_time,
                    'exit_time': timestamp,
                    'holding_period': (timestamp - position.entry_time).total_seconds() / 86400,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'cost_basis': cost,
                    'proceeds': proceeds
                })

                # Remove position
                self.positions[symbol].popleft()

            else:
                # Partially close position
                closed_quantity = remaining_to_sell

                # Calculate P&L for partial close
                proceeds = closed_quantity * price
                cost = closed_quantity * position.entry_price
                profit_loss = proceeds - cost
                profit_loss_pct = (profit_loss / cost) * 100

                closed_lots.append({
                    'symbol': symbol,
                    'quantity': closed_quantity,
                    'entry_price': position.entry_price,
                    'exit_price': price,
                    'entry_time': position.entry_time,
                    'exit_time': timestamp,
                    'holding_period': (timestamp - position.entry_time).total_seconds() / 86400,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'cost_basis': cost,
                    'proceeds': proceeds
                })

                # Reduce position quantity
                position.quantity -= closed_quantity
                position.cost_basis = position.quantity * position.entry_price

                remaining_to_sell = 0

        # Store in history
        self.position_history.extend(closed_lots)

        # Clean up empty positions
        if symbol in self.positions and not self.positions[symbol]:
            del self.positions[symbol]

        logger.debug(f"Closed {len(closed_lots)} lots for {symbol}")

        return closed_lots

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get aggregate position for a symbol."""

        if symbol not in self.positions or not self.positions[symbol]:
            return {
                'symbol': symbol,
                'total_quantity': 0,
                'avg_entry_price': 0,
                'total_cost_basis': 0,
                'lots': []
            }

        lots = list(self.positions[symbol])
        total_quantity = sum(lot.quantity for lot in lots)
        total_cost = sum(lot.cost_basis for lot in lots)
        avg_price = total_cost / total_quantity if total_quantity > 0 else 0

        return {
            'symbol': symbol,
            'total_quantity': total_quantity,
            'avg_entry_price': avg_price,
            'total_cost_basis': total_cost,
            'lots': [
                {
                    'quantity': lot.quantity,
                    'entry_price': lot.entry_price,
                    'entry_time': lot.entry_time,
                    'order_id': lot.order_id
                }
                for lot in lots
            ]
        }

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all current positions."""

        all_positions = {}
        for symbol in self.positions.keys():
            all_positions[symbol] = self.get_position(symbol)

        return all_positions

    def get_position_value(
        self,
        symbol: str,
        current_price: float
    ) -> Dict[str, Any]:
        """Get current value and unrealized P&L for a position."""

        position = self.get_position(symbol)

        if position['total_quantity'] == 0:
            return {
                'symbol': symbol,
                'quantity': 0,
                'current_value': 0,
                'unrealized_pl': 0,
                'unrealized_pl_pct': 0
            }

        current_value = position['total_quantity'] * current_price
        unrealized_pl = current_value - position['total_cost_basis']
        unrealized_pl_pct = (unrealized_pl / position['total_cost_basis']) * 100

        return {
            'symbol': symbol,
            'quantity': position['total_quantity'],
            'avg_entry_price': position['avg_entry_price'],
            'current_price': current_price,
            'cost_basis': position['total_cost_basis'],
            'current_value': current_value,
            'unrealized_pl': unrealized_pl,
            'unrealized_pl_pct': unrealized_pl_pct
        }

    def get_realized_pl(self, days: int = None) -> pd.DataFrame:
        """Get realized P&L from closed positions."""

        if not self.position_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.position_history)

        if days:
            cutoff = datetime.now() - pd.Timedelta(days=days)
            df = df[df['exit_time'] >= cutoff]

        return df

    def get_total_realized_pl(self, days: int = None) -> float:
        """Get total realized profit/loss."""

        df = self.get_realized_pl(days)

        if df.empty:
            return 0.0

        return df['profit_loss'].sum()

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position for a symbol."""

        return symbol in self.positions and len(self.positions[symbol]) > 0

    def get_holding_period(self, symbol: str) -> float:
        """Get average holding period for a symbol in days."""

        if not self.has_position(symbol):
            return 0.0

        lots = list(self.positions[symbol])
        now = datetime.now()

        holding_periods = [(now - lot.entry_time).total_seconds() / 86400 for lot in lots]

        return sum(holding_periods) / len(holding_periods)

    def clear_all(self):
        """Clear all positions (use with caution)."""

        self.positions.clear()
        logger.warning("All positions cleared")

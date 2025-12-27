"""Order execution module for live and paper trading with trade logging."""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.database import Database
from utils.trade_logger import TradeLogger, TradeReason, get_trade_logger
from config import config

logger = get_logger(__name__)

# Try to import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    ALPACA_TRADING_AVAILABLE = True
except ImportError:
    ALPACA_TRADING_AVAILABLE = False
    logger.warning("Alpaca trading SDK not available")


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class OrderExecutor:
    """Handles order execution for live and paper trading."""

    def __init__(self, mode: str = "paper"):
        """
        Initialize order executor.

        Args:
            mode: "paper" for Alpaca paper trading, "live" for real trading
        """
        self.mode = mode
        self.db = Database()
        self.trade_logger = get_trade_logger()

        # Alpaca client - used for both paper and live trading
        self.alpaca_client = None
        if ALPACA_TRADING_AVAILABLE:
            self._init_alpaca_client()

        if not self.alpaca_client:
            raise RuntimeError("Alpaca client failed to initialize - cannot trade without Alpaca connection")

        logger.info(f"OrderExecutor initialized in {mode} mode (Alpaca connected)")

    def _init_alpaca_client(self):
        """Initialize Alpaca trading client."""
        try:
            api_key = os.getenv('ALPACA_API_KEY') or config.get('api_keys.alpaca_key')
            api_secret = os.getenv('ALPACA_SECRET_KEY') or config.get('api_keys.alpaca_secret')

            if api_key and api_secret:
                # Use paper=True for paper mode, paper=False for live mode
                use_paper = (self.mode == "paper")
                self.alpaca_client = TradingClient(api_key, api_secret, paper=use_paper)
                logger.info(f"Alpaca trading client initialized (paper={use_paper})")
            else:
                logger.warning("Alpaca API keys not found")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca trading client: {e}")

    def execute_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: float,
        order_type: str = "market",
        limit_price: float = None,
        strategy_name: str = None,
        strategy_params: Dict = None,
        reason: TradeReason = None,
        market_snapshot: Dict = None,
        portfolio_value: float = None
    ) -> OrderResult:
        """
        Execute a trading order.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            order_type: "market" or "limit"
            limit_price: Price for limit orders
            strategy_name: Name of the strategy that generated the signal
            strategy_params: Parameters of the strategy
            reason: TradeReason object with decision reasoning
            market_snapshot: Current market data snapshot
            portfolio_value: Current portfolio value

        Returns:
            OrderResult with execution details
        """
        logger.info(f"Executing {side} order: {quantity} {symbol} ({order_type}) via Alpaca")

        result = self._execute_alpaca_order(
            symbol, side, quantity, order_type, limit_price
        )

        # Log the trade with reasoning
        if result.success and self.trade_logger:
            self._log_trade(
                result,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                reason=reason,
                market_snapshot=market_snapshot,
                portfolio_value=portfolio_value
            )

        return result

    def _execute_alpaca_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        limit_price: float = None
    ) -> OrderResult:
        """Execute an order through Alpaca."""
        if not self.alpaca_client:
            return OrderResult(
                success=False,
                order_id="",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=0,
                status=OrderStatus.FAILED,
                message="Alpaca client not initialized"
            )

        try:
            # Convert side to Alpaca enum
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Create order request
            if order_type == "market":
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY
                )
            else:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )

            # Submit order
            order = self.alpaca_client.submit_order(order_request)

            logger.info(f"Alpaca order submitted: {order.id} - {side} {quantity} {symbol}")

            # Determine status
            status_map = {
                'new': OrderStatus.SUBMITTED,
                'accepted': OrderStatus.SUBMITTED,
                'pending_new': OrderStatus.PENDING,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'filled': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED
            }

            order_status = status_map.get(str(order.status).lower(), OrderStatus.PENDING)

            return OrderResult(
                success=order_status in [OrderStatus.SUBMITTED, OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED],
                order_id=str(order.id),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=float(order.limit_price or order.filled_avg_price or 0),
                status=order_status,
                filled_quantity=float(order.filled_qty or 0),
                filled_price=float(order.filled_avg_price or 0),
                message=f"Order {order.status}"
            )

        except Exception as e:
            logger.error(f"Alpaca order failed: {e}")
            return OrderResult(
                success=False,
                order_id="",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=0,
                status=OrderStatus.FAILED,
                message=str(e)
            )

    def _log_trade(
        self,
        result: OrderResult,
        strategy_name: str = None,
        strategy_params: Dict = None,
        reason: TradeReason = None,
        market_snapshot: Dict = None,
        portfolio_value: float = None
    ):
        """Log a trade with full reasoning."""
        if not self.trade_logger:
            return

        # Create default reason if not provided
        if reason is None:
            reason = TradeReason(
                primary_signal="manual_trade",
                signal_value=0,
                threshold=0,
                direction="n/a",
                explanation="Trade executed without detailed reasoning"
            )

        try:
            self.trade_logger.log_trade(
                symbol=result.symbol,
                action=result.side.upper(),
                quantity=result.filled_quantity or result.quantity,
                price=result.filled_price or result.price,
                strategy_name=strategy_name or "unknown",
                strategy_params=strategy_params or {},
                reason=reason,
                mode="live" if self.mode == "live" else "paper",
                side="long" if result.side.lower() == "buy" else "short",
                portfolio_value_before=portfolio_value,
                market_snapshot=market_snapshot,
                timestamp=result.timestamp
            )

            logger.debug(f"Trade logged: {result.order_id}")

        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of an order from Alpaca."""
        try:
            order = self.alpaca_client.get_order_by_id(order_id)
            return {
                'order_id': str(order.id),
                'symbol': order.symbol,
                'side': str(order.side),
                'quantity': float(order.qty),
                'filled_quantity': float(order.filled_qty or 0),
                'status': str(order.status),
                'filled_price': float(order.filled_avg_price or 0)
            }
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order on Alpaca."""
        try:
            self.alpaca_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders from Alpaca."""
        try:
            orders = self.alpaca_client.get_orders()
            return [
                {
                    'order_id': str(o.id),
                    'symbol': o.symbol,
                    'side': str(o.side),
                    'quantity': float(o.qty),
                    'status': str(o.status)
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_account_info(self) -> Optional[Dict]:
        """Get account information from Alpaca."""
        try:
            account = self.alpaca_client.get_account()
            return {
                'mode': self.mode,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity)
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions from Alpaca."""
        positions = {}

        if self.alpaca_client:
            try:
                alpaca_positions = self.alpaca_client.get_all_positions()
                for pos in alpaca_positions:
                    positions[pos.symbol] = {
                        'quantity': float(pos.qty),
                        'avg_cost': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'market_value': float(pos.market_value),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_pl_pct': float(pos.unrealized_plpc) * 100
                    }
            except Exception as e:
                logger.error(f"Failed to get positions: {e}")

        return positions

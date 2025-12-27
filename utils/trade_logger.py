"""
Trade Logger - Comprehensive logging of all trades with reasoning.

Logs every trade (real or backtest) with:
- Trade details (symbol, action, quantity, price)
- Strategy used and parameters
- Deciding factors and reasoning
- Market conditions at time of trade
- Technical indicators that triggered the signal
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np

from utils.logger import get_logger
from utils.database import Database

logger = get_logger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class TradeReason:
    """Captures the reasoning behind a trade decision."""

    primary_signal: str  # Main indicator/signal that triggered the trade
    signal_value: float  # Value of the primary signal
    threshold: float  # Threshold that was crossed
    direction: str  # 'above' or 'below' threshold

    # Supporting indicators
    supporting_indicators: Dict[str, float] = field(default_factory=dict)

    # Market context
    trend_direction: str = ""  # 'bullish', 'bearish', 'neutral'
    volatility_level: str = ""  # 'low', 'medium', 'high'
    volume_condition: str = ""  # 'above_average', 'below_average', 'normal'

    # Confirmation signals
    confirmations: List[str] = field(default_factory=list)

    # Human-readable explanation
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class TradeLogEntry:
    """Complete log entry for a single trade."""

    # Identification
    trade_id: str
    timestamp: datetime

    # Trade details
    symbol: str
    action: str  # 'BUY' or 'SELL'
    side: str  # 'long' or 'short'
    quantity: float
    price: float
    total_value: float

    # Strategy info
    strategy_name: str
    strategy_params: Dict[str, Any]

    # Trade reasoning
    reason: TradeReason

    # Context
    mode: str  # 'backtest', 'paper', 'live'
    portfolio_value_before: float
    position_size_pct: float

    # Market snapshot at trade time
    market_snapshot: Dict[str, float] = field(default_factory=dict)

    # Optional: linked to entry trade (for exits)
    entry_trade_id: Optional[str] = None
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    holding_period_days: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['reason'] = self.reason.to_dict()
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class TradeLogger:
    """
    Comprehensive trade logging system.

    Logs all trades to:
    1. Database (for querying and analysis)
    2. CSV file (for easy export)
    3. JSON file (for detailed backup)
    """

    def __init__(self, log_dir: str = "trade_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.db = Database()
        self._ensure_trade_log_table()

        # In-memory buffer for current session
        self.session_trades: List[TradeLogEntry] = []
        self.trade_counter = 0

        # Track logged trades to prevent duplicates (key: symbol_action_timestamp_strategy)
        self._logged_trade_keys: set = set()

        logger.info(f"TradeLogger initialized. Logs saved to: {self.log_dir}")

    def _ensure_trade_log_table(self):
        """Create trade_log table if it doesn't exist."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    total_value REAL NOT NULL,
                    strategy_name TEXT NOT NULL,
                    strategy_params TEXT,
                    mode TEXT NOT NULL,
                    portfolio_value_before REAL,
                    position_size_pct REAL,

                    -- Reasoning fields
                    primary_signal TEXT,
                    signal_value REAL,
                    threshold REAL,
                    direction TEXT,
                    supporting_indicators TEXT,
                    trend_direction TEXT,
                    volatility_level TEXT,
                    volume_condition TEXT,
                    confirmations TEXT,
                    explanation TEXT,

                    -- Market snapshot
                    market_snapshot TEXT,

                    -- Exit trade fields
                    entry_trade_id TEXT,
                    realized_pnl REAL,
                    realized_pnl_pct REAL,
                    holding_period_days REAL,

                    -- Full JSON for backup
                    full_json TEXT
                )
            """)

            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_log_timestamp
                ON trade_log(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_log_symbol
                ON trade_log(symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_log_strategy
                ON trade_log(strategy_name)
            """)

    def generate_trade_id(self, mode: str = "BT") -> str:
        """Generate unique trade ID."""
        self.trade_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{mode}-{timestamp}-{self.trade_counter:04d}"

    def _make_trade_key(self, symbol: str, action: str, timestamp: datetime, strategy_name: str, price: float) -> str:
        """Create a unique key for a trade to detect duplicates."""
        # Round timestamp to minute and price to 2 decimals to catch near-duplicates
        ts_key = timestamp.strftime("%Y%m%d%H%M") if timestamp else ""
        price_key = f"{price:.2f}"
        return f"{symbol}_{action}_{ts_key}_{strategy_name}_{price_key}"

    def log_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        reason: TradeReason,
        mode: str = "backtest",
        side: str = "long",
        portfolio_value_before: float = 0,
        market_snapshot: Dict[str, float] = None,
        entry_trade_id: str = None,
        realized_pnl: float = None,
        realized_pnl_pct: float = None,
        holding_period_days: float = None,
        timestamp: datetime = None
    ) -> Optional[TradeLogEntry]:
        """
        Log a trade with full details and reasoning.

        Returns the TradeLogEntry for reference, or None if it was a duplicate.
        """

        if timestamp is None:
            timestamp = datetime.now()

        # Check for duplicate trade
        trade_key = self._make_trade_key(symbol, action, timestamp, strategy_name, price)
        if trade_key in self._logged_trade_keys:
            logger.debug(f"Skipping duplicate trade: {symbol} {action} @ {timestamp}")
            return None
        self._logged_trade_keys.add(trade_key)

        mode_prefix = {"backtest": "BT", "paper": "PP", "live": "LV"}.get(mode, "XX")
        trade_id = self.generate_trade_id(mode_prefix)

        total_value = quantity * price
        position_size_pct = (total_value / portfolio_value_before * 100) if portfolio_value_before > 0 else 0

        entry = TradeLogEntry(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol,
            action=action.upper(),
            side=side,
            quantity=quantity,
            price=price,
            total_value=total_value,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            reason=reason,
            mode=mode,
            portfolio_value_before=portfolio_value_before,
            position_size_pct=position_size_pct,
            market_snapshot=market_snapshot or {},
            entry_trade_id=entry_trade_id,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            holding_period_days=holding_period_days
        )

        # Add to session buffer
        self.session_trades.append(entry)

        # Save to database
        self._save_to_database(entry)

        # Log to file
        self._append_to_csv(entry)

        logger.info(
            f"Trade logged: {trade_id} | {action} {quantity:.2f} {symbol} @ ${price:.2f} | "
            f"Reason: {reason.primary_signal} {reason.direction} {reason.threshold}"
        )

        return entry

    def _save_to_database(self, entry: TradeLogEntry):
        """Save trade log entry to database."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trade_log (
                        trade_id, timestamp, symbol, action, side, quantity, price,
                        total_value, strategy_name, strategy_params, mode,
                        portfolio_value_before, position_size_pct,
                        primary_signal, signal_value, threshold, direction,
                        supporting_indicators, trend_direction, volatility_level,
                        volume_condition, confirmations, explanation,
                        market_snapshot, entry_trade_id, realized_pnl,
                        realized_pnl_pct, holding_period_days, full_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.trade_id,
                    entry.timestamp.isoformat(),
                    entry.symbol,
                    entry.action,
                    entry.side,
                    float(entry.quantity),
                    float(entry.price),
                    float(entry.total_value),
                    entry.strategy_name,
                    json.dumps(convert_numpy_types(entry.strategy_params)),
                    entry.mode,
                    float(entry.portfolio_value_before) if entry.portfolio_value_before else 0,
                    float(entry.position_size_pct) if entry.position_size_pct else 0,
                    entry.reason.primary_signal,
                    float(entry.reason.signal_value) if entry.reason.signal_value else 0,
                    float(entry.reason.threshold) if entry.reason.threshold else 0,
                    entry.reason.direction,
                    json.dumps(convert_numpy_types(entry.reason.supporting_indicators)),
                    entry.reason.trend_direction,
                    entry.reason.volatility_level,
                    entry.reason.volume_condition,
                    json.dumps(convert_numpy_types(entry.reason.confirmations)),
                    entry.reason.explanation,
                    json.dumps(convert_numpy_types(entry.market_snapshot)),
                    entry.entry_trade_id,
                    float(entry.realized_pnl) if entry.realized_pnl else None,
                    float(entry.realized_pnl_pct) if entry.realized_pnl_pct else None,
                    float(entry.holding_period_days) if entry.holding_period_days else None,
                    entry.to_json()
                ))
        except Exception as e:
            logger.error(f"Failed to save trade to database: {e}")

    def _append_to_csv(self, entry: TradeLogEntry):
        """Append trade to CSV log file."""
        csv_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m')}.csv"

        file_exists = csv_file.exists()

        try:
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                if not file_exists:
                    # Write header
                    writer.writerow([
                        'trade_id', 'timestamp', 'symbol', 'action', 'side',
                        'quantity', 'price', 'total_value', 'strategy',
                        'mode', 'primary_signal', 'signal_value', 'threshold',
                        'direction', 'trend', 'explanation', 'realized_pnl'
                    ])

                writer.writerow([
                    entry.trade_id,
                    entry.timestamp.isoformat(),
                    entry.symbol,
                    entry.action,
                    entry.side,
                    f"{entry.quantity:.4f}",
                    f"{entry.price:.2f}",
                    f"{entry.total_value:.2f}",
                    entry.strategy_name,
                    entry.mode,
                    entry.reason.primary_signal,
                    f"{entry.reason.signal_value:.4f}" if entry.reason.signal_value else "",
                    f"{entry.reason.threshold:.4f}" if entry.reason.threshold else "",
                    entry.reason.direction,
                    entry.reason.trend_direction,
                    entry.reason.explanation[:100] if entry.reason.explanation else "",
                    f"{entry.realized_pnl:.2f}" if entry.realized_pnl else ""
                ])
        except Exception as e:
            logger.error(f"Failed to write trade to CSV: {e}")

    def get_trades(
        self,
        symbol: str = None,
        strategy: str = None,
        mode: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        action: str = None
    ) -> pd.DataFrame:
        """Query trades from database with filters."""

        query = "SELECT * FROM trade_log WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if strategy:
            query += " AND strategy_name = ?"
            params.append(strategy)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        if action:
            query += " AND action = ?"
            params.append(action.upper())

        query += " ORDER BY timestamp DESC"

        try:
            with self.db.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                return df
        except Exception as e:
            logger.error(f"Failed to query trades: {e}")
            return pd.DataFrame()

    def get_trade_summary(self, mode: str = None) -> Dict[str, Any]:
        """Get summary statistics for trades."""

        df = self.get_trades(mode=mode)

        if df.empty:
            return {"total_trades": 0}

        buys = df[df['action'] == 'BUY']
        sells = df[df['action'] == 'SELL']

        # Calculate P&L for completed trades
        completed = df[df['realized_pnl'].notna()]

        return {
            "total_trades": len(df),
            "buy_trades": len(buys),
            "sell_trades": len(sells),
            "completed_trades": len(completed),
            "total_realized_pnl": completed['realized_pnl'].sum() if not completed.empty else 0,
            "avg_realized_pnl": completed['realized_pnl'].mean() if not completed.empty else 0,
            "win_rate": (completed['realized_pnl'] > 0).mean() * 100 if not completed.empty else 0,
            "strategies_used": df['strategy_name'].unique().tolist(),
            "symbols_traded": df['symbol'].unique().tolist(),
            "most_used_strategy": df['strategy_name'].mode().iloc[0] if not df.empty else None,
            "avg_position_size_pct": df['position_size_pct'].mean() if 'position_size_pct' in df else 0
        }

    def export_session(self, filename: str = None) -> str:
        """Export current session trades to JSON file."""

        if not filename:
            filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.log_dir / filename

        data = {
            "session_start": self.session_trades[0].timestamp.isoformat() if self.session_trades else None,
            "session_end": datetime.now().isoformat(),
            "total_trades": len(self.session_trades),
            "trades": [t.to_dict() for t in self.session_trades]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Session exported to: {filepath}")
        return str(filepath)

    def print_recent_trades(self, n: int = 10):
        """Print recent trades in a formatted table."""

        df = self.get_trades()

        if df.empty:
            print("No trades logged yet.")
            return

        df = df.head(n)

        print("\n" + "=" * 100)
        print("RECENT TRADES")
        print("=" * 100)
        print(f"{'Trade ID':<20} {'Time':<20} {'Symbol':<8} {'Action':<6} {'Qty':<10} {'Price':<10} {'Reason':<30}")
        print("-" * 100)

        for _, row in df.iterrows():
            print(f"{row['trade_id']:<20} {row['timestamp'][:19]:<20} {row['symbol']:<8} "
                  f"{row['action']:<6} {row['quantity']:<10.2f} ${row['price']:<9.2f} "
                  f"{row['primary_signal'][:30] if row['primary_signal'] else 'N/A':<30}")

        print("=" * 100)


# Helper function to build trade reasons from strategy signals
def build_trade_reason(
    signal_name: str,
    signal_value: float,
    threshold: float,
    direction: str,
    data: pd.DataFrame = None,
    additional_indicators: Dict[str, float] = None
) -> TradeReason:
    """
    Build a TradeReason from strategy signal data.

    Args:
        signal_name: Name of the primary signal (e.g., 'RSI', 'MACD_crossover')
        signal_value: Current value of the signal
        threshold: Threshold that triggered the trade
        direction: 'above' or 'below'
        data: Market data DataFrame for calculating context
        additional_indicators: Dict of other indicator values
    """

    reason = TradeReason(
        primary_signal=signal_name,
        signal_value=signal_value,
        threshold=threshold,
        direction=direction,
        supporting_indicators=additional_indicators or {}
    )

    # Add market context if data provided
    if data is not None and len(data) >= 20:
        # Trend direction
        sma_20 = data['close'].tail(20).mean()
        current_price = data['close'].iloc[-1]
        if current_price > sma_20 * 1.02:
            reason.trend_direction = "bullish"
        elif current_price < sma_20 * 0.98:
            reason.trend_direction = "bearish"
        else:
            reason.trend_direction = "neutral"

        # Volatility
        returns = data['close'].pct_change().tail(20)
        volatility = returns.std()
        if volatility > 0.03:
            reason.volatility_level = "high"
        elif volatility < 0.01:
            reason.volatility_level = "low"
        else:
            reason.volatility_level = "medium"

        # Volume
        if 'volume' in data.columns:
            avg_volume = data['volume'].tail(20).mean()
            current_volume = data['volume'].iloc[-1]
            if current_volume > avg_volume * 1.5:
                reason.volume_condition = "above_average"
            elif current_volume < avg_volume * 0.5:
                reason.volume_condition = "below_average"
            else:
                reason.volume_condition = "normal"

    # Build explanation
    action = "BUY" if direction == "below" or (direction == "above" and "bullish" in signal_name.lower()) else "SELL"
    reason.explanation = (
        f"{action} signal: {signal_name} = {signal_value:.4f} crossed {direction} {threshold:.4f}. "
        f"Market trend: {reason.trend_direction}. Volatility: {reason.volatility_level}."
    )

    return reason


# Global trade logger instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get or create the global trade logger instance."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger

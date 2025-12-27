"""Database management for AI Trading Bot."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
import pandas as pd

from config import config
from utils.logger import get_logger

logger = get_logger(__name__)


class Database:
    """Database handler for storing predictions, trades, and performance data."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = config.get('database.path', 'database/trading_bot.db')

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()
        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    predicted_value REAL,
                    predicted_direction TEXT,
                    confidence REAL,
                    features TEXT,
                    model_version TEXT,
                    actual_value REAL,
                    actual_direction TEXT,
                    accuracy REAL,
                    profit_impact REAL,
                    evaluation_timestamp DATETIME
                )
            """)

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    order_type TEXT,
                    strategy TEXT,
                    status TEXT,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    holding_period INTEGER,
                    commission REAL,
                    slippage REAL,
                    exit_reason TEXT,
                    exit_timestamp DATETIME,
                    metadata TEXT
                )
            """)

            # Portfolio history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    daily_return REAL,
                    cumulative_return REAL,
                    positions TEXT,
                    metrics TEXT
                )
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    period TEXT NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    total_profit REAL,
                    total_loss REAL,
                    net_profit REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown REAL,
                    average_win REAL,
                    average_loss REAL,
                    largest_win REAL,
                    largest_loss REAL,
                    average_holding_period REAL,
                    metrics TEXT
                )
            """)

            # News sentiment table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    headline TEXT,
                    summary TEXT,
                    source TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    relevance_score REAL,
                    url TEXT
                )
            """)

            # Backtests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    strategy_name TEXT NOT NULL,
                    start_date DATE,
                    end_date DATE,
                    initial_capital REAL,
                    final_capital REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    parameters TEXT,
                    results TEXT
                )
            """)

            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    evaluation_period TEXT,
                    training_data_size INTEGER,
                    hyperparameters TEXT
                )
            """)

            # Learning log table (tracks what the bot learns)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    learning_type TEXT NOT NULL,
                    description TEXT,
                    previous_behavior TEXT,
                    new_behavior TEXT,
                    trigger_event TEXT,
                    expected_improvement REAL,
                    metadata TEXT
                )
            """)

            conn.commit()
            logger.info("Database tables initialized")

    # Prediction methods
    def store_prediction(
        self,
        symbol: str,
        prediction_type: str,
        predicted_value: Optional[float] = None,
        predicted_direction: Optional[str] = None,
        confidence: float = 0.0,
        features: Dict[str, Any] = None,
        model_version: str = "1.0"
    ) -> int:
        """Store a prediction for later evaluation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, symbol, prediction_type, predicted_value,
                    predicted_direction, confidence, features, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                symbol,
                prediction_type,
                predicted_value,
                predicted_direction,
                confidence,
                json.dumps(features) if features else None,
                model_version
            ))
            return cursor.lastrowid

    def evaluate_prediction(
        self,
        prediction_id: int,
        actual_value: Optional[float] = None,
        actual_direction: Optional[str] = None,
        profit_impact: float = 0.0
    ):
        """Evaluate a prediction against actual results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get the original prediction
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            prediction = cursor.fetchone()

            if not prediction:
                logger.warning(f"Prediction {prediction_id} not found")
                return

            # Calculate accuracy
            accuracy = None
            if actual_value and prediction['predicted_value']:
                error = abs(actual_value - prediction['predicted_value'])
                accuracy = max(0, 1 - (error / prediction['predicted_value']))
            elif actual_direction and prediction['predicted_direction']:
                accuracy = 1.0 if actual_direction == prediction['predicted_direction'] else 0.0

            cursor.execute("""
                UPDATE predictions
                SET actual_value = ?,
                    actual_direction = ?,
                    accuracy = ?,
                    profit_impact = ?,
                    evaluation_timestamp = ?
                WHERE id = ?
            """, (
                actual_value,
                actual_direction,
                accuracy,
                profit_impact,
                datetime.now(),
                prediction_id
            ))

    def get_prediction_performance(self, days: int = 30) -> pd.DataFrame:
        """Get prediction performance statistics."""
        query = """
            SELECT
                symbol,
                prediction_type,
                AVG(accuracy) as avg_accuracy,
                AVG(confidence) as avg_confidence,
                AVG(profit_impact) as avg_profit_impact,
                COUNT(*) as total_predictions
            FROM predictions
            WHERE evaluation_timestamp IS NOT NULL
                AND timestamp >= datetime('now', '-{} days')
            GROUP BY symbol, prediction_type
        """.format(days)

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    # Trade methods
    def store_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        order_type: str = "market",
        strategy: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> int:
        """Store a trade entry."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, symbol, side, quantity, entry_price,
                    order_type, strategy, status, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                symbol,
                side,
                quantity,
                entry_price,
                order_type,
                strategy,
                'open',
                json.dumps(metadata) if metadata else None
            ))
            return cursor.lastrowid

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str = "manual",
        commission: float = 0.0,
        slippage: float = 0.0
    ):
        """Close a trade and calculate profit/loss."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get the original trade
            cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            trade = cursor.fetchone()

            if not trade:
                logger.warning(f"Trade {trade_id} not found")
                return

            # Calculate P&L
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            side = trade['side']

            if side == 'buy':
                profit_loss = (exit_price - entry_price) * quantity - commission - slippage
            else:  # sell/short
                profit_loss = (entry_price - exit_price) * quantity - commission - slippage

            profit_loss_pct = (profit_loss / (entry_price * quantity)) * 100

            # Calculate holding period
            entry_time = datetime.fromisoformat(trade['timestamp'])
            holding_period = (datetime.now() - entry_time).total_seconds()

            cursor.execute("""
                UPDATE trades
                SET exit_price = ?,
                    status = 'closed',
                    profit_loss = ?,
                    profit_loss_pct = ?,
                    holding_period = ?,
                    commission = ?,
                    slippage = ?,
                    exit_reason = ?,
                    exit_timestamp = ?
                WHERE id = ?
            """, (
                exit_price,
                profit_loss,
                profit_loss_pct,
                holding_period,
                commission,
                slippage,
                exit_reason,
                datetime.now(),
                trade_id
            ))

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open trades."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'open'")
            return [dict(row) for row in cursor.fetchall()]

    def get_trades_history(self, days: int = 30) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        query = """
            SELECT * FROM trades
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        """.format(days)

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    # Portfolio methods
    def store_portfolio_snapshot(
        self,
        total_value: float,
        cash: float,
        positions_value: float,
        daily_return: float,
        cumulative_return: float,
        positions: Dict[str, Any],
        metrics: Dict[str, Any] = None
    ):
        """Store a portfolio snapshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO portfolio_history (
                    timestamp, total_value, cash, positions_value,
                    daily_return, cumulative_return, positions, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                total_value,
                cash,
                positions_value,
                daily_return,
                cumulative_return,
                json.dumps(positions),
                json.dumps(metrics) if metrics else None
            ))

    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio history."""
        query = """
            SELECT * FROM portfolio_history
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp ASC
        """.format(days)

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    # Learning log methods
    def log_learning(
        self,
        learning_type: str,
        description: str,
        previous_behavior: str,
        new_behavior: str,
        trigger_event: str,
        expected_improvement: float = 0.0,
        metadata: Dict[str, Any] = None
    ):
        """Log what the bot learns and how it adapts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO learning_log (
                    timestamp, learning_type, description,
                    previous_behavior, new_behavior, trigger_event,
                    expected_improvement, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                learning_type,
                description,
                previous_behavior,
                new_behavior,
                trigger_event,
                expected_improvement,
                json.dumps(metadata) if metadata else None
            ))

    def get_learning_history(self, days: int = 30) -> pd.DataFrame:
        """Get learning history."""
        query = """
            SELECT * FROM learning_log
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        """.format(days)

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    # Performance methods
    def store_performance_metrics(
        self,
        period: str,
        metrics: Dict[str, Any]
    ):
        """Store performance metrics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (
                    timestamp, period, total_trades, winning_trades,
                    losing_trades, win_rate, total_profit, total_loss,
                    net_profit, profit_factor, sharpe_ratio, sortino_ratio,
                    max_drawdown, average_win, average_loss, largest_win,
                    largest_loss, average_holding_period, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                period,
                metrics.get('total_trades', 0),
                metrics.get('winning_trades', 0),
                metrics.get('losing_trades', 0),
                metrics.get('win_rate', 0.0),
                metrics.get('total_profit', 0.0),
                metrics.get('total_loss', 0.0),
                metrics.get('net_profit', 0.0),
                metrics.get('profit_factor', 0.0),
                metrics.get('sharpe_ratio', 0.0),
                metrics.get('sortino_ratio', 0.0),
                metrics.get('max_drawdown', 0.0),
                metrics.get('average_win', 0.0),
                metrics.get('average_loss', 0.0),
                metrics.get('largest_win', 0.0),
                metrics.get('largest_loss', 0.0),
                metrics.get('average_holding_period', 0.0),
                json.dumps(metrics)
            ))

    # News sentiment methods
    def store_news_sentiment(
        self,
        symbol: str,
        headline: str,
        summary: str,
        source: str,
        sentiment_score: float,
        sentiment_label: str,
        relevance_score: float = 1.0,
        url: str = None
    ):
        """Store news sentiment analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO news_sentiment (
                    timestamp, symbol, headline, summary, source,
                    sentiment_score, sentiment_label, relevance_score, url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                symbol,
                headline,
                summary,
                source,
                sentiment_score,
                sentiment_label,
                relevance_score,
                url
            ))

    def get_recent_sentiment(self, symbol: str = None, hours: int = 24) -> pd.DataFrame:
        """Get recent sentiment for a symbol or all symbols."""
        if symbol:
            query = """
                SELECT * FROM news_sentiment
                WHERE symbol = ? AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours)
            params = (symbol,)
        else:
            query = """
                SELECT * FROM news_sentiment
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours)
            params = ()

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def cleanup_old_data(self, days: int = 365):
        """Clean up data older than specified days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            tables = [
                'news_sentiment',
                'portfolio_history'
            ]

            for table in tables:
                cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE timestamp < datetime('now', '-{days} days')
                """)

            logger.info(f"Cleaned up data older than {days} days")

    def store_backtest_result(
        self,
        strategy_name: str,
        start_date,
        end_date,
        initial_capital: float,
        final_capital: float,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_trades: int,
        parameters: str,
        results: str
    ):
        """Store backtest results."""
        # Convert pandas Timestamps to ISO format strings for SQLite
        if hasattr(start_date, 'isoformat'):
            start_date = start_date.isoformat()
        else:
            start_date = str(start_date)

        if hasattr(end_date, 'isoformat'):
            end_date = end_date.isoformat()
        else:
            end_date = str(end_date)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backtests (
                    timestamp, strategy_name, start_date, end_date,
                    initial_capital, final_capital, total_return,
                    sharpe_ratio, max_drawdown, win_rate, total_trades,
                    parameters, results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                strategy_name,
                start_date,
                end_date,
                initial_capital,
                final_capital,
                total_return,
                sharpe_ratio,
                max_drawdown,
                win_rate,
                total_trades,
                parameters,
                results
            ))

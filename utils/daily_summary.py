"""
Daily Summary Report Generator for AI Trading Bot.

Generates end-of-day reports including:
- Portfolio performance (starting vs ending balance)
- Best and worst performing trades/stocks
- What the bot learned today
- Daily goals and whether they were achieved
- Key market insights
"""

import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np

from utils.logger import get_logger
from utils.database import Database
from config import config

logger = get_logger(__name__)


@dataclass
class DailyGoal:
    """Represents a daily goal for the trading bot."""
    goal_id: str
    date: date
    goal_type: str  # 'profit', 'win_rate', 'new_stock', 'risk_management', 'learning'
    description: str
    target_value: float
    actual_value: float = 0.0
    achieved: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['date'] = self.date.isoformat()
        return data


@dataclass
class DailySummary:
    """Complete daily summary for the trading bot."""
    date: date

    # Portfolio Performance
    starting_cash: float
    ending_cash: float
    starting_portfolio_value: float
    ending_portfolio_value: float
    daily_pnl: float
    daily_return_pct: float

    # Trade Statistics
    total_trades: int
    buy_trades: int
    sell_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float

    # Best/Worst Performers
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]
    best_performing_stock: str
    worst_performing_stock: str

    # Strategies
    most_used_strategy: str
    best_strategy: str
    strategies_used: List[str]

    # Learning & Insights
    lessons_learned: List[str]
    market_conditions: str
    volatility_level: str

    # Goals
    daily_goals: List[DailyGoal]
    goals_achieved: int
    goals_total: int

    # Additional Metrics
    largest_position: Dict[str, Any]
    avg_trade_size: float
    avg_holding_time: float
    sharpe_ratio_today: float
    max_drawdown_today: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['date'] = self.date.isoformat()
        data['daily_goals'] = [g.to_dict() if hasattr(g, 'to_dict') else g for g in self.daily_goals]
        return data


class DailySummaryGenerator:
    """Generates daily summary reports for the trading bot."""

    def __init__(self, output_dir: str = "logs/daily_summaries"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database()
        self._ensure_goals_table()

        # Load Alpaca for account info if available
        self.alpaca_client = None
        self._init_alpaca()

    def _init_alpaca(self):
        """Initialize Alpaca client for account data."""
        try:
            from alpaca.trading.client import TradingClient
            api_key = config.get('alpaca.api_key')
            secret_key = config.get('alpaca.secret_key')
            paper = config.get('alpaca.paper', True)

            if api_key and secret_key:
                self.alpaca_client = TradingClient(api_key, secret_key, paper=paper)
                logger.info("Alpaca client initialized for daily summary")
        except Exception as e:
            logger.warning(f"Could not initialize Alpaca client: {e}")

    def _ensure_goals_table(self):
        """Create daily goals table if it doesn't exist."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT UNIQUE NOT NULL,
                    date DATE NOT NULL,
                    goal_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    target_value REAL,
                    actual_value REAL DEFAULT 0,
                    achieved INTEGER DEFAULT 0,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    starting_cash REAL,
                    ending_cash REAL,
                    starting_portfolio_value REAL,
                    ending_portfolio_value REAL,
                    daily_pnl REAL,
                    daily_return_pct REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    goals_achieved INTEGER,
                    goals_total INTEGER,
                    full_summary TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get_account_balance(self) -> Tuple[float, float]:
        """Get current account balance from Alpaca."""
        if self.alpaca_client:
            try:
                account = self.alpaca_client.get_account()
                cash = float(account.cash)
                portfolio_value = float(account.portfolio_value)
                return cash, portfolio_value
            except Exception as e:
                logger.warning(f"Could not get Alpaca account balance: {e}")

        # Fallback to config or default
        return config.get('trading.initial_capital', 100000.0), config.get('trading.initial_capital', 100000.0)

    def set_daily_goals(self, target_date: date = None) -> List[DailyGoal]:
        """
        Set daily goals for the trading bot.

        Goals are based on:
        - Recent performance
        - Market conditions
        - Learning objectives
        """
        if target_date is None:
            target_date = date.today()

        goals = []

        # Get recent performance to set realistic goals
        recent_stats = self._get_recent_stats(days=7)

        # Goal 1: Daily profit target (based on recent avg, aim for 10% improvement)
        avg_daily_pnl = recent_stats.get('avg_daily_pnl', 0)
        profit_target = max(50, avg_daily_pnl * 1.1) if avg_daily_pnl > 0 else 100

        goals.append(DailyGoal(
            goal_id=f"PROFIT-{target_date.isoformat()}",
            date=target_date,
            goal_type='profit',
            description=f"Achieve ${profit_target:.2f} in daily profit",
            target_value=profit_target
        ))

        # Goal 2: Win rate target
        recent_win_rate = recent_stats.get('win_rate', 50)
        win_rate_target = min(75, recent_win_rate + 5)  # Aim for 5% improvement, max 75%

        goals.append(DailyGoal(
            goal_id=f"WINRATE-{target_date.isoformat()}",
            date=target_date,
            goal_type='win_rate',
            description=f"Maintain win rate above {win_rate_target:.0f}%",
            target_value=win_rate_target
        ))

        # Goal 3: Risk management - keep max drawdown under control
        goals.append(DailyGoal(
            goal_id=f"RISK-{target_date.isoformat()}",
            date=target_date,
            goal_type='risk_management',
            description="Keep intraday drawdown under 2%",
            target_value=2.0
        ))

        # Goal 4: Learning goal - varies by day
        learning_goals = [
            "Identify 1 new pattern in winning trades",
            "Review and adjust signal weights based on recent performance",
            "Analyze correlation between volatility and trade success",
            "Test a strategy parameter adjustment",
            "Document insights from best/worst trades"
        ]
        learning_goal = learning_goals[target_date.weekday() % len(learning_goals)]

        goals.append(DailyGoal(
            goal_id=f"LEARN-{target_date.isoformat()}",
            date=target_date,
            goal_type='learning',
            description=learning_goal,
            target_value=1.0  # Binary: achieved or not
        ))

        # Goal 5: Exploration - find opportunities
        goals.append(DailyGoal(
            goal_id=f"EXPLORE-{target_date.isoformat()}",
            date=target_date,
            goal_type='new_stock',
            description="Screen for 3 new potential stocks to add to watchlist",
            target_value=3.0
        ))

        # Save goals to database
        self._save_goals(goals)

        return goals

    def _save_goals(self, goals: List[DailyGoal]):
        """Save goals to database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for goal in goals:
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_goals
                    (goal_id, date, goal_type, description, target_value, actual_value, achieved, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    goal.goal_id,
                    goal.date.isoformat(),
                    goal.goal_type,
                    goal.description,
                    goal.target_value,
                    goal.actual_value,
                    1 if goal.achieved else 0,
                    goal.notes
                ))

    def _get_recent_stats(self, days: int = 7) -> Dict[str, float]:
        """Get recent trading statistics for goal setting."""
        try:
            with self.db.get_connection() as conn:
                # Get recent trades
                query = """
                    SELECT realized_pnl, timestamp
                    FROM trade_log
                    WHERE realized_pnl IS NOT NULL
                    AND timestamp >= datetime('now', '-{} days')
                """.format(days)
                df = pd.read_sql_query(query, conn)

                if df.empty:
                    return {'avg_daily_pnl': 0, 'win_rate': 50}

                wins = (df['realized_pnl'] > 0).sum()
                total = len(df)

                return {
                    'avg_daily_pnl': df['realized_pnl'].sum() / days,
                    'win_rate': (wins / total * 100) if total > 0 else 50
                }
        except Exception as e:
            logger.warning(f"Could not get recent stats: {e}")
            return {'avg_daily_pnl': 0, 'win_rate': 50}

    def get_todays_goals(self) -> List[DailyGoal]:
        """Get goals for today."""
        today = date.today()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT goal_id, date, goal_type, description, target_value,
                       actual_value, achieved, notes
                FROM daily_goals
                WHERE date = ?
            """, (today.isoformat(),))

            rows = cursor.fetchall()

            if not rows:
                # No goals set yet, create them
                return self.set_daily_goals(today)

            goals = []
            for row in rows:
                goals.append(DailyGoal(
                    goal_id=row[0],
                    date=datetime.strptime(row[1], '%Y-%m-%d').date(),
                    goal_type=row[2],
                    description=row[3],
                    target_value=row[4],
                    actual_value=row[5],
                    achieved=bool(row[6]),
                    notes=row[7] or ""
                ))

            return goals

    def update_goal_progress(self, goal_id: str, actual_value: float, notes: str = ""):
        """Update progress on a specific goal."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get target value
            cursor.execute("SELECT target_value, goal_type FROM daily_goals WHERE goal_id = ?", (goal_id,))
            row = cursor.fetchone()

            if row:
                target = row[0]
                goal_type = row[1]

                # Determine if achieved based on goal type
                if goal_type == 'risk_management':
                    achieved = actual_value <= target  # Lower is better
                else:
                    achieved = actual_value >= target  # Higher is better

                cursor.execute("""
                    UPDATE daily_goals
                    SET actual_value = ?, achieved = ?, notes = ?
                    WHERE goal_id = ?
                """, (actual_value, 1 if achieved else 0, notes, goal_id))

    def generate_summary(self, target_date: date = None, date_from: date = None, date_to: date = None) -> DailySummary:
        """
        Generate a comprehensive daily summary.
        
        Args:
            target_date: Single date for summary (legacy support)
            date_from: Start of date range
            date_to: End of date range
        """
        if target_date is None:
            target_date = date.today()
        
        # Support date range
        if date_from is None:
            date_from = target_date
        if date_to is None:
            date_to = target_date

        # Get trades for the date range
        trades_df = self._get_trades_in_range(date_from, date_to)

        # Get account balances
        starting_cash, starting_portfolio = self._get_start_of_day_balance(target_date)
        ending_cash, ending_portfolio = self.get_account_balance()

        # Calculate P&L
        daily_pnl = ending_portfolio - starting_portfolio
        daily_return_pct = (daily_pnl / starting_portfolio * 100) if starting_portfolio > 0 else 0

        # Trade statistics
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['action'] == 'BUY']) if not trades_df.empty else 0
        sell_trades = len(trades_df[trades_df['action'] == 'SELL']) if not trades_df.empty else 0

        completed_trades = trades_df[trades_df['realized_pnl'].notna()] if not trades_df.empty else pd.DataFrame()
        winning_trades = len(completed_trades[completed_trades['realized_pnl'] > 0]) if not completed_trades.empty else 0
        losing_trades = len(completed_trades[completed_trades['realized_pnl'] < 0]) if not completed_trades.empty else 0
        win_rate = (winning_trades / len(completed_trades) * 100) if len(completed_trades) > 0 else 0

        total_profit = completed_trades[completed_trades['realized_pnl'] > 0]['realized_pnl'].sum() if not completed_trades.empty else 0
        total_loss = abs(completed_trades[completed_trades['realized_pnl'] < 0]['realized_pnl'].sum()) if not completed_trades.empty else 0

        # Best/worst trades
        best_trade = {}
        worst_trade = {}
        if not completed_trades.empty:
            best_idx = completed_trades['realized_pnl'].idxmax()
            worst_idx = completed_trades['realized_pnl'].idxmin()
            best_trade = completed_trades.loc[best_idx].to_dict() if pd.notna(best_idx) else {}
            worst_trade = completed_trades.loc[worst_idx].to_dict() if pd.notna(worst_idx) else {}

        # Best/worst performing stocks
        best_stock, worst_stock = self._get_best_worst_stocks(completed_trades)

        # Strategy analysis
        most_used_strategy = ""
        best_strategy = ""
        strategies_used = []
        if not trades_df.empty and 'strategy_name' in trades_df.columns:
            strategies_used = trades_df['strategy_name'].unique().tolist()
            most_used_strategy = trades_df['strategy_name'].mode().iloc[0] if len(trades_df) > 0 else ""

            # Find best strategy by average P&L
            if not completed_trades.empty:
                strategy_pnl = completed_trades.groupby('strategy_name')['realized_pnl'].mean()
                if not strategy_pnl.empty:
                    best_strategy = strategy_pnl.idxmax()

        # Generate lessons learned
        lessons = self._analyze_lessons_learned(completed_trades, daily_pnl)

        # Determine market conditions
        market_conditions, volatility = self._assess_market_conditions()

        # Get goals and update them
        goals = self.get_todays_goals()
        self._update_goals_with_actual_values(goals, daily_pnl, win_rate, daily_return_pct)
        goals_achieved = sum(1 for g in goals if g.achieved)

        # Additional metrics
        avg_trade_size = trades_df['total_value'].mean() if not trades_df.empty and 'total_value' in trades_df.columns else 0
        avg_holding_time = completed_trades['holding_period_days'].mean() if not completed_trades.empty and 'holding_period_days' in completed_trades.columns else 0

        # Create largest position info
        largest_position = {}
        if not trades_df.empty and 'total_value' in trades_df.columns:
            largest_idx = trades_df['total_value'].idxmax()
            largest_position = trades_df.loc[largest_idx].to_dict() if pd.notna(largest_idx) else {}

        # Calculate today's Sharpe (simplified)
        sharpe_today = (daily_return_pct / 2) if daily_return_pct != 0 else 0  # Simplified
        max_drawdown_today = abs(daily_return_pct) if daily_return_pct < 0 else 0

        summary = DailySummary(
            date=target_date,
            starting_cash=starting_cash,
            ending_cash=ending_cash,
            starting_portfolio_value=starting_portfolio,
            ending_portfolio_value=ending_portfolio,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            total_trades=total_trades,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_loss=total_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            best_performing_stock=best_stock,
            worst_performing_stock=worst_stock,
            most_used_strategy=most_used_strategy,
            best_strategy=best_strategy,
            strategies_used=strategies_used,
            lessons_learned=lessons,
            market_conditions=market_conditions,
            volatility_level=volatility,
            daily_goals=goals,
            goals_achieved=goals_achieved,
            goals_total=len(goals),
            largest_position=largest_position,
            avg_trade_size=avg_trade_size,
            avg_holding_time=avg_holding_time,
            sharpe_ratio_today=sharpe_today,
            max_drawdown_today=max_drawdown_today
        )

        # Save summary
        self._save_summary(summary)

        return summary

    def _get_days_trades(self, target_date: date) -> pd.DataFrame:
        """Get all trades for a specific date."""
        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT * FROM trade_log
                    WHERE DATE(timestamp) = ?
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=(target_date.isoformat(),))
                return df
        except Exception as e:
            logger.warning(f"Could not get day's trades: {e}")
            return pd.DataFrame()

    def _get_trades_in_range(self, date_from: date, date_to: date) -> pd.DataFrame:
        """Get all trades within a date range."""
        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT * FROM trade_log
                    WHERE DATE(timestamp) >= ? AND DATE(timestamp) <= ?
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=(date_from.isoformat(), date_to.isoformat()))
                return df
        except Exception as e:
            logger.warning(f"Could not get trades in range: {e}")
            return pd.DataFrame()

    def _get_start_of_day_balance(self, target_date: date) -> Tuple[float, float]:
        """Get the balance at the start of the day."""
        # Try to get from previous day's summary
        prev_date = target_date - timedelta(days=1)

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ending_cash, ending_portfolio_value
                    FROM daily_summaries
                    WHERE date = ?
                """, (prev_date.isoformat(),))
                row = cursor.fetchone()

                if row:
                    return row[0], row[1]
        except Exception as e:
            logger.debug(f"Could not get previous day values: {e}")

        # Fallback to current balance (for first day or missing data)
        return self.get_account_balance()

    def _get_best_worst_stocks(self, completed_trades: pd.DataFrame) -> Tuple[str, str]:
        """Get the best and worst performing stocks."""
        if completed_trades.empty or 'symbol' not in completed_trades.columns:
            return "", ""

        stock_pnl = completed_trades.groupby('symbol')['realized_pnl'].sum()

        if stock_pnl.empty:
            return "", ""

        best_stock = stock_pnl.idxmax()
        worst_stock = stock_pnl.idxmin()

        return best_stock, worst_stock

    def _analyze_lessons_learned(self, completed_trades: pd.DataFrame, daily_pnl: float) -> List[str]:
        """Analyze trades and generate lessons learned."""
        lessons = []

        if completed_trades.empty:
            lessons.append("No completed trades today - market may have been slow or no signals triggered.")
            return lessons

        # Analyze win/loss patterns
        wins = completed_trades[completed_trades['realized_pnl'] > 0]
        losses = completed_trades[completed_trades['realized_pnl'] < 0]

        if len(wins) > len(losses):
            lessons.append(f"Good day with {len(wins)} wins vs {len(losses)} losses. Strategy selection was effective.")
        elif len(losses) > len(wins):
            lessons.append(f"Challenging day with more losses ({len(losses)}) than wins ({len(wins)}). Review signal thresholds.")

        # Analyze strategies
        if 'strategy_name' in completed_trades.columns:
            strategy_perf = completed_trades.groupby('strategy_name')['realized_pnl'].agg(['sum', 'count'])

            best_strat = strategy_perf['sum'].idxmax() if not strategy_perf.empty else None
            worst_strat = strategy_perf['sum'].idxmin() if not strategy_perf.empty else None

            if best_strat and strategy_perf.loc[best_strat, 'sum'] > 0:
                lessons.append(f"{best_strat} was the most profitable strategy today. Consider increasing its weight.")

            if worst_strat and strategy_perf.loc[worst_strat, 'sum'] < 0:
                lessons.append(f"{worst_strat} underperformed today. Monitor for continued issues.")

        # Analyze trade sizes
        if 'total_value' in completed_trades.columns:
            avg_winner = wins['total_value'].mean() if not wins.empty else 0
            avg_loser = losses['total_value'].mean() if not losses.empty else 0

            if avg_loser > avg_winner * 1.5:
                lessons.append("Losing trades were larger than winners. Consider tighter position sizing on uncertain signals.")

        # Overall P&L insight
        if daily_pnl > 0:
            lessons.append(f"Profitable day with ${daily_pnl:.2f} gain. Key was good signal confirmation.")
        elif daily_pnl < 0:
            lessons.append(f"Loss of ${abs(daily_pnl):.2f} today. Review if market conditions matched strategy assumptions.")

        return lessons

    def _assess_market_conditions(self) -> Tuple[str, str]:
        """Assess current market conditions."""
        # This would ideally pull from market data
        # Simplified version
        try:
            # Try to get SPY data for market assessment
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period="5d")

            if not hist.empty:
                today_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
                five_day_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                volatility = hist['Close'].pct_change().std() * 100

                # Determine conditions
                if today_return > 1:
                    conditions = "Bullish - Strong upward movement"
                elif today_return < -1:
                    conditions = "Bearish - Significant downward pressure"
                elif abs(today_return) < 0.3:
                    conditions = "Ranging - Market moving sideways"
                else:
                    conditions = "Mixed - Moderate volatility"

                if volatility > 2:
                    vol_level = "High"
                elif volatility > 1:
                    vol_level = "Medium"
                else:
                    vol_level = "Low"

                return conditions, vol_level
        except Exception as e:
            logger.debug(f"Could not assess market conditions: {e}")

        return "Unknown - Could not assess market", "Unknown"

    def _update_goals_with_actual_values(self, goals: List[DailyGoal], daily_pnl: float,
                                          win_rate: float, daily_return_pct: float):
        """Update goals with actual achieved values."""
        for goal in goals:
            if goal.goal_type == 'profit':
                goal.actual_value = daily_pnl
                goal.achieved = daily_pnl >= goal.target_value
            elif goal.goal_type == 'win_rate':
                goal.actual_value = win_rate
                goal.achieved = win_rate >= goal.target_value
            elif goal.goal_type == 'risk_management':
                goal.actual_value = abs(daily_return_pct) if daily_return_pct < 0 else 0
                goal.achieved = goal.actual_value <= goal.target_value

            # Save updated goal
            self.update_goal_progress(goal.goal_id, goal.actual_value,
                                     "Achieved!" if goal.achieved else "Not achieved")

    def _save_summary(self, summary: DailySummary):
        """Save summary to database and file."""
        # Save to database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO daily_summaries
                (date, starting_cash, ending_cash, starting_portfolio_value,
                 ending_portfolio_value, daily_pnl, daily_return_pct, total_trades,
                 winning_trades, losing_trades, win_rate, goals_achieved, goals_total,
                 full_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.date.isoformat(),
                summary.starting_cash,
                summary.ending_cash,
                summary.starting_portfolio_value,
                summary.ending_portfolio_value,
                summary.daily_pnl,
                summary.daily_return_pct,
                summary.total_trades,
                summary.winning_trades,
                summary.losing_trades,
                summary.win_rate,
                summary.goals_achieved,
                summary.goals_total,
                json.dumps(summary.to_dict(), default=str)
            ))

        # Save to file
        self._save_to_file(summary)

    def _save_to_file(self, summary: DailySummary):
        """Save summary to a readable file."""
        filename = self.output_dir / f"daily_summary_{summary.date.isoformat()}.txt"

        lines = []
        lines.append("=" * 80)
        lines.append(f"            DAILY TRADING SUMMARY - {summary.date.strftime('%B %d, %Y')}")
        lines.append("=" * 80)
        lines.append("")

        # Portfolio Performance
        lines.append("PORTFOLIO PERFORMANCE")
        lines.append("-" * 40)
        lines.append(f"  Starting Portfolio Value: ${summary.starting_portfolio_value:,.2f}")
        lines.append(f"  Ending Portfolio Value:   ${summary.ending_portfolio_value:,.2f}")
        lines.append(f"  Cash Available:           ${summary.ending_cash:,.2f}")
        lines.append("")

        pnl_emoji = "📈" if summary.daily_pnl >= 0 else "📉"
        pnl_color = "PROFIT" if summary.daily_pnl >= 0 else "LOSS"
        lines.append(f"  Daily P&L: {pnl_emoji} ${summary.daily_pnl:+,.2f} ({summary.daily_return_pct:+.2f}%) - {pnl_color}")
        lines.append("")

        # Trade Statistics
        lines.append("TRADING ACTIVITY")
        lines.append("-" * 40)
        lines.append(f"  Total Trades: {summary.total_trades}")
        lines.append(f"    - Buy orders:  {summary.buy_trades}")
        lines.append(f"    - Sell orders: {summary.sell_trades}")
        lines.append("")
        lines.append(f"  Winning Trades: {summary.winning_trades}")
        lines.append(f"  Losing Trades:  {summary.losing_trades}")
        lines.append(f"  Win Rate:       {summary.win_rate:.1f}%")
        lines.append("")
        lines.append(f"  Total Profits: +${summary.total_profit:,.2f}")
        lines.append(f"  Total Losses:  -${summary.total_loss:,.2f}")
        lines.append("")

        # Best/Worst
        lines.append("PERFORMANCE HIGHLIGHTS")
        lines.append("-" * 40)
        if summary.best_trade:
            bt = summary.best_trade
            lines.append(f"  Best Trade:  {bt.get('symbol', 'N/A')} - ${bt.get('realized_pnl', 0):+,.2f}")
        if summary.worst_trade:
            wt = summary.worst_trade
            lines.append(f"  Worst Trade: {wt.get('symbol', 'N/A')} - ${wt.get('realized_pnl', 0):+,.2f}")
        lines.append(f"  Best Stock:  {summary.best_performing_stock or 'N/A'}")
        lines.append(f"  Worst Stock: {summary.worst_performing_stock or 'N/A'}")
        lines.append("")

        # Strategies
        lines.append("STRATEGY ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"  Most Used Strategy: {summary.most_used_strategy or 'N/A'}")
        lines.append(f"  Best Strategy Today: {summary.best_strategy or 'N/A'}")
        lines.append(f"  Strategies Used: {', '.join(summary.strategies_used) if summary.strategies_used else 'None'}")
        lines.append("")

        # Market Conditions
        lines.append("MARKET CONDITIONS")
        lines.append("-" * 40)
        lines.append(f"  Market: {summary.market_conditions}")
        lines.append(f"  Volatility: {summary.volatility_level}")
        lines.append("")

        # Daily Goals
        lines.append("DAILY GOALS")
        lines.append("-" * 40)
        lines.append(f"  Goals Achieved: {summary.goals_achieved}/{summary.goals_total}")
        lines.append("")
        for goal in summary.daily_goals:
            status = "✅" if goal.achieved else "❌"
            lines.append(f"  {status} {goal.description}")
            lines.append(f"      Target: {goal.target_value:.2f} | Actual: {goal.actual_value:.2f}")
        lines.append("")

        # Lessons Learned
        lines.append("WHAT I LEARNED TODAY")
        lines.append("-" * 40)
        for i, lesson in enumerate(summary.lessons_learned, 1):
            lines.append(f"  {i}. {lesson}")
        lines.append("")

        # Additional Metrics
        lines.append("ADDITIONAL METRICS")
        lines.append("-" * 40)
        lines.append(f"  Average Trade Size: ${summary.avg_trade_size:,.2f}")
        lines.append(f"  Avg Holding Time:   {summary.avg_holding_time:.2f} days")
        lines.append(f"  Today's Sharpe:     {summary.sharpe_ratio_today:.2f}")
        lines.append(f"  Max Drawdown:       {summary.max_drawdown_today:.2f}%")
        lines.append("")

        lines.append("=" * 80)
        lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        logger.info(f"Daily summary saved to: {filename}")

    def print_summary(self, summary: DailySummary = None):
        """Print summary to console."""
        if summary is None:
            summary = self.generate_summary()

        print("\n" + "=" * 60)
        print(f"    DAILY SUMMARY - {summary.date.strftime('%B %d, %Y')}")
        print("=" * 60)

        print(f"\n💰 Portfolio: ${summary.ending_portfolio_value:,.2f}")
        pnl_sign = "+" if summary.daily_pnl >= 0 else ""
        print(f"📊 Daily P&L: {pnl_sign}${summary.daily_pnl:,.2f} ({pnl_sign}{summary.daily_return_pct:.2f}%)")

        print(f"\n📈 Trades: {summary.total_trades} total")
        print(f"   Win Rate: {summary.win_rate:.1f}% ({summary.winning_trades}W / {summary.losing_trades}L)")

        print(f"\n🎯 Goals: {summary.goals_achieved}/{summary.goals_total} achieved")

        print("\n📚 Lessons:")
        for lesson in summary.lessons_learned[:3]:
            print(f"   • {lesson[:70]}...")

        print("\n" + "=" * 60)


# Global instance
_daily_summary_generator: Optional[DailySummaryGenerator] = None


def get_daily_summary_generator() -> DailySummaryGenerator:
    """Get or create the global daily summary generator."""
    global _daily_summary_generator
    if _daily_summary_generator is None:
        _daily_summary_generator = DailySummaryGenerator()
    return _daily_summary_generator


def generate_daily_summary() -> DailySummary:
    """Generate today's daily summary."""
    generator = get_daily_summary_generator()
    return generator.generate_summary()


def set_daily_goals() -> List[DailyGoal]:
    """Set goals for today."""
    generator = get_daily_summary_generator()
    return generator.set_daily_goals()


def print_daily_summary():
    """Print today's summary to console."""
    generator = get_daily_summary_generator()
    summary = generator.generate_summary()
    generator.print_summary(summary)


class SelfReflectionAnalyzer:
    """
    Analyzes the bot's own performance and identifies areas for improvement.

    The bot reviews its trading history and generates actionable insights
    on what it needs to work on.
    """

    def __init__(self):
        self.db = Database()

    def analyze_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Comprehensive self-analysis of trading performance.

        Returns insights on:
        - Strategy effectiveness
        - Timing issues
        - Position sizing problems
        - Pattern recognition gaps
        - Emotional/momentum trading indicators
        """
        analysis = {
            'overall_assessment': '',
            'strengths': [],
            'weaknesses': [],
            'areas_to_improve': [],
            'recommended_actions': [],
            'confidence_calibration': {},
            'strategy_insights': {},
            'timing_analysis': {},
            'risk_assessment': {}
        }

        trades_df = self._get_trade_history(days)

        if trades_df.empty:
            analysis['overall_assessment'] = "Insufficient data for analysis. Need more trading history."
            return analysis

        # Analyze each aspect
        analysis['strategy_insights'] = self._analyze_strategies(trades_df)
        analysis['timing_analysis'] = self._analyze_timing(trades_df)
        analysis['risk_assessment'] = self._analyze_risk(trades_df)
        analysis['confidence_calibration'] = self._analyze_confidence(trades_df)

        # Generate strengths and weaknesses
        self._identify_strengths_weaknesses(analysis, trades_df)

        # Generate improvement recommendations
        self._generate_recommendations(analysis, trades_df)

        # Overall assessment
        analysis['overall_assessment'] = self._generate_overall_assessment(analysis, trades_df)

        return analysis

    def _get_trade_history(self, days: int) -> pd.DataFrame:
        """Get trade history for analysis."""
        try:
            with self.db.get_connection() as conn:
                query = f"""
                    SELECT * FROM trade_log
                    WHERE timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp ASC
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.warning(f"Could not get trade history: {e}")
            return pd.DataFrame()

    def _analyze_strategies(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze strategy effectiveness."""
        insights = {}

        if 'strategy_name' not in trades_df.columns:
            return {'error': 'No strategy data available'}

        completed = trades_df[trades_df['realized_pnl'].notna()]

        if completed.empty:
            return {'error': 'No completed trades to analyze'}

        # Per-strategy analysis
        strategy_stats = completed.groupby('strategy_name').agg({
            'realized_pnl': ['sum', 'mean', 'count'],
            'realized_pnl_pct': 'mean'
        }).round(2)

        strategy_stats.columns = ['total_pnl', 'avg_pnl', 'trade_count', 'avg_return_pct']

        # Win rate per strategy
        for strategy in strategy_stats.index:
            strat_trades = completed[completed['strategy_name'] == strategy]
            wins = (strat_trades['realized_pnl'] > 0).sum()
            total = len(strat_trades)
            strategy_stats.loc[strategy, 'win_rate'] = (wins / total * 100) if total > 0 else 0

        insights['strategy_performance'] = strategy_stats.to_dict()

        # Identify best and worst
        insights['best_strategy'] = strategy_stats['total_pnl'].idxmax()
        insights['worst_strategy'] = strategy_stats['total_pnl'].idxmin()

        # Strategy consistency
        pnl_std = completed.groupby('strategy_name')['realized_pnl'].std()
        insights['most_consistent'] = pnl_std.idxmin() if not pnl_std.empty else None
        insights['least_consistent'] = pnl_std.idxmax() if not pnl_std.empty else None

        return insights

    def _analyze_timing(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade timing effectiveness."""
        insights = {}

        if trades_df.empty or 'timestamp' not in trades_df.columns:
            return {'error': 'No timestamp data available'}

        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_df['day_of_week'] = trades_df['timestamp'].dt.day_name()

        completed = trades_df[trades_df['realized_pnl'].notna()]

        if completed.empty:
            return {'error': 'No completed trades to analyze'}

        # Best/worst hours
        hour_pnl = completed.groupby('hour')['realized_pnl'].mean()
        insights['best_trading_hour'] = int(hour_pnl.idxmax()) if not hour_pnl.empty else None
        insights['worst_trading_hour'] = int(hour_pnl.idxmin()) if not hour_pnl.empty else None

        # Best/worst days
        day_pnl = completed.groupby('day_of_week')['realized_pnl'].mean()
        insights['best_trading_day'] = day_pnl.idxmax() if not day_pnl.empty else None
        insights['worst_trading_day'] = day_pnl.idxmin() if not day_pnl.empty else None

        # Holding period analysis
        if 'holding_period_days' in completed.columns:
            completed_with_hold = completed[completed['holding_period_days'].notna()]
            if not completed_with_hold.empty:
                # Optimal holding period
                holding_bins = pd.cut(completed_with_hold['holding_period_days'],
                                      bins=[0, 0.5, 1, 3, 7, float('inf')],
                                      labels=['<0.5 days', '0.5-1 day', '1-3 days', '3-7 days', '7+ days'])
                hold_pnl = completed_with_hold.groupby(holding_bins)['realized_pnl'].mean()
                insights['optimal_holding_period'] = str(hold_pnl.idxmax()) if not hold_pnl.empty else None

        return insights

    def _analyze_risk(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk management effectiveness."""
        insights = {}

        completed = trades_df[trades_df['realized_pnl'].notna()]

        if completed.empty:
            return {'error': 'No completed trades to analyze'}

        # Win/loss ratio
        wins = completed[completed['realized_pnl'] > 0]['realized_pnl']
        losses = completed[completed['realized_pnl'] < 0]['realized_pnl'].abs()

        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = losses.mean() if not losses.empty else 0

        insights['avg_win'] = avg_win
        insights['avg_loss'] = avg_loss
        insights['win_loss_ratio'] = (avg_win / avg_loss) if avg_loss > 0 else float('inf')

        # Risk/reward assessment
        if insights['win_loss_ratio'] < 1.5:
            insights['risk_reward_issue'] = "Average wins are not significantly larger than losses. Consider widening take-profit or tightening stop-loss."
        else:
            insights['risk_reward_issue'] = None

        # Consecutive losses analysis
        pnl_series = completed['realized_pnl'].values
        max_consecutive_losses = 0
        current_streak = 0
        for pnl in pnl_series:
            if pnl < 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        insights['max_consecutive_losses'] = max_consecutive_losses

        if max_consecutive_losses >= 5:
            insights['drawdown_warning'] = f"Experienced {max_consecutive_losses} consecutive losses. Review strategy parameters."
        else:
            insights['drawdown_warning'] = None

        # Position sizing consistency
        if 'total_value' in completed.columns:
            position_std = completed['total_value'].std()
            position_mean = completed['total_value'].mean()
            insights['position_size_cv'] = (position_std / position_mean) if position_mean > 0 else 0

        return insights

    def _analyze_confidence(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confidence calibration - are high-confidence trades actually better?"""
        insights = {}

        # This would integrate with signal confidence scores if available
        # For now, analyze based on signal strength indicators

        completed = trades_df[trades_df['realized_pnl'].notna()]

        if completed.empty:
            return {'error': 'No completed trades to analyze'}

        # If we have signal_value (indicator strength), analyze it
        if 'signal_value' in completed.columns:
            completed_with_signal = completed[completed['signal_value'].notna()]
            if not completed_with_signal.empty:
                # Correlation between signal strength and outcome
                correlation = completed_with_signal['signal_value'].corr(completed_with_signal['realized_pnl'])
                insights['signal_strength_correlation'] = correlation

                if abs(correlation) < 0.2:
                    insights['calibration_issue'] = "Signal strength has low correlation with actual outcomes. Consider recalibrating thresholds."
                elif correlation > 0.5:
                    insights['calibration_issue'] = None
                    insights['calibration_note'] = "Signal strength correlates well with outcomes."

        return insights

    def _identify_strengths_weaknesses(self, analysis: Dict, trades_df: pd.DataFrame):
        """Identify bot's strengths and weaknesses."""
        strengths = []
        weaknesses = []

        completed = trades_df[trades_df['realized_pnl'].notna()]
        if completed.empty:
            return

        # Win rate analysis
        win_rate = (completed['realized_pnl'] > 0).mean() * 100

        if win_rate >= 60:
            strengths.append(f"Strong win rate of {win_rate:.1f}% - good at selecting profitable trades")
        elif win_rate < 45:
            weaknesses.append(f"Win rate of {win_rate:.1f}% is below target - need to improve signal quality")

        # Risk/reward
        risk_insights = analysis.get('risk_assessment', {})
        if risk_insights.get('win_loss_ratio', 0) >= 2:
            strengths.append("Excellent risk/reward ratio - letting winners run and cutting losses")
        elif risk_insights.get('win_loss_ratio', 0) < 1.5:
            weaknesses.append("Risk/reward ratio needs improvement - average loss is too close to average win")

        # Strategy analysis
        strategy_insights = analysis.get('strategy_insights', {})
        if strategy_insights.get('best_strategy'):
            strengths.append(f"{strategy_insights['best_strategy']} strategy is performing well")
        if strategy_insights.get('worst_strategy') and strategy_insights.get('worst_strategy') != strategy_insights.get('best_strategy'):
            weaknesses.append(f"{strategy_insights['worst_strategy']} strategy needs review or parameter tuning")

        # Timing
        timing = analysis.get('timing_analysis', {})
        if timing.get('best_trading_hour'):
            strengths.append(f"Trading during hour {timing['best_trading_hour']} tends to be profitable")
        if timing.get('worst_trading_hour'):
            weaknesses.append(f"Trading during hour {timing['worst_trading_hour']} shows poor results - consider avoiding")

        # Consecutive losses
        if risk_insights.get('max_consecutive_losses', 0) <= 3:
            strengths.append("Good at avoiding long losing streaks")
        elif risk_insights.get('max_consecutive_losses', 0) >= 6:
            weaknesses.append(f"Experienced {risk_insights['max_consecutive_losses']} consecutive losses - need circuit breaker")

        analysis['strengths'] = strengths
        analysis['weaknesses'] = weaknesses

    def _generate_recommendations(self, analysis: Dict, trades_df: pd.DataFrame):
        """Generate actionable improvement recommendations."""
        recommendations = []
        areas_to_improve = []

        completed = trades_df[trades_df['realized_pnl'].notna()]
        if completed.empty:
            analysis['areas_to_improve'] = ["Gather more trading data for analysis"]
            analysis['recommended_actions'] = ["Continue paper trading to build history"]
            return

        # Based on weaknesses
        for weakness in analysis.get('weaknesses', []):
            if 'win rate' in weakness.lower():
                areas_to_improve.append("Signal Quality")
                recommendations.append("Review and tighten entry signal thresholds")
                recommendations.append("Add confirmation indicators before entering trades")

            if 'risk/reward' in weakness.lower():
                areas_to_improve.append("Position Management")
                recommendations.append("Implement trailing stop-losses to protect gains")
                recommendations.append("Consider partial profit-taking at key levels")

            if 'strategy' in weakness.lower():
                areas_to_improve.append("Strategy Selection")
                recommendations.append("Reduce allocation to underperforming strategies")
                recommendations.append("Run backtests with updated parameters")

            if 'hour' in weakness.lower():
                areas_to_improve.append("Trade Timing")
                recommendations.append("Avoid trading during historically unprofitable hours")
                recommendations.append("Focus activity during high-probability windows")

            if 'consecutive' in weakness.lower():
                areas_to_improve.append("Risk Management")
                recommendations.append("Implement daily loss limit to pause trading after X losses")
                recommendations.append("Reduce position size after 2+ consecutive losses")

        # General recommendations based on data
        risk_insights = analysis.get('risk_assessment', {})
        if risk_insights.get('position_size_cv', 0) > 0.5:
            areas_to_improve.append("Position Sizing Consistency")
            recommendations.append("Standardize position sizing to reduce variance")

        # If no specific issues found
        if not recommendations:
            recommendations.append("Continue current approach - metrics are satisfactory")
            recommendations.append("Monitor for changes in market conditions")

        analysis['areas_to_improve'] = list(set(areas_to_improve))
        analysis['recommended_actions'] = recommendations

    def _generate_overall_assessment(self, analysis: Dict, trades_df: pd.DataFrame) -> str:
        """Generate overall assessment summary."""
        completed = trades_df[trades_df['realized_pnl'].notna()]
        if completed.empty:
            return "Insufficient trading data for a comprehensive assessment."

        total_pnl = completed['realized_pnl'].sum()
        win_rate = (completed['realized_pnl'] > 0).mean() * 100
        num_trades = len(completed)

        # Build assessment
        assessment_parts = []

        if total_pnl > 0:
            assessment_parts.append(f"Overall profitable with ${total_pnl:,.2f} total P&L across {num_trades} trades.")
        else:
            assessment_parts.append(f"Currently in a loss position of ${abs(total_pnl):,.2f} across {num_trades} trades.")

        if win_rate >= 55:
            assessment_parts.append(f"Win rate of {win_rate:.1f}% is solid.")
        else:
            assessment_parts.append(f"Win rate of {win_rate:.1f}% needs improvement.")

        num_strengths = len(analysis.get('strengths', []))
        num_weaknesses = len(analysis.get('weaknesses', []))

        if num_strengths > num_weaknesses:
            assessment_parts.append("The bot has more strengths than weaknesses, indicating good overall design.")
        elif num_weaknesses > num_strengths:
            assessment_parts.append("Several areas need attention. Focus on the recommended improvements.")
        else:
            assessment_parts.append("Performance is balanced with room for optimization.")

        areas = analysis.get('areas_to_improve', [])
        if areas:
            assessment_parts.append(f"Priority areas to work on: {', '.join(areas[:3])}.")

        return " ".join(assessment_parts)

    def generate_self_reflection_report(self, days: int = 30) -> str:
        """Generate a human-readable self-reflection report."""
        analysis = self.analyze_performance(days)

        lines = []
        lines.append("=" * 80)
        lines.append("            BOT SELF-REFLECTION & IMPROVEMENT REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Analysis Period: Last {days} days")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Overall Assessment
        lines.append("OVERALL ASSESSMENT")
        lines.append("-" * 40)
        lines.append(analysis.get('overall_assessment', 'No assessment available'))
        lines.append("")

        # Strengths
        lines.append("💪 MY STRENGTHS")
        lines.append("-" * 40)
        for strength in analysis.get('strengths', ['No strengths identified yet']):
            lines.append(f"  ✓ {strength}")
        lines.append("")

        # Weaknesses
        lines.append("⚠️ MY WEAKNESSES")
        lines.append("-" * 40)
        for weakness in analysis.get('weaknesses', ['No weaknesses identified']):
            lines.append(f"  ✗ {weakness}")
        lines.append("")

        # Areas to Improve
        lines.append("🎯 AREAS I NEED TO WORK ON")
        lines.append("-" * 40)
        for i, area in enumerate(analysis.get('areas_to_improve', ['Continue learning']), 1):
            lines.append(f"  {i}. {area}")
        lines.append("")

        # Recommended Actions
        lines.append("📋 MY ACTION PLAN")
        lines.append("-" * 40)
        for i, action in enumerate(analysis.get('recommended_actions', ['Keep trading and learning']), 1):
            lines.append(f"  {i}. {action}")
        lines.append("")

        # Strategy Insights
        lines.append("📊 STRATEGY PERFORMANCE REVIEW")
        lines.append("-" * 40)
        strat_insights = analysis.get('strategy_insights', {})
        if strat_insights.get('best_strategy'):
            lines.append(f"  Best Performer: {strat_insights['best_strategy']}")
        if strat_insights.get('worst_strategy'):
            lines.append(f"  Needs Work: {strat_insights['worst_strategy']}")
        if strat_insights.get('most_consistent'):
            lines.append(f"  Most Consistent: {strat_insights['most_consistent']}")
        lines.append("")

        # Timing Analysis
        lines.append("⏰ TIMING INSIGHTS")
        lines.append("-" * 40)
        timing = analysis.get('timing_analysis', {})
        if timing.get('best_trading_hour'):
            lines.append(f"  Best Hour: {timing['best_trading_hour']}:00")
        if timing.get('worst_trading_hour'):
            lines.append(f"  Avoid Hour: {timing['worst_trading_hour']}:00")
        if timing.get('best_trading_day'):
            lines.append(f"  Best Day: {timing['best_trading_day']}")
        if timing.get('optimal_holding_period'):
            lines.append(f"  Optimal Hold Time: {timing['optimal_holding_period']}")
        lines.append("")

        # Risk Assessment
        lines.append("⚖️ RISK MANAGEMENT CHECK")
        lines.append("-" * 40)
        risk = analysis.get('risk_assessment', {})
        if risk.get('win_loss_ratio'):
            lines.append(f"  Win/Loss Ratio: {risk['win_loss_ratio']:.2f}")
        if risk.get('max_consecutive_losses'):
            lines.append(f"  Max Losing Streak: {risk['max_consecutive_losses']} trades")
        if risk.get('risk_reward_issue'):
            lines.append(f"  Issue: {risk['risk_reward_issue']}")
        if risk.get('drawdown_warning'):
            lines.append(f"  Warning: {risk['drawdown_warning']}")
        lines.append("")

        lines.append("=" * 80)
        lines.append("I will continue to learn and improve! 🤖📈")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_reflection_report(self, days: int = 30):
        """Save self-reflection report to file."""
        report = self.generate_self_reflection_report(days)

        output_dir = Path("logs/self_reflection")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_dir / f"reflection_{date.today().isoformat()}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Self-reflection report saved to: {filename}")
        return str(filename)


def get_self_reflection_analyzer() -> SelfReflectionAnalyzer:
    """Get a self-reflection analyzer instance."""
    return SelfReflectionAnalyzer()


def generate_self_reflection(days: int = 30) -> str:
    """Generate and save a self-reflection report."""
    analyzer = SelfReflectionAnalyzer()
    analyzer.save_reflection_report(days)
    return analyzer.generate_self_reflection_report(days)


if __name__ == "__main__":
    # Generate and print today's summary
    print_daily_summary()

    print("\n\n")

    # Generate self-reflection
    analyzer = SelfReflectionAnalyzer()
    print(analyzer.generate_self_reflection_report())

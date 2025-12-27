"""
Generate human-readable trade log from the technical trade data.

Converts technical trading jargon into plain English explanations.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from utils.database import Database
from utils.logger import get_logger

logger = get_logger(__name__)


class ReadableTradeLog:
    """Generates human-readable trade logs."""

    def __init__(self):
        self.db = Database()

    def get_all_trades(self, filter_type: str = 'all', date_filter: str = None, 
                       date_from: str = None, date_to: str = None) -> pd.DataFrame:
        """
        Get trades from database with optional filtering.
        
        Args:
            filter_type: 'all', 'backtest', 'live', 'paper', 'live_paper'
            date_filter: 'today', 'week', 'month', 'custom', or None for all time
            date_from: Start date for custom range (YYYY-MM-DD)
            date_to: End date for custom range (YYYY-MM-DD)
        
        Returns:
            DataFrame of filtered trades
        """
        # Build WHERE clause based on filter
        conditions = []
        
        # Filter by trade type
        if filter_type == 'backtest':
            conditions.append("mode = 'backtest'")
        elif filter_type == 'live':
            conditions.append("mode = 'live'")
        elif filter_type == 'paper':
            conditions.append("mode = 'paper'")
        elif filter_type == 'live_paper':
            conditions.append("(mode = 'live' OR mode = 'paper')")
        # 'all' = no filter
        
        # Filter by date
        if date_filter == 'today':
            conditions.append("DATE(timestamp) = DATE('now')")
        elif date_filter == 'week':
            conditions.append("timestamp >= DATE('now', '-7 days')")
        elif date_filter == 'month':
            conditions.append("timestamp >= DATE('now', '-30 days')")
        elif date_filter == 'custom':
            if date_from:
                conditions.append(f"DATE(timestamp) >= '{date_from}'")
            if date_to:
                conditions.append(f"DATE(timestamp) <= '{date_to}'")
        
        # Build query
        query = "SELECT * FROM trade_log"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp ASC"
        
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
        return df

    def get_trade_counts(self) -> dict:
        """Get counts of trades by type for display."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            counts = {
                'total': 0,
                'backtest': 0,
                'live': 0,
                'paper': 0,
                'today': 0
            }
            
            # Total
            cursor.execute("SELECT COUNT(*) FROM trade_log")
            counts['total'] = cursor.fetchone()[0]
            
            # By mode
            cursor.execute("SELECT mode, COUNT(*) FROM trade_log GROUP BY mode")
            for row in cursor.fetchall():
                if row[0] in counts:
                    counts[row[0]] = row[1]
            
            # Today
            cursor.execute("SELECT COUNT(*) FROM trade_log WHERE DATE(timestamp) = DATE('now')")
            counts['today'] = cursor.fetchone()[0]
            
        return counts

    def explain_strategy(self, strategy_name: str) -> str:
        """Get plain English description of a strategy."""
        strategies = {
            'momentum_strategy': 'Momentum - Buys stocks that are going up, sells when they stop',
            'momentum': 'Momentum - Buys stocks that are going up, sells when they stop',
            'mean_reversion_strategy': 'Mean Reversion - Buys when stock drops too far, expecting a bounce back',
            'mean_reversion': 'Mean Reversion - Buys when stock drops too far, expecting a bounce back',
            'trend_following_strategy': 'Trend Following - Follows the overall direction of the market',
            'trend_following': 'Trend Following - Follows the overall direction of the market',
            'breakout_strategy': 'Breakout - Buys when price breaks above recent highs',
            'breakout': 'Breakout - Buys when price breaks above recent highs',
            'rsi_strategy': 'RSI - Buys oversold stocks, sells overbought stocks',
            'rsi': 'RSI - Buys oversold stocks, sells overbought stocks',
            'macd_strategy': 'MACD - Uses moving average crossovers to find buy/sell signals',
            'macd': 'MACD - Uses moving average crossovers to find buy/sell signals',
            'pairs_trading_strategy': 'Pairs Trading - Trades based on statistical relationships',
            'pairs_trading': 'Pairs Trading - Trades based on statistical relationships',
            'ml_hybrid_strategy': 'AI Hybrid - Uses multiple signals combined with machine learning',
            'ml_hybrid': 'AI Hybrid - Uses multiple signals combined with machine learning',
            'ai_prediction': 'AI Prediction - Uses confidence-weighted technical analysis',
            'ai_prediction_strategy': 'AI Prediction - Uses confidence-weighted technical analysis'
        }
        return strategies.get(strategy_name, strategy_name)

    def explain_signal(self, signal: str, value: float, threshold: float, direction: str) -> str:
        """Convert technical signal into plain English."""

        if signal == 'momentum':
            pct = value * 100
            thresh_pct = threshold * 100
            if direction == 'above':
                return f"Stock rose {pct:.1f}% in 20 days (above {thresh_pct:.1f}% trigger) - upward momentum detected"
            else:
                return f"Stock fell {abs(pct):.1f}% in 20 days (below {thresh_pct:.1f}% trigger) - momentum died"

        elif signal == 'bollinger_lower_band':
            return f"Price dropped to ${value:.2f}, below the safety band at ${threshold:.2f} - stock is oversold"

        elif signal == 'bollinger_mean_reversion':
            return f"Price bounced back to ${value:.2f}, reaching the average at ${threshold:.2f} - target hit"

        elif signal == 'ma_crossover_bullish':
            return "20-day average crossed ABOVE 50-day average - trend turning positive (Golden Cross)"

        elif signal == 'ma_crossover_bearish':
            return "20-day average crossed BELOW 50-day average - trend turning negative (Death Cross)"

        elif signal == 'channel_breakout_up':
            return f"Price ${value:.2f} broke above resistance at ${threshold:.2f} - bullish breakout"

        elif signal == 'channel_breakdown':
            return f"Price ${value:.2f} broke below support at ${threshold:.2f} - exiting to prevent larger losses"

        elif signal == 'rsi_oversold':
            return f"RSI at {value:.1f} (below {threshold:.0f}) - stock is oversold, expecting bounce"

        elif signal == 'rsi_overbought':
            return f"RSI at {value:.1f} (above {threshold:.0f}) - stock is overbought, time to sell"

        elif signal == 'macd_bullish_crossover':
            return "MACD crossed above signal line - momentum turning positive"

        elif signal == 'macd_bearish_crossover':
            return "MACD crossed below signal line - momentum turning negative"

        elif signal == 'z_score_oversold':
            return f"Price is {abs(value):.1f} standard deviations below average - extremely oversold"

        elif signal == 'z_score_mean_reversion':
            return "Price returned to near its average - taking profits"

        elif signal == 'multi_indicator_bullish':
            return f"{int(value)} out of {int(threshold)} indicators are bullish - strong buy signal"

        elif signal == 'backtest_end':
            return "Backtest period ended - closing all positions"

        else:
            return f"{signal}: value={value:.4f}, threshold={threshold:.4f}, direction={direction}"

    def format_money(self, amount: float) -> str:
        """Format money with proper sign and commas."""
        if amount >= 0:
            return f"+${amount:,.2f}"
        else:
            return f"-${abs(amount):,.2f}"

    def format_trade(self, trade: pd.Series, trade_num: int) -> str:
        """Format a single trade into readable text."""
        lines = []

        # Parse timestamp
        ts = pd.to_datetime(trade['timestamp'])
        date_str = ts.strftime("%B %d, %Y")

        lines.append("-" * 78)
        lines.append(f"TRADE #{trade_num} - {date_str}")
        lines.append("-" * 78)
        lines.append(f"Stock: {trade['symbol']}")

        action = "BOUGHT" if trade['action'] == 'BUY' else "SOLD"
        lines.append(f"Action: {action} {trade['quantity']:.2f} shares at ${trade['price']:.2f} each")
        lines.append(f"Total: ${trade['total_value']:,.2f}")
        lines.append(f"Strategy: {self.explain_strategy(trade['strategy_name'])}")
        lines.append("")

        # Why the bot did this
        lines.append(f"WHY: {self.explain_signal(trade['primary_signal'], trade['signal_value'] or 0, trade['threshold'] or 0, trade['direction'] or '')}")

        # Result if it's a sell with P&L
        if trade['realized_pnl'] is not None and pd.notna(trade['realized_pnl']):
            pnl = trade['realized_pnl']
            if pnl >= 0:
                lines.append(f"\nRESULT: {self.format_money(pnl)} PROFIT!")
            else:
                lines.append(f"\nRESULT: {self.format_money(pnl)} LOSS")

        lines.append("")
        return "\n".join(lines)

    def generate_readable_report(self, output_path: str = None, filter_type: str = 'all', 
                                   date_filter: str = None, date_from: str = None, date_to: str = None) -> str:
        """
        Generate a complete readable trade report.
        
        Args:
            output_path: Where to save the report
            filter_type: 'all', 'backtest', 'live', 'paper', 'live_paper'
            date_filter: 'today', 'week', 'month', 'custom', or None
            date_from: Start date for custom range (YYYY-MM-DD)
            date_to: End date for custom range (YYYY-MM-DD)
        """

        if output_path is None:
            output_path = "trade_logs/trades_readable.txt"

        df = self.get_all_trades(filter_type=filter_type, date_filter=date_filter, 
                                  date_from=date_from, date_to=date_to)

        if df.empty:
            filter_desc = self._get_filter_description(filter_type, date_filter, date_from, date_to)
            return f"No trades found matching filter: {filter_desc}"

        # Build title based on filter
        filter_desc = self._get_filter_description(filter_type, date_filter, date_from, date_to)

        lines = []
        lines.append("=" * 78)
        lines.append("              TRADING BOT - TRADE LOG (EASY TO READ)")
        lines.append("=" * 78)
        lines.append("")
        lines.append(f"Filter: {filter_desc}")
        lines.append(f"Total trades in this report: {len(df)}")
        lines.append("")
        lines.append("This report shows trades made by your trading bot in plain English.")
        lines.append("")
        lines.append("QUICK GUIDE:")
        lines.append("  - BOUGHT = Bot purchased shares (expecting price to go up)")
        lines.append("  - SOLD = Bot sold shares (taking profits or cutting losses)")
        lines.append("  - PROFIT = Trade made money")
        lines.append("  - LOSS = Trade lost money")
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"                         TRADES ({filter_desc.upper()})")
        lines.append("=" * 78)
        lines.append("")

        for i, (_, trade) in enumerate(df.iterrows(), 1):
            lines.append(self.format_trade(trade, i))

        # Summary section
        lines.append("=" * 78)
        lines.append("                         SUMMARY")
        lines.append("=" * 78)
        lines.append("")

        # Group by strategy
        strategy_stats = df.groupby('strategy_name').agg({
            'trade_id': 'count',
            'realized_pnl': 'sum'
        }).reset_index()
        strategy_stats.columns = ['Strategy', 'Trades', 'Total P&L']

        lines.append("RESULTS BY STRATEGY:")
        lines.append("-" * 50)
        for _, row in strategy_stats.iterrows():
            pnl = row['Total P&L'] if pd.notna(row['Total P&L']) else 0
            lines.append(f"  {self.explain_strategy(row['Strategy'])[:40]:<40}")
            lines.append(f"    Trades: {row['Trades']}, Net Result: {self.format_money(pnl)}")
            lines.append("")

        # Overall stats
        total_trades = len(df)
        sells_with_pnl = df[df['realized_pnl'].notna()]
        total_pnl = sells_with_pnl['realized_pnl'].sum() if not sells_with_pnl.empty else 0
        wins = (sells_with_pnl['realized_pnl'] > 0).sum() if not sells_with_pnl.empty else 0
        losses = (sells_with_pnl['realized_pnl'] < 0).sum() if not sells_with_pnl.empty else 0

        lines.append("-" * 50)
        lines.append(f"OVERALL: {total_trades} trades total")
        lines.append(f"  Winning trades: {wins}")
        lines.append(f"  Losing trades: {losses}")
        lines.append(f"  Net Result: {self.format_money(total_pnl)}")

        if wins + losses > 0:
            win_rate = (wins / (wins + losses)) * 100
            lines.append(f"  Win Rate: {win_rate:.1f}%")

        lines.append("")
        lines.append("=" * 78)
        lines.append(f"Report generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        lines.append("=" * 78)

        report = "\n".join(lines)

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)

        logger.info(f"Readable trade log saved to: {output_path}")
        return report

    def _get_filter_description(self, filter_type: str, date_filter: str, 
                                 date_from: str = None, date_to: str = None) -> str:
        """Get human-readable description of the filter."""
        parts = []
        
        # Trade type
        type_desc = {
            'all': 'All trades',
            'backtest': 'Backtest trades only',
            'live': 'Live trades only',
            'paper': 'Paper trades only',
            'live_paper': 'Live & Paper trades'
        }
        parts.append(type_desc.get(filter_type, filter_type))
        
        # Date filter
        if date_filter == 'today':
            parts.append('from today')
        elif date_filter == 'week':
            parts.append('from last 7 days')
        elif date_filter == 'month':
            parts.append('from last 30 days')
        elif date_filter == 'custom':
            if date_from and date_to:
                parts.append(f'from {date_from} to {date_to}')
            elif date_from:
                parts.append(f'from {date_from} onwards')
            elif date_to:
                parts.append(f'up to {date_to}')
        
        return ' '.join(parts)

    def generate_simple_csv(self, output_path: str = None, filter_type: str = 'all', 
                            date_filter: str = None, date_from: str = None, date_to: str = None) -> str:
        """
        Generate a simplified CSV with plain English columns.
        
        Args:
            output_path: Where to save the CSV
            filter_type: 'all', 'backtest', 'live', 'paper', 'live_paper'
            date_filter: 'today', 'week', 'month', 'custom', or None
            date_from: Start date for custom range (YYYY-MM-DD)
            date_to: End date for custom range (YYYY-MM-DD)
        """

        if output_path is None:
            output_path = "trade_logs/trades_simple.csv"

        df = self.get_all_trades(filter_type=filter_type, date_filter=date_filter,
                                  date_from=date_from, date_to=date_to)

        if df.empty:
            return "No trades found."

        simple_rows = []
        for _, trade in df.iterrows():
            ts = pd.to_datetime(trade['timestamp'])

            action = "Bought" if trade['action'] == 'BUY' else "Sold"
            
            # Get trade type from mode
            trade_type = trade.get('mode', 'unknown')
            if trade_type == 'backtest':
                trade_type = 'Backtest'
            elif trade_type == 'paper':
                trade_type = 'Paper'
            elif trade_type == 'live':
                trade_type = 'LIVE'

            # Determine result
            result = ""
            if trade['realized_pnl'] is not None and pd.notna(trade['realized_pnl']):
                pnl = trade['realized_pnl']
                if pnl >= 0:
                    result = f"Made ${pnl:,.2f}"
                else:
                    result = f"Lost ${abs(pnl):,.2f}"

            simple_rows.append({
                'Date': ts.strftime("%m/%d/%Y"),
                'Time': ts.strftime("%I:%M %p"),
                'Type': trade_type,
                'Stock': trade['symbol'],
                'What Happened': f"{action} {trade['quantity']:.0f} shares",
                'Price Per Share': f"${trade['price']:.2f}",
                'Total Amount': f"${trade['total_value']:,.2f}",
                'Strategy Used': self.explain_strategy(trade['strategy_name']).split(' - ')[0],
                'Why': self.explain_signal(
                    trade['primary_signal'],
                    trade['signal_value'] or 0,
                    trade['threshold'] or 0,
                    trade['direction'] or ''
                )[:80],
                'Result': result
            })

        simple_df = pd.DataFrame(simple_rows)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        simple_df.to_csv(output_file, index=False)

        logger.info(f"Simple CSV saved to: {output_path}")
        return str(output_file)


def generate_readable_logs():
    """Generate all readable log formats."""
    reader = ReadableTradeLog()

    print("\nGenerating readable trade logs...")
    print("-" * 50)

    # Generate text report
    txt_path = "trade_logs/trades_readable.txt"
    reader.generate_readable_report(txt_path)
    print(f"Created: {txt_path}")

    # Generate simple CSV
    csv_path = "trade_logs/trades_simple.csv"
    reader.generate_simple_csv(csv_path)
    print(f"Created: {csv_path}")

    print("-" * 50)
    print("Done! Check the trade_logs folder for the readable files.")


if __name__ == "__main__":
    generate_readable_logs()

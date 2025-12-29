"""
TAZ TRADER - Tazmanian Devil Aggressive Trading System
=======================================================

Purpose: Grow small accounts ($500-$1000) as FAST as possible.

This trader combines:
- Aggressive momentum strategies
- RL-guided decision making
- Volatility scanning for opportunities
- Both stocks AND crypto (24/7 trading)

Author: Built for maximum profit velocity
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest, StockLatestQuoteRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# Local imports
from scanner.taz_scanner import TazScanner, TazOpportunity
from rl.taz_rl_agent import TazRLTrainer, TazTradingEnvironment

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')


@dataclass
class TazPosition:
    """Active Taz position."""
    symbol: str
    asset_type: str  # 'stock' or 'crypto'
    shares: float
    entry_price: float
    entry_time: datetime
    strategy: str
    target_profit_pct: float = 0.03  # 3% default target
    stop_loss_pct: float = 0.05  # 5% stop loss
    max_hold_minutes: int = 480  # 8 hours max


@dataclass
class TazStats:
    """Trading statistics."""
    trades_executed: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    biggest_win: float = 0.0
    biggest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    best_streak: int = 0
    worst_streak: int = 0


class TazTrader:
    """
    The Tazmanian Devil - Aggressive profit-seeking trading bot.

    Designed for small accounts looking for rapid growth.
    Trades both stocks and crypto for maximum opportunities.
    """

    def __init__(
        self,
        initial_capital: float = 1000,
        paper: bool = True,
        max_position_pct: float = 0.15,  # REDUCED from 40% to 15% for safety
        max_positions: int = 3,
        stop_loss_pct: float = 0.02,  # TIGHTENED from 5% to 2%
        take_profit_pct: float = 0.03,
        check_interval: int = 30,  # Seconds
        use_rl: bool = True,
        trade_crypto: bool = True,
        trade_stocks: bool = False  # Disabled by default to avoid PDT
    ):
        """
        Initialize Taz Trader.

        Args:
            initial_capital: Starting capital (for tracking)
            paper: Paper trading mode
            max_position_pct: Max % of portfolio per position
            max_positions: Max concurrent positions
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            check_interval: Seconds between checks
            use_rl: Use RL agent for decisions
            trade_crypto: Enable crypto trading
            trade_stocks: Enable stock trading (disabled by default to avoid PDT)
        """
        self.initial_capital = initial_capital
        self.paper = paper
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.check_interval = check_interval
        self.use_rl = use_rl
        self.trade_crypto = trade_crypto
        self.trade_stocks = trade_stocks

        # Alpaca clients
        self.trading_client = TradingClient(API_KEY, API_SECRET, paper=paper)
        self.stock_data = StockHistoricalDataClient(API_KEY, API_SECRET)
        self.crypto_data = CryptoHistoricalDataClient(API_KEY, API_SECRET)

        # Scanner
        self.scanner = TazScanner(min_volatility=2.0, min_volume_ratio=1.0)

        # RL Trainer
        self.rl_trainer = TazRLTrainer() if use_rl else None

        # Positions
        self.positions: Dict[str, TazPosition] = {}

        # Stats
        self.stats = TazStats()

        # State
        self.running = False
        self._stop_event = threading.Event()

        # Data directory
        self.data_dir = Path(__file__).parent / 'data'
        self.data_dir.mkdir(exist_ok=True)

        # Last scan time
        self.last_scan_time = None
        self.scan_interval_minutes = 15

        print(f"""
================================================================
          TAZ TRADER INITIALIZED - Tazmanian Devil Mode
================================================================
  Capital: ${initial_capital:,.2f}
  Mode: {'PAPER' if paper else 'LIVE'} Trading
  Max Position: {max_position_pct*100:.0f}%
  Stop Loss: {stop_loss_pct*100:.0f}% | Take Profit: {take_profit_pct*100:.0f}%
  Crypto: {'Enabled' if trade_crypto else 'Disabled'}
  Stocks: {'Enabled' if trade_stocks else 'Disabled (No PDT)'}
  RL Agent: {'Enabled' if use_rl else 'Disabled'}
================================================================
        """)

    def get_account(self) -> dict:
        """Get account info."""
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            print(f"[TAZ] Error getting account: {e}")
            return {'equity': 0, 'cash': 0, 'buying_power': 0}

    def get_current_positions(self) -> Dict[str, dict]:
        """Get all current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return {
                pos.symbol: {
                    'shares': float(pos.qty),
                    'avg_cost': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'pnl_pct': float(pos.unrealized_plpc) * 100
                }
                for pos in positions
            }
        except Exception as e:
            print(f"[TAZ] Error getting positions: {e}")
            return {}

    def get_current_price(self, symbol: str, asset_type: str = 'stock') -> Optional[float]:
        """Get current price for a symbol."""
        try:
            if asset_type == 'crypto':
                request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.crypto_data.get_crypto_latest_quote(request)
            else:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.stock_data.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                return float(quote.ask_price or quote.bid_price)
            return None
        except Exception as e:
            print(f"[TAZ] Error getting price for {symbol}: {e}")
            return None

    def execute_buy(self, symbol: str, strategy: str, asset_type: str = 'stock',
                    position_pct: float = None) -> bool:
        """
        Execute a buy order.

        Args:
            symbol: Symbol to buy
            strategy: Strategy name that triggered the buy
            asset_type: 'stock' or 'crypto'
            position_pct: Override position size (default: max_position_pct)
        """
        try:
            # Check position count
            current_positions = self.get_current_positions()
            if len(current_positions) >= self.max_positions:
                print(f"[TAZ] Max positions ({self.max_positions}) reached")
                return False

            # Already have position?
            if symbol in current_positions or symbol.replace('/USD', '') in current_positions:
                print(f"[TAZ] Already have position in {symbol}")
                return False

            # Get account and calculate position size
            account = self.get_account()
            equity = account['equity']

            position_pct = position_pct or self.max_position_pct
            position_value = equity * position_pct

            # Get current price
            price = self.get_current_price(symbol, asset_type)
            if not price:
                return False

            # Calculate shares (fractional for crypto)
            if asset_type == 'crypto':
                shares = position_value / price
                # Round to reasonable precision
                shares = round(shares, 6)
            else:
                shares = int(position_value / price)

            if shares <= 0:
                return False

            # Execute order
            # For crypto, use the symbol without /USD for Alpaca
            order_symbol = symbol.replace('/USD', 'USD') if asset_type == 'crypto' else symbol

            order = MarketOrderRequest(
                symbol=order_symbol,
                qty=shares if asset_type == 'stock' else None,
                notional=position_value if asset_type == 'crypto' else None,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC if asset_type == 'crypto' else TimeInForce.DAY
            )

            result = self.trading_client.submit_order(order)

            # Track position
            self.positions[symbol] = TazPosition(
                symbol=symbol,
                asset_type=asset_type,
                shares=shares,
                entry_price=price,
                entry_time=datetime.now(),
                strategy=strategy,
                target_profit_pct=self.take_profit_pct,
                stop_loss_pct=self.stop_loss_pct
            )

            print(f"[TAZ] >>> BUY {shares:.4f} {symbol} @ ${price:.4f} | Strategy: {strategy}")
            self.stats.trades_executed += 1
            self._save_state()

            return True

        except Exception as e:
            print(f"[TAZ] Buy failed for {symbol}: {e}")
            return False

    def execute_sell(self, symbol: str, reason: str = "signal") -> bool:
        """Execute a sell order."""
        try:
            current_positions = self.get_current_positions()

            # Handle crypto symbol format
            lookup_symbol = symbol.replace('/USD', 'USD') if '/USD' in symbol else symbol

            if lookup_symbol not in current_positions:
                print(f"[TAZ] No position in {symbol} to sell")
                return False

            pos = current_positions[lookup_symbol]
            shares = pos['shares']

            # Execute sell
            order = MarketOrderRequest(
                symbol=lookup_symbol,
                qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC if 'USD' in lookup_symbol and lookup_symbol != 'USD' else TimeInForce.DAY
            )

            result = self.trading_client.submit_order(order)

            # Calculate P&L
            pnl = pos['unrealized_pnl']
            pnl_pct = pos['pnl_pct']

            # Update stats
            self.stats.total_profit += pnl
            self.stats.trades_executed += 1

            if pnl > 0:
                self.stats.winning_trades += 1
                self.stats.consecutive_wins += 1
                self.stats.consecutive_losses = 0
                if self.stats.consecutive_wins > self.stats.best_streak:
                    self.stats.best_streak = self.stats.consecutive_wins
                if pnl > self.stats.biggest_win:
                    self.stats.biggest_win = pnl
                result_str = f"WIN +${pnl:.2f} ({pnl_pct:+.1f}%)"
            else:
                self.stats.losing_trades += 1
                self.stats.consecutive_losses += 1
                self.stats.consecutive_wins = 0
                if self.stats.consecutive_losses > self.stats.worst_streak:
                    self.stats.worst_streak = self.stats.consecutive_losses
                if pnl < self.stats.biggest_loss:
                    self.stats.biggest_loss = pnl
                result_str = f"LOSS ${pnl:.2f} ({pnl_pct:.1f}%)"

            # Remove from tracking
            if symbol in self.positions:
                del self.positions[symbol]

            print(f"[TAZ] $$$ SELL {shares:.4f} {symbol} @ ${pos['current_price']:.4f} | {result_str} | Reason: {reason}")
            self._save_state()

            return True

        except Exception as e:
            print(f"[TAZ] Sell failed for {symbol}: {e}")
            return False

    def check_positions(self):
        """Check all positions for exit conditions."""
        current_positions = self.get_current_positions()

        for symbol, taz_pos in list(self.positions.items()):
            lookup_symbol = symbol.replace('/USD', 'USD') if '/USD' in symbol else symbol

            if lookup_symbol not in current_positions:
                continue

            pos = current_positions[lookup_symbol]
            pnl_pct = pos['pnl_pct'] / 100  # Convert to decimal

            # Check stop loss
            if pnl_pct <= -taz_pos.stop_loss_pct:
                print(f"[TAZ] !!! STOP LOSS triggered for {symbol}")
                self.execute_sell(symbol, f"stop_loss_{pnl_pct:.1%}")
                continue

            # Check take profit
            if pnl_pct >= taz_pos.target_profit_pct:
                print(f"[TAZ] +++ TAKE PROFIT triggered for {symbol}")
                self.execute_sell(symbol, f"take_profit_{pnl_pct:.1%}")
                continue

            # Check max hold time
            hold_minutes = (datetime.now() - taz_pos.entry_time).total_seconds() / 60
            if hold_minutes >= taz_pos.max_hold_minutes:
                print(f"[TAZ] --- MAX HOLD TIME reached for {symbol}")
                self.execute_sell(symbol, f"max_hold_{hold_minutes:.0f}min")
                continue

            # Trailing stop: if we're up 2%+, tighten stop to breakeven
            if pnl_pct > 0.02 and taz_pos.stop_loss_pct > 0.01:
                taz_pos.stop_loss_pct = 0.01  # Tighten to 1%
                print(f"[TAZ] Trailing stop tightened for {symbol}")

    def scan_for_opportunities(self):
        """Scan for new trading opportunities."""
        # Check if we need to scan
        if self.last_scan_time:
            minutes_since_scan = (datetime.now() - self.last_scan_time).total_seconds() / 60
            if minutes_since_scan < self.scan_interval_minutes:
                return

        print(f"\n[TAZ] Scanning for opportunities...")

        # Check if we can take new positions
        current_positions = self.get_current_positions()
        if len(current_positions) >= self.max_positions:
            print(f"[TAZ] Max positions reached, skipping scan")
            return

        # Scan for opportunities
        opportunities = []

        # Scan stocks if enabled
        if self.trade_stocks:
            stock_opps = self.scanner.scan_stocks()
            opportunities.extend(stock_opps)

        # Scan crypto if enabled
        if self.trade_crypto:
            crypto_opps = self.scanner.scan_crypto()
            opportunities.extend(crypto_opps)

        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)

        # Get BUY signals
        buy_signals = [o for o in opportunities if o.signal == 'BUY']

        if buy_signals:
            print(f"[TAZ] Found {len(buy_signals)} BUY signals")

            # Take top opportunities up to available slots
            available_slots = self.max_positions - len(current_positions)

            for opp in buy_signals[:available_slots]:
                # Skip if already have position
                if opp.symbol in self.positions:
                    continue

                # Additional RL check if enabled
                if self.use_rl and self.rl_trainer:
                    try:
                        rl_action = self.rl_trainer.predict_action(opp.symbol, opp.asset_type)
                        if rl_action.get('action') not in ['SMALL_BUY', 'BIG_BUY']:
                            print(f"[TAZ] RL says HOLD/SELL for {opp.symbol}, skipping")
                            continue
                    except Exception as e:
                        # No RL model, proceed with scanner signal
                        pass

                # Execute buy
                self.execute_buy(
                    symbol=opp.symbol,
                    strategy=opp.strategy,
                    asset_type=opp.asset_type,
                    position_pct=self.max_position_pct if opp.score > 80 else self.max_position_pct * 0.7
                )

        self.last_scan_time = datetime.now()
        self.scanner.opportunities = opportunities
        self.scanner._save_results()

    def is_trading_hours(self, asset_type: str = 'stock') -> bool:
        """Check if it's trading hours."""
        if asset_type == 'crypto':
            return True  # Crypto trades 24/7

        et = ZoneInfo('America/New_York')
        now = datetime.now(et)

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours (9:30 AM - 4:00 PM ET)
        # Extended hours: 8:00 AM - 6:00 PM
        market_open = dt_time(8, 0)  # Extended hours
        market_close = dt_time(18, 0)

        return market_open <= now.time() <= market_close

    def run(self):
        """Main trading loop."""
        print("\n" + "="*60)
        print("[TAZ] === STARTING TAZMANIAN DEVIL TRADER ===")
        print("="*60)

        # Load previous state
        self._load_state()

        # Initial scan
        self.scan_for_opportunities()

        self.running = True
        self._stop_event.clear()

        try:
            while not self._stop_event.is_set():
                # Always check positions (crypto trades 24/7)
                self.check_positions()

                # Check if we should scan
                can_trade_stocks = self.trade_stocks and self.is_trading_hours('stock')
                can_trade_crypto = self.trade_crypto  # Crypto trades 24/7

                if can_trade_stocks or can_trade_crypto:
                    self.scan_for_opportunities()
                else:
                    # Print waiting message occasionally
                    if datetime.now().second < self.check_interval:
                        print(f"[TAZ] Waiting for trading hours...")

                # Wait
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n[TAZ] Stopping...")

        self.running = False
        self._save_state()
        self.print_summary()
        print("[TAZ] Stopped.")

    def stop(self):
        """Stop the trader."""
        self._stop_event.set()

    def _save_state(self):
        """Save state to file."""
        state = {
            'positions': {
                s: {
                    'symbol': p.symbol,
                    'asset_type': p.asset_type,
                    'shares': p.shares,
                    'entry_price': p.entry_price,
                    'entry_time': p.entry_time.isoformat(),
                    'strategy': p.strategy
                }
                for s, p in self.positions.items()
            },
            'stats': {
                'trades_executed': self.stats.trades_executed,
                'winning_trades': self.stats.winning_trades,
                'losing_trades': self.stats.losing_trades,
                'total_profit': self.stats.total_profit,
                'biggest_win': self.stats.biggest_win,
                'biggest_loss': self.stats.biggest_loss,
                'best_streak': self.stats.best_streak,
                'worst_streak': self.stats.worst_streak
            }
        }

        with open(self.data_dir / 'taz_state.json', 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from file."""
        state_file = self.data_dir / 'taz_state.json'
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Restore stats
            stats = state.get('stats', {})
            self.stats.trades_executed = stats.get('trades_executed', 0)
            self.stats.winning_trades = stats.get('winning_trades', 0)
            self.stats.losing_trades = stats.get('losing_trades', 0)
            self.stats.total_profit = stats.get('total_profit', 0.0)
            self.stats.biggest_win = stats.get('biggest_win', 0.0)
            self.stats.biggest_loss = stats.get('biggest_loss', 0.0)
            self.stats.best_streak = stats.get('best_streak', 0)
            self.stats.worst_streak = stats.get('worst_streak', 0)

            print(f"[TAZ] Loaded state: {self.stats.trades_executed} trades, ${self.stats.total_profit:.2f} profit")

        except Exception as e:
            print(f"[TAZ] Error loading state: {e}")

    def print_summary(self):
        """Print trading summary."""
        account = self.get_account()
        win_rate = (self.stats.winning_trades / max(self.stats.trades_executed, 1)) * 100

        print(f"""
================================================================
                    TAZ TRADING SUMMARY
================================================================
  Account Equity: ${account['equity']:,.2f}
  Total Profit: ${self.stats.total_profit:,.2f}

  Trades: {self.stats.trades_executed}
  Winners: {self.stats.winning_trades} | Losers: {self.stats.losing_trades}
  Win Rate: {win_rate:.1f}%

  Biggest Win: ${self.stats.biggest_win:,.2f}
  Biggest Loss: ${self.stats.biggest_loss:,.2f}
  Best Streak: {self.stats.best_streak} | Worst: {self.stats.worst_streak}
================================================================
        """)

    def get_status(self) -> dict:
        """Get current status."""
        account = self.get_account()
        positions = self.get_current_positions()

        return {
            'running': self.running,
            'paper': self.paper,
            'account': account,
            'positions': positions,
            'tracked_positions': {
                s: {
                    'symbol': p.symbol,
                    'asset_type': p.asset_type,
                    'entry_price': p.entry_price,
                    'strategy': p.strategy
                }
                for s, p in self.positions.items()
            },
            'stats': {
                'trades': self.stats.trades_executed,
                'win_rate': (self.stats.winning_trades / max(self.stats.trades_executed, 1)) * 100,
                'total_profit': self.stats.total_profit
            },
            'scanner': self.scanner.get_status() if hasattr(self.scanner, 'get_status') else {},
            'market_open': self.is_trading_hours('stock'),
            'crypto_enabled': self.trade_crypto,
            'stocks_enabled': self.trade_stocks
        }


def main():
    """Run Taz Trader."""
    import argparse

    parser = argparse.ArgumentParser(description='Taz Trader - Aggressive Growth')
    parser.add_argument('--capital', type=float, default=1000, help='Initial capital')
    parser.add_argument('--live', action='store_true', help='Live trading (default: paper)')
    parser.add_argument('--no-crypto', action='store_true', help='Disable crypto trading')
    parser.add_argument('--stocks', action='store_true', help='Enable stock trading (disabled by default to avoid PDT)')
    parser.add_argument('--no-rl', action='store_true', help='Disable RL agent')
    parser.add_argument('--position-size', type=float, default=0.15, help='Max position % (default: 15%)')
    parser.add_argument('--max-positions', type=int, default=3, help='Max concurrent positions')

    args = parser.parse_args()

    trader = TazTrader(
        initial_capital=args.capital,
        paper=not args.live,
        max_position_pct=args.position_size,
        max_positions=args.max_positions,
        trade_crypto=not args.no_crypto,
        trade_stocks=args.stocks,
        use_rl=not args.no_rl
    )

    trader.run()


if __name__ == '__main__':
    main()

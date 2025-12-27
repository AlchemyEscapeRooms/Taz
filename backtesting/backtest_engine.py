"""Comprehensive backtesting engine."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from utils.database import Database
from utils.trade_logger import TradeLogger, TradeReason, get_trade_logger
from config import config
from core.market_monitor import MarketMonitor, get_market_monitor, Prediction

logger = get_logger(__name__)


class Position:
    """Represents a trading position."""

    def __init__(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        entry_time: datetime,
        side: str = 'long'
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.side = side
        self.exit_price = None
        self.exit_time = None
        self.profit_loss = 0.0
        self.profit_loss_pct = 0.0

    def close(self, exit_price: float, exit_time: datetime):
        """Close the position."""
        self.exit_price = exit_price
        self.exit_time = exit_time

        if self.side == 'long':
            self.profit_loss = (exit_price - self.entry_price) * self.quantity
        else:  # short
            self.profit_loss = (self.entry_price - exit_price) * self.quantity

        self.profit_loss_pct = (self.profit_loss / (self.entry_price * self.quantity)) * 100

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def holding_period(self) -> float:
        """Holding period in days."""
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 86400
        return 0


class BacktestEngine:
    """Engine for backtesting trading strategies."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        enable_trade_logging: bool = True,
        max_position_size: float = None
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        # Max position size - defaults to config value (10%) if not specified
        self.max_position_size = max_position_size or config.get('trading.max_position_size', 0.1)

        self.positions: List[Position] = []
        self.open_positions: Dict[str, Position] = {}
        self.equity_curve = []
        self.trades = []

        # Trade logging
        self.enable_trade_logging = enable_trade_logging
        self.trade_logger = get_trade_logger() if enable_trade_logging else None

        # Current strategy info (set during run_backtest)
        self.current_strategy_name = ""
        self.current_strategy_params = {}

        # Map entry trades to their trade log IDs
        self.entry_trade_ids: Dict[str, str] = {}

        self.db = Database()

        # AI Learning System integration
        self.market_monitor = get_market_monitor()
        self.backtest_predictions: Dict[str, Prediction] = {}  # Track predictions by symbol

        # Walk-forward learning settings
        self.resolved_predictions_count = 0
        self.learn_every_n_trades = 10  # Trigger learning every N resolved trades
        self.enable_walkforward_learning = True  # Enable incremental learning during backtest

    def reset(self):
        """Reset the backtest state."""
        self.capital = self.initial_capital
        self.positions = []
        self.open_positions = {}
        self.equity_curve = []
        self.trades = []
        self.entry_trade_ids = {}
        self.backtest_predictions = {}
        self.resolved_predictions_count = 0

    def enter_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        side: str = 'long'
    ) -> bool:
        """Enter a new position."""

        # Calculate costs
        position_cost = quantity * price
        commission_cost = position_cost * self.commission
        slippage_cost = position_cost * self.slippage
        total_cost = position_cost + commission_cost + slippage_cost

        # Check if we have enough capital
        if total_cost > self.capital:
            logger.warning(f"Insufficient capital for {symbol}: need ${total_cost:.2f}, have ${self.capital:.2f}")
            return False

        # Adjust price for slippage
        if side == 'long':
            entry_price = price * (1 + self.slippage)
        else:
            entry_price = price * (1 - self.slippage)

        # Create position
        position = Position(symbol, quantity, entry_price, timestamp, side)
        self.positions.append(position)
        self.open_positions[symbol] = position

        # Update capital
        self.capital -= total_cost

        logger.debug(f"Entered {side} position: {symbol} x{quantity} @ ${entry_price:.2f}")
        return True

    def exit_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime
    ) -> bool:
        """Exit an open position."""

        if symbol not in self.open_positions:
            return False

        position = self.open_positions[symbol]

        # Adjust price for slippage
        if position.side == 'long':
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)

        # Close position
        position.close(exit_price, timestamp)

        # Calculate proceeds
        proceeds = position.quantity * exit_price
        commission_cost = proceeds * self.commission

        # Update capital
        self.capital += proceeds - commission_cost

        # Record trade
        self.trades.append({
            'symbol': symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'profit_loss': position.profit_loss,
            'profit_loss_pct': position.profit_loss_pct,
            'holding_period': position.holding_period
        })

        # Remove from open positions
        del self.open_positions[symbol]

        logger.debug(f"Exited position: {symbol} P&L: ${position.profit_loss:.2f} ({position.profit_loss_pct:.2f}%)")
        return True

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""

        # Cash
        portfolio_value = self.capital

        # Open positions
        for symbol, position in self.open_positions.items():
            if symbol in current_prices:
                current_value = position.quantity * current_prices[symbol]
                portfolio_value += current_value

        return portfolio_value

    def record_equity(self, timestamp: datetime, current_prices: Dict[str, float]):
        """Record current equity for equity curve."""

        portfolio_value = self.get_portfolio_value(current_prices)

        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': portfolio_value,
            'cash': self.capital,
            'positions_value': portfolio_value - self.capital,
            'return': (portfolio_value / self.initial_capital - 1) * 100
        })

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func,
        strategy_params: Dict[str, Any] = None,
        strategy_name: str = None
    ) -> Dict[str, Any]:
        """Run a backtest with a given strategy."""

        self.reset()

        # Store strategy info for trade logging
        self.current_strategy_name = strategy_name or strategy_func.__name__
        self.current_strategy_params = strategy_params or {}

        logger.info(f"Running backtest from {data.index[0]} to {data.index[-1]}")
        logger.info(f"Initial capital: ${self.initial_capital:.2f}")

        strategy_params = strategy_params or {}

        # Iterate through each time period
        for i in range(len(data)):
            current_date = data.index[i]
            current_data = data.iloc[:i+1]

            # Get current prices
            current_prices = {
                symbol: data['close'].iloc[i]
                for symbol in [data['symbol'].iloc[i]] if 'symbol' in data.columns
            }

            if not current_prices:
                current_prices = {'DEFAULT': data['close'].iloc[i]}

            # Call strategy function to get signals
            signals = strategy_func(current_data, self, strategy_params)

            # Process signals
            if signals:
                for signal in signals:
                    # Extract reason from signal if present
                    reason_data = signal.get('reason', {})

                    if signal['action'] == 'buy':
                        success = self.enter_position(
                            symbol=signal['symbol'],
                            quantity=signal['quantity'],
                            price=signal['price'],
                            timestamp=current_date,
                            side='long'
                        )

                        # Log the trade with reasoning
                        if success and self.trade_logger and reason_data:
                            self._log_entry_trade(
                                signal, current_date, current_data, reason_data
                            )

                        # Track prediction for AI Learning System
                        if success:
                            self._track_entry_prediction(signal, current_date, current_data, reason_data)

                    elif signal['action'] == 'sell':
                        # Get entry info before exiting
                        entry_trade_id = self.entry_trade_ids.get(signal['symbol'])
                        position = self.open_positions.get(signal['symbol'])

                        success = self.exit_position(
                            symbol=signal['symbol'],
                            price=signal['price'],
                            timestamp=current_date
                        )

                        # Log the exit trade with reasoning
                        if success and self.trade_logger and position:
                            self._log_exit_trade(
                                signal, current_date, position, entry_trade_id, reason_data
                            )

                        # Resolve prediction for AI Learning System
                        if success and position:
                            self._resolve_prediction(signal['symbol'], signal['price'], position)

            # Record equity
            self.record_equity(current_date, current_prices)

        # Close any remaining open positions at end of backtest
        final_date = data.index[-1]
        final_prices = current_prices

        for symbol in list(self.open_positions.keys()):
            if symbol in final_prices:
                position = self.open_positions.get(symbol)
                entry_trade_id = self.entry_trade_ids.get(symbol)

                self.exit_position(symbol, final_prices[symbol], final_date)

                # Log forced exit
                if self.trade_logger and position:
                    self._log_exit_trade(
                        {'symbol': symbol, 'price': final_prices[symbol]},
                        final_date,
                        position,
                        entry_trade_id,
                        {
                            'primary_signal': 'backtest_end',
                            'signal_value': 0,
                            'threshold': 0,
                            'direction': 'n/a',
                            'explanation': 'Position closed at end of backtest period.'
                        }
                    )

                # Resolve prediction for AI Learning System
                if position:
                    self._resolve_prediction(symbol, final_prices[symbol], position)

        # Run AI Learning System after backtest completes
        self._run_post_backtest_learning()

        # Calculate performance metrics
        results = self.calculate_performance()

        logger.info(f"Backtest complete. Final capital: ${self.capital:.2f}")
        logger.info(f"Total return: {results['total_return']:.2f}%")
        logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {results['max_drawdown']:.2f}%")
        logger.info(f"Win rate: {results['win_rate']:.2f}%")

        return results

    def _log_entry_trade(self, signal: Dict, timestamp, data: pd.DataFrame, reason_data: Dict):
        """Log an entry trade with full reasoning."""
        if not self.trade_logger:
            return

        # Build market snapshot
        market_snapshot = {
            'open': data['open'].iloc[-1] if 'open' in data.columns else 0,
            'high': data['high'].iloc[-1] if 'high' in data.columns else 0,
            'low': data['low'].iloc[-1] if 'low' in data.columns else 0,
            'close': data['close'].iloc[-1],
            'volume': data['volume'].iloc[-1] if 'volume' in data.columns else 0
        }

        # Build TradeReason
        reason = TradeReason(
            primary_signal=reason_data.get('primary_signal', 'unknown'),
            signal_value=reason_data.get('signal_value', 0),
            threshold=reason_data.get('threshold', 0),
            direction=reason_data.get('direction', 'unknown'),
            supporting_indicators=reason_data.get('supporting_indicators', {}),
            confirmations=reason_data.get('confirmations', []),
            explanation=reason_data.get('explanation', '')
        )

        # Convert pandas timestamp to datetime if needed
        if hasattr(timestamp, 'to_pydatetime'):
            timestamp = timestamp.to_pydatetime()

        entry = self.trade_logger.log_trade(
            symbol=signal['symbol'],
            action='BUY',
            quantity=signal['quantity'],
            price=signal['price'],
            strategy_name=self.current_strategy_name,
            strategy_params=self.current_strategy_params,
            reason=reason,
            mode='backtest',
            side='long',
            portfolio_value_before=self.get_portfolio_value({signal['symbol']: signal['price']}),
            market_snapshot=market_snapshot,
            timestamp=timestamp
        )

        # Store the trade ID for linking to exit
        self.entry_trade_ids[signal['symbol']] = entry.trade_id

    def _log_exit_trade(self, signal: Dict, timestamp, position, entry_trade_id: str, reason_data: Dict):
        """Log an exit trade with P&L and reasoning."""
        if not self.trade_logger:
            return

        # Calculate P&L
        exit_price = signal['price']
        realized_pnl = (exit_price - position.entry_price) * position.quantity
        realized_pnl_pct = ((exit_price / position.entry_price) - 1) * 100

        # Calculate holding period
        if hasattr(timestamp, 'to_pydatetime'):
            exit_dt = timestamp.to_pydatetime()
        else:
            exit_dt = timestamp

        if hasattr(position.entry_time, 'to_pydatetime'):
            entry_dt = position.entry_time.to_pydatetime()
        else:
            entry_dt = position.entry_time

        holding_period = (exit_dt - entry_dt).total_seconds() / 86400

        # Build TradeReason
        reason = TradeReason(
            primary_signal=reason_data.get('primary_signal', 'unknown'),
            signal_value=reason_data.get('signal_value', 0),
            threshold=reason_data.get('threshold', 0),
            direction=reason_data.get('direction', 'unknown'),
            supporting_indicators=reason_data.get('supporting_indicators', {}),
            confirmations=reason_data.get('confirmations', []),
            explanation=reason_data.get('explanation', '')
        )

        self.trade_logger.log_trade(
            symbol=signal['symbol'],
            action='SELL',
            quantity=position.quantity,
            price=exit_price,
            strategy_name=self.current_strategy_name,
            strategy_params=self.current_strategy_params,
            reason=reason,
            mode='backtest',
            side='long',
            portfolio_value_before=self.get_portfolio_value({signal['symbol']: exit_price}),
            entry_trade_id=entry_trade_id,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            holding_period_days=holding_period,
            timestamp=exit_dt
        )

        # Clear the entry trade ID
        if signal['symbol'] in self.entry_trade_ids:
            del self.entry_trade_ids[signal['symbol']]

    def _track_entry_prediction(self, signal: Dict, timestamp, data: pd.DataFrame, reason_data: Dict):
        """Track a prediction for AI Learning System when entering a position."""
        try:
            symbol = signal['symbol']
            entry_price = signal['price']

            # Extract signals from the reason_data
            signals_dict = {}
            if reason_data:
                signals_dict['primary_signal'] = reason_data.get('primary_signal', 'unknown')
                signals_dict['signal_value'] = reason_data.get('signal_value', 0)
                if 'supporting_indicators' in reason_data:
                    signals_dict.update(reason_data['supporting_indicators'])

            # Calculate technical indicators for the prediction
            if len(data) >= 20:
                # RSI
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, 0.0001)
                rsi = 100 - (100 / (1 + rs))
                signals_dict['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

                # MACD
                ema12 = data['close'].ewm(span=12).mean()
                ema26 = data['close'].ewm(span=26).mean()
                macd = ema12 - ema26
                signals_dict['macd'] = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0

                # Momentum
                momentum = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1) * 100
                signals_dict['momentum_5d'] = float(momentum) if not pd.isna(momentum) else 0

            # Create prediction
            prediction = Prediction(
                prediction_id=self.market_monitor.prediction_tracker.generate_prediction_id(),
                timestamp=timestamp if isinstance(timestamp, datetime) else timestamp.to_pydatetime(),
                symbol=symbol,
                predicted_direction='up',  # We're buying, so we predict up
                confidence=70.0,  # Default confidence for backtest entries
                predicted_change_pct=2.0,  # Estimated target
                timeframe='backtest',
                target_price=entry_price * 1.02,  # 2% target
                entry_price=entry_price,
                signals=signals_dict,
                reasoning=f"Backtest entry: {reason_data.get('primary_signal', 'strategy signal')}"
            )

            # Store prediction
            self.backtest_predictions[symbol] = prediction

            # Add to market monitor's prediction tracker
            self.market_monitor.prediction_tracker.add_prediction(prediction)

            logger.debug(f"Tracked prediction for {symbol} at ${entry_price:.2f}")

        except Exception as e:
            logger.warning(f"Error tracking prediction: {e}")

    def _resolve_prediction(self, symbol: str, exit_price: float, position):
        """Resolve a prediction when a position is closed."""
        try:
            if symbol not in self.backtest_predictions:
                return

            prediction = self.backtest_predictions[symbol]

            # Resolve the prediction in the tracker (it calculates actual_change internally)
            self.market_monitor.prediction_tracker.resolve_prediction(
                prediction.prediction_id,
                exit_price
            )

            # Calculate actual change for logging
            entry_price = position.entry_price
            actual_change_pct = ((exit_price / entry_price) - 1) * 100
            was_correct = actual_change_pct > 0

            # Remove from our tracking
            del self.backtest_predictions[symbol]

            # Increment counter and trigger walk-forward learning if enabled
            self.resolved_predictions_count += 1
            if self.enable_walkforward_learning and self.resolved_predictions_count % self.learn_every_n_trades == 0:
                self._run_incremental_learning()

            logger.debug(f"Resolved prediction for {symbol}: {'correct' if was_correct else 'wrong'} "
                        f"({actual_change_pct:+.2f}%)")

        except Exception as e:
            logger.warning(f"Error resolving prediction: {e}")

    def _run_incremental_learning(self):
        """Run incremental learning during backtest (walk-forward style)."""
        try:
            # Get recent accuracy stats (use shorter window for incremental learning)
            stats = self.market_monitor.prediction_tracker.get_accuracy_stats(days=30)

            if stats['total_predictions'] >= 5:  # Need minimum predictions
                # Run learning to adjust weights based on recent performance
                self.market_monitor._learn_from_history()

                logger.debug(f"Walk-forward learning: {self.resolved_predictions_count} trades, "
                           f"{stats['accuracy']:.1f}% accuracy, weights updated")

        except Exception as e:
            logger.debug(f"Incremental learning skipped: {e}")

    def _run_post_backtest_learning(self):
        """Run AI Learning System after backtest to update signal weights."""
        try:
            # Get accuracy stats
            stats = self.market_monitor.prediction_tracker.get_accuracy_stats(days=365)

            if stats['total_predictions'] > 0:
                # Report walk-forward learning stats
                learning_cycles = self.resolved_predictions_count // self.learn_every_n_trades
                if self.enable_walkforward_learning and learning_cycles > 0:
                    logger.info(f"AI Learning: Walk-forward mode - {learning_cycles} incremental updates during backtest")

                logger.info(f"AI Learning: {stats['total_predictions']} predictions tracked, "
                           f"{stats['accuracy']:.1f}% accuracy")

                # Final learning pass to ensure weights are current
                self.market_monitor._learn_from_history()

                logger.info("AI Learning: Final signal weights after walk-forward backtest:")
                for signal, weight in sorted(self.market_monitor.signal_weights.items(),
                                            key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"  {signal}: {weight:.3f}")

        except Exception as e:
            logger.warning(f"Error in post-backtest learning: {e}")

    def calculate_performance(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""

        if not self.trades:
            return self._empty_results()

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        # Basic metrics
        final_capital = self.capital
        total_return = (final_capital / self.initial_capital - 1) * 100

        # Trade statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        losing_trades = len(trades_df[trades_df['profit_loss'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit metrics
        total_profit = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum()
        total_loss = abs(trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum())
        net_profit = total_profit - total_loss
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

        # Average metrics
        avg_win = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['profit_loss'] < 0]['profit_loss'].mean() if losing_trades > 0 else 0
        avg_trade = trades_df['profit_loss'].mean()

        # Largest metrics
        largest_win = trades_df['profit_loss'].max() if total_trades > 0 else 0
        largest_loss = trades_df['profit_loss'].min() if total_trades > 0 else 0

        # Holding period
        avg_holding_period = trades_df['holding_period'].mean() if total_trades > 0 else 0

        # Risk-adjusted returns
        if len(equity_df) > 1:
            # Calculate daily returns from equity values (not from cumulative return column)
            equity_series = equity_df['equity']
            daily_returns = equity_series.pct_change().dropna()

            if len(daily_returns) > 0 and daily_returns.std() > 0:
                # Annualized Sharpe ratio (assuming 252 trading days)
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0

            # Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
            else:
                sortino_ratio = 0

            # Maximum drawdown from equity curve
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown = drawdown.min() * 100  # Convert to percentage
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0

        # Consecutive metrics
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for trade in self.trades:
            if trade['profit_loss'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'net_profit': net_profit,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_holding_period': avg_holding_period,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'equity_curve': equity_df.to_dict('records'),
            'trades': trades_df.to_dict('records')
        }

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': 0,
            'net_profit': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0
        }

    def compare_strategies(
        self,
        data: pd.DataFrame,
        strategies: Dict[str, tuple]
    ) -> pd.DataFrame:
        """Compare multiple strategies on the same data."""

        logger.info(f"Comparing {len(strategies)} strategies")

        results = []

        for strategy_name, (strategy_func, params) in strategies.items():
            logger.info(f"Testing strategy: {strategy_name}")

            perf = self.run_backtest(data, strategy_func, params)
            perf['strategy_name'] = strategy_name

            results.append(perf)

            # Store in database
            self.db.store_backtest_result(
                strategy_name=strategy_name,
                start_date=data.index[0],
                end_date=data.index[-1],
                initial_capital=perf['initial_capital'],
                final_capital=perf['final_capital'],
                total_return=perf['total_return'],
                sharpe_ratio=perf['sharpe_ratio'],
                max_drawdown=perf['max_drawdown'],
                win_rate=perf['win_rate'],
                total_trades=perf['total_trades'],
                parameters=str(params),
                results=str(perf)
            )

        # Convert to DataFrame for easy comparison
        comparison_df = pd.DataFrame(results)

        # Rank strategies
        comparison_df['sharpe_rank'] = comparison_df['sharpe_ratio'].rank(ascending=False)
        comparison_df['return_rank'] = comparison_df['total_return'].rank(ascending=False)
        comparison_df['drawdown_rank'] = comparison_df['max_drawdown'].abs().rank(ascending=True)

        comparison_df['overall_rank'] = (
            comparison_df['sharpe_rank'] +
            comparison_df['return_rank'] +
            comparison_df['drawdown_rank']
        ) / 3

        comparison_df = comparison_df.sort_values('overall_rank')

        logger.info("Strategy comparison complete")
        logger.info("\nTop 3 strategies:")
        header = f"{'Strategy':<20} {'Return %':>10} {'Sharpe':>10} {'Max DD %':>10}"
        logger.info(header)
        logger.info("-" * len(header))
        for _, row in comparison_df.head(3).iterrows():
            logger.info(f"{row['strategy_name']:<20} {row['total_return']:>10.2f} {row['sharpe_ratio']:>10.2f} {row['max_drawdown']:>10.2f}")

        return comparison_df

    def optimize_parameters(
        self,
        data: pd.DataFrame,
        strategy_func,
        param_grid: Dict[str, List[Any]],
        optimization_metric: str = 'sharpe_ratio'
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimize strategy parameters using grid search."""

        from itertools import product

        logger.info("Starting parameter optimization")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        results = []

        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))

            logger.debug(f"Testing parameters {i+1}/{len(param_combinations)}: {params}")

            perf = self.run_backtest(data, strategy_func, params)
            perf['params'] = params

            results.append(perf)

        # Find best parameters
        results_df = pd.DataFrame(results)

        if optimization_metric in results_df.columns:
            best_idx = results_df[optimization_metric].idxmax()
            best_params = results_df.loc[best_idx, 'params']
            best_performance = results_df.loc[best_idx].to_dict()
        else:
            best_params = {}
            best_performance = {}

        logger.info(f"Optimization complete. Best {optimization_metric}: {best_performance.get(optimization_metric, 'N/A')}")
        logger.info(f"Best parameters: {best_params}")

        return best_params, best_performance

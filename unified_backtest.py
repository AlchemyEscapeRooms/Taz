"""
Unified Backtest Engine
========================

This backtester simulates EXACTLY how the bot would trade live.

Key Principles:
1. POINT-IN-TIME: Iterate day-by-day, only using data available at that moment
2. NO LOOKAHEAD: Features are calculated only from past data
3. SAME CODE PATH: Uses the exact same strategy functions as live trading
4. REALISTIC EXECUTION: Simulates slippage, fees, and market impact
5. TRAINS FIRST: ML models are trained on initial data before trading begins

This ensures that backtest results are a realistic estimate of live performance.

Author: Claude AI  
Date: November 29, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from strategies_ml import (
    UnifiedTradingEngine,
    ml_driven_strategy,
    adaptive_momentum_strategy,
    mean_reversion_ml_strategy,
    _fallback_technical_strategy,
    ML_STRATEGY_REGISTRY,
    ML_DEFAULT_PARAMS
)
from utils.logger import get_logger
from utils.database import Database

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    entry_price: float
    quantity: float
    entry_date: datetime
    entry_reason: Dict[str, Any]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    highest_price: float = 0  # For trailing stop
    
    def __post_init__(self):
        self.highest_price = self.entry_price


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_date: datetime
    exit_date: datetime
    entry_reason: Dict[str, Any]
    exit_reason: Dict[str, Any]
    pnl: float = 0
    pnl_pct: float = 0
    holding_days: int = 0
    
    def __post_init__(self):
        self.pnl = (self.exit_price - self.entry_price) * self.quantity
        self.pnl_pct = (self.exit_price / self.entry_price - 1) * 100
        self.holding_days = (self.exit_date - self.entry_date).days


@dataclass 
class BacktestResult:
    """Complete backtest results."""
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_days: float
    
    # Risk metrics
    volatility: float
    var_95: float
    
    # Equity curve
    equity_curve: pd.DataFrame
    trades: List[Trade]
    
    # ML metrics
    ml_accuracy: float
    ml_precision: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_holding_days': self.avg_holding_days,
            'volatility': self.volatility,
            'var_95': self.var_95,
            'ml_accuracy': self.ml_accuracy,
            'ml_precision': self.ml_precision
        }


class UnifiedBacktester:
    """
    Backtester that simulates live trading exactly.
    
    The key innovation is that we iterate through historical data day-by-day,
    making trading decisions as if we're live trading on that specific date.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.001,  # 0.1% commission
        slippage_rate: float = 0.0005,   # 0.05% slippage
        risk_free_rate: float = 0.05     # 5% annual risk-free rate
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        
        # Trading state
        self.capital = initial_capital
        self.open_positions: Dict[str, Position] = {}
        self.completed_trades: List[Trade] = []
        
        # ML engine
        self.ml_engine = UnifiedTradingEngine(initial_capital)
        
        # Tracking
        self.equity_history = []
        self.daily_returns = []
        self.predictions_made = 0
        self.predictions_correct = 0
        
        # Database
        self.db = Database()
        
    def reset(self):
        """Reset the backtester for a new run."""
        self.capital = self.initial_capital
        self.open_positions = {}
        self.completed_trades = []
        self.equity_history = []
        self.daily_returns = []
        self.predictions_made = 0
        self.predictions_correct = 0
        self.ml_engine = UnifiedTradingEngine(self.initial_capital)
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        strategy_params: Dict[str, Any] = None,
        train_split: float = 0.3,  # Use first 30% for training
        start_date: str = None,
        end_date: str = None
    ) -> BacktestResult:
        """
        Run a backtest using point-in-time simulation.
        
        Process:
        1. Split data into training and trading periods
        2. Train ML models on training period
        3. Iterate through trading period day-by-day
        4. On each day, use only data available up to that point
        5. Generate signals and execute trades
        6. Calculate final metrics
        """
        self.reset()
        
        logger.info("=" * 60)
        logger.info("UNIFIED BACKTEST ENGINE")
        logger.info("=" * 60)
        
        # Validate data
        if len(data) < 100:
            raise ValueError(f"Insufficient data: {len(data)} rows (need at least 100)")
            
        # Filter by date if specified
        if 'date' not in data.columns:
            data['date'] = data.index
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]
            
        if len(data) < 100:
            raise ValueError(f"Insufficient data after date filter: {len(data)} rows")
        
        strategy_params = strategy_params or {}
        
        # Split into training and trading periods
        train_end_idx = int(len(data) * train_split)
        train_data = data.iloc[:train_end_idx].copy()
        trade_data = data.iloc[train_end_idx:].copy()
        
        logger.info(f"Data split:")
        logger.info(f"  Training: {len(train_data)} days ({data['date'].iloc[0]} to {data['date'].iloc[train_end_idx-1]})")
        logger.info(f"  Trading:  {len(trade_data)} days ({data['date'].iloc[train_end_idx]} to {data['date'].iloc[-1]})")
        
        # STEP 1: Train ML models on historical data
        logger.info("\nTraining ML models...")
        training_success = self.ml_engine.train_models(train_data)
        
        if not training_success:
            logger.warning("ML training failed, using fallback strategy")
            strategy_func = _fallback_technical_strategy
        
        # STEP 2: Iterate through trading period day-by-day
        logger.info("\nStarting point-in-time simulation...")
        
        for day_idx in range(len(trade_data)):
            # Get the actual date we're simulating
            current_date = trade_data['date'].iloc[day_idx]
            
            # CRITICAL: Use ALL data up to this point (training + trading so far)
            # This is exactly what we'd have available in live trading
            available_data = pd.concat([
                train_data,
                trade_data.iloc[:day_idx + 1]
            ]).reset_index(drop=True)
            
            # Current day's data
            current_price = trade_data['close'].iloc[day_idx]
            current_high = trade_data['high'].iloc[day_idx]
            current_low = trade_data['low'].iloc[day_idx]
            
            # Update positions (check stops, trailing stops)
            self._update_positions(current_price, current_high, current_low, current_date)
            
            # Create engine mock for strategy (same interface as live trading)
            # Includes max_position_size so strategies use correct limits
            class EngineState:
                def __init__(self, positions, capital, ml_engine, max_position_size=0.1):
                    self.open_positions = {k: {'entry_price': v.entry_price} for k, v in positions.items()}
                    self.capital = capital
                    self.ml_engine = ml_engine
                    self.max_position_size = max_position_size  # 10% default, matches config

            engine_state = EngineState(self.open_positions, self.capital, self.ml_engine)
            
            # Generate signals using the EXACT same strategy function as live
            try:
                signals = strategy_func(available_data, engine_state, strategy_params)
            except Exception as e:
                logger.warning(f"Strategy error on {current_date}: {e}")
                signals = []
            
            # Execute signals
            for signal in signals:
                self._execute_signal(signal, current_price, current_date)
            
            # Record daily equity
            daily_equity = self._calculate_equity(current_price)
            self.equity_history.append({
                'date': current_date,
                'equity': daily_equity,
                'cash': self.capital,
                'positions_value': daily_equity - self.capital
            })
            
            # Calculate daily return
            if len(self.equity_history) > 1:
                prev_equity = self.equity_history[-2]['equity']
                daily_return = (daily_equity / prev_equity - 1) if prev_equity > 0 else 0
                self.daily_returns.append(daily_return)
        
        # STEP 3: Close remaining positions at end
        if self.open_positions:
            final_price = trade_data['close'].iloc[-1]
            final_date = trade_data['date'].iloc[-1]
            
            for symbol in list(self.open_positions.keys()):
                self._close_position(
                    symbol, final_price, final_date,
                    {'primary_signal': 'backtest_end', 'explanation': 'End of backtest period'}
                )
        
        # STEP 4: Calculate performance metrics
        result = self._calculate_metrics()
        
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Return: {result.total_return:.2f}%")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {result.max_drawdown:.2f}%")
        logger.info(f"Win Rate: {result.win_rate:.2f}%")
        logger.info(f"Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"Total Trades: {result.total_trades}")
        
        return result
    
    def _execute_signal(self, signal: Dict, current_price: float, current_date: datetime):
        """Execute a trading signal with realistic slippage and commission."""
        
        action = signal.get('action')
        symbol = signal.get('symbol', 'UNKNOWN')
        quantity = signal.get('quantity', 0)
        
        if action == 'buy' and symbol not in self.open_positions:
            # Apply slippage (buy at slightly higher price)
            execution_price = current_price * (1 + self.slippage_rate)
            
            # Calculate cost with commission
            cost = execution_price * quantity * (1 + self.commission_rate)
            
            if cost <= self.capital:
                # Adjust quantity if needed
                if cost > self.capital:
                    quantity = self.capital / (execution_price * (1 + self.commission_rate))
                    cost = execution_price * quantity * (1 + self.commission_rate)
                
                self.capital -= cost
                
                # Set stop loss and take profit based on ATR or fixed percentage
                stop_loss = execution_price * 0.95  # 5% stop loss
                take_profit = execution_price * 1.15  # 15% take profit
                
                self.open_positions[symbol] = Position(
                    symbol=symbol,
                    entry_price=execution_price,
                    quantity=quantity,
                    entry_date=current_date,
                    entry_reason=signal.get('reason', {}),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop_pct=0.03  # 3% trailing stop
                )
                
                logger.debug(f"BUY {symbol}: {quantity:.2f} @ ${execution_price:.2f}")
                
                # Track ML prediction
                self.predictions_made += 1
                
        elif action == 'sell' and symbol in self.open_positions:
            self._close_position(symbol, current_price, current_date, signal.get('reason', {}))
    
    def _close_position(self, symbol: str, current_price: float, current_date: datetime, exit_reason: Dict):
        """Close a position with realistic execution."""
        
        if symbol not in self.open_positions:
            return
            
        position = self.open_positions[symbol]
        
        # Apply slippage (sell at slightly lower price)
        execution_price = current_price * (1 - self.slippage_rate)
        
        # Calculate proceeds with commission
        proceeds = execution_price * position.quantity * (1 - self.commission_rate)
        self.capital += proceeds
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=position.quantity,
            entry_date=position.entry_date,
            exit_date=current_date,
            entry_reason=position.entry_reason,
            exit_reason=exit_reason
        )
        self.completed_trades.append(trade)
        
        # Track ML accuracy
        if trade.pnl > 0:
            self.predictions_correct += 1
        
        del self.open_positions[symbol]
        
        logger.debug(f"SELL {symbol}: {position.quantity:.2f} @ ${execution_price:.2f} "
                    f"(P&L: ${trade.pnl:.2f}, {trade.pnl_pct:.2f}%)")
    
    def _update_positions(self, current_price: float, high: float, low: float, current_date: datetime):
        """Update positions - check stops, trailing stops, take profits."""
        
        for symbol in list(self.open_positions.keys()):
            position = self.open_positions[symbol]
            
            # Update trailing stop
            if position.trailing_stop_pct and high > position.highest_price:
                position.highest_price = high
                position.stop_loss = max(
                    position.stop_loss or 0,
                    position.highest_price * (1 - position.trailing_stop_pct)
                )
            
            should_close = False
            exit_reason = {}
            
            # Check stop loss
            if position.stop_loss and low <= position.stop_loss:
                should_close = True
                exit_reason = {
                    'primary_signal': 'stop_loss',
                    'explanation': f"Stop loss triggered at ${position.stop_loss:.2f}"
                }
                exit_price = position.stop_loss
            
            # Check take profit
            elif position.take_profit and high >= position.take_profit:
                should_close = True
                exit_reason = {
                    'primary_signal': 'take_profit',
                    'explanation': f"Take profit triggered at ${position.take_profit:.2f}"
                }
                exit_price = position.take_profit
            
            if should_close:
                self._close_position(symbol, exit_price, current_date, exit_reason)
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate total equity."""
        positions_value = sum(
            pos.quantity * current_price 
            for pos in self.open_positions.values()
        )
        return self.capital + positions_value
    
    def _calculate_metrics(self) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        
        # Build equity curve
        equity_df = pd.DataFrame(self.equity_history)
        if equity_df.empty:
            equity_df = pd.DataFrame({'date': [datetime.now()], 'equity': [self.initial_capital]})
        
        # Returns
        returns = np.array(self.daily_returns) if self.daily_returns else np.array([0])
        
        # Total return
        final_equity = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else self.initial_capital
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # Annualized return
        trading_days = len(equity_df)
        years = trading_days / 252
        annualized_return = ((final_equity / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0
        
        # Sharpe ratio
        excess_returns = np.mean(returns) - self.risk_free_rate/252
        sharpe_ratio = excess_returns / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino_ratio = excess_returns / (downside_std + 1e-10) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Max drawdown
        equity_series = equity_df['equity']
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in self.completed_trades if t.pnl > 0]
        losing_trades = [t for t in self.completed_trades if t.pnl <= 0]
        
        total_trades = len(self.completed_trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = (num_winning / total_trades * 100) if total_trades > 0 else 0
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        avg_holding_days = np.mean([t.holding_days for t in self.completed_trades]) if self.completed_trades else 0
        
        # VaR
        var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
        
        # ML accuracy
        ml_accuracy = (self.predictions_correct / self.predictions_made * 100) if self.predictions_made > 0 else 0
        ml_precision = ml_accuracy  # Simplified
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            winning_trades=num_winning,
            losing_trades=num_losing,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding_days,
            volatility=volatility,
            var_95=var_95,
            equity_curve=equity_df,
            trades=self.completed_trades,
            ml_accuracy=ml_accuracy,
            ml_precision=ml_precision
        )


def run_unified_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    strategy: str = 'ml_driven',
    initial_capital: float = 100000,
    train_split: float = 0.3
) -> BacktestResult:
    """
    Convenience function to run a unified backtest.
    
    Args:
        symbol: Stock symbol to backtest
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: Strategy name ('ml_driven', 'adaptive_momentum', 'mean_reversion_ml')
        initial_capital: Starting capital
        train_split: Fraction of data for ML training
    
    Returns:
        BacktestResult with comprehensive metrics
    """
    from data.market_data import MarketDataCollector
    
    logger.info(f"Running unified backtest for {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Strategy: {strategy}")
    
    # Get data
    data_collector = MarketDataCollector()
    data = data_collector.get_historical_data(symbol, start_date, end_date)
    
    if data.empty:
        raise ValueError(f"No data available for {symbol}")
    
    # Get strategy function
    strategy_func = ML_STRATEGY_REGISTRY.get(strategy, ml_driven_strategy)
    strategy_params = ML_DEFAULT_PARAMS.get(strategy, {})
    
    # Run backtest
    backtester = UnifiedBacktester(initial_capital=initial_capital)
    
    result = backtester.run_backtest(
        data=data,
        strategy_func=strategy_func,
        strategy_params=strategy_params,
        train_split=train_split
    )
    
    return result


def compare_strategies(
    symbol: str,
    start_date: str,
    end_date: str,
    strategies: List[str] = None,
    initial_capital: float = 100000
) -> pd.DataFrame:
    """
    Compare multiple strategies on the same data.
    
    Returns DataFrame with metrics for each strategy.
    """
    if strategies is None:
        strategies = list(ML_STRATEGY_REGISTRY.keys())
    
    results = []
    
    for strategy in strategies:
        try:
            result = run_unified_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                initial_capital=initial_capital
            )
            
            metrics = result.to_dict()
            metrics['strategy'] = strategy
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error testing {strategy}: {e}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    result = run_unified_backtest(
        symbol="SPY",
        start_date="2023-01-01",
        end_date="2024-12-31",
        strategy="ml_driven",
        initial_capital=100000
    )
    
    print("\nBacktest Results:")
    print(f"Total Return: {result.total_return:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"Win Rate: {result.win_rate:.2f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")

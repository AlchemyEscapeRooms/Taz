"""
Historical Learning Module
==========================

Fast-forward the learning process by running through historical data.

Instead of waiting months to collect predictions, we:
1. Load years of historical data
2. Walk through hour-by-hour (or day-by-day)
3. Make predictions using ONLY data available at that moment
4. Immediately verify against what actually happened
5. Update weights

This gives the bot a "head start" - it arrives at live trading already educated.

Author: Claude AI
Date: November 29, 2025
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import field
from datetime import datetime, timedelta, time
from dataclasses import dataclass
import json
from pathlib import Path

# Alpaca imports for historical data
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from learning_trader import (
    LearningTrader,
    PredictionDatabase,
    FeatureExtractor,
    StockLearningProfile,
    Prediction,
    PredictionHorizon,
    PredictionDirection
)
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class HistoricalTrainer:
    """
    Train the learning system on historical data.
    
    Simulates what would have happened if the bot was running for years.
    """
    
    def __init__(
        self,
        symbols: List[str],
        api_key: str = None,
        api_secret: str = None,
        db_path: str = "data/predictions_historical.db"
    ):
        self.symbols = symbols
        # Try multiple config paths for API keys, also check environment variables
        self.api_key = (api_key
                       or config.get('api_keys.alpaca.api_key')
                       or config.get('alpaca.api_key')
                       or os.environ.get('ALPACA_API_KEY'))
        self.api_secret = (api_secret
                          or config.get('api_keys.alpaca.secret_key')
                          or config.get('alpaca.api_secret')
                          or os.environ.get('ALPACA_SECRET_KEY'))
        
        # Initialize Alpaca client
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )
        
        # Database for historical training (separate from live)
        self.db = PredictionDatabase(db_path)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Stock profiles
        self.profiles: Dict[str, StockLearningProfile] = {}
        self._init_profiles()

        # Training stats
        self.total_predictions = 0
        self.correct_predictions = 0

        # Portfolio simulation
        self.initial_capital = 100000.0
        self.cash = self.initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {shares, entry_price, entry_time}
        self.trades: List[Dict] = []  # List of completed trades
        self.equity_curve: List[Dict] = []  # Track equity over time

        # Trading parameters (match live trading)
        self.max_position_pct = config.get('service.max_position_pct', 0.1)
        # Lower confidence threshold for backtest to generate more trades
        # Live uses 0.55, but backtest uses 0.51 to see more activity
        self.min_confidence = 0.51
        self.stop_loss_pct = config.get('trading.stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('service.default_take_profit_pct', 0.05)

        logger.info(f"HistoricalTrainer initialized for {len(symbols)} symbols")
    
    def _init_profiles(self):
        """Initialize learning profiles for all symbols."""
        for symbol in self.symbols:
            profile = self.db.get_stock_profile(symbol)
            if profile is None:
                profile = StockLearningProfile(
                    symbol=symbol,
                    feature_weights=self._get_default_weights()
                )
            self.profiles[symbol] = profile
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Default feature weights."""
        return {
            'rsi_14': 1.0,
            'rsi_7': 0.8,
            'macd_hist': 1.0,
            'macd_accelerating': 0.8,
            'trend_strength': 1.0,
            'momentum_score': 1.0,
            'bb_position': 1.0,
            'volume_ratio': 0.5,
            'price_change_5': 1.0,
            'price_change_10': 0.8,
            'price_vs_sma20': 1.0,
            'mean_reversion_score': 1.0,
            'volatility_ratio': 0.5,
            'range_position': 0.7,
            'consecutive_days': 0.6,
        }
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1Hour"
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: "1Hour" for hourly, "1Day" for daily
        """
        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min"),
            "15Min": TimeFrame(15, "Min"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day
        }
        
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf_map.get(timeframe, TimeFrame.Hour),
                start=datetime.strptime(start_date, "%Y-%m-%d"),
                end=datetime.strptime(end_date, "%Y-%m-%d")
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            if symbol not in bars.data:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'vwap': bar.vwap
            } for bar in bars.data[symbol]])
            
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_all_historical_data(
        self,
        start_date: str,
        end_date: str,
        timeframe: str = "1Hour"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols."""
        
        all_data = {}
        
        for symbol in self.symbols:
            df = self.fetch_historical_data(symbol, start_date, end_date, timeframe)
            if not df.empty:
                all_data[symbol] = df
        
        logger.info(f"Fetched data for {len(all_data)}/{len(self.symbols)} symbols")
        return all_data
    
    def _make_prediction_at_time(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_idx: int,
        horizon: PredictionHorizon
    ) -> Optional[Tuple[Prediction, float]]:
        """
        Make a prediction using only data available up to current_idx.
        
        Returns (prediction, actual_price_at_target) or None if not enough data.
        """
        # Need at least 50 bars of history
        if current_idx < 50:
            return None
        
        # Get data slice - ONLY data up to current point
        historical_data = df.iloc[:current_idx + 1].copy()
        
        current_bar = df.iloc[current_idx]
        current_price = current_bar['close']
        current_time = df.index[current_idx]
        
        # Calculate target index based on horizon
        if horizon == PredictionHorizon.ONE_HOUR:
            target_idx = current_idx + 1  # Next bar (1 hour later)
        elif horizon == PredictionHorizon.END_OF_DAY:
            # Find end of current day
            current_date = current_time.date()
            target_idx = current_idx
            for i in range(current_idx + 1, len(df)):
                if df.index[i].date() > current_date:
                    break
                target_idx = i
            if target_idx == current_idx:
                target_idx = current_idx + 1
        else:  # NEXT_DAY
            # Find end of next trading day
            current_date = current_time.date()
            found_next_day = False
            target_idx = current_idx
            for i in range(current_idx + 1, len(df)):
                bar_date = df.index[i].date()
                if bar_date > current_date:
                    found_next_day = True
                    current_date = bar_date
                if found_next_day and df.index[i].date() > current_date:
                    break
                target_idx = i
            if target_idx == current_idx:
                return None  # No future data available
        
        # Check if target is within bounds
        if target_idx >= len(df):
            return None
        
        target_price = df.iloc[target_idx]['close']
        target_time = df.index[target_idx]
        
        # Extract features from historical data only
        features = self.feature_extractor.extract_all_features(historical_data)
        if not features:
            if current_idx < 55:  # Log first few failures
                logger.info(f"No features extracted for {symbol} at idx {current_idx}, historical_data len={len(historical_data)}")
            return None
        
        # Make prediction using current weights
        profile = self.profiles[symbol]
        weights = profile.feature_weights
        
        # Calculate weighted score (matches live trading logic in background_service.py)
        score = 0
        total_weight = 0
        signals_used = {}

        for feature_name, weight in weights.items():
            if feature_name not in features:
                continue

            value = features[feature_name]
            signals_used[feature_name] = value

            # Normalize and weight (same logic as background_service._generate_prediction)
            if feature_name in ['rsi_14', 'rsi_7']:
                contribution = (50 - value) / 50 * weight
            elif feature_name == 'bb_position':
                contribution = (0.5 - value) * 2 * weight
            elif feature_name in ['trend_strength', 'momentum_score', 'macd_hist']:
                contribution = value / 100 * weight
            elif feature_name == 'mean_reversion_score':
                if features.get('rsi_14', 50) < 40:
                    contribution = value / 100 * weight
                else:
                    contribution = -value / 100 * weight
            elif 'price_change' in feature_name:
                contribution = value / 5 * weight
            else:
                contribution = value * weight * 0.01

            score += contribution
            total_weight += abs(weight)

        if total_weight > 0:
            score = score / total_weight
        score = max(-1, min(1, score))

        # Determine direction and confidence
        # Note: Using lower threshold for backtest since score calculations produce small values
        threshold = config.get('learning.direction_threshold', 0.1) * 0.3  # 0.03 instead of 0.1
        if score > threshold:
            direction = PredictionDirection.UP
            confidence = min(0.9, 0.5 + abs(score) * 0.4)
        elif score < -threshold:
            direction = PredictionDirection.DOWN
            confidence = min(0.9, 0.5 + abs(score) * 0.4)
        else:
            direction = PredictionDirection.FLAT
            confidence = 0.5

        # Debug: Log first few predictions to understand score distribution
        if profile.total_predictions < 2 and horizon == PredictionHorizon.ONE_HOUR:
            logger.info(f"DEBUG {symbol} Prediction #{profile.total_predictions}: score={score:.4f}, threshold={threshold}, direction={direction.value}, confidence={confidence:.2%}")
            logger.info(f"  Features used: rsi_14={signals_used.get('rsi_14', 'N/A'):.1f}, trend_strength={signals_used.get('trend_strength', 'N/A')}, macd_hist={signals_used.get('macd_hist', 'N/A'):.4f}")
            logger.info(f"  total_weight={total_weight:.2f}, raw_score_before_div={score * total_weight:.4f}")

        # Adjust confidence by historical accuracy (matches live)
        # But only apply adjustment after enough predictions to be meaningful
        if profile.total_predictions >= 50:
            if horizon == PredictionHorizon.ONE_HOUR:
                accuracy = profile.accuracy_1h
            elif horizon == PredictionHorizon.END_OF_DAY:
                accuracy = profile.accuracy_eod
            else:
                accuracy = profile.accuracy_next_day

            confidence = confidence * (0.5 + accuracy * 0.5)

        # Debug: Log confidence for 1-hour UP predictions (potential trades)
        if horizon == PredictionHorizon.ONE_HOUR and direction == PredictionDirection.UP:
            if self.total_predictions < 10:  # Only log first few
                logger.info(f"Trade candidate: {symbol} UP, confidence={confidence:.2%}, score={score:.3f}, threshold={self.min_confidence:.2%}")
        
        # Expected move
        atr_pct = features.get('atr_pct', 1.5)
        if horizon == PredictionHorizon.ONE_HOUR:
            expected_move = atr_pct * 0.2
        elif horizon == PredictionHorizon.END_OF_DAY:
            expected_move = atr_pct * 0.5
        else:
            expected_move = atr_pct * 1.0
        
        if direction == PredictionDirection.DOWN:
            expected_move = -expected_move
        elif direction == PredictionDirection.FLAT:
            expected_move = 0
        
        pred_id = f"hist_{symbol}_{horizon.value}_{current_time.strftime('%Y%m%d%H%M')}"
        
        prediction = Prediction(
            id=pred_id,
            symbol=symbol,
            horizon=horizon,
            prediction_time=current_time.to_pydatetime(),
            price_at_prediction=current_price,
            predicted_direction=direction,
            predicted_change_pct=expected_move,
            confidence=confidence,
            signals_used=signals_used,
            target_time=target_time.to_pydatetime()
        )
        
        return (prediction, target_price)
    
    def _verify_and_learn(
        self,
        prediction: Prediction,
        actual_price: float
    ) -> bool:
        """
        Verify prediction and update learning weights.
        
        Returns True if prediction was correct.
        """
        actual_change_pct = (actual_price / prediction.price_at_prediction - 1) * 100
        
        # Determine actual direction
        flat_threshold = 0.3
        if abs(actual_change_pct) < flat_threshold:
            actual_direction = PredictionDirection.FLAT
        elif actual_change_pct > 0:
            actual_direction = PredictionDirection.UP
        else:
            actual_direction = PredictionDirection.DOWN
        
        was_correct = (prediction.predicted_direction == actual_direction)
        
        # Update prediction record
        prediction.actual_price = actual_price
        prediction.actual_change_pct = actual_change_pct
        prediction.was_correct = was_correct
        prediction.verified_at = prediction.target_time
        
        # Update profile
        profile = self.profiles[prediction.symbol]
        profile.total_predictions += 1
        if was_correct:
            profile.total_correct += 1
        
        # Update horizon-specific accuracy
        alpha = 0.05  # Slower learning rate for historical (more data)
        
        if prediction.horizon == PredictionHorizon.ONE_HOUR:
            profile.predictions_1h += 1
            profile.accuracy_1h = profile.accuracy_1h * (1 - alpha) + (1 if was_correct else 0) * alpha
        elif prediction.horizon == PredictionHorizon.END_OF_DAY:
            profile.predictions_eod += 1
            profile.accuracy_eod = profile.accuracy_eod * (1 - alpha) + (1 if was_correct else 0) * alpha
        else:
            profile.predictions_next_day += 1
            profile.accuracy_next_day = profile.accuracy_next_day * (1 - alpha) + (1 if was_correct else 0) * alpha
        
        # Update feature weights (use config values to match live)
        weight_adj = config.get('learning.weight_adjustment', 0.02)
        min_weight = config.get('learning.min_weight', 0.5)
        max_weight = config.get('learning.max_weight', 2.0)
        weight_adjustment = weight_adj if was_correct else -weight_adj

        for feature_name in prediction.signals_used:
            if feature_name in profile.feature_weights:
                profile.feature_weights[feature_name] *= (1 + weight_adjustment)
                profile.feature_weights[feature_name] = max(min_weight, min(max_weight, profile.feature_weights[feature_name]))
        
        # Track stats
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        
        return was_correct

    # =========================================================================
    # PORTFOLIO SIMULATION - Trade execution matching live trading logic
    # =========================================================================

    def _get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value (cash + positions)."""
        total = self.cash
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total += position['shares'] * current_prices[symbol]
        return total

    def _execute_buy(self, symbol: str, price: float, timestamp, confidence: float) -> Optional[Dict]:
        """Execute a buy order in simulation."""
        # Calculate position size (same as live trading)
        portfolio_value = self._get_portfolio_value({symbol: price})
        max_position_value = portfolio_value * self.max_position_pct
        shares = int(max_position_value / price)

        if shares <= 0:
            return None

        cost = shares * price

        if cost > self.cash:
            # Adjust shares to available cash
            shares = int(self.cash / price)
            if shares <= 0:
                return None
            cost = shares * price

        # Execute buy
        self.cash -= cost
        self.positions[symbol] = {
            'shares': shares,
            'entry_price': price,
            'entry_time': timestamp,
            'confidence': confidence,
            'stop_loss': price * (1 - self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct)
        }

        trade = {
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'value': cost,
            'timestamp': timestamp,
            'confidence': confidence
        }

        logger.debug(f"BUY {shares} {symbol} @ ${price:.2f} (confidence: {confidence:.1%})")
        return trade

    def _execute_sell(self, symbol: str, price: float, timestamp, reason: str) -> Optional[Dict]:
        """Execute a sell order in simulation."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        shares = position['shares']
        entry_price = position['entry_price']

        # Calculate P&L
        proceeds = shares * price
        cost_basis = shares * entry_price
        pnl = proceeds - cost_basis
        pnl_pct = (price / entry_price - 1) * 100

        # Execute sell
        self.cash += proceeds
        del self.positions[symbol]

        trade = {
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'value': proceeds,
            'timestamp': timestamp,
            'entry_price': entry_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_time': str(timestamp - position['entry_time']) if hasattr(timestamp, '__sub__') else 'N/A'
        }

        self.trades.append(trade)
        logger.debug(f"SELL {shares} {symbol} @ ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}")
        return trade

    def _check_stop_loss_take_profit(self, symbol: str, current_price: float, timestamp) -> Optional[Dict]:
        """Check if stop loss or take profit should trigger."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        if current_price <= position['stop_loss']:
            return self._execute_sell(symbol, current_price, timestamp, 'stop_loss')
        elif current_price >= position['take_profit']:
            return self._execute_sell(symbol, current_price, timestamp, 'take_profit')

        return None

    def _evaluate_trade_signal(
        self,
        prediction: 'Prediction',
        current_price: float,
        timestamp
    ) -> Optional[Dict]:
        """
        Evaluate if a prediction should trigger a trade.
        Matches the logic in background_service._evaluate_trade_signal
        """
        symbol = prediction.symbol

        # Only trade on 1-hour predictions (most actionable)
        if prediction.horizon != PredictionHorizon.ONE_HOUR:
            return None

        # Check if we already have a position
        if symbol in self.positions:
            # Exit on DOWN signal with lower confidence threshold than entry
            # Entry needs 51%, exit only needs the prediction to be DOWN (not FLAT)
            if prediction.predicted_direction == PredictionDirection.DOWN and prediction.confidence >= 0.50:
                logger.info(f"SELL SIGNAL: {symbol} DOWN prediction (conf={prediction.confidence:.1%}), exiting position")
                return self._execute_sell(symbol, current_price, timestamp, 'signal_reversal')
            return None

        # For new entries, require minimum confidence
        if prediction.confidence < self.min_confidence:
            return None

        # Entry signal - only buy on UP predictions
        if prediction.predicted_direction == PredictionDirection.UP:
            logger.info(f"BUY SIGNAL: {symbol} UP prediction, confidence={prediction.confidence:.1%}, entering position @ ${current_price:.2f}")
            return self._execute_buy(symbol, current_price, timestamp, prediction.confidence)

        return None

    def train_on_historical(
        self,
        start_date: str,
        end_date: str,
        prediction_interval: int = 1,  # Make prediction every N bars
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the learning system on historical data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            prediction_interval: How often to make predictions (in bars)
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        logger.info("=" * 60)
        logger.info("HISTORICAL TRAINING STARTED")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("=" * 60)
        
        # Fetch all historical data
        all_data = self.fetch_all_historical_data(start_date, end_date, "1Hour")

        if not all_data:
            logger.error("No historical data available")
            return {'error': 'No historical data available', 'total_predictions': 0}

        for symbol, df in all_data.items():
            logger.info(f"Data for {symbol}: {len(df)} bars, from {df.index[0] if len(df) > 0 else 'N/A'} to {df.index[-1] if len(df) > 0 else 'N/A'}")
        
        # Reset stats
        self.total_predictions = 0
        self.correct_predictions = 0

        # Reset portfolio for fresh simulation
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # Process each symbol
        for symbol, df in all_data.items():
            if verbose:
                logger.info(f"\nTraining on {symbol} ({len(df)} bars)...")

            symbol_predictions = 0
            symbol_correct = 0

            # Walk through data
            logger.info(f"Walking through {symbol}: indices 50 to {len(df) - 10}, step={prediction_interval}")
            predictions_made = 0
            up_predictions = 0
            down_predictions = 0
            flat_predictions = 0

            for idx in range(50, len(df) - 10, prediction_interval):  # Need 50 bars history, leave 10 for verification

                current_bar = df.iloc[idx]
                current_price = current_bar['close']
                current_time = df.index[idx]

                # Check stop loss / take profit on existing positions
                self._check_stop_loss_take_profit(symbol, current_price, current_time)

                # Track equity at each bar
                self.equity_curve.append({
                    'timestamp': current_time,
                    'equity': self._get_portfolio_value({symbol: current_price}),
                    'cash': self.cash,
                    'positions_value': self._get_portfolio_value({symbol: current_price}) - self.cash
                })

                # Make predictions for all horizons
                for horizon in PredictionHorizon:
                    result = self._make_prediction_at_time(symbol, df, idx, horizon)

                    if result is None:
                        continue

                    prediction, actual_price = result
                    was_correct = self._verify_and_learn(prediction, actual_price)

                    symbol_predictions += 1
                    predictions_made += 1
                    if was_correct:
                        symbol_correct += 1

                    # Track direction distribution
                    if prediction.predicted_direction == PredictionDirection.UP:
                        up_predictions += 1
                    elif prediction.predicted_direction == PredictionDirection.DOWN:
                        down_predictions += 1
                    else:
                        flat_predictions += 1

                    # Execute trade based on prediction (simulated, not real!)
                    self._evaluate_trade_signal(prediction, current_price, current_time)

            # Close any remaining positions at end of backtest
            if symbol in self.positions:
                final_price = df.iloc[-1]['close']
                final_time = df.index[-1]
                self._execute_sell(symbol, final_price, final_time, 'end_of_backtest')

            # Save updated profile
            self.db.update_stock_profile(self.profiles[symbol])

            if verbose and symbol_predictions > 0:
                accuracy = symbol_correct / symbol_predictions
                logger.info(f"  {symbol}: {symbol_predictions} predictions, {accuracy:.1%} accuracy")
                logger.info(f"  {symbol}: Directions - UP: {up_predictions}, DOWN: {down_predictions}, FLAT: {flat_predictions}")
                logger.info(f"  {symbol}: {len(self.trades)} trades executed so far")
        
        # Calculate final stats
        overall_accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0

        # Calculate trading P&L stats
        final_equity = self.cash  # All positions should be closed
        total_pnl = final_equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        total_gains = sum(t.get('pnl', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf') if total_gains > 0 else 0

        avg_win = total_gains / len(winning_trades) if winning_trades else 0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0

        # Max drawdown from equity curve
        max_drawdown = 0
        peak_equity = self.initial_capital
        for point in self.equity_curve:
            if point['equity'] > peak_equity:
                peak_equity = point['equity']
            drawdown = (peak_equity - point['equity']) / peak_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        results = {
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'overall_accuracy': overall_accuracy,
            'symbols_trained': len(all_data),
            # Trading results
            'trading': {
                'initial_capital': self.initial_capital,
                'final_equity': final_equity,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'trades': self.trades[-20:]  # Last 20 trades for display
            },
            'profiles': {symbol: {
                'accuracy_1h': p.accuracy_1h,
                'accuracy_eod': p.accuracy_eod,
                'accuracy_next_day': p.accuracy_next_day,
                'total_predictions': p.total_predictions,
                'overall_accuracy': p.overall_accuracy,
                'top_features': sorted(
                    p.feature_weights.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            } for symbol, p in self.profiles.items()}
        }

        logger.info("\n" + "=" * 60)
        logger.info("HISTORICAL BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total predictions: {self.total_predictions}")
        logger.info(f"Correct: {self.correct_predictions}")
        logger.info(f"Overall accuracy: {overall_accuracy:.1%}")
        logger.info("-" * 40)
        logger.info("TRADING RESULTS (Simulated)")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Equity: ${final_equity:,.2f}")
        logger.info(f"Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.1%}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.1%}")

        return results
    
    def export_learned_weights(self, output_path: str = "data/learned_weights.json"):
        """Export learned weights to JSON for use in live trading."""
        
        weights_data = {
            'trained_at': datetime.now().isoformat(),
            'total_predictions': self.total_predictions,
            'overall_accuracy': self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0,
            'symbols': {}
        }
        
        for symbol, profile in self.profiles.items():
            weights_data['symbols'][symbol] = {
                'feature_weights': profile.feature_weights,
                'accuracy_1h': profile.accuracy_1h,
                'accuracy_eod': profile.accuracy_eod,
                'accuracy_next_day': profile.accuracy_next_day,
                'total_predictions': profile.total_predictions,
                'min_confidence_to_trade': max(0.6, 1.0 - profile.overall_accuracy)
            }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        logger.info(f"Exported learned weights to {output_path}")
        return output_path
    
    def get_training_report(self) -> str:
        """Generate a human-readable training report."""
        
        report = []
        report.append("=" * 60)
        report.append("HISTORICAL TRAINING REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Total Predictions: {self.total_predictions}")
        report.append(f"Correct: {self.correct_predictions}")
        report.append(f"Overall Accuracy: {self.correct_predictions / self.total_predictions:.1%}" if self.total_predictions > 0 else "N/A")
        report.append("")
        report.append("-" * 60)
        report.append("PER-STOCK RESULTS")
        report.append("-" * 60)
        
        for symbol, profile in sorted(self.profiles.items(), key=lambda x: x[1].overall_accuracy, reverse=True):
            report.append("")
            report.append(f"{symbol}:")
            report.append(f"  Predictions: {profile.total_predictions}")
            report.append(f"  Overall Accuracy: {profile.overall_accuracy:.1%}")
            report.append(f"  1-Hour Accuracy: {profile.accuracy_1h:.1%}")
            report.append(f"  EOD Accuracy: {profile.accuracy_eod:.1%}")
            report.append(f"  Next-Day Accuracy: {profile.accuracy_next_day:.1%}")
            report.append(f"  Min Confidence to Trade: {max(0.6, 1.0 - profile.overall_accuracy):.1%}")
            
            # Top features
            top_features = sorted(profile.feature_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            report.append(f"  Top Features:")
            for feat, weight in top_features:
                report.append(f"    - {feat}: {weight:.2f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


class HistoricalLearningTrader(LearningTrader):
    """
    Learning trader that can be initialized with historically-trained weights.
    """
    
    def __init__(
        self,
        symbols: List[str],
        weights_path: str = None,
        **kwargs
    ):
        super().__init__(symbols, **kwargs)
        
        if weights_path:
            self.load_historical_weights(weights_path)
    
    def load_historical_weights(self, weights_path: str):
        """Load pre-trained weights from historical training."""
        
        try:
            with open(weights_path, 'r') as f:
                weights_data = json.load(f)
            
            logger.info(f"Loading weights trained at {weights_data.get('trained_at')}")
            logger.info(f"Historical accuracy: {weights_data.get('overall_accuracy', 0):.1%}")
            
            for symbol in self.symbols:
                if symbol in weights_data.get('symbols', {}):
                    symbol_data = weights_data['symbols'][symbol]
                    
                    profile = self.profiles.get(symbol, StockLearningProfile(symbol=symbol))
                    profile.feature_weights = symbol_data.get('feature_weights', profile.feature_weights)
                    profile.accuracy_1h = symbol_data.get('accuracy_1h', 0.5)
                    profile.accuracy_eod = symbol_data.get('accuracy_eod', 0.5)
                    profile.accuracy_next_day = symbol_data.get('accuracy_next_day', 0.5)
                    profile.total_predictions = symbol_data.get('total_predictions', 0)
                    profile.min_confidence_to_trade = symbol_data.get('min_confidence_to_trade', 0.7)
                    
                    # Calculate total_correct from accuracy
                    profile.total_correct = int(profile.total_predictions * profile.overall_accuracy)
                    
                    self.profiles[symbol] = profile
                    self.db.update_stock_profile(profile)
                    
                    logger.info(f"  {symbol}: Loaded {profile.total_predictions} predictions, {profile.overall_accuracy:.1%} accuracy")
            
            logger.info("Historical weights loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading weights: {e}")


def train_and_export(
    symbols: List[str],
    start_date: str = "2022-01-01",
    end_date: str = "2024-11-01",
    output_path: str = "data/learned_weights.json"
) -> str:
    """
    Convenience function to train on historical data and export weights.
    
    Args:
        symbols: List of stock symbols
        start_date: Training start date
        end_date: Training end date
        output_path: Where to save learned weights
        
    Returns:
        Path to exported weights file
    """
    trainer = HistoricalTrainer(symbols)
    
    # Train
    results = trainer.train_on_historical(start_date, end_date)
    
    # Print report
    print(trainer.get_training_report())
    
    # Export
    return trainer.export_learned_weights(output_path)


def main():
    """Example usage."""
    
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "META", "NVDA", "AMD", "NFLX", "SPY"
    ]
    
    # Train on 2 years of historical data
    weights_path = train_and_export(
        symbols=symbols,
        start_date="2022-01-01",
        end_date="2024-11-01"
    )
    
    print(f"\nWeights exported to: {weights_path}")
    
    # Now create a live trader with pre-trained weights
    live_trader = HistoricalLearningTrader(
        symbols=symbols,
        weights_path=weights_path,
        learning_mode=True  # Still learn, but with a head start
    )
    
    live_trader.print_status()
    
    # Ready to run live
    # live_trader.run_loop(interval_minutes=60)


if __name__ == "__main__":
    main()

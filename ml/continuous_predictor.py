"""
Continuous Prediction Engine

The bot makes predictions NON-STOP, 24/7, on EVERYTHING.

WHY THIS IS CRITICAL:
====================
- Most bots only "think" when trading → Learn slowly
- This bot "thinks" constantly → Learns exponentially faster
- Makes predictions every minute on every stock being monitored
- Every prediction gets scored → Bot learns from THOUSANDS of data points daily

EXAMPLE OF CONTINUOUS LEARNING:
===============================

9:30 AM: Predicts AAPL movement for next 5min, 15min, 1hour, 1day
9:35 AM: Scores 5min prediction, makes new set of predictions
9:45 AM: Scores 15min prediction, makes new set
10:30 AM: Scores 1hour prediction, makes new set
4:00 PM: Scores 1day prediction, makes new set

Result: For 1 stock, bot makes ~100 predictions per day
        For 100 stocks being monitored = 10,000 predictions/day
        365 days = 3.65 MILLION learning events per year!

This is how the bot gets SMART - it's not waiting for trades, it's learning
from every single market movement.

KEY FEATURES:
=============
1. Multi-timeframe predictions (1min, 5min, 15min, 1hour, 4hour, 1day, 1week)
2. Multiple prediction types (price, direction, volatility, volume)
3. Automatic scoring when timeframe elapses
4. Feeds predictions back to RL engine for learning
5. Maintains prediction database for analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time
import json
import logging
from pathlib import Path

from ml.reinforcement_learner import (
    MarketState,
    Prediction,
    ReinforcementLearningEngine
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionTarget:
    """What to predict and when to score it"""
    timeframe: str  # '1min', '5min', '15min', '1h', '4h', '1d', '1w'
    prediction_types: List[str]  # ['price_move', 'direction', 'volatility']
    enabled: bool = True

    def get_timedelta(self) -> timedelta:
        """Convert timeframe to timedelta"""
        mapping = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1w': timedelta(weeks=1)
        }
        return mapping.get(self.timeframe, timedelta(hours=1))


@dataclass
class PredictionBatch:
    """Batch of predictions for a symbol at a specific time"""
    symbol: str
    timestamp: datetime
    predictions: List[Prediction] = field(default_factory=list)


class ContinuousPredictionEngine:
    """
    Makes predictions continuously, non-stop, on all monitored stocks.

    This is the "always thinking" component that makes the bot smart.
    """

    def __init__(
        self,
        rl_engine: ReinforcementLearningEngine,
        prediction_targets: Optional[List[PredictionTarget]] = None,
        max_concurrent_symbols: int = 100,
        prediction_interval_seconds: int = 60,  # Make predictions every minute
        auto_score: bool = True,
        save_dir: str = "ml/predictions"
    ):
        self.rl_engine = rl_engine
        self.max_concurrent_symbols = max_concurrent_symbols
        self.prediction_interval_seconds = prediction_interval_seconds
        self.auto_score = auto_score

        # Default prediction targets
        if prediction_targets is None:
            self.prediction_targets = [
                PredictionTarget('5min', ['direction', 'price_move']),
                PredictionTarget('15min', ['direction', 'price_move']),
                PredictionTarget('1h', ['direction', 'price_move', 'volatility']),
                PredictionTarget('4h', ['direction', 'price_move']),
                PredictionTarget('1d', ['direction', 'price_move', 'volatility']),
                PredictionTarget('1w', ['direction', 'price_move'])
            ]
        else:
            self.prediction_targets = prediction_targets

        # Predictions waiting to be scored
        self.pending_predictions: Dict[str, List[Prediction]] = defaultdict(list)

        # Prediction history for analysis
        self.prediction_batches: List[PredictionBatch] = []

        # Symbols being monitored
        self.monitored_symbols: List[str] = []

        # Current market data (symbol → latest state)
        self.latest_states: Dict[str, MarketState] = {}

        # Price history for scoring (symbol → timestamp → price)
        self.price_history: Dict[str, Dict[datetime, float]] = defaultdict(dict)

        # Statistics
        self.total_predictions_made = 0
        self.total_predictions_scored = 0
        self.predictions_by_timeframe: Dict[str, int] = defaultdict(int)
        self.accuracy_by_timeframe: Dict[str, List[float]] = defaultdict(list)

        # Threading
        self.is_running = False
        self.prediction_thread: Optional[threading.Thread] = None
        self.scoring_thread: Optional[threading.Thread] = None

        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Continuous Prediction Engine initialized: "
                   f"{len(self.prediction_targets)} timeframes, "
                   f"interval={prediction_interval_seconds}s")

    def add_symbol(self, symbol: str):
        """Add symbol to monitoring list"""
        if symbol not in self.monitored_symbols and len(self.monitored_symbols) < self.max_concurrent_symbols:
            self.monitored_symbols.append(symbol)
            logger.info(f"Added {symbol} to prediction monitoring ({len(self.monitored_symbols)} total)")

    def remove_symbol(self, symbol: str):
        """Remove symbol from monitoring"""
        if symbol in self.monitored_symbols:
            self.monitored_symbols.remove(symbol)
            logger.info(f"Removed {symbol} from prediction monitoring")

    def update_market_state(self, symbol: str, state: MarketState):
        """
        Update latest market state for a symbol.

        This is called continuously as new market data arrives.
        """
        self.latest_states[symbol] = state
        self.price_history[symbol][state.timestamp] = state.price

    def make_predictions_for_symbol(
        self,
        symbol: str,
        state: Optional[MarketState] = None
    ) -> PredictionBatch:
        """
        Make all configured predictions for a symbol.

        This is called every prediction_interval_seconds for each symbol.
        """

        if state is None:
            state = self.latest_states.get(symbol)
            if state is None:
                logger.warning(f"No market state available for {symbol}")
                return PredictionBatch(symbol, datetime.now(), [])

        batch = PredictionBatch(
            symbol=symbol,
            timestamp=datetime.now(),
            predictions=[]
        )

        # Make predictions for each timeframe
        for target in self.prediction_targets:
            if not target.enabled:
                continue

            for pred_type in target.prediction_types:
                prediction = self.rl_engine.make_prediction(
                    symbol=symbol,
                    market_state=state,
                    prediction_type=pred_type,
                    timeframe=target.timeframe
                )

                batch.predictions.append(prediction)
                self.pending_predictions[symbol].append(prediction)

                self.total_predictions_made += 1
                self.predictions_by_timeframe[target.timeframe] += 1

        logger.info(
            f"Made {len(batch.predictions)} predictions for {symbol} "
            f"(total pending: {len(self.pending_predictions[symbol])})"
        )

        self.prediction_batches.append(batch)

        return batch

    def score_predictions(self, symbol: str):
        """
        Score all predictions for a symbol that are ready to be scored.

        A prediction is ready when its timeframe has elapsed.
        """

        if symbol not in self.pending_predictions:
            return

        current_time = datetime.now()
        pending = self.pending_predictions[symbol]
        still_pending = []
        scored_count = 0

        for prediction in pending:
            # Get target timeframe
            target_delta = self._get_timeframe_delta(prediction.timeframe)
            score_time = prediction.timestamp + target_delta

            # Ready to score?
            if current_time >= score_time:
                # Get actual price movement
                actual_price = self._get_price_at_time(symbol, score_time)

                if actual_price is not None:
                    entry_price = self.price_history[symbol].get(prediction.timestamp)

                    if entry_price is not None:
                        # Calculate actual movement
                        actual_move = (actual_price - entry_price) / entry_price

                        # Determine direction
                        if actual_move > 0.001:
                            actual_direction = 'up'
                        elif actual_move < -0.001:
                            actual_direction = 'down'
                        else:
                            actual_direction = 'neutral'

                        # Score it
                        reward = self.rl_engine.score_prediction(
                            prediction.prediction_id,
                            actual_value=actual_move,
                            actual_direction=actual_direction
                        )

                        # Track accuracy
                        if prediction.prediction_accuracy is not None:
                            self.accuracy_by_timeframe[prediction.timeframe].append(
                                prediction.prediction_accuracy
                            )

                        self.total_predictions_scored += 1
                        scored_count += 1

                        logger.debug(
                            f"Scored {symbol} {prediction.timeframe} prediction: "
                            f"predicted={prediction.predicted_value:+.2%}, "
                            f"actual={actual_move:+.2%}, reward={reward:+.3f}"
                        )
                    else:
                        still_pending.append(prediction)
                else:
                    still_pending.append(prediction)
            else:
                still_pending.append(prediction)

        # Update pending list
        self.pending_predictions[symbol] = still_pending

        if scored_count > 0:
            logger.info(f"Scored {scored_count} predictions for {symbol} "
                       f"({len(still_pending)} still pending)")

    def start_continuous_predictions(self):
        """
        Start making predictions continuously in background.

        This runs 24/7, non-stop!
        """

        if self.is_running:
            logger.warning("Continuous predictions already running")
            return

        self.is_running = True

        # Start prediction thread
        self.prediction_thread = threading.Thread(
            target=self._prediction_loop,
            name="ContinuousPredictionThread",
            daemon=True
        )
        self.prediction_thread.start()

        # Start scoring thread
        if self.auto_score:
            self.scoring_thread = threading.Thread(
                target=self._scoring_loop,
                name="PredictionScoringThread",
                daemon=True
            )
            self.scoring_thread.start()

        logger.info("Continuous prediction engine started! "
                   "Bot is now thinking 24/7...")

    def stop_continuous_predictions(self):
        """Stop continuous predictions"""
        self.is_running = False
        logger.info("Stopping continuous prediction engine...")

    def _prediction_loop(self):
        """Main loop that makes predictions continuously"""

        logger.info(f"Prediction loop started: {len(self.monitored_symbols)} symbols, "
                   f"{self.prediction_interval_seconds}s interval")

        while self.is_running:
            try:
                # Make predictions for all monitored symbols
                for symbol in self.monitored_symbols:
                    if symbol in self.latest_states:
                        self.make_predictions_for_symbol(symbol)

                # Wait before next round
                time.sleep(self.prediction_interval_seconds)

            except Exception as e:
                logger.error(f"Error in prediction loop: {e}", exc_info=True)
                time.sleep(self.prediction_interval_seconds)

        logger.info("Prediction loop stopped")

    def _scoring_loop(self):
        """Main loop that scores predictions continuously"""

        logger.info("Scoring loop started")

        while self.is_running:
            try:
                # Score predictions for all symbols
                for symbol in self.monitored_symbols:
                    self.score_predictions(symbol)

                # Check every 30 seconds
                time.sleep(30)

            except Exception as e:
                logger.error(f"Error in scoring loop: {e}", exc_info=True)
                time.sleep(30)

        logger.info("Scoring loop stopped")

    def _get_timeframe_delta(self, timeframe: str) -> timedelta:
        """Convert timeframe string to timedelta"""
        for target in self.prediction_targets:
            if target.timeframe == timeframe:
                return target.get_timedelta()
        return timedelta(hours=1)

    def _get_price_at_time(self, symbol: str, target_time: datetime) -> Optional[float]:
        """Get price at specific time (or closest available)"""

        if symbol not in self.price_history:
            return None

        history = self.price_history[symbol]

        # Find closest timestamp
        closest_time = min(
            history.keys(),
            key=lambda t: abs((t - target_time).total_seconds()),
            default=None
        )

        if closest_time:
            # Only use if within 5 minutes of target
            if abs((closest_time - target_time).total_seconds()) < 300:
                return history[closest_time]

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction engine statistics"""

        # Calculate accuracy by timeframe
        accuracy_stats = {}
        for timeframe, accuracies in self.accuracy_by_timeframe.items():
            if accuracies:
                accuracy_stats[timeframe] = {
                    'count': len(accuracies),
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies)
                }

        # Pending by symbol
        pending_by_symbol = {
            symbol: len(preds)
            for symbol, preds in self.pending_predictions.items()
        }

        return {
            'is_running': self.is_running,
            'monitored_symbols': len(self.monitored_symbols),
            'symbols': self.monitored_symbols,
            'total_predictions_made': self.total_predictions_made,
            'total_predictions_scored': self.total_predictions_scored,
            'pending_predictions': sum(pending_by_symbol.values()),
            'pending_by_symbol': pending_by_symbol,
            'predictions_by_timeframe': dict(self.predictions_by_timeframe),
            'accuracy_by_timeframe': accuracy_stats,
            'prediction_batches_stored': len(self.prediction_batches)
        }

    def get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get prediction performance for specific symbol"""

        # Get all scored predictions for this symbol
        scored = [
            pred for batch in self.prediction_batches
            if batch.symbol == symbol
            for pred in batch.predictions
            if pred.is_resolved()
        ]

        if not scored:
            return {'symbol': symbol, 'predictions': 0}

        # Calculate metrics
        total = len(scored)
        correct = sum(1 for p in scored if p.predicted_direction == p.actual_direction)
        avg_accuracy = np.mean([p.prediction_accuracy for p in scored if p.prediction_accuracy])
        avg_reward = np.mean([p.reward for p in scored if p.reward])

        # By timeframe
        by_timeframe = defaultdict(list)
        for pred in scored:
            by_timeframe[pred.timeframe].append(pred)

        timeframe_stats = {}
        for tf, preds in by_timeframe.items():
            tf_correct = sum(1 for p in preds if p.predicted_direction == p.actual_direction)
            timeframe_stats[tf] = {
                'total': len(preds),
                'correct': tf_correct,
                'accuracy': tf_correct / len(preds),
                'avg_reward': np.mean([p.reward for p in preds if p.reward])
            }

        return {
            'symbol': symbol,
            'total_predictions': total,
            'correct_predictions': correct,
            'direction_accuracy': correct / total,
            'avg_prediction_accuracy': avg_accuracy,
            'avg_reward': avg_reward,
            'by_timeframe': timeframe_stats
        }

    def save_predictions(self, filename: Optional[str] = None):
        """Save prediction history"""

        if filename is None:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.save_dir / filename

        data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'prediction_batches': [
                {
                    'symbol': batch.symbol,
                    'timestamp': batch.timestamp.isoformat(),
                    'predictions': [
                        {
                            'id': p.prediction_id,
                            'type': p.prediction_type,
                            'timeframe': p.timeframe,
                            'predicted_value': p.predicted_value,
                            'predicted_direction': p.predicted_direction,
                            'confidence': p.confidence,
                            'actual_value': p.actual_value,
                            'actual_direction': p.actual_direction,
                            'accuracy': p.prediction_accuracy,
                            'reward': p.reward
                        }
                        for p in batch.predictions[-100:]  # Last 100 per batch
                    ]
                }
                for batch in self.prediction_batches[-100:]  # Last 100 batches
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved predictions to {filepath}")

        return str(filepath)


if __name__ == '__main__':
    # Test continuous predictor
    print("=" * 80)
    print("CONTINUOUS PREDICTION ENGINE TEST")
    print("=" * 80)

    from ml.reinforcement_learner import MarketState

    # Create RL engine
    rl_engine = ReinforcementLearningEngine()

    # Create prediction engine
    pred_engine = ContinuousPredictionEngine(
        rl_engine=rl_engine,
        prediction_interval_seconds=5,  # Fast for testing
        auto_score=True
    )

    # Add symbols to monitor
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    for symbol in symbols:
        pred_engine.add_symbol(symbol)

    # Create sample market states
    for symbol in symbols:
        state = MarketState(
            timestamp=datetime.now(),
            symbol=symbol,
            price=150.0 + np.random.randn() * 5,
            price_change_1h=np.random.randn() * 0.01,
            price_change_1d=np.random.randn() * 0.02,
            price_change_1w=np.random.randn() * 0.05,
            rsi=50 + np.random.randn() * 20,
            macd=np.random.randn(),
            sma_20=148.0,
            sma_50=145.0,
            sma_200=140.0,
            volatility=0.02,
            volume_ratio=1.0 + np.random.randn() * 0.3,
            regime='trending_up' if np.random.random() > 0.5 else 'ranging',
            regime_confidence=0.7 + np.random.random() * 0.3,
            position_pnl=0.0,
            portfolio_cash_pct=0.3,
            portfolio_risk=0.05,
            recent_win_rate=0.6,
            spy_change=np.random.randn() * 0.01,
            market_breadth=0.5 + np.random.random() * 0.3,
            vix=15 + np.random.randn() * 5
        )
        pred_engine.update_market_state(symbol, state)

    print("\n1. Making predictions for all symbols:")
    print("-" * 80)
    for symbol in symbols:
        batch = pred_engine.make_predictions_for_symbol(symbol)
        print(f"{symbol}: {len(batch.predictions)} predictions made")

    print("\n2. Statistics:")
    print("-" * 80)
    stats = pred_engine.get_statistics()
    print(f"Total predictions: {stats['total_predictions_made']}")
    print(f"Pending predictions: {stats['pending_predictions']}")
    print(f"By timeframe: {stats['predictions_by_timeframe']}")

    print("\n3. Starting continuous predictions (5 seconds):")
    print("-" * 80)
    pred_engine.start_continuous_predictions()
    print("Running...")
    time.sleep(5)
    pred_engine.stop_continuous_predictions()
    time.sleep(1)  # Let threads finish

    print("\n4. Final statistics:")
    print("-" * 80)
    final_stats = pred_engine.get_statistics()
    print(f"Total predictions: {final_stats['total_predictions_made']}")
    print(f"Scored predictions: {final_stats['total_predictions_scored']}")

    print("\n" + "=" * 80)
    print("Continuous prediction engine tests complete!")

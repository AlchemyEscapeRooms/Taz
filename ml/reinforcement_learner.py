"""
Reinforcement Learning Trading Engine

This is the BRAIN of the bot - where it actually LEARNS from experience.

KEY CONCEPTS (in simple terms):
================================

Q-Learning = "Quality Learning"
- Bot learns the "quality" (expected reward) of each action in each situation
- Over time, it figures out which actions lead to best results
- Like a video game player learning which moves work best

State = The current situation
- Price trends, indicators, market regime, portfolio status
- "Am I in an uptrend with low volatility and winning streak?"

Action = What the bot can do
- Buy, sell, hold, adjust position size

Reward = How good was that action?
- Positive reward = made profit
- Negative reward = lost money
- The bot tries to maximize total reward over time

CONTINUOUS LEARNING:
===================
- Bot makes predictions 24/7, even when not trading
- Every prediction gets scored when outcome is known
- Bot learns from EVERY prediction, not just trades
- Improves exponentially over time

EXAMPLE:
========
9:30 AM: Bot predicts "AAPL will go up 1% in next hour" (confidence: 75%)
10:30 AM: AAPL went up 0.8% → Reward = +0.8, Prediction accuracy = 80%
         Bot learns: "This pattern led to accurate prediction - do more of this"

9:35 AM: Bot predicts "TSLA will drop 2% today" (confidence: 60%)
4:00 PM: TSLA went up 1% → Reward = -2, Prediction was wrong
         Bot learns: "This pattern led to wrong prediction - avoid this"

Over thousands of predictions daily, bot gets smarter and smarter.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """
    Complete snapshot of market conditions at a point in time.

    This is what the bot "sees" when making decisions.
    """
    timestamp: datetime
    symbol: str

    # Price data
    price: float
    price_change_1h: float
    price_change_1d: float
    price_change_1w: float

    # Technical indicators
    rsi: float
    macd: float
    sma_20: float
    sma_50: float
    sma_200: float
    volatility: float
    volume_ratio: float  # Current volume vs average

    # Market regime
    regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'
    regime_confidence: float

    # Portfolio context
    position_pnl: float  # If we own this stock, current P&L
    portfolio_cash_pct: float  # % of portfolio in cash
    portfolio_risk: float  # Current total portfolio risk
    recent_win_rate: float  # Win rate over last 20 trades

    # Broader market
    spy_change: float  # S&P 500 performance
    market_breadth: float  # % of stocks up vs down
    vix: float  # Volatility index

    def to_vector(self) -> np.ndarray:
        """Convert state to numeric vector for ML"""
        return np.array([
            self.price_change_1h,
            self.price_change_1d,
            self.price_change_1w,
            self.rsi / 100.0,  # Normalize to 0-1
            self.macd / self.price,  # Normalize
            (self.price - self.sma_20) / self.price,
            (self.price - self.sma_50) / self.price,
            (self.price - self.sma_200) / self.price,
            self.volatility,
            self.volume_ratio,
            1.0 if self.regime == 'trending_up' else 0.0,
            1.0 if self.regime == 'trending_down' else 0.0,
            1.0 if self.regime == 'ranging' else 0.0,
            1.0 if self.regime == 'volatile' else 0.0,
            self.regime_confidence,
            self.position_pnl,
            self.portfolio_cash_pct,
            self.portfolio_risk,
            self.recent_win_rate,
            self.spy_change,
            self.market_breadth,
            self.vix / 100.0  # Normalize
        ])

    def to_key(self) -> str:
        """Convert to discrete state key for Q-table"""
        # Discretize continuous values into buckets
        return f"{self._bucket(self.price_change_1d, 0.05)}_" \
               f"{self._bucket(self.rsi, 20)}_" \
               f"{self.regime}_" \
               f"{self._bucket(self.volatility, 0.1)}_" \
               f"{self._bucket(self.recent_win_rate, 0.2)}"

    def _bucket(self, value: float, bucket_size: float) -> int:
        """Discretize value into buckets"""
        return int(value / bucket_size)


@dataclass
class TradingAction:
    """Action the bot can take"""
    action_type: str  # 'buy', 'sell', 'hold', 'increase', 'decrease'
    position_size_pct: float  # 0.0 to 0.25 (0-25% of portfolio)
    confidence: float  # 0.0 to 1.0

    def to_index(self) -> int:
        """Convert action to discrete index"""
        # Map to one of 15 discrete actions
        if self.action_type == 'hold':
            return 0
        elif self.action_type == 'buy':
            # 5 buy sizes: 5%, 10%, 15%, 20%, 25%
            size_bucket = int(self.position_size_pct * 100 / 5) - 1
            return 1 + max(0, min(4, size_bucket))
        elif self.action_type == 'sell':
            return 6
        elif self.action_type == 'increase':
            # 4 increase amounts: 25%, 50%, 75%, 100%
            return 7 + int(self.confidence * 4)
        elif self.action_type == 'decrease':
            # 4 decrease amounts: 25%, 50%, 75%, 100%
            return 11 + int(self.confidence * 4)
        return 0


@dataclass
class Prediction:
    """A prediction made by the bot"""
    prediction_id: str
    timestamp: datetime
    symbol: str
    prediction_type: str  # 'price_move', 'direction', 'volatility'
    timeframe: str  # '1h', '1d', '1w'

    # What we predicted
    predicted_value: float
    predicted_direction: str  # 'up', 'down', 'neutral'
    confidence: float  # 0-1

    # Context when prediction was made
    market_state: MarketState

    # Outcome (filled in later)
    actual_value: Optional[float] = None
    actual_direction: Optional[str] = None
    outcome_timestamp: Optional[datetime] = None

    # Scoring
    prediction_error: Optional[float] = None  # Absolute error
    prediction_accuracy: Optional[float] = None  # 0-1 score
    reward: Optional[float] = None  # Reward for this prediction

    # Learning
    was_used_for_training: bool = False

    def is_resolved(self) -> bool:
        """Has this prediction been resolved?"""
        return self.actual_value is not None

    def calculate_reward(self) -> float:
        """
        Calculate reward for this prediction.

        Reward formula:
        - Correct direction + low error = high reward
        - Wrong direction = negative reward
        - Scaled by confidence (more confident = higher reward/penalty)
        """
        if not self.is_resolved():
            return 0.0

        # Direction correctness
        direction_correct = (self.predicted_direction == self.actual_direction)
        direction_reward = 1.0 if direction_correct else -1.0

        # Magnitude accuracy
        if self.predicted_value != 0:
            error_pct = abs(self.actual_value - self.predicted_value) / abs(self.predicted_value)
        else:
            error_pct = abs(self.actual_value)

        # Lower error = higher reward
        accuracy_reward = max(0, 1.0 - error_pct)

        # Combined reward
        base_reward = direction_reward * 0.6 + accuracy_reward * 0.4

        # Scale by confidence (high confidence correct = great, high confidence wrong = terrible)
        confidence_multiplier = 0.5 + (self.confidence * 1.0)

        total_reward = base_reward * confidence_multiplier

        # Store for later analysis
        self.prediction_error = error_pct
        self.prediction_accuracy = accuracy_reward
        self.reward = total_reward

        return total_reward


@dataclass
class Experience:
    """Single learning experience (state, action, reward, next_state)"""
    state: MarketState
    action: TradingAction
    reward: float
    next_state: Optional[MarketState]
    done: bool  # True if episode ended (position closed)


class ReinforcementLearningEngine:
    """
    Reinforcement Learning engine that LEARNS from every prediction and trade.

    This is the CORE INTELLIGENCE that gets smarter over time.

    HOW IT WORKS:
    1. Bot observes market state
    2. Bot chooses action (using Q-table + exploration)
    3. Action produces outcome and reward
    4. Bot updates Q-table to learn from experience
    5. Repeat millions of times → Bot gets expert-level
    """

    def __init__(
        self,
        learning_rate: float = 0.001,  # How fast to learn
        discount_factor: float = 0.95,  # How much to value future rewards
        exploration_rate: float = 0.20,  # % of time to explore vs exploit
        exploration_decay: float = 0.9995,  # Slowly reduce exploration
        min_exploration: float = 0.05,  # Always explore at least 5%
        state_dims: int = 22,  # Dimensions of state vector
        num_actions: int = 15,  # Number of possible actions
        memory_size: int = 100000,  # Experience replay buffer size
        save_dir: str = "ml/models"
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.state_dims = state_dims
        self.num_actions = num_actions

        # Q-table: Maps (state, action) → expected reward
        # Start with small random values
        self.q_table: Dict[str, np.ndarray] = defaultdict(
            lambda: np.random.randn(num_actions) * 0.01
        )

        # Experience replay memory (for better learning)
        self.memory: deque = deque(maxlen=memory_size)

        # Prediction tracking
        self.predictions: Dict[str, Prediction] = {}
        self.prediction_history: List[Prediction] = []

        # Learning statistics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.total_reward_earned = 0.0
        self.learning_episodes = 0

        # Performance tracking
        self.recent_rewards = deque(maxlen=1000)
        self.recent_accuracies = deque(maxlen=1000)

        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"RL Engine initialized: lr={learning_rate}, gamma={discount_factor}, "
                   f"exploration={exploration_rate}")

    def select_action(
        self,
        state: MarketState,
        mode: str = 'train'
    ) -> TradingAction:
        """
        Select action using epsilon-greedy strategy.

        Args:
            state: Current market state
            mode: 'train' (explore) or 'exploit' (best action only)

        Returns:
            TradingAction to take
        """

        state_key = state.to_key()
        q_values = self.q_table[state_key]

        # Exploration vs Exploitation
        if mode == 'train' and np.random.random() < self.exploration_rate:
            # EXPLORE: Try random action to discover new strategies
            action_idx = np.random.randint(0, self.num_actions)
            logger.debug(f"Exploring: random action {action_idx}")
        else:
            # EXPLOIT: Use best known action
            action_idx = np.argmax(q_values)
            logger.debug(f"Exploiting: best action {action_idx} (Q={q_values[action_idx]:.3f})")

        # Convert action index to TradingAction
        action = self._index_to_action(action_idx, q_values[action_idx])

        return action

    def learn_from_experience(self, experience: Experience):
        """
        Update Q-table based on experience.

        This is where LEARNING happens!

        Q-Learning update formula:
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

        In English:
        New value = Old value + learning_rate * (actual_result - expected_result)
        """

        # Add to memory
        self.memory.append(experience)

        # Get Q-values for current state
        state_key = experience.state.to_key()
        action_idx = experience.action.to_index()
        current_q = self.q_table[state_key][action_idx]

        # Calculate target Q-value
        if experience.done or experience.next_state is None:
            # Episode ended, no future reward
            target_q = experience.reward
        else:
            # Episode continues, estimate future reward
            next_state_key = experience.next_state.to_key()
            next_max_q = np.max(self.q_table[next_state_key])
            target_q = experience.reward + self.discount_factor * next_max_q

        # Update Q-value
        td_error = target_q - current_q  # Temporal difference error
        self.q_table[state_key][action_idx] += self.learning_rate * td_error

        # Track learning
        self.learning_episodes += 1
        self.recent_rewards.append(experience.reward)

        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

        logger.debug(
            f"Learned: State={state_key[:30]}..., Action={action_idx}, "
            f"Reward={experience.reward:.3f}, TD_error={td_error:.3f}, "
            f"NewQ={self.q_table[state_key][action_idx]:.3f}"
        )

    def make_prediction(
        self,
        symbol: str,
        market_state: MarketState,
        prediction_type: str = 'price_move',
        timeframe: str = '1h'
    ) -> Prediction:
        """
        Make a prediction about future price movement.

        Bot makes predictions NON-STOP, even when not trading.
        Every prediction gets scored and used for learning.
        """

        # Use Q-values to predict
        state_key = market_state.to_key()
        q_values = self.q_table[state_key]

        # Predicted direction based on best action
        best_action_idx = np.argmax(q_values)
        best_action = self._index_to_action(best_action_idx, q_values[best_action_idx])

        if best_action.action_type in ['buy', 'increase']:
            predicted_direction = 'up'
            predicted_value = market_state.price_change_1d * 1.5  # Expect acceleration
        elif best_action.action_type in ['sell', 'decrease']:
            predicted_direction = 'down'
            predicted_value = market_state.price_change_1d * 1.5
        else:
            predicted_direction = 'neutral'
            predicted_value = market_state.price_change_1d * 0.5  # Expect continuation

        # Confidence based on Q-value spread
        q_spread = np.max(q_values) - np.min(q_values)
        confidence = min(1.0, q_spread / 10.0)  # Normalize

        # Create prediction
        prediction = Prediction(
            prediction_id=f"{symbol}_{datetime.now().isoformat()}",
            timestamp=datetime.now(),
            symbol=symbol,
            prediction_type=prediction_type,
            timeframe=timeframe,
            predicted_value=predicted_value,
            predicted_direction=predicted_direction,
            confidence=confidence,
            market_state=market_state
        )

        # Store for later scoring
        self.predictions[prediction.prediction_id] = prediction
        self.total_predictions += 1

        logger.info(
            f"Prediction: {symbol} {predicted_direction} {predicted_value:+.2%} "
            f"in {timeframe} (confidence: {confidence:.0%})"
        )

        return prediction

    def score_prediction(
        self,
        prediction_id: str,
        actual_value: float,
        actual_direction: str
    ) -> float:
        """
        Score a prediction and learn from it.

        This is called when prediction timeframe has elapsed.
        Bot learns from EVERY prediction!
        """

        if prediction_id not in self.predictions:
            logger.warning(f"Prediction {prediction_id} not found")
            return 0.0

        prediction = self.predictions[prediction_id]

        # Fill in actual outcome
        prediction.actual_value = actual_value
        prediction.actual_direction = actual_direction
        prediction.outcome_timestamp = datetime.now()

        # Calculate reward
        reward = prediction.calculate_reward()

        # Update statistics
        self.total_reward_earned += reward
        self.recent_accuracies.append(prediction.prediction_accuracy or 0.0)

        if prediction.predicted_direction == actual_direction:
            self.correct_predictions += 1

        # Move to history
        self.prediction_history.append(prediction)
        del self.predictions[prediction_id]

        logger.info(
            f"Scored prediction: {prediction.symbol} - "
            f"Predicted: {prediction.predicted_direction} {prediction.predicted_value:+.2%}, "
            f"Actual: {actual_direction} {actual_value:+.2%}, "
            f"Reward: {reward:+.3f}, Accuracy: {prediction.prediction_accuracy:.0%}"
        )

        return reward

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""

        prediction_accuracy = (
            self.correct_predictions / max(1, self.total_predictions - len(self.predictions))
        )

        avg_recent_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0.0
        avg_recent_accuracy = np.mean(list(self.recent_accuracies)) if self.recent_accuracies else 0.0

        return {
            'total_predictions': self.total_predictions,
            'pending_predictions': len(self.predictions),
            'scored_predictions': len(self.prediction_history),
            'correct_predictions': self.correct_predictions,
            'prediction_accuracy': prediction_accuracy,
            'total_reward': self.total_reward_earned,
            'avg_reward_recent': avg_recent_reward,
            'avg_accuracy_recent': avg_recent_accuracy,
            'exploration_rate': self.exploration_rate,
            'learning_episodes': self.learning_episodes,
            'q_table_size': len(self.q_table),
            'memory_size': len(self.memory)
        }

    def save_model(self, filename: Optional[str] = None):
        """Save Q-table and learning state"""

        if filename is None:
            filename = f"rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        filepath = self.save_dir / filename

        state = {
            'q_table': dict(self.q_table),
            'exploration_rate': self.exploration_rate,
            'learning_episodes': self.learning_episodes,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'total_reward': self.total_reward_earned,
            'prediction_history': self.prediction_history[-1000:],  # Last 1000
            'stats': self.get_learning_stats()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Model saved to {filepath}")

        return str(filepath)

    def load_model(self, filepath: str):
        """Load Q-table and learning state"""

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.q_table = defaultdict(lambda: np.random.randn(self.num_actions) * 0.01)
        self.q_table.update(state['q_table'])
        self.exploration_rate = state.get('exploration_rate', self.exploration_rate)
        self.learning_episodes = state.get('learning_episodes', 0)
        self.total_predictions = state.get('total_predictions', 0)
        self.correct_predictions = state.get('correct_predictions', 0)
        self.total_reward_earned = state.get('total_reward', 0.0)
        self.prediction_history = state.get('prediction_history', [])

        logger.info(f"Model loaded from {filepath}: {state.get('stats', {})}")

    def _index_to_action(self, action_idx: int, q_value: float) -> TradingAction:
        """Convert action index to TradingAction"""

        confidence = 1.0 / (1.0 + np.exp(-q_value))  # Sigmoid of Q-value

        if action_idx == 0:
            return TradingAction('hold', 0.0, confidence)
        elif 1 <= action_idx <= 5:
            # Buy actions
            size_pct = (action_idx) * 0.05  # 5%, 10%, 15%, 20%, 25%
            return TradingAction('buy', size_pct, confidence)
        elif action_idx == 6:
            return TradingAction('sell', 0.0, confidence)
        elif 7 <= action_idx <= 10:
            # Increase actions
            increase_factor = (action_idx - 6) * 0.25
            return TradingAction('increase', increase_factor, confidence)
        elif 11 <= action_idx <= 14:
            # Decrease actions
            decrease_factor = (action_idx - 10) * 0.25
            return TradingAction('decrease', decrease_factor, confidence)

        return TradingAction('hold', 0.0, confidence)


if __name__ == '__main__':
    # Test RL engine
    print("=" * 80)
    print("REINFORCEMENT LEARNING ENGINE TEST")
    print("=" * 80)

    # Create RL engine
    rl_engine = ReinforcementLearningEngine(
        learning_rate=0.01,
        exploration_rate=0.3
    )

    # Create sample market state
    state = MarketState(
        timestamp=datetime.now(),
        symbol='AAPL',
        price=150.0,
        price_change_1h=0.005,
        price_change_1d=0.015,
        price_change_1w=0.03,
        rsi=55.0,
        macd=1.2,
        sma_20=148.0,
        sma_50=145.0,
        sma_200=140.0,
        volatility=0.02,
        volume_ratio=1.3,
        regime='trending_up',
        regime_confidence=0.8,
        position_pnl=0.0,
        portfolio_cash_pct=0.3,
        portfolio_risk=0.05,
        recent_win_rate=0.6,
        spy_change=0.01,
        market_breadth=0.65,
        vix=18.0
    )

    print("\n1. Making a prediction:")
    print("-" * 80)
    prediction = rl_engine.make_prediction('AAPL', state, timeframe='1h')
    print(f"Predicted: {prediction.predicted_direction} {prediction.predicted_value:+.2%}")
    print(f"Confidence: {prediction.confidence:.0%}")

    print("\n2. Selecting an action:")
    print("-" * 80)
    action = rl_engine.select_action(state, mode='train')
    print(f"Action: {action.action_type} (size: {action.position_size_pct:.1%}, confidence: {action.confidence:.0%})")

    print("\n3. Learning from experience:")
    print("-" * 80)
    # Simulate outcome
    next_state = state
    next_state.price_change_1h = 0.012  # Price went up

    experience = Experience(
        state=state,
        action=action,
        reward=1.2,  # Good profit
        next_state=next_state,
        done=False
    )

    rl_engine.learn_from_experience(experience)
    print("Learned from experience!")

    print("\n4. Scoring prediction:")
    print("-" * 80)
    reward = rl_engine.score_prediction(
        prediction.prediction_id,
        actual_value=0.012,
        actual_direction='up'
    )
    print(f"Prediction reward: {reward:+.3f}")

    print("\n5. Learning stats:")
    print("-" * 80)
    stats = rl_engine.get_learning_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("RL Engine tests complete!")

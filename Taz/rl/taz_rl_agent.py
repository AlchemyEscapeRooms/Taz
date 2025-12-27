"""
TAZ Reinforcement Learning Agent
================================
Aggressive RL agent designed for maximum profit velocity.
Trained to find and exploit volatile price movements.

Key differences from standard RL trader:
- Reward function prioritizes raw profit over risk-adjusted returns
- Faster training with aggressive exploration
- Optimized for volatile assets and quick trades
- Supports both stocks and crypto
"""

import os
import sys
import json
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TazRL')


@dataclass
class TazRLConfig:
    """Configuration for aggressive RL agent."""

    # Environment - smaller window for faster reaction
    lookback_window: int = 15
    initial_balance: float = 1000  # Small account focus
    transaction_cost: float = 0.001

    # Aggressive position sizing
    max_position_pct: float = 0.5  # Up to 50% in one trade
    min_position_pct: float = 0.2  # At least 20% when entering

    # Agent - faster learning
    state_size: int = 0
    action_size: int = 5  # Hold, Small Buy, Big Buy, Small Sell, Big Sell
    learning_rate: float = 0.002  # Higher for faster learning
    gamma: float = 0.90  # More focus on immediate rewards
    epsilon: float = 1.0
    epsilon_min: float = 0.05  # Still explore 5% of time
    epsilon_decay: float = 0.98  # Fast decay

    # Training
    batch_size: int = 64
    memory_size: int = 5000  # Smaller memory, more recent focus
    target_update_freq: int = 50  # More frequent updates

    # Network
    hidden_layers: List[int] = None

    # Reward tuning
    profit_multiplier: float = 2.0  # Boost profit rewards
    loss_multiplier: float = 0.5  # Reduce loss penalties
    hold_penalty: float = 0.02  # Penalize doing nothing
    quick_profit_bonus: float = 0.5
    big_move_threshold: float = 0.03  # 3% = big move
    big_move_bonus: float = 1.0

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]


class TazTradingEnvironment:
    """
    Aggressive trading environment for RL agent.
    Rewards profit velocity over risk management.
    """

    # Actions
    HOLD = 0
    SMALL_BUY = 1  # 25% of max position
    BIG_BUY = 2    # 50% of max position
    SMALL_SELL = 3  # Sell half
    BIG_SELL = 4    # Sell all

    ACTION_NAMES = ['HOLD', 'SMALL_BUY', 'BIG_BUY', 'SMALL_SELL', 'BIG_SELL']

    def __init__(self, df: pd.DataFrame, config: TazRLConfig):
        self.config = config
        self.df = df.copy()
        self._prepare_features()

        # State tracking
        self.current_step = 0
        self.balance = config.initial_balance
        self.shares = 0
        self.avg_cost = 0
        self.entry_step = 0  # Track when we entered

        # Episode tracking
        self.episode_trades = []
        self.episode_values = []
        self.total_profit = 0
        self.winning_trades = 0
        self.total_trades = 0

    def _prepare_features(self):
        """Calculate features optimized for volatility trading."""
        df = self.df

        # Fast returns (1, 5, 15 bars)
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_15'] = df['close'].pct_change(15)

        # Volatility (fast, 5-period)
        df['volatility'] = df['returns_1'].rolling(5).std()

        # Volume features
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(float)

        # Fast RSI (7-period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # -1 to 1

        # Fast MACD (8, 17, 9)
        ema8 = df['close'].ewm(span=8).mean()
        ema17 = df['close'].ewm(span=17).mean()
        macd = ema8 - ema17
        signal = macd.ewm(span=9).mean()
        df['macd_hist'] = macd - signal
        df['macd_normalized'] = df['macd_hist'] / df['close']

        # Bollinger (15-period, 2 std)
        sma15 = df['close'].rolling(15).mean()
        std15 = df['close'].rolling(15).std()
        df['bb_upper'] = sma15 + (std15 * 2)
        df['bb_lower'] = sma15 - (std15 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR for volatility context
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(7).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # Momentum
        df['momentum'] = df['close'].pct_change(10)

        # Price position (where is price relative to recent range)
        df['range_position'] = (df['close'] - df['low'].rolling(10).min()) / \
                               (df['high'].rolling(10).max() - df['low'].rolling(10).min() + 0.0001)

        # Feature columns
        self.feature_columns = [
            'returns_1', 'returns_5', 'returns_15',
            'volatility', 'volume_ratio', 'volume_spike',
            'rsi_normalized', 'macd_normalized', 'bb_position',
            'atr_pct', 'momentum', 'range_position'
        ]

        # Fill and clip
        for col in self.feature_columns:
            df[col] = df[col].fillna(0).clip(-5, 5)

        self.df = df
        self.state_size = len(self.feature_columns) * self.config.lookback_window + 4
        self.config.state_size = self.state_size

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_step = self.config.lookback_window
        self.balance = self.config.initial_balance
        self.shares = 0
        self.avg_cost = 0
        self.entry_step = 0
        self.total_profit = 0
        self.winning_trades = 0
        self.total_trades = 0
        self.episode_trades = []
        self.episode_values = [self.config.initial_balance]

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state."""
        idx = self.current_step
        lookback = self.config.lookback_window

        # Feature window
        features = []
        for col in self.feature_columns:
            window = self.df[col].iloc[idx-lookback:idx].values
            features.extend(window)

        # Position features
        current_price = self.df['close'].iloc[idx]
        position_value = self.shares * current_price
        total_value = self.balance + position_value

        position_features = [
            float(self.shares > 0),
            position_value / total_value if total_value > 0 else 0,
            (current_price - self.avg_cost) / self.avg_cost if self.avg_cost > 0 else 0,
            (self.current_step - self.entry_step) / 100 if self.shares > 0 else 0
        ]
        features.extend(position_features)

        return np.array(features, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action with aggressive reward calculation."""
        current_price = self.df['close'].iloc[self.current_step]
        prev_value = self.balance + self.shares * current_price

        reward = 0
        trade_info = None

        # === EXECUTE ACTION ===
        if action == self.SMALL_BUY and self.shares == 0:
            # Small buy: 25% of balance
            spend = self.balance * 0.25
            cost_per_share = current_price * (1 + self.config.transaction_cost)
            shares_to_buy = int(spend / cost_per_share)

            if shares_to_buy > 0:
                total_cost = shares_to_buy * cost_per_share
                self.balance -= total_cost
                self.shares = shares_to_buy
                self.avg_cost = current_price
                self.entry_step = self.current_step
                trade_info = ('SMALL_BUY', shares_to_buy, current_price)

        elif action == self.BIG_BUY and self.shares == 0:
            # Big buy: 50% of balance
            spend = self.balance * self.config.max_position_pct
            cost_per_share = current_price * (1 + self.config.transaction_cost)
            shares_to_buy = int(spend / cost_per_share)

            if shares_to_buy > 0:
                total_cost = shares_to_buy * cost_per_share
                self.balance -= total_cost
                self.shares = shares_to_buy
                self.avg_cost = current_price
                self.entry_step = self.current_step
                trade_info = ('BIG_BUY', shares_to_buy, current_price)

        elif action == self.SMALL_SELL and self.shares > 0:
            # Sell half
            shares_to_sell = self.shares // 2
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.config.transaction_cost)
                pnl = proceeds - (shares_to_sell * self.avg_cost)
                self.balance += proceeds
                self.shares -= shares_to_sell
                self.total_profit += pnl
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                trade_info = ('SMALL_SELL', shares_to_sell, current_price, pnl)

        elif action == self.BIG_SELL and self.shares > 0:
            # Sell all
            proceeds = self.shares * current_price * (1 - self.config.transaction_cost)
            pnl = proceeds - (self.shares * self.avg_cost)
            trade_info = ('BIG_SELL', self.shares, current_price, pnl)
            self.balance += proceeds
            self.total_profit += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            self.shares = 0
            self.avg_cost = 0

        # Move forward
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # === CALCULATE REWARD ===
        new_price = self.df['close'].iloc[self.current_step]
        new_value = self.balance + self.shares * new_price

        # Base reward: portfolio change
        portfolio_return = (new_value - prev_value) / prev_value
        reward = portfolio_return * 100  # Scale

        # === AGGRESSIVE REWARD MODIFIERS ===

        # Bonus for profitable trades
        if trade_info and len(trade_info) > 3:
            pnl = trade_info[3]
            pnl_pct = pnl / (trade_info[1] * trade_info[2])

            if pnl > 0:
                # Profit bonus (scaled by magnitude)
                reward += pnl_pct * 100 * self.config.profit_multiplier

                # Quick profit bonus (sold within 10 bars)
                bars_held = self.current_step - self.entry_step
                if bars_held < 10:
                    reward += self.config.quick_profit_bonus

                # Big move bonus
                if pnl_pct > self.config.big_move_threshold:
                    reward += self.config.big_move_bonus
            else:
                # Smaller loss penalty (encourage risk taking)
                reward += pnl_pct * 100 * self.config.loss_multiplier

        # Holding penalty (encourage action)
        if action == self.HOLD:
            # Missed opportunity penalty if we could have profited
            if self.shares == 0 and portfolio_return > 0.005:
                reward -= self.config.hold_penalty
            # But small bonus for holding winners
            elif self.shares > 0 and portfolio_return > 0:
                reward += 0.01

        # Track values
        self.episode_values.append(new_value)
        if trade_info:
            self.episode_trades.append({
                'step': self.current_step,
                'action': trade_info[0],
                'shares': trade_info[1],
                'price': trade_info[2],
                'pnl': trade_info[3] if len(trade_info) > 3 else None
            })

        info = {
            'portfolio_value': new_value,
            'balance': self.balance,
            'shares': self.shares,
            'trade': trade_info,
            'total_profit': self.total_profit
        }

        return self._get_state(), reward, done, info

    def get_episode_stats(self) -> dict:
        """Get episode statistics."""
        final_value = self.episode_values[-1] if self.episode_values else self.config.initial_balance
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance

        return {
            'final_value': final_value,
            'total_return': total_return,
            'total_profit': self.total_profit,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'trades': self.episode_trades
        }


class TazDQNAgent:
    """Aggressive DQN agent for fast learning."""

    def __init__(self, state_size: int, action_size: int, config: TazRLConfig):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.memory = deque(maxlen=config.memory_size)
        self.epsilon = config.epsilon

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.train_step = 0

    def _build_model(self) -> keras.Model:
        """Build neural network."""
        inputs = keras.Input(shape=(self.state_size,))

        x = layers.Dense(self.config.hidden_layers[0], activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        for units in self.config.hidden_layers[1:]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Dueling DQN architecture
        value = layers.Dense(32, activation='relu')(x)
        value = layers.Dense(1)(value)

        advantage = layers.Dense(32, activation='relu')(x)
        advantage = layers.Dense(self.action_size)(advantage)

        # Combine value and advantage using Keras ops (not TF directly)
        advantage_mean = keras.ops.mean(advantage, axis=1, keepdims=True)
        outputs = value + (advantage - advantage_mean)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss='huber',
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate)
        )

        return model

    def update_target_model(self):
        """Copy weights to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action."""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self) -> float:
        """Train on batch."""
        if len(self.memory) < self.config.batch_size:
            return 0

        batch = random.sample(self.memory, self.config.batch_size)

        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Double DQN
        current_q = self.model.predict(states, verbose=0)
        next_q_model = self.model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)

        for i in range(self.config.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                best_action = np.argmax(next_q_model[i])
                current_q[i][actions[i]] = rewards[i] + self.config.gamma * next_q_target[i][best_action]

        history = self.model.fit(states, current_q, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

        # Update target network
        self.train_step += 1
        if self.train_step % self.config.target_update_freq == 0:
            self.update_target_model()

        return loss

    def save(self, filepath: str):
        """Save model."""
        self.model.save_weights(filepath + '_model.weights.h5')
        self.target_model.save_weights(filepath + '_target.weights.h5')

        config = {
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'config': asdict(self.config)
        }
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, filepath: str):
        """Load model."""
        self.model.load_weights(filepath + '_model.weights.h5')
        self.target_model.load_weights(filepath + '_target.weights.h5')

        with open(filepath + '_config.json', 'r') as f:
            config = json.load(f)
        self.epsilon = config['epsilon']
        self.train_step = config['train_step']


class TazRLTrainer:
    """Main trainer for Taz RL agent."""

    def __init__(self, config: TazRLConfig = None):
        self.config = config or TazRLConfig()
        self.agent = None
        self.model_dir = Path(__file__).parent / 'taz_models'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Alpaca clients
        self.stock_client = None
        self.crypto_client = None
        self._init_clients()

    def _init_clients(self):
        """Initialize data clients."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient

            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')

            if api_key and secret_key:
                self.stock_client = StockHistoricalDataClient(api_key, secret_key)
                self.crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
                logger.info("Alpaca clients initialized")
        except Exception as e:
            logger.error(f"Failed to init clients: {e}")

    def fetch_data(self, symbol: str, days: int = 30, asset_type: str = 'stock') -> pd.DataFrame:
        """Fetch historical data."""
        from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = datetime.now()
        start = end - timedelta(days=days)

        if asset_type == 'crypto':
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )
            bars = self.crypto_client.get_crypto_bars(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )
            bars = self.stock_client.get_stock_bars(request)

        df = bars.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')

        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.sort_index(inplace=True)

        return df

    def train(self, symbol: str, episodes: int = 200, days: int = 60,
              asset_type: str = 'stock') -> dict:
        """
        Train the Taz RL agent on historical data.

        Args:
            symbol: Symbol to train on
            episodes: Training episodes
            days: Days of historical data
            asset_type: 'stock' or 'crypto'
        """
        logger.info(f"[TAZ] Training on {symbol} ({asset_type}) for {episodes} episodes...")

        df = self.fetch_data(symbol, days, asset_type)
        logger.info(f"[TAZ] Got {len(df)} bars")

        if len(df) < 100:
            raise ValueError(f"Not enough data for {symbol}")

        env = TazTradingEnvironment(df, self.config)
        self.agent = TazDQNAgent(env.state_size, self.config.action_size, self.config)

        training_stats = {
            'episode_returns': [],
            'episode_trades': [],
            'losses': []
        }

        best_return = -float('inf')

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            losses = []

            while True:
                action = self.agent.act(state)
                next_state, reward, done, info = env.step(action)

                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.replay()
                if loss > 0:
                    losses.append(loss)

                total_reward += reward
                state = next_state

                if done:
                    break

            stats = env.get_episode_stats()
            training_stats['episode_returns'].append(stats['total_return'])
            training_stats['episode_trades'].append(stats['total_trades'])
            if losses:
                training_stats['losses'].append(np.mean(losses))

            # Log progress
            if (episode + 1) % 10 == 0:
                avg_return = np.mean(training_stats['episode_returns'][-10:])
                logger.info(
                    f"[TAZ] Episode {episode+1}/{episodes} | "
                    f"Return: {stats['total_return']:.2%} | "
                    f"Avg(10): {avg_return:.2%} | "
                    f"Trades: {stats['total_trades']} | "
                    f"WinRate: {stats['win_rate']:.1%} | "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )

            # Save best
            if stats['total_return'] > best_return:
                best_return = stats['total_return']
                self.save_model(symbol, 'best')

        # Final save
        self.save_model(symbol, 'final')

        final_stats = {
            'symbol': symbol,
            'asset_type': asset_type,
            'episodes': episodes,
            'best_return': best_return,
            'avg_return': np.mean(training_stats['episode_returns']),
            'avg_trades': np.mean(training_stats['episode_trades'])
        }

        logger.info(f"\n[TAZ] Training Complete!")
        logger.info(f"Best Return: {best_return:.2%}")
        logger.info(f"Avg Return: {final_stats['avg_return']:.2%}")

        return final_stats

    def train_multi_symbol(self, symbols: List[str], episodes_per_symbol: int = 100,
                           asset_type: str = 'stock') -> dict:
        """Train on multiple symbols for generalization."""
        results = {}

        for symbol in symbols:
            try:
                logger.info(f"\n[TAZ] Training on {symbol}...")
                result = self.train(symbol, episodes_per_symbol, asset_type=asset_type)
                results[symbol] = result
            except Exception as e:
                logger.error(f"[TAZ] Failed to train {symbol}: {e}")
                results[symbol] = {'error': str(e)}

        return results

    def save_model(self, symbol: str, suffix: str = 'latest'):
        """Save model."""
        if self.agent:
            filepath = self.model_dir / f'taz_{symbol}_{suffix}'
            self.agent.save(str(filepath))
            logger.info(f"[TAZ] Model saved: {filepath}")

    def load_model(self, symbol: str, suffix: str = 'best') -> bool:
        """Load model."""
        filepath = self.model_dir / f'taz_{symbol}_{suffix}'
        config_path = f'{filepath}_config.json'

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            self.config.state_size = config_data['state_size']
            self.agent = TazDQNAgent(
                config_data['state_size'],
                config_data['action_size'],
                self.config
            )
            self.agent.load(str(filepath))
            logger.info(f"[TAZ] Model loaded: {filepath}")
            return True
        return False

    def predict_action(self, symbol: str, asset_type: str = 'stock') -> dict:
        """Get action prediction for current market state."""
        if not self.load_model(symbol):
            return {'error': f'No model for {symbol}'}

        df = self.fetch_data(symbol, days=7, asset_type=asset_type)
        env = TazTradingEnvironment(df, self.config)

        env.current_step = len(df) - 1
        state = env._get_state()

        action = self.agent.act(state, training=False)
        action_name = TazTradingEnvironment.ACTION_NAMES[action]

        return {
            'symbol': symbol,
            'action': action_name,
            'action_id': action,
            'current_price': df['close'].iloc[-1],
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Train Taz RL agent."""
    import argparse

    parser = argparse.ArgumentParser(description='Taz RL Trainer')
    parser.add_argument('--symbol', default='TSLA', help='Symbol to train')
    parser.add_argument('--episodes', type=int, default=200, help='Training episodes')
    parser.add_argument('--days', type=int, default=60, help='Days of data')
    parser.add_argument('--type', default='stock', choices=['stock', 'crypto'])

    args = parser.parse_args()

    trainer = TazRLTrainer()

    print(f"\n[TAZ] Starting aggressive RL training on {args.symbol}...")
    print(f"[TAZ] Episodes: {args.episodes}, Days: {args.days}")

    result = trainer.train(
        symbol=args.symbol,
        episodes=args.episodes,
        days=args.days,
        asset_type=args.type
    )

    print(f"\n[TAZ] Training results:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()

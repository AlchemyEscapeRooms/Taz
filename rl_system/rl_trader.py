"""
Reinforcement Learning Trading Agent
====================================
A separate RL-based trading system that learns optimal trading strategies
through experience. Runs independently from SimpleTrader.

Components:
- TradingEnvironment: Gym-like environment for training
- DQNAgent: Deep Q-Network agent with experience replay
- RLTrader: Main class for training and shadow trading

Usage:
    # Train on historical data
    python rl_trader.py train --symbol AAPL --episodes 1000

    # Shadow trade (log decisions without executing)
    python rl_trader.py shadow --symbols AAPL,TSLA,MSFT

    # Evaluate performance
    python rl_trader.py evaluate --symbol AAPL
"""

import os
import sys
import json
import random
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

# TensorFlow setup - suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RLTrader')


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class RLConfig:
    """Configuration for RL trading agent"""
    # Environment
    lookback_window: int = 30          # Days of history for state
    initial_balance: float = 100000    # Starting cash
    transaction_cost: float = 0.001    # 0.1% per trade
    max_position_pct: float = 0.2      # Max 20% in single position

    # Agent
    state_size: int = 0                # Calculated based on features
    action_size: int = 3               # Hold, Buy, Sell
    learning_rate: float = 0.001
    gamma: float = 0.95                # Discount factor
    epsilon: float = 1.0               # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995

    # Training
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 100      # Update target network every N steps

    # Neural network
    hidden_layers: List[int] = None

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]


# =============================================================================
# INDICATORS (same as SimpleTrader for consistency)
# =============================================================================
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram"""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_bollinger(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================
class TradingEnvironment:
    """
    Gym-like trading environment for RL agent.

    State: [price_features, indicator_features, position_features]
    Actions: 0=Hold, 1=Buy, 2=Sell
    Reward: Risk-adjusted returns with transaction cost penalty
    """

    # Actions
    HOLD = 0
    BUY = 1
    SELL = 2
    ACTION_NAMES = ['HOLD', 'BUY', 'SELL']

    def __init__(self, df: pd.DataFrame, config: RLConfig):
        """
        Initialize environment with OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            config: RLConfig instance
        """
        self.config = config
        self.df = df.copy()
        self._prepare_features()

        # State
        self.current_step = 0
        self.balance = config.initial_balance
        self.shares = 0
        self.avg_cost = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0

        # Episode tracking
        self.episode_trades = []
        self.episode_values = []

    def _prepare_features(self):
        """Calculate all features for state representation"""
        df = self.df

        # Price features (normalized)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_20'] = df['close'].pct_change(20)

        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Technical indicators
        df['rsi'] = calculate_rsi(df['close'])
        macd, signal, hist = calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        upper, middle, lower = calculate_bollinger(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_position'] = (df['close'] - lower) / (upper - lower)

        df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        df['atr_pct'] = df['atr'] / df['close']

        # Trend indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend'] = (df['sma_20'] - df['sma_50']) / df['sma_50']

        # Normalize features to [-1, 1] or [0, 1] range
        self.feature_columns = [
            'returns', 'volatility', 'momentum_5', 'momentum_20',
            'volume_ratio', 'rsi', 'macd_hist', 'bb_position',
            'atr_pct', 'trend'
        ]

        # Fill NaN with 0 (beginning of series)
        for col in self.feature_columns:
            df[col] = df[col].fillna(0)

        # Clip extreme values
        for col in self.feature_columns:
            df[col] = df[col].clip(-10, 10)

        self.df = df

        # Set state size
        # Features + lookback window + position features
        self.state_size = len(self.feature_columns) * self.config.lookback_window + 3
        self.config.state_size = self.state_size

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.config.lookback_window
        self.balance = self.config.initial_balance
        self.shares = 0
        self.avg_cost = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.episode_trades = []
        self.episode_values = [self.config.initial_balance]

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        idx = self.current_step
        lookback = self.config.lookback_window

        # Get lookback window of features
        features = []
        for col in self.feature_columns:
            window = self.df[col].iloc[idx-lookback:idx].values
            features.extend(window)

        # Add position features
        current_price = self.df['close'].iloc[idx]
        position_value = self.shares * current_price
        total_value = self.balance + position_value

        position_features = [
            self.shares > 0,  # Has position (0 or 1)
            position_value / total_value if total_value > 0 else 0,  # Position ratio
            (current_price - self.avg_cost) / self.avg_cost if self.avg_cost > 0 else 0  # Unrealized P&L %
        ]
        features.extend(position_features)

        return np.array(features, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return next state, reward, done, info.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            state: Next state
            reward: Reward for action
            done: Episode finished
            info: Additional info
        """
        current_price = self.df['close'].iloc[self.current_step]
        prev_value = self.balance + self.shares * current_price

        reward = 0
        trade_info = None

        if action == self.BUY and self.shares == 0:
            # Buy: Use max_position_pct of balance
            max_spend = self.balance * self.config.max_position_pct
            cost_per_share = current_price * (1 + self.config.transaction_cost)
            shares_to_buy = int(max_spend / cost_per_share)

            if shares_to_buy > 0:
                total_cost = shares_to_buy * cost_per_share
                self.balance -= total_cost
                self.shares = shares_to_buy
                self.avg_cost = current_price
                self.total_trades += 1
                trade_info = ('BUY', shares_to_buy, current_price)

        elif action == self.SELL and self.shares > 0:
            # Sell: Sell all shares
            proceeds = self.shares * current_price * (1 - self.config.transaction_cost)
            pnl = proceeds - (self.shares * self.avg_cost)
            self.balance += proceeds

            if pnl > 0:
                self.winning_trades += 1
            self.total_pnl += pnl

            trade_info = ('SELL', self.shares, current_price, pnl)
            self.shares = 0
            self.avg_cost = 0
            self.total_trades += 1

        # Move to next step
        self.current_step += 1

        # Check if done
        done = self.current_step >= len(self.df) - 1

        # Calculate reward
        new_price = self.df['close'].iloc[self.current_step]
        new_value = self.balance + self.shares * new_price

        # Reward = change in portfolio value (normalized)
        returns = (new_value - prev_value) / prev_value

        # Risk-adjusted reward (penalize large drawdowns)
        reward = returns * 100  # Scale up for learning

        # Small penalty for holding cash (encourage trading)
        if self.shares == 0 and returns > 0.001:
            reward -= 0.01  # Missed opportunity penalty

        # Bonus for winning trades
        if trade_info and trade_info[0] == 'SELL':
            pnl = trade_info[3]
            if pnl > 0:
                reward += 0.5  # Bonus for profitable trade
            else:
                reward -= 0.2  # Penalty for losing trade

        # Track episode values
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
            'total_pnl': self.total_pnl
        }

        return self._get_state(), reward, done, info

    def get_episode_stats(self) -> dict:
        """Get statistics for the episode"""
        final_value = self.episode_values[-1] if self.episode_values else self.config.initial_balance
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance

        # Calculate Sharpe ratio
        if len(self.episode_values) > 1:
            returns = pd.Series(self.episode_values).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        values = pd.Series(self.episode_values)
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        max_drawdown = drawdown.min()

        win_rate = self.winning_trades / max(self.total_trades, 1)

        return {
            'final_value': final_value,
            'total_return': total_return,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'trades': self.episode_trades
        }


# =============================================================================
# DQN AGENT
# =============================================================================
class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    """

    def __init__(self, state_size: int, action_size: int, config: RLConfig):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Experience replay memory
        self.memory = deque(maxlen=config.memory_size)

        # Exploration
        self.epsilon = config.epsilon

        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Training step counter
        self.train_step = 0

    def _build_model(self) -> keras.Model:
        """Build the neural network"""
        # Use functional API to avoid deprecation warning
        inputs = keras.Input(shape=(self.state_size,))

        # First hidden layer
        x = layers.Dense(self.config.hidden_layers[0], activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Additional hidden layers
        for units in self.config.hidden_layers[1:]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Output layer
        outputs = layers.Dense(self.action_size, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss='huber',  # More robust than MSE
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate)
        )

        return model

    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self) -> float:
        """Train on a batch from replay memory"""
        if len(self.memory) < self.config.batch_size:
            return 0

        batch = random.sample(self.memory, self.config.batch_size)

        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Current Q values
        current_q = self.model.predict(states, verbose=0)

        # Next Q values from target network
        next_q = self.target_model.predict(next_states, verbose=0)

        # Update Q values with Bellman equation
        for i in range(self.config.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.config.gamma * np.max(next_q[i])

        # Train
        history = self.model.fit(states, current_q, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.config.target_update_freq == 0:
            self.update_target_model()

        return loss

    def save(self, filepath: str):
        """Save model weights and config"""
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
        """Load model weights and config"""
        self.model.load_weights(filepath + '_model.weights.h5')
        self.target_model.load_weights(filepath + '_target.weights.h5')

        with open(filepath + '_config.json', 'r') as f:
            config = json.load(f)
        self.epsilon = config['epsilon']
        self.train_step = config['train_step']


# =============================================================================
# RL TRADER (Main Class)
# =============================================================================
class RLTrader:
    """
    Main RL trading system that can train and shadow trade.
    """

    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        self.agent = None
        self.data_dir = Path(__file__).parent / 'models'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Alpaca client for live data
        self.trading_client = None
        self.data_client = None
        self._init_alpaca()

        # Shadow trading state
        self.shadow_positions = {}
        self.shadow_trades = []

    def _init_alpaca(self):
        """Initialize Alpaca clients"""
        try:
            from dotenv import load_dotenv
            # Load .env from parent directory (Improved folder)
            env_path = Path(__file__).parent.parent / '.env'
            load_dotenv(env_path)

            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')

            if api_key and secret_key:
                from alpaca.trading.client import TradingClient
                from alpaca.data.historical import StockHistoricalDataClient

                self.trading_client = TradingClient(api_key, secret_key, paper=True)
                self.data_client = StockHistoricalDataClient(api_key, secret_key)
                logger.info("Alpaca clients initialized")
            else:
                logger.warning("Alpaca credentials not found")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca: {e}")

    def fetch_historical_data(self, symbol: str, days: int = 365,
                               timeframe: str = 'day') -> pd.DataFrame:
        """Fetch historical OHLCV data from Alpaca"""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        # Ensure symbol is uppercase for Alpaca API
        symbol = symbol.upper()

        end = datetime.now()
        start = end - timedelta(days=days)

        tf = TimeFrame.Day if timeframe == 'day' else TimeFrame.Hour

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end
        )

        bars = self.data_client.get_stock_bars(request)
        df = bars.df

        if df is None or len(df) == 0:
            raise ValueError(f"No data for {symbol}")

        # Handle MultiIndex (symbol level)
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')

        # Ensure we have required columns
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.sort_index(inplace=True)

        logger.info(f"Fetched {len(df)} bars for {symbol}")
        return df

    def train(self, symbol: str, episodes: int = 500, days: int = 365,
              save_interval: int = 100) -> dict:
        """
        Train the RL agent on historical data.

        Args:
            symbol: Stock symbol to train on
            episodes: Number of training episodes
            days: Days of historical data to use
            save_interval: Save model every N episodes

        Returns:
            Training statistics
        """
        logger.info(f"Fetching {days} days of data for {symbol}...")
        df = self.fetch_historical_data(symbol, days)
        logger.info(f"Got {len(df)} bars")

        # Create environment
        env = TradingEnvironment(df, self.config)

        # Create agent
        self.agent = DQNAgent(env.state_size, self.config.action_size, self.config)

        # Training loop
        training_stats = {
            'episode_returns': [],
            'episode_trades': [],
            'episode_sharpes': [],
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

            # Episode stats
            stats = env.get_episode_stats()
            training_stats['episode_returns'].append(stats['total_return'])
            training_stats['episode_trades'].append(stats['total_trades'])
            training_stats['episode_sharpes'].append(stats['sharpe_ratio'])
            if losses:
                training_stats['losses'].append(np.mean(losses))

            # Log progress
            if (episode + 1) % 10 == 0:
                avg_return = np.mean(training_stats['episode_returns'][-10:])
                logger.info(
                    f"Episode {episode+1}/{episodes} | "
                    f"Return: {stats['total_return']:.2%} | "
                    f"Avg Return (10): {avg_return:.2%} | "
                    f"Trades: {stats['total_trades']} | "
                    f"Win Rate: {stats['win_rate']:.1%} | "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )

            # Save best model
            if stats['total_return'] > best_return:
                best_return = stats['total_return']
                self.save_model(symbol, 'best')

            # Periodic save
            if (episode + 1) % save_interval == 0:
                self.save_model(symbol, f'ep{episode+1}')

        # Final save
        self.save_model(symbol, 'final')

        # Summary
        final_stats = {
            'symbol': symbol,
            'episodes': episodes,
            'best_return': best_return,
            'final_epsilon': self.agent.epsilon,
            'avg_return': np.mean(training_stats['episode_returns']),
            'avg_sharpe': np.mean(training_stats['episode_sharpes']),
            'avg_trades': np.mean(training_stats['episode_trades'])
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"Training Complete for {symbol}")
        logger.info(f"Best Return: {best_return:.2%}")
        logger.info(f"Avg Return: {final_stats['avg_return']:.2%}")
        logger.info(f"Avg Sharpe: {final_stats['avg_sharpe']:.2f}")
        logger.info(f"{'='*50}\n")

        return final_stats

    def save_model(self, symbol: str, suffix: str = 'latest'):
        """Save model to disk"""
        if self.agent:
            filepath = self.data_dir / f'{symbol}_{suffix}'
            self.agent.save(str(filepath))
            logger.info(f"Model saved: {filepath}")

    def load_model(self, symbol: str, suffix: str = 'best'):
        """Load model from disk"""
        filepath = self.data_dir / f'{symbol}_{suffix}'

        # Load config
        config_path = f'{filepath}_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            self.config.state_size = config_data['state_size']
            self.agent = DQNAgent(
                config_data['state_size'],
                config_data['action_size'],
                self.config
            )
            self.agent.load(str(filepath))
            logger.info(f"Model loaded: {filepath}")
            return True
        return False

    def evaluate(self, symbol: str, days: int = 90) -> dict:
        """
        Evaluate trained model on recent data.

        Args:
            symbol: Stock symbol
            days: Days to evaluate

        Returns:
            Evaluation statistics
        """
        # Load model
        if not self.load_model(symbol):
            raise ValueError(f"No trained model found for {symbol}")

        # Get data
        df = self.fetch_historical_data(symbol, days)

        # Create environment
        env = TradingEnvironment(df, self.config)

        # Run evaluation (no exploration)
        state = env.reset()

        while True:
            action = self.agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                break

        stats = env.get_episode_stats()

        # Compare to buy & hold
        buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]

        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results for {symbol}")
        logger.info(f"Period: {days} days")
        logger.info(f"{'='*50}")
        logger.info(f"RL Agent Return:    {stats['total_return']:.2%}")
        logger.info(f"Buy & Hold Return:  {buy_hold_return:.2%}")
        logger.info(f"Outperformance:     {stats['total_return'] - buy_hold_return:.2%}")
        logger.info(f"Sharpe Ratio:       {stats['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown:       {stats['max_drawdown']:.2%}")
        logger.info(f"Total Trades:       {stats['total_trades']}")
        logger.info(f"Win Rate:           {stats['win_rate']:.1%}")
        logger.info(f"{'='*50}\n")

        stats['buy_hold_return'] = buy_hold_return
        stats['outperformance'] = stats['total_return'] - buy_hold_return

        return stats

    def shadow_trade(self, symbols: List[str], interval_seconds: int = 60):
        """
        Run shadow trading - log decisions without executing.
        Runs alongside SimpleTrader without interfering.

        Args:
            symbols: List of symbols to shadow trade
            interval_seconds: Check interval
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        import time

        logger.info(f"Starting shadow trading for {symbols}")
        logger.info("This will log decisions WITHOUT executing trades")

        # Load models
        agents = {}
        for symbol in symbols:
            if self.load_model(symbol):
                agents[symbol] = self.agent
                self.agent = None  # Reset for next
            else:
                logger.warning(f"No model for {symbol}, skipping")

        if not agents:
            logger.error("No models loaded, exiting")
            return

        # Initialize shadow positions
        for symbol in agents:
            self.shadow_positions[symbol] = {'shares': 0, 'avg_cost': 0}

        # Shadow trading loop
        logger.info(f"Shadow trading {list(agents.keys())}")

        while True:
            try:
                for symbol, agent in agents.items():
                    # Get latest data
                    df = self.fetch_historical_data(symbol, days=60)

                    # Create environment with latest data
                    env = TradingEnvironment(df, self.config)

                    # Get state for latest bar
                    env.current_step = len(df) - 1
                    env.shares = self.shadow_positions[symbol]['shares']
                    env.avg_cost = self.shadow_positions[symbol]['avg_cost']

                    state = env._get_state()

                    # Get action
                    action = agent.act(state, training=False)
                    action_name = TradingEnvironment.ACTION_NAMES[action]

                    current_price = df['close'].iloc[-1]

                    # Log decision
                    trade_log = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'action': action_name,
                        'price': current_price,
                        'shadow_shares': self.shadow_positions[symbol]['shares'],
                        'executed': False  # Shadow only
                    }

                    # Update shadow position
                    if action == TradingEnvironment.BUY and self.shadow_positions[symbol]['shares'] == 0:
                        shadow_shares = int(100000 * 0.2 / current_price)  # 20% of $100k
                        self.shadow_positions[symbol] = {
                            'shares': shadow_shares,
                            'avg_cost': current_price
                        }
                        trade_log['shadow_shares'] = shadow_shares
                        logger.info(f"[SHADOW BUY] {symbol}: {shadow_shares} @ ${current_price:.2f}")

                    elif action == TradingEnvironment.SELL and self.shadow_positions[symbol]['shares'] > 0:
                        shares = self.shadow_positions[symbol]['shares']
                        avg_cost = self.shadow_positions[symbol]['avg_cost']
                        pnl = shares * (current_price - avg_cost)
                        trade_log['pnl'] = pnl
                        self.shadow_positions[symbol] = {'shares': 0, 'avg_cost': 0}
                        result = 'WIN' if pnl > 0 else 'LOSS'
                        logger.info(f"[SHADOW SELL] {symbol}: {shares} @ ${current_price:.2f} | {result} ${pnl:.2f}")

                    else:
                        logger.debug(f"[SHADOW HOLD] {symbol} @ ${current_price:.2f}")

                    self.shadow_trades.append(trade_log)

                # Save shadow trades
                self._save_shadow_trades()

                # Wait for next interval
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Shadow trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in shadow trading: {e}")
                time.sleep(interval_seconds)

    def _save_shadow_trades(self):
        """Save shadow trades to file"""
        filepath = self.data_dir / 'shadow_trades.json'
        with open(filepath, 'w') as f:
            json.dump(self.shadow_trades, f, indent=2)

    def compare_to_simple_trader(self, symbol: str, days: int = 30) -> dict:
        """
        Compare RL agent performance to SimpleTrader on same data.

        Args:
            symbol: Symbol to compare
            days: Days to evaluate

        Returns:
            Comparison statistics
        """
        # Get RL performance
        rl_stats = self.evaluate(symbol, days)

        # Get SimpleTrader performance (from API)
        try:
            import requests
            response = requests.get(f'http://localhost:8000/api/brain')
            if response.ok:
                brain_data = response.json()
                st_strategy = brain_data.get('per_stock', {}).get(symbol, {})

                logger.info(f"\n{'='*50}")
                logger.info(f"Comparison: RL Agent vs SimpleTrader for {symbol}")
                logger.info(f"{'='*50}")
                logger.info(f"RL Agent Return:      {rl_stats['total_return']:.2%}")
                logger.info(f"SimpleTrader Strategy: {st_strategy.get('strategy_name', 'N/A')}")
                logger.info(f"{'='*50}\n")
        except Exception as e:
            logger.warning(f"Could not get SimpleTrader data: {e}")

        return rl_stats


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='RL Trading Agent')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train agent on historical data')
    train_parser.add_argument('--symbol', required=True, help='Stock symbol')
    train_parser.add_argument('--episodes', type=int, default=500, help='Training episodes')
    train_parser.add_argument('--days', type=int, default=365, help='Days of historical data')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained agent')
    eval_parser.add_argument('--symbol', required=True, help='Stock symbol')
    eval_parser.add_argument('--days', type=int, default=90, help='Days to evaluate')

    # Shadow trade command
    shadow_parser = subparsers.add_parser('shadow', help='Run shadow trading')
    shadow_parser.add_argument('--symbols', required=True, help='Comma-separated symbols')
    shadow_parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare to SimpleTrader')
    compare_parser.add_argument('--symbol', required=True, help='Stock symbol')
    compare_parser.add_argument('--days', type=int, default=30, help='Days to compare')

    args = parser.parse_args()

    trader = RLTrader()

    if args.command == 'train':
        trader.train(args.symbol, args.episodes, args.days)
    elif args.command == 'evaluate':
        trader.evaluate(args.symbol, args.days)
    elif args.command == 'shadow':
        symbols = [s.strip() for s in args.symbols.split(',')]
        trader.shadow_trade(symbols, args.interval)
    elif args.command == 'compare':
        trader.compare_to_simple_trader(args.symbol, args.days)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

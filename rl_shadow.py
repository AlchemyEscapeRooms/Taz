"""
RL Shadow Trader - The Kid with the Coloring Book
==================================================
Runs alongside SimpleTrader, watches everything, makes recommendations,
but has ZERO influence on actual trading. Just observing and logging.

Think of it as bringing your kid to work - they sit in the corner with
crayons while you do real work. We'll check their drawings later to see
if they're any good.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger('RLShadow')

# Global instance
_rl_shadow: Optional['RLShadow'] = None


@dataclass
class ShadowTrade:
    """Record of what RL would have done."""
    timestamp: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    q_values: Dict[str, float]
    price: float
    simple_trader_action: str  # What SimpleTrader actually did
    agreed: bool  # Did RL and SimpleTrader agree?


class RLShadow:
    """
    Shadow trader that watches but doesn't touch.

    Like a kid at work with a coloring book - they're there,
    they're watching, but they have no authority.
    """

    ACTION_NAMES = ['HOLD', 'BUY', 'SELL']

    def __init__(self):
        self.models: Dict[str, any] = {}  # symbol -> agent
        self.shadow_trades: List[ShadowTrade] = []
        self.shadow_positions: Dict[str, dict] = {}  # What RL thinks it owns
        self.recommendations: Dict[str, dict] = {}  # Current recommendations

        self.models_dir = Path(__file__).parent / "rl_system" / "models"
        self.shadow_log_path = Path(__file__).parent / "data" / "rl_shadow_log.json"

        # Track agreement rate
        self.total_signals = 0
        self.agreements = 0

        self._load_available_models()
        self._load_shadow_log()

    def _load_available_models(self):
        """Load all trained RL models."""
        if not self.models_dir.exists():
            logger.warning("No RL models directory found")
            return

        # Find all model configs
        for config_file in self.models_dir.glob("*_best_config.json"):
            symbol = config_file.stem.replace("_best_config", "").upper()
            try:
                self._load_model(symbol)
            except Exception as e:
                logger.error(f"Failed to load model for {symbol}: {e}")

    def _load_model(self, symbol: str):
        """Load a single RL model."""
        # Try uppercase first, then lowercase
        for sym in [symbol.upper(), symbol.lower()]:
            filepath = self.models_dir / f"{sym}_best"
            config_path = f"{filepath}_config.json"

            if os.path.exists(config_path):
                try:
                    # Import here to avoid loading TF unless needed
                    from rl_system.rl_trader import DQNAgent, RLConfig

                    with open(config_path, 'r') as f:
                        config_data = json.load(f)

                    config = RLConfig()
                    config.state_size = config_data['state_size']

                    agent = DQNAgent(
                        config_data['state_size'],
                        config_data['action_size'],
                        config
                    )
                    agent.load(str(filepath))

                    self.models[symbol.upper()] = {
                        'agent': agent,
                        'config': config,
                        'train_steps': config_data.get('train_step', 0),
                        'epsilon': config_data.get('epsilon', 0)
                    }

                    logger.info(f"RL Shadow: Loaded model for {symbol.upper()}")
                    return True
                except Exception as e:
                    logger.error(f"Error loading {symbol}: {e}")

        return False

    def _load_shadow_log(self):
        """Load previous shadow trades."""
        if self.shadow_log_path.exists():
            try:
                with open(self.shadow_log_path, 'r') as f:
                    data = json.load(f)
                    self.shadow_trades = [ShadowTrade(**t) for t in data.get('trades', [])]
                    self.total_signals = data.get('total_signals', 0)
                    self.agreements = data.get('agreements', 0)
            except:
                pass

    def _save_shadow_log(self):
        """Save shadow trades to disk."""
        try:
            self.shadow_log_path.parent.mkdir(exist_ok=True)
            data = {
                'trades': [asdict(t) for t in self.shadow_trades[-1000:]],  # Keep last 1000
                'total_signals': self.total_signals,
                'agreements': self.agreements,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.shadow_log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save shadow log: {e}")

    def get_recommendation(self, symbol: str, df: pd.DataFrame,
                           has_position: bool = False,
                           position_pct: float = 0,
                           unrealized_pnl_pct: float = 0) -> Optional[dict]:
        """
        Get RL's recommendation for a symbol.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data (needs at least 30 bars)
            has_position: Whether we currently hold this stock
            position_pct: What % of portfolio is in this position
            unrealized_pnl_pct: Current unrealized P&L %

        Returns:
            Dict with recommendation or None if no model
        """
        symbol = symbol.upper()

        if symbol not in self.models:
            return None

        if df is None or len(df) < 30:
            return None

        try:
            model_data = self.models[symbol]
            agent = model_data['agent']

            # Build state manually (simplified version)
            state = self._build_state(df, has_position, position_pct, unrealized_pnl_pct)

            if state is None:
                return None

            # Get Q-values
            state_reshaped = np.reshape(state, [1, len(state)])
            q_values = agent.model.predict(state_reshaped, verbose=0)[0]

            # Get action
            action_idx = np.argmax(q_values)
            action = self.ACTION_NAMES[action_idx]

            # Calculate confidence (how much better is best action vs others)
            confidence = float(np.max(q_values) - np.mean(q_values))

            recommendation = {
                'symbol': symbol,
                'action': action,
                'q_values': {
                    'HOLD': float(q_values[0]),
                    'BUY': float(q_values[1]),
                    'SELL': float(q_values[2])
                },
                'confidence': confidence,
                'has_model': True,
                'train_steps': model_data['train_steps']
            }

            # Cache it
            self.recommendations[symbol] = recommendation

            return recommendation

        except Exception as e:
            logger.error(f"RL recommendation error for {symbol}: {e}")
            return None

    def _build_state(self, df: pd.DataFrame, has_position: bool,
                     position_pct: float, unrealized_pnl_pct: float) -> Optional[np.ndarray]:
        """Build state vector from market data."""
        try:
            # Calculate features
            df = df.copy()

            # Price features
            df['returns'] = df['close'].pct_change().fillna(0).clip(-10, 10)
            df['volatility'] = df['returns'].rolling(20).std().fillna(0).clip(-10, 10)
            df['momentum_5'] = df['close'].pct_change(5).fillna(0).clip(-10, 10)
            df['momentum_20'] = df['close'].pct_change(20).fillna(0).clip(-10, 10)

            # Volume
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = (df['volume'] / df['volume_ma']).fillna(1).clip(-10, 10)

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = (100 - (100 / (1 + rs))).fillna(50) / 100  # Normalize to 0-1
            df['rsi'] = df['rsi'].clip(-10, 10)

            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            df['macd_hist'] = ((macd - signal) / df['close']).fillna(0).clip(-10, 10)

            # Bollinger
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            upper = sma20 + 2 * std20
            lower = sma20 - 2 * std20
            df['bb_position'] = ((df['close'] - lower) / (upper - lower)).fillna(0.5).clip(-10, 10)

            # ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_pct'] = (tr.rolling(14).mean() / df['close']).fillna(0).clip(-10, 10)

            # Trend
            sma50 = df['close'].rolling(50).mean()
            df['trend'] = ((sma20 - sma50) / sma50).fillna(0).clip(-10, 10)

            # Feature columns
            feature_cols = [
                'returns', 'volatility', 'momentum_5', 'momentum_20',
                'volume_ratio', 'rsi', 'macd_hist', 'bb_position',
                'atr_pct', 'trend'
            ]

            # Get last 30 bars of each feature
            features = []
            lookback = 30

            for col in feature_cols:
                window = df[col].iloc[-lookback:].values
                if len(window) < lookback:
                    # Pad with zeros if not enough data
                    window = np.pad(window, (lookback - len(window), 0), 'constant')
                features.extend(window)

            # Add position features
            features.extend([
                float(has_position),
                float(position_pct),
                float(unrealized_pnl_pct)
            ])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"State building error: {e}")
            return None

    def record_signal(self, symbol: str, rl_action: str, simple_trader_action: str,
                      q_values: Dict[str, float], price: float):
        """Record what RL recommended vs what SimpleTrader did."""
        agreed = (rl_action == simple_trader_action)

        trade = ShadowTrade(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action=rl_action,
            q_values=q_values,
            price=price,
            simple_trader_action=simple_trader_action,
            agreed=agreed
        )

        self.shadow_trades.append(trade)
        self.total_signals += 1
        if agreed:
            self.agreements += 1

        self._save_shadow_log()

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with trained models."""
        return list(self.models.keys())

    def get_stats(self) -> dict:
        """Get shadow trading statistics."""
        agreement_rate = self.agreements / max(self.total_signals, 1)

        return {
            'models_loaded': len(self.models),
            'symbols': list(self.models.keys()),
            'total_signals': self.total_signals,
            'agreements': self.agreements,
            'agreement_rate': agreement_rate,
            'shadow_trades': len(self.shadow_trades),
            'current_recommendations': self.recommendations
        }


def get_rl_shadow() -> RLShadow:
    """Get or create the global RL shadow instance."""
    global _rl_shadow
    if _rl_shadow is None:
        _rl_shadow = RLShadow()
    return _rl_shadow

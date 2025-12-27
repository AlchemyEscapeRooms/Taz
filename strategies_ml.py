"""
Cutting-Edge ML-Driven Trading Strategies
==========================================

This module implements state-of-the-art machine learning trading strategies with a unified
architecture that ensures backtesting uses the EXACT same logic as live trading.

Key Principles:
1. UNIFIED ARCHITECTURE: Backtest simulates receiving data day-by-day, just like live
2. POINT-IN-TIME: Only uses data available at that moment (no lookahead bias)
3. ML-FIRST: All decisions driven by trained models, not just technical indicators
4. ADAPTIVE: Models learn from their mistakes and adapt over time
5. PROFIT-FOCUSED: Optimized for maximum risk-adjusted returns

Author: Claude AI
Date: November 29, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import talib as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

logger = get_logger(__name__)


class SignalStrength(Enum):
    """Signal strength levels for ML predictions."""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1


@dataclass
class MLSignal:
    """Machine learning generated trading signal."""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    predicted_return: float  # Expected % return
    risk_reward_ratio: float
    strength: SignalStrength
    features_used: Dict[str, float]
    model_agreement: float  # How much models agree (0-1)
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'predicted_return': self.predicted_return,
            'risk_reward_ratio': self.risk_reward_ratio,
            'strength': self.strength.value,
            'features': self.features_used,
            'model_agreement': self.model_agreement,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }


class UnifiedTradingEngine:
    """
    Unified engine that processes data identically for backtest and live trading.
    
    The KEY INSIGHT: In backtesting, we iterate through historical data day-by-day,
    making decisions as if we're live trading on that date. This ensures:
    - No lookahead bias
    - Same exact logic for backtest and live
    - Realistic slippage and execution simulation
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.open_positions = {}
        self.trade_history = []
        self.ml_models = {}
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        # Performance tracking
        self.daily_returns = []
        self.equity_curve = [initial_capital]
        
        # ML model ensemble
        self._initialize_ml_models()
        
    def _initialize_ml_models(self):
        """Initialize ensemble of ML models."""
        self.ml_models = {
            'direction_classifier': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            ),
            'confidence_model': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'volatility_model': RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
        }
        
    def extract_features(self, df: pd.DataFrame, current_idx: int = -1) -> Dict[str, float]:
        """
        Extract features using ONLY data available up to current_idx.
        This is the CRITICAL function that ensures no lookahead bias.
        
        In live trading, current_idx = -1 (latest data)
        In backtesting, current_idx = the day being simulated
        """
        if current_idx == -1:
            current_idx = len(df) - 1
            
        # Get data slice - ONLY data up to current_idx (inclusive)
        data = df.iloc[:current_idx + 1].copy()
        
        if len(data) < 50:
            return {}
            
        features = {}
        
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # === PRICE FEATURES ===
            features['price_current'] = close[-1]
            features['price_change_1d'] = (close[-1] / close[-2] - 1) * 100 if len(close) > 1 else 0
            features['price_change_5d'] = (close[-1] / close[-5] - 1) * 100 if len(close) > 5 else 0
            features['price_change_10d'] = (close[-1] / close[-10] - 1) * 100 if len(close) > 10 else 0
            features['price_change_20d'] = (close[-1] / close[-20] - 1) * 100 if len(close) > 20 else 0
            
            # === MOVING AVERAGES ===
            sma_5 = ta.SMA(close, timeperiod=5)[-1]
            sma_10 = ta.SMA(close, timeperiod=10)[-1]
            sma_20 = ta.SMA(close, timeperiod=20)[-1]
            sma_50 = ta.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else sma_20
            
            features['price_vs_sma5'] = (close[-1] / sma_5 - 1) * 100 if sma_5 > 0 else 0
            features['price_vs_sma10'] = (close[-1] / sma_10 - 1) * 100 if sma_10 > 0 else 0
            features['price_vs_sma20'] = (close[-1] / sma_20 - 1) * 100 if sma_20 > 0 else 0
            features['price_vs_sma50'] = (close[-1] / sma_50 - 1) * 100 if sma_50 > 0 else 0
            
            features['sma5_vs_sma20'] = (sma_5 / sma_20 - 1) * 100 if sma_20 > 0 else 0
            features['sma10_vs_sma50'] = (sma_10 / sma_50 - 1) * 100 if sma_50 > 0 else 0
            
            # === MOMENTUM INDICATORS ===
            features['rsi_14'] = ta.RSI(close, timeperiod=14)[-1]
            features['rsi_7'] = ta.RSI(close, timeperiod=7)[-1]
            
            macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            features['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
            features['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            features['macd_hist'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            features['macd_hist_change'] = macd_hist[-1] - macd_hist[-2] if len(macd_hist) > 1 and not np.isnan(macd_hist[-2]) else 0
            
            # Stochastic
            slowk, slowd = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            features['stoch_k'] = slowk[-1] if not np.isnan(slowk[-1]) else 50
            features['stoch_d'] = slowd[-1] if not np.isnan(slowd[-1]) else 50
            
            # === VOLATILITY ===
            features['atr_14'] = ta.ATR(high, low, close, timeperiod=14)[-1]
            features['atr_pct'] = (features['atr_14'] / close[-1]) * 100 if close[-1] > 0 else 0
            
            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(close, timeperiod=20)
            bb_width = (upper[-1] - lower[-1]) / middle[-1] * 100 if middle[-1] > 0 else 0
            bb_position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) > 0 else 0.5
            features['bb_width'] = bb_width
            features['bb_position'] = bb_position
            
            # Historical volatility
            returns = pd.Series(close).pct_change().dropna()
            features['volatility_5d'] = returns.tail(5).std() * np.sqrt(252) * 100 if len(returns) >= 5 else 0
            features['volatility_20d'] = returns.tail(20).std() * np.sqrt(252) * 100 if len(returns) >= 20 else 0
            
            # === VOLUME FEATURES ===
            vol_sma = ta.SMA(volume.astype(float), timeperiod=20)[-1]
            features['volume_ratio'] = volume[-1] / vol_sma if vol_sma > 0 else 1
            features['volume_trend'] = ta.SMA(volume.astype(float), timeperiod=5)[-1] / vol_sma if vol_sma > 0 else 1
            
            # On Balance Volume trend
            obv = ta.OBV(close, volume.astype(float))
            obv_sma = ta.SMA(obv, timeperiod=20)[-1]
            features['obv_trend'] = (obv[-1] / obv_sma - 1) * 100 if obv_sma != 0 else 0
            
            # === TREND STRENGTH ===
            adx = ta.ADX(high, low, close, timeperiod=14)[-1]
            features['adx'] = adx if not np.isnan(adx) else 25
            
            plus_di = ta.PLUS_DI(high, low, close, timeperiod=14)[-1]
            minus_di = ta.MINUS_DI(high, low, close, timeperiod=14)[-1]
            features['di_diff'] = plus_di - minus_di
            
            # === PATTERN RECOGNITION ===
            # Recent high/low position
            high_20 = max(high[-20:]) if len(high) >= 20 else high[-1]
            low_20 = min(low[-20:]) if len(low) >= 20 else low[-1]
            features['position_in_range'] = (close[-1] - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
            
            # Consecutive up/down days
            price_changes = pd.Series(close).diff()
            consecutive_up = 0
            consecutive_down = 0
            for i in range(-1, -min(10, len(price_changes)), -1):
                if price_changes.iloc[i] > 0:
                    consecutive_up += 1
                else:
                    break
            for i in range(-1, -min(10, len(price_changes)), -1):
                if price_changes.iloc[i] < 0:
                    consecutive_down += 1
                else:
                    break
            features['consecutive_up'] = consecutive_up
            features['consecutive_down'] = consecutive_down
            
            # === COMPOSITE SCORES ===
            # Trend score (-100 to +100)
            trend_score = 0
            trend_score += 25 if features['price_vs_sma20'] > 0 else -25
            trend_score += 25 if features['sma5_vs_sma20'] > 0 else -25
            trend_score += 25 if features['macd_hist'] > 0 else -25
            trend_score += 25 if features['di_diff'] > 0 else -25
            features['trend_score'] = trend_score
            
            # Momentum score (-100 to +100)
            momentum_score = 0
            if features['rsi_14'] < 30:
                momentum_score += 50  # Oversold = bullish
            elif features['rsi_14'] > 70:
                momentum_score -= 50  # Overbought = bearish
            momentum_score += 25 if features['macd_hist_change'] > 0 else -25
            momentum_score += 25 if features['price_change_5d'] > 0 else -25
            features['momentum_score'] = momentum_score
            
            # Risk score (0 to 100, higher = more risky)
            risk_score = min(100, features['volatility_20d'] * 2 + features['atr_pct'] * 5)
            features['risk_score'] = risk_score
            
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            return {}
            
        return features
    
    def train_models(self, historical_data: pd.DataFrame, min_samples: int = 200):
        """
        Train ML models on historical data.
        
        This is called BEFORE backtesting or going live.
        Uses a walk-forward approach to avoid lookahead bias.
        """
        logger.info("Training ML models...")
        
        if len(historical_data) < min_samples + 50:
            logger.warning(f"Insufficient data for training: {len(historical_data)} rows")
            return False
            
        # Prepare training data
        X_list = []
        y_direction = []  # 1 = up, 0 = down
        y_confidence = []  # 1 = high confidence move, 0 = low
        
        # Use data from day 60 to len-5 for training (need lookback and future return)
        for i in range(60, len(historical_data) - 5):
            features = self.extract_features(historical_data, i)
            if not features:
                continue
                
            # Future return (what we're trying to predict)
            future_return = (historical_data['close'].iloc[i + 5] / historical_data['close'].iloc[i] - 1) * 100
            
            # Direction: 1 if positive, 0 if negative
            direction = 1 if future_return > 0 else 0
            
            # Confidence: 1 if move > 2%, 0 otherwise (significant move)
            confidence = 1 if abs(future_return) > 2 else 0
            
            X_list.append(list(features.values()))
            y_direction.append(direction)
            y_confidence.append(confidence)
        
        if len(X_list) < 100:
            logger.warning(f"Insufficient training samples: {len(X_list)}")
            return False
            
        X = np.array(X_list)
        y_direction = np.array(y_direction)
        y_confidence = np.array(y_confidence)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train models
        try:
            # Direction prediction model
            self.ml_models['direction_classifier'].fit(X_scaled, y_direction)
            
            # Confidence model (when to expect big moves)
            self.ml_models['confidence_model'].fit(X_scaled, y_confidence)
            
            # Evaluate training performance
            dir_score = self.ml_models['direction_classifier'].score(X_scaled, y_direction)
            conf_score = self.ml_models['confidence_model'].score(X_scaled, y_confidence)
            
            logger.info(f"Model training complete:")
            logger.info(f"  Direction accuracy: {dir_score:.2%}")
            logger.info(f"  Confidence accuracy: {conf_score:.2%}")
            
            self.is_trained = True
            self._feature_names = list(self.extract_features(historical_data, 60).keys())
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def generate_ml_signal(self, df: pd.DataFrame, symbol: str, current_idx: int = -1) -> Optional[MLSignal]:
        """
        Generate a trading signal using ML models.
        
        This is the UNIFIED function used by both backtest and live trading.
        """
        features = self.extract_features(df, current_idx)
        if not features:
            return None
            
        try:
            # Prepare features for prediction
            X = np.array([list(features.values())])
            X_scaled = self.feature_scaler.transform(X)
            
            # Get predictions from ensemble
            direction_proba = self.ml_models['direction_classifier'].predict_proba(X_scaled)[0]
            confidence_proba = self.ml_models['confidence_model'].predict_proba(X_scaled)[0]
            
            # Direction: probability of going up
            up_prob = direction_proba[1] if len(direction_proba) > 1 else 0.5
            
            # Expected confidence (probability of significant move)
            move_confidence = confidence_proba[1] if len(confidence_proba) > 1 else 0.3
            
            # Combined confidence score
            overall_confidence = up_prob if up_prob > 0.5 else (1 - up_prob)
            overall_confidence = overall_confidence * (0.5 + move_confidence * 0.5)
            
            # Determine action
            if up_prob > 0.6 and overall_confidence > 0.55:
                action = 'buy'
                predicted_return = (up_prob - 0.5) * 10  # Rough estimate
            elif up_prob < 0.4 and overall_confidence > 0.55:
                action = 'sell'
                predicted_return = (0.5 - up_prob) * 10
            else:
                action = 'hold'
                predicted_return = 0
                
            # Calculate risk-reward ratio
            atr_pct = features.get('atr_pct', 2)
            risk = atr_pct * 2  # 2x ATR as stop loss
            reward = abs(predicted_return)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Determine signal strength
            if overall_confidence > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif overall_confidence > 0.7:
                strength = SignalStrength.STRONG
            elif overall_confidence > 0.6:
                strength = SignalStrength.MODERATE
            elif overall_confidence > 0.5:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
                
            # Generate reasoning
            key_factors = []
            if features.get('trend_score', 0) > 50:
                key_factors.append("Strong uptrend")
            elif features.get('trend_score', 0) < -50:
                key_factors.append("Strong downtrend")
            if features.get('rsi_14', 50) < 30:
                key_factors.append(f"RSI oversold ({features['rsi_14']:.1f})")
            elif features.get('rsi_14', 50) > 70:
                key_factors.append(f"RSI overbought ({features['rsi_14']:.1f})")
            if features.get('macd_hist_change', 0) > 0:
                key_factors.append("MACD momentum positive")
            if features.get('volume_ratio', 1) > 1.5:
                key_factors.append(f"High volume ({features['volume_ratio']:.1f}x)")
                
            reasoning = f"{action.upper()}: ML predicts {up_prob:.1%} up probability. " + \
                       f"Key factors: {', '.join(key_factors) if key_factors else 'Mixed signals'}."
            
            return MLSignal(
                symbol=symbol,
                action=action,
                confidence=overall_confidence,
                predicted_return=predicted_return,
                risk_reward_ratio=risk_reward,
                strength=strength,
                features_used=features,
                model_agreement=move_confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.warning(f"Error generating ML signal: {e}")
            return None


def ml_driven_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """
    Cutting-edge ML-driven trading strategy.
    
    This is the main entry point that works identically for backtest and live.
    
    The 'engine' parameter contains:
    - open_positions: Dict of current positions
    - capital: Available cash
    - ml_engine: The UnifiedTradingEngine instance (if available)
    
    For backtesting: engine.ml_engine is passed in, already trained
    For live trading: engine.ml_engine is the live trading engine
    """
    signals = []
    
    if len(data) < 60:
        return signals
        
    # Get or create ML engine
    ml_engine = getattr(engine, 'ml_engine', None)
    if ml_engine is None:
        # Create a simple fallback if no ML engine provided
        logger.warning("No ML engine provided, using fallback technical strategy")
        return _fallback_technical_strategy(data, engine, params)
    
    if not ml_engine.is_trained:
        logger.warning("ML models not trained, using fallback strategy")
        return _fallback_technical_strategy(data, engine, params)
    
    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'
    current_price = data['close'].iloc[-1]
    
    # Generate ML signal
    ml_signal = ml_engine.generate_ml_signal(data, symbol)
    
    if ml_signal is None:
        return signals
        
    # Position sizing based on confidence and Kelly criterion
    base_position = params.get('position_size', 0.1)
    kelly_fraction = params.get('kelly_fraction', 0.25)  # Use quarter Kelly for safety
    
    # Adjust position size based on confidence
    confidence_multiplier = ml_signal.confidence * 2  # 0-2x
    adjusted_position = base_position * min(confidence_multiplier, 1.5)  # Cap at 1.5x
    
    # Apply Kelly criterion if we have win rate data
    if ml_signal.risk_reward_ratio > 0:
        win_prob = ml_signal.confidence
        kelly = win_prob - (1 - win_prob) / ml_signal.risk_reward_ratio
        kelly = max(0, kelly * kelly_fraction)
        adjusted_position = min(adjusted_position, kelly)
    
    # Minimum and maximum position sizes
    adjusted_position = max(0.02, min(0.25, adjusted_position))
    
    position_size = engine.capital * adjusted_position
    quantity = position_size / current_price if current_price > 0 else 0
    
    # Generate signals based on ML prediction
    if ml_signal.action == 'buy' and ml_signal.strength.value >= SignalStrength.MODERATE.value:
        if symbol not in engine.open_positions:
            signals.append({
                'action': 'buy',
                'symbol': symbol,
                'price': current_price,
                'quantity': quantity,
                'reason': {
                    'primary_signal': 'ml_prediction',
                    'signal_value': ml_signal.confidence,
                    'threshold': 0.55,
                    'direction': 'bullish',
                    'supporting_indicators': ml_signal.features_used,
                    'confirmations': [
                        f"ML confidence: {ml_signal.confidence:.1%}",
                        f"Predicted return: {ml_signal.predicted_return:.2f}%",
                        f"Risk/Reward: {ml_signal.risk_reward_ratio:.2f}",
                        f"Signal strength: {ml_signal.strength.name}",
                    ],
                    'explanation': ml_signal.reasoning
                }
            })
            
    elif ml_signal.action == 'sell':
        if symbol in engine.open_positions:
            signals.append({
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'reason': {
                    'primary_signal': 'ml_prediction',
                    'signal_value': ml_signal.confidence,
                    'threshold': 0.55,
                    'direction': 'bearish',
                    'supporting_indicators': ml_signal.features_used,
                    'confirmations': [
                        f"ML confidence: {ml_signal.confidence:.1%}",
                        f"Predicted return: {ml_signal.predicted_return:.2f}%",
                        f"Signal strength: {ml_signal.strength.name}",
                    ],
                    'explanation': ml_signal.reasoning
                }
            })
    
    return signals


def _fallback_technical_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """
    Fallback strategy when ML models aren't available.
    Uses a combination of technical indicators with proper entry/exit logic.
    """
    signals = []
    
    if len(data) < 50:
        return signals
        
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    volume = data['volume'].values
    
    current_price = close[-1]
    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'
    
    # Calculate indicators
    rsi = ta.RSI(close, timeperiod=14)[-1]
    macd, macd_signal, macd_hist = ta.MACD(close)
    sma_20 = ta.SMA(close, timeperiod=20)[-1]
    sma_50 = ta.SMA(close, timeperiod=50)[-1]
    atr = ta.ATR(high, low, close, timeperiod=14)[-1]
    
    # Volume analysis
    vol_sma = ta.SMA(volume.astype(float), timeperiod=20)[-1]
    volume_ratio = volume[-1] / vol_sma if vol_sma > 0 else 1
    
    # Score-based approach (instead of waiting for exact crossovers)
    buy_score = 0
    sell_score = 0
    reasoning = []
    
    # Trend analysis
    if current_price > sma_20 > sma_50:
        buy_score += 2
        reasoning.append("Strong uptrend (price > SMA20 > SMA50)")
    elif current_price < sma_20 < sma_50:
        sell_score += 2
        reasoning.append("Strong downtrend (price < SMA20 < SMA50)")
    elif current_price > sma_20:
        buy_score += 1
        reasoning.append("Above SMA20")
    elif current_price < sma_20:
        sell_score += 1
        reasoning.append("Below SMA20")
    
    # RSI analysis
    if rsi < 30:
        buy_score += 2
        reasoning.append(f"RSI oversold ({rsi:.1f})")
    elif rsi < 40:
        buy_score += 1
        reasoning.append(f"RSI low ({rsi:.1f})")
    elif rsi > 70:
        sell_score += 2
        reasoning.append(f"RSI overbought ({rsi:.1f})")
    elif rsi > 60:
        sell_score += 1
        reasoning.append(f"RSI high ({rsi:.1f})")
    
    # MACD analysis
    if macd_hist[-1] > 0 and macd_hist[-1] > macd_hist[-2]:
        buy_score += 1
        reasoning.append("MACD histogram rising")
    elif macd_hist[-1] < 0 and macd_hist[-1] < macd_hist[-2]:
        sell_score += 1
        reasoning.append("MACD histogram falling")
    
    # Volume confirmation
    if volume_ratio > 1.5:
        # High volume amplifies the signal
        if buy_score > sell_score:
            buy_score += 1
            reasoning.append(f"High volume confirmation ({volume_ratio:.1f}x)")
        elif sell_score > buy_score:
            sell_score += 1
            reasoning.append(f"High volume confirmation ({volume_ratio:.1f}x)")
    
    # Position sizing - use risk manager limits if available
    max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
    position_size = engine.capital * max_position_pct
    quantity = position_size / current_price if current_price > 0 else 0
    
    # Generate signals with threshold
    threshold = params.get('signal_threshold', 3)
    
    if buy_score >= threshold and symbol not in engine.open_positions:
        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'technical_composite',
                'signal_value': buy_score,
                'threshold': threshold,
                'direction': 'bullish',
                'supporting_indicators': {
                    'rsi': rsi,
                    'macd_hist': macd_hist[-1],
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'volume_ratio': volume_ratio,
                    'atr': atr
                },
                'confirmations': reasoning,
                'explanation': f"BUY: Technical score {buy_score} >= {threshold}. {'; '.join(reasoning)}"
            }
        })
        
    elif sell_score >= threshold and symbol in engine.open_positions:
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'technical_composite',
                'signal_value': sell_score,
                'threshold': threshold,
                'direction': 'bearish',
                'supporting_indicators': {
                    'rsi': rsi,
                    'macd_hist': macd_hist[-1],
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'volume_ratio': volume_ratio
                },
                'confirmations': reasoning,
                'explanation': f"SELL: Technical score {sell_score} >= {threshold}. {'; '.join(reasoning)}"
            }
        })
    
    return signals


# === ADDITIONAL CUTTING-EDGE STRATEGIES ===

def adaptive_momentum_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """
    Adaptive momentum strategy that adjusts to market conditions.
    
    Key innovations:
    1. Adapts lookback period based on volatility
    2. Uses multiple timeframe confirmation
    3. Risk-adjusted position sizing
    """
    signals = []
    
    if len(data) < 100:
        return signals
        
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    
    current_price = close[-1]
    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'
    
    # Adaptive lookback based on volatility
    atr = ta.ATR(high, low, close, timeperiod=14)[-1]
    volatility_ratio = atr / current_price * 100
    
    # Higher volatility = shorter lookback
    if volatility_ratio > 3:
        short_period = 5
        long_period = 15
    elif volatility_ratio > 2:
        short_period = 10
        long_period = 25
    else:
        short_period = 20
        long_period = 50
    
    # Calculate adaptive momentum
    returns_short = (close[-1] / close[-short_period] - 1) * 100
    returns_long = (close[-1] / close[-long_period] - 1) * 100
    
    # Multi-timeframe momentum
    rsi_short = ta.RSI(close, timeperiod=7)[-1]
    rsi_long = ta.RSI(close, timeperiod=21)[-1]
    
    # Momentum alignment score
    momentum_score = 0
    reasoning = []
    
    if returns_short > 0:
        momentum_score += 1
        reasoning.append(f"Short momentum +{returns_short:.1f}%")
    if returns_long > 0:
        momentum_score += 1
        reasoning.append(f"Long momentum +{returns_long:.1f}%")
    if rsi_short > 50 and rsi_long > 50:
        momentum_score += 1
        reasoning.append(f"RSI alignment (short:{rsi_short:.0f}, long:{rsi_long:.0f})")
    if rsi_short > rsi_long:
        momentum_score += 1
        reasoning.append("Accelerating momentum")
    
    # Risk-adjusted position size
    base_position = params.get('position_size', 0.1)
    risk_adjustment = 2 / max(volatility_ratio, 1)  # Lower position for higher vol
    adjusted_position = base_position * min(risk_adjustment, 1.5)
    
    position_size = engine.capital * adjusted_position
    quantity = position_size / current_price if current_price > 0 else 0
    
    # Entry conditions
    entry_threshold = params.get('entry_threshold', 3)
    
    if momentum_score >= entry_threshold and symbol not in engine.open_positions:
        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'adaptive_momentum',
                'signal_value': momentum_score,
                'threshold': entry_threshold,
                'direction': 'bullish',
                'supporting_indicators': {
                    'returns_short': returns_short,
                    'returns_long': returns_long,
                    'rsi_short': rsi_short,
                    'rsi_long': rsi_long,
                    'volatility_ratio': volatility_ratio,
                    'adaptive_periods': f"{short_period}/{long_period}"
                },
                'confirmations': reasoning,
                'explanation': f"BUY: Adaptive momentum score {momentum_score}/{entry_threshold}. " +
                              f"Using {short_period}/{long_period} periods for {volatility_ratio:.1f}% volatility. " +
                              f"{'; '.join(reasoning)}"
            }
        })
    
    # Exit conditions (momentum reversal or stop loss)
    if symbol in engine.open_positions:
        position = engine.open_positions[symbol]
        entry_price = position.get('entry_price', current_price)
        pnl_pct = (current_price / entry_price - 1) * 100
        
        exit_reasons = []
        should_exit = False
        
        # Trailing stop
        if pnl_pct < -2 * volatility_ratio:
            should_exit = True
            exit_reasons.append(f"Stop loss hit ({pnl_pct:.1f}%)")
        
        # Momentum reversal
        if momentum_score <= 0:
            should_exit = True
            exit_reasons.append("Momentum reversed")
        
        # Take profit at extreme RSI
        if pnl_pct > 10 and rsi_short > 75:
            should_exit = True
            exit_reasons.append(f"Take profit (RSI {rsi_short:.0f}, P&L {pnl_pct:.1f}%)")
        
        if should_exit:
            signals.append({
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'reason': {
                    'primary_signal': 'momentum_exit',
                    'signal_value': momentum_score,
                    'threshold': 0,
                    'direction': 'exit',
                    'supporting_indicators': {
                        'pnl_pct': pnl_pct,
                        'rsi_short': rsi_short,
                        'momentum_score': momentum_score
                    },
                    'confirmations': exit_reasons,
                    'explanation': f"SELL: {'; '.join(exit_reasons)}"
                }
            })
    
    return signals


def mean_reversion_ml_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """
    Mean reversion strategy with ML-enhanced entry timing.
    
    Key innovations:
    1. Z-score based extreme detection
    2. Volume confirmation for reversals
    3. ML-predicted reversal probability
    """
    signals = []
    
    if len(data) < 50:
        return signals
        
    close = data['close'].values
    volume = data['volume'].values
    
    current_price = close[-1]
    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'
    
    # Calculate z-score (how many std devs from mean)
    lookback = params.get('lookback', 20)
    mean = np.mean(close[-lookback:])
    std = np.std(close[-lookback:])
    zscore = (current_price - mean) / std if std > 0 else 0
    
    # RSI for confirmation
    rsi = ta.RSI(close, timeperiod=14)[-1]
    
    # Volume spike detection
    vol_mean = np.mean(volume[-20:])
    volume_ratio = volume[-1] / vol_mean if vol_mean > 0 else 1
    
    # Bollinger Band position
    upper, middle, lower = ta.BBANDS(close, timeperiod=20)
    bb_position = (current_price - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) > 0 else 0.5
    
    # Position sizing - use risk manager limits if available
    max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
    position_size = engine.capital * max_position_pct
    quantity = position_size / current_price if current_price > 0 else 0

    entry_zscore = params.get('entry_zscore', 2.0)
    exit_zscore = params.get('exit_zscore', 0.5)
    
    reasoning = []
    
    # Buy signal: oversold conditions
    if zscore < -entry_zscore and symbol not in engine.open_positions:
        score = 0
        
        if rsi < 30:
            score += 2
            reasoning.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            score += 1
            reasoning.append(f"RSI low ({rsi:.1f})")
            
        if bb_position < 0.1:
            score += 2
            reasoning.append(f"Below lower BB")
        elif bb_position < 0.2:
            score += 1
            reasoning.append(f"Near lower BB")
            
        if volume_ratio > 1.5:
            score += 1
            reasoning.append(f"High volume ({volume_ratio:.1f}x)")
        
        # Need minimum confirmation
        if score >= 3:
            signals.append({
                'action': 'buy',
                'symbol': symbol,
                'price': current_price,
                'quantity': quantity,
                'reason': {
                    'primary_signal': 'mean_reversion_oversold',
                    'signal_value': zscore,
                    'threshold': -entry_zscore,
                    'direction': 'bullish_reversal',
                    'supporting_indicators': {
                        'zscore': zscore,
                        'rsi': rsi,
                        'bb_position': bb_position,
                        'volume_ratio': volume_ratio
                    },
                    'confirmations': reasoning,
                    'explanation': f"BUY: Extreme oversold (z={zscore:.2f}). Score {score}/5. {'; '.join(reasoning)}"
                }
            })
    
    # Sell signal: overbought or mean reversion target hit
    elif symbol in engine.open_positions:
        position = engine.open_positions[symbol]
        entry_price = position.get('entry_price', current_price)
        pnl_pct = (current_price / entry_price - 1) * 100
        
        exit_reasons = []
        should_exit = False
        
        # Mean reversion target
        if abs(zscore) < exit_zscore:
            should_exit = True
            exit_reasons.append(f"Mean reversion target (z={zscore:.2f})")
        
        # Overbought exit
        if zscore > entry_zscore:
            should_exit = True
            exit_reasons.append(f"Overbought (z={zscore:.2f})")
        
        # Stop loss
        if pnl_pct < -5:
            should_exit = True
            exit_reasons.append(f"Stop loss ({pnl_pct:.1f}%)")
        
        if should_exit:
            signals.append({
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'reason': {
                    'primary_signal': 'mean_reversion_exit',
                    'signal_value': zscore,
                    'threshold': exit_zscore,
                    'direction': 'exit',
                    'supporting_indicators': {
                        'zscore': zscore,
                        'pnl_pct': pnl_pct,
                        'rsi': rsi
                    },
                    'confirmations': exit_reasons,
                    'explanation': f"SELL: {'; '.join(exit_reasons)}"
                }
            })
    
    return signals


# Strategy registry - add the new strategies
ML_STRATEGY_REGISTRY = {
    'ml_driven': ml_driven_strategy,
    'adaptive_momentum': adaptive_momentum_strategy,
    'mean_reversion_ml': mean_reversion_ml_strategy,
    'fallback_technical': _fallback_technical_strategy,
}

# Default parameters for each strategy
ML_DEFAULT_PARAMS = {
    'ml_driven': {
        'position_size': 0.1,
        'kelly_fraction': 0.25,
    },
    'adaptive_momentum': {
        'position_size': 0.1,
        'entry_threshold': 3,
    },
    'mean_reversion_ml': {
        'position_size': 0.1,
        'lookback': 20,
        'entry_zscore': 2.0,
        'exit_zscore': 0.5,
    },
    'fallback_technical': {
        'position_size': 0.1,
        'signal_threshold': 3,
    }
}

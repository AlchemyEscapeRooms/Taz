"""
Learning Trader Bot
===================

A trading bot that LEARNS before it trades.

Core Loop:
1. OBSERVE - Pull batch market data from Alpaca
2. PREDICT - Make predictions for 1hr, EOD, next day (NO TRADING)
3. TRACK - Log predictions with timestamps
4. VERIFY - Check predictions against actual outcomes
5. LEARN - Update weights based on what worked/failed
6. TRADE - Only after proving accuracy (optional, can stay in learn mode)

Prediction Horizons:
- 1 Hour: Fast feedback loop
- End of Day: Daily patterns
- Next Day Close: Swing patterns

Author: Claude AI
Date: November 29, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import sqlite3
import json
import threading
import time as time_module
from pathlib import Path

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import talib as ta

from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class PredictionHorizon(Enum):
    """Prediction time horizons."""
    ONE_HOUR = "1h"
    END_OF_DAY = "eod"
    NEXT_DAY = "next_day"


class PredictionDirection(Enum):
    """Price direction prediction."""
    UP = "up"
    DOWN = "down"
    FLAT = "flat"  # Less than threshold movement


@dataclass
class Prediction:
    """A single prediction with all context."""
    id: str
    symbol: str
    horizon: PredictionHorizon
    
    # When prediction was made
    prediction_time: datetime
    price_at_prediction: float
    
    # The prediction itself
    predicted_direction: PredictionDirection
    predicted_change_pct: float  # Expected % change
    confidence: float  # 0-1
    
    # What signals led to this prediction
    signals_used: Dict[str, float]
    
    # When to check outcome
    target_time: datetime
    
    # Filled in later when we verify
    actual_price: Optional[float] = None
    actual_change_pct: Optional[float] = None
    was_correct: Optional[bool] = None
    verified_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'horizon': self.horizon.value,
            'prediction_time': self.prediction_time.isoformat(),
            'price_at_prediction': self.price_at_prediction,
            'predicted_direction': self.predicted_direction.value,
            'predicted_change_pct': self.predicted_change_pct,
            'confidence': self.confidence,
            'signals_used': json.dumps(self.signals_used),
            'target_time': self.target_time.isoformat(),
            'actual_price': self.actual_price,
            'actual_change_pct': self.actual_change_pct,
            'was_correct': self.was_correct,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None
        }


@dataclass
class StockLearningProfile:
    """Learning profile for a specific stock."""
    symbol: str
    
    # Accuracy tracking per horizon
    accuracy_1h: float = 0.5
    accuracy_eod: float = 0.5
    accuracy_next_day: float = 0.5
    
    # Sample counts
    predictions_1h: int = 0
    predictions_eod: int = 0
    predictions_next_day: int = 0
    
    # Feature weights - what signals work for THIS stock
    feature_weights: Dict[str, float] = field(default_factory=dict)
    
    # Trading thresholds (earned by accuracy)
    min_confidence_to_trade: float = 0.7
    
    # Performance
    total_correct: int = 0
    total_predictions: int = 0
    
    @property
    def overall_accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.total_correct / self.total_predictions


class PredictionDatabase:
    """SQLite database for storing predictions and learning data."""
    
    def __init__(self, db_path: str = "data/predictions.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,
                prediction_time TEXT NOT NULL,
                price_at_prediction REAL NOT NULL,
                predicted_direction TEXT NOT NULL,
                predicted_change_pct REAL NOT NULL,
                confidence REAL NOT NULL,
                signals_used TEXT,
                target_time TEXT NOT NULL,
                actual_price REAL,
                actual_change_pct REAL,
                was_correct INTEGER,
                verified_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Stock learning profiles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_profiles (
                symbol TEXT PRIMARY KEY,
                accuracy_1h REAL DEFAULT 0.5,
                accuracy_eod REAL DEFAULT 0.5,
                accuracy_next_day REAL DEFAULT 0.5,
                predictions_1h INTEGER DEFAULT 0,
                predictions_eod INTEGER DEFAULT 0,
                predictions_next_day INTEGER DEFAULT 0,
                feature_weights TEXT DEFAULT '{}',
                min_confidence_to_trade REAL DEFAULT 0.7,
                total_correct INTEGER DEFAULT 0,
                total_predictions INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Feature performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL NOT NULL,
                horizon TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                was_correct INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_horizon ON predictions(horizon)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_verified ON predictions(was_correct)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_time)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Prediction database initialized at {self.db_path}")
    
    def save_prediction(self, pred: Prediction):
        """Save a new prediction."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (
                id, symbol, horizon, prediction_time, price_at_prediction,
                predicted_direction, predicted_change_pct, confidence,
                signals_used, target_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred.id, pred.symbol, pred.horizon.value,
            pred.prediction_time.isoformat(), pred.price_at_prediction,
            pred.predicted_direction.value, pred.predicted_change_pct,
            pred.confidence, json.dumps(pred.signals_used),
            pred.target_time.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_pending_predictions(self, before_time: datetime = None) -> List[Dict]:
        """Get predictions that need verification."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if before_time is None:
            before_time = datetime.now()
        
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE was_correct IS NULL 
            AND target_time <= ?
            ORDER BY target_time ASC
        """, (before_time.isoformat(),))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def update_prediction_outcome(self, pred_id: str, actual_price: float, 
                                   actual_change_pct: float, was_correct: bool):
        """Update a prediction with its actual outcome."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictions 
            SET actual_price = ?, actual_change_pct = ?, 
                was_correct = ?, verified_at = ?
            WHERE id = ?
        """, (actual_price, actual_change_pct, int(was_correct), 
              datetime.now().isoformat(), pred_id))
        
        conn.commit()
        conn.close()
    
    def get_stock_profile(self, symbol: str) -> Optional[StockLearningProfile]:
        """Get learning profile for a stock."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM stock_profiles WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        columns = ['symbol', 'accuracy_1h', 'accuracy_eod', 'accuracy_next_day',
                   'predictions_1h', 'predictions_eod', 'predictions_next_day',
                   'feature_weights', 'min_confidence_to_trade', 
                   'total_correct', 'total_predictions', 'updated_at']
        data = dict(zip(columns, row))
        
        return StockLearningProfile(
            symbol=data['symbol'],
            accuracy_1h=data['accuracy_1h'],
            accuracy_eod=data['accuracy_eod'],
            accuracy_next_day=data['accuracy_next_day'],
            predictions_1h=data['predictions_1h'],
            predictions_eod=data['predictions_eod'],
            predictions_next_day=data['predictions_next_day'],
            feature_weights=json.loads(data['feature_weights']),
            min_confidence_to_trade=data['min_confidence_to_trade'],
            total_correct=data['total_correct'],
            total_predictions=data['total_predictions']
        )
    
    def update_stock_profile(self, profile: StockLearningProfile):
        """Update or create a stock learning profile."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO stock_profiles (
                symbol, accuracy_1h, accuracy_eod, accuracy_next_day,
                predictions_1h, predictions_eod, predictions_next_day,
                feature_weights, min_confidence_to_trade,
                total_correct, total_predictions, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                accuracy_1h = excluded.accuracy_1h,
                accuracy_eod = excluded.accuracy_eod,
                accuracy_next_day = excluded.accuracy_next_day,
                predictions_1h = excluded.predictions_1h,
                predictions_eod = excluded.predictions_eod,
                predictions_next_day = excluded.predictions_next_day,
                feature_weights = excluded.feature_weights,
                min_confidence_to_trade = excluded.min_confidence_to_trade,
                total_correct = excluded.total_correct,
                total_predictions = excluded.total_predictions,
                updated_at = excluded.updated_at
        """, (
            profile.symbol, profile.accuracy_1h, profile.accuracy_eod,
            profile.accuracy_next_day, profile.predictions_1h,
            profile.predictions_eod, profile.predictions_next_day,
            json.dumps(profile.feature_weights), profile.min_confidence_to_trade,
            profile.total_correct, profile.total_predictions,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def log_feature_performance(self, symbol: str, feature_name: str, 
                                 feature_value: float, horizon: str,
                                 predicted_direction: str, was_correct: bool):
        """Log how a specific feature performed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feature_performance (
                symbol, feature_name, feature_value, horizon,
                predicted_direction, was_correct
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, feature_name, feature_value, horizon,
              predicted_direction, int(was_correct)))
        
        conn.commit()
        conn.close()
    
    def get_feature_accuracy(self, symbol: str = None, 
                             feature_name: str = None) -> pd.DataFrame:
        """Get accuracy stats for features."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                symbol,
                feature_name,
                horizon,
                COUNT(*) as total,
                SUM(was_correct) as correct,
                AVG(was_correct) as accuracy
            FROM feature_performance
            WHERE was_correct IS NOT NULL
        """
        
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if feature_name:
            query += " AND feature_name = ?"
            params.append(feature_name)
        
        query += " GROUP BY symbol, feature_name, horizon"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_prediction_stats(self, symbol: str = None, 
                             days: int = 30) -> Dict[str, Any]:
        """Get prediction statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = """
            SELECT 
                horizon,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(CASE WHEN was_correct IS NOT NULL THEN was_correct ELSE NULL END) as accuracy,
                AVG(confidence) as avg_confidence,
                AVG(ABS(predicted_change_pct)) as avg_predicted_move,
                AVG(ABS(actual_change_pct)) as avg_actual_move
            FROM predictions
            WHERE prediction_time >= ?
        """
        
        params = [cutoff]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " GROUP BY horizon"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        return {r['horizon']: r for r in results}


class MarketDataBatcher:
    """Handles batch market data requests from Alpaca."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or config.get('alpaca.api_key')
        self.api_secret = api_secret or config.get('alpaca.api_secret')
        
        # Data client for market data
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )
        
        # Trading client for orders (when we're ready)
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=True  # Always start with paper trading
        )
        
        logger.info("MarketDataBatcher initialized with Alpaca API")
    
    def get_batch_bars(
        self, 
        symbols: List[str], 
        timeframe: str = "5Min",
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical bars for multiple symbols in ONE API call.
        
        Args:
            symbols: List of stock symbols (max ~200 per call)
            timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"
            limit: Number of bars per symbol
            
        Returns:
            Dict mapping symbol -> DataFrame with OHLCV data
        """
        # Map timeframe string to Alpaca TimeFrame
        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min"),
            "15Min": TimeFrame(15, "Min"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day
        }
        
        timeframe_obj = tf_map.get(timeframe, TimeFrame(5, "Min"))
        
        # Calculate start time based on limit and timeframe
        now = datetime.now()
        if "Min" in timeframe:
            minutes = int(timeframe.replace("Min", "")) if timeframe != "1Min" else 1
            start = now - timedelta(minutes=minutes * limit * 2)  # Extra buffer
        elif timeframe == "1Hour":
            start = now - timedelta(hours=limit * 2)
        else:
            start = now - timedelta(days=limit * 2)
        
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe_obj,
                start=start,
                limit=limit
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            # Convert to dict of DataFrames
            result = {}
            for symbol in symbols:
                if symbol in bars.data:
                    symbol_bars = bars.data[symbol]
                    df = pd.DataFrame([{
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'vwap': bar.vwap
                    } for bar in symbol_bars])
                    
                    if not df.empty:
                        df.set_index('timestamp', inplace=True)
                        result[symbol] = df
            
            logger.info(f"Fetched bars for {len(result)}/{len(symbols)} symbols")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching batch bars: {e}")
            return {}
    
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            result = {}
            for symbol in symbols:
                if symbol in quotes:
                    # Use midpoint of bid/ask
                    quote = quotes[symbol]
                    if quote.bid_price and quote.ask_price:
                        result[symbol] = (quote.bid_price + quote.ask_price) / 2
                    elif quote.ask_price:
                        result[symbol] = quote.ask_price
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching latest prices: {e}")
            return {}


class FeatureExtractor:
    """Extract features from market data for predictions."""
    
    @staticmethod
    def extract_all_features(df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all features from OHLCV data.
        Returns dict of feature_name -> value.
        """
        if len(df) < 50:
            return {}
        
        features = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            current_price = close[-1]
            
            # === MOMENTUM FEATURES ===
            
            # Price changes over different periods
            for period in [1, 5, 10, 20]:
                if len(close) > period:
                    change = (close[-1] / close[-period] - 1) * 100
                    features[f'price_change_{period}'] = change
            
            # RSI
            rsi_14 = ta.RSI(close, timeperiod=14)
            features['rsi_14'] = rsi_14[-1] if not np.isnan(rsi_14[-1]) else 50
            
            rsi_7 = ta.RSI(close, timeperiod=7)
            features['rsi_7'] = rsi_7[-1] if not np.isnan(rsi_7[-1]) else 50
            
            # RSI divergence (is RSI trending differently than price?)
            if len(rsi_14) > 5:
                price_trend = close[-1] > close[-5]
                rsi_trend = rsi_14[-1] > rsi_14[-5]
                features['rsi_divergence'] = 1 if price_trend != rsi_trend else 0
            
            # === TREND FEATURES ===
            
            # Moving averages
            sma_5 = ta.SMA(close, timeperiod=5)[-1]
            sma_10 = ta.SMA(close, timeperiod=10)[-1]
            sma_20 = ta.SMA(close, timeperiod=20)[-1]
            sma_50 = ta.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else sma_20
            
            features['price_vs_sma5'] = (current_price / sma_5 - 1) * 100
            features['price_vs_sma10'] = (current_price / sma_10 - 1) * 100
            features['price_vs_sma20'] = (current_price / sma_20 - 1) * 100
            features['price_vs_sma50'] = (current_price / sma_50 - 1) * 100
            
            features['sma5_vs_sma20'] = (sma_5 / sma_20 - 1) * 100
            features['sma10_vs_sma50'] = (sma_10 / sma_50 - 1) * 100
            
            # Trend strength
            features['trend_strength'] = (
                (1 if current_price > sma_5 else -1) +
                (1 if current_price > sma_10 else -1) +
                (1 if current_price > sma_20 else -1) +
                (1 if sma_5 > sma_20 else -1)
            ) / 4 * 100  # -100 to +100
            
            # === MACD ===
            macd, macd_signal, macd_hist = ta.MACD(close)
            features['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
            features['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            features['macd_hist'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            
            if len(macd_hist) > 1 and not np.isnan(macd_hist[-2]):
                features['macd_hist_change'] = macd_hist[-1] - macd_hist[-2]
                features['macd_accelerating'] = 1 if features['macd_hist_change'] > 0 else -1
            
            # === VOLATILITY FEATURES ===
            
            # ATR
            atr = ta.ATR(high, low, close, timeperiod=14)[-1]
            features['atr'] = atr
            features['atr_pct'] = (atr / current_price) * 100
            
            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(close, timeperiod=20)
            bb_width = (upper[-1] - lower[-1]) / middle[-1] * 100
            bb_position = (current_price - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) > 0 else 0.5
            
            features['bb_width'] = bb_width
            features['bb_position'] = bb_position
            features['bb_squeeze'] = 1 if bb_width < 4 else 0  # Low volatility squeeze
            
            # Historical volatility
            returns = pd.Series(close).pct_change().dropna()
            features['volatility_5'] = returns.tail(5).std() * np.sqrt(252) * 100 if len(returns) >= 5 else 0
            features['volatility_20'] = returns.tail(20).std() * np.sqrt(252) * 100 if len(returns) >= 20 else 0
            
            # Volatility ratio (recent vs longer term)
            if features['volatility_20'] > 0:
                features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
            
            # === VOLUME FEATURES ===
            
            vol_sma = ta.SMA(volume.astype(float), timeperiod=20)[-1]
            features['volume_ratio'] = volume[-1] / vol_sma if vol_sma > 0 else 1
            
            # Volume trend
            vol_sma_5 = ta.SMA(volume.astype(float), timeperiod=5)[-1]
            features['volume_trend'] = vol_sma_5 / vol_sma if vol_sma > 0 else 1
            
            # Price-volume relationship
            if len(returns) >= 5:
                pv_corr = pd.Series(close[-20:]).pct_change().corr(pd.Series(volume[-20:].astype(float)).pct_change())
                features['price_volume_corr'] = pv_corr if not np.isnan(pv_corr) else 0
            
            # === SUPPORT/RESISTANCE ===
            
            high_20 = max(high[-20:])
            low_20 = min(low[-20:])
            features['distance_from_high'] = (current_price / high_20 - 1) * 100
            features['distance_from_low'] = (current_price / low_20 - 1) * 100
            features['range_position'] = (current_price - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
            
            # === PATTERN FEATURES ===
            
            # Consecutive up/down
            price_changes = pd.Series(close).diff()
            consecutive = 0
            direction = 1 if price_changes.iloc[-1] > 0 else -1
            for i in range(-1, -min(10, len(price_changes)), -1):
                if (price_changes.iloc[i] > 0 and direction > 0) or (price_changes.iloc[i] < 0 and direction < 0):
                    consecutive += direction
                else:
                    break
            features['consecutive_days'] = consecutive
            
            # Gap
            if len(df) > 1:
                features['gap_pct'] = (df['open'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            
            # === STOCHASTIC ===
            slowk, slowd = ta.STOCH(high, low, close)
            features['stoch_k'] = slowk[-1] if not np.isnan(slowk[-1]) else 50
            features['stoch_d'] = slowd[-1] if not np.isnan(slowd[-1]) else 50
            features['stoch_oversold'] = 1 if features['stoch_k'] < 20 else 0
            features['stoch_overbought'] = 1 if features['stoch_k'] > 80 else 0
            
            # === COMPOSITE SCORES ===
            
            # Momentum score (-100 to +100)
            momentum_score = 0
            if features['rsi_14'] < 30:
                momentum_score += 30
            elif features['rsi_14'] > 70:
                momentum_score -= 30
            
            momentum_score += features.get('price_change_5', 0) * 2
            momentum_score += features.get('macd_hist', 0) * 10
            features['momentum_score'] = max(-100, min(100, momentum_score))
            
            # Mean reversion score (0 to 100, higher = more likely to revert)
            mr_score = 0
            if features['rsi_14'] < 30 or features['rsi_14'] > 70:
                mr_score += 30
            if features['bb_position'] < 0.1 or features['bb_position'] > 0.9:
                mr_score += 30
            if abs(features.get('price_vs_sma20', 0)) > 5:
                mr_score += 20
            if features.get('consecutive_days', 0) >= 3 or features.get('consecutive_days', 0) <= -3:
                mr_score += 20
            features['mean_reversion_score'] = mr_score
            
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
        
        return features


class LearningTrader:
    """
    The main learning trader bot.
    
    Observes -> Predicts -> Tracks -> Learns -> (Eventually) Trades
    """
    
    def __init__(
        self,
        symbols: List[str],
        api_key: str = None,
        api_secret: str = None,
        db_path: str = "data/predictions.db",
        learning_mode: bool = True  # Start in learning mode (no real trades)
    ):
        self.symbols = symbols
        self.learning_mode = learning_mode
        
        # Components
        self.db = PredictionDatabase(db_path)
        self.data_batcher = MarketDataBatcher(api_key, api_secret)
        self.feature_extractor = FeatureExtractor()
        
        # Stock profiles (loaded from DB or initialized)
        self.profiles: Dict[str, StockLearningProfile] = {}
        self._load_profiles()
        
        # Global feature weights (learned across all stocks)
        self.global_feature_weights: Dict[str, float] = {}
        
        # Trading day times
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Minimum predictions before considering trading
        self.min_predictions_to_trade = 100
        self.min_accuracy_to_trade = 0.55
        
        # Running state
        self.is_running = False
        self.last_prediction_time: Optional[datetime] = None
        
        logger.info(f"LearningTrader initialized with {len(symbols)} symbols")
        logger.info(f"Learning mode: {learning_mode}")
    
    def _load_profiles(self):
        """Load or create profiles for all symbols."""
        for symbol in self.symbols:
            profile = self.db.get_stock_profile(symbol)
            if profile is None:
                profile = StockLearningProfile(
                    symbol=symbol,
                    feature_weights=self._get_default_weights()
                )
                self.db.update_stock_profile(profile)
            self.profiles[symbol] = profile
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Default feature weights before learning."""
        return {
            'rsi_14': 1.0,
            'macd_hist': 1.0,
            'trend_strength': 1.0,
            'momentum_score': 1.0,
            'bb_position': 1.0,
            'volume_ratio': 0.5,
            'price_change_5': 1.0,
            'mean_reversion_score': 1.0,
        }
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now()
        current_time = now.time()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        return self.market_open <= current_time <= self.market_close
    
    def _get_target_time(self, horizon: PredictionHorizon) -> datetime:
        """Calculate target time for a prediction horizon."""
        now = datetime.now()
        
        if horizon == PredictionHorizon.ONE_HOUR:
            return now + timedelta(hours=1)
        
        elif horizon == PredictionHorizon.END_OF_DAY:
            # Today at market close
            target = now.replace(hour=16, minute=0, second=0, microsecond=0)
            if now.time() >= self.market_close:
                # Market already closed, use next trading day
                target += timedelta(days=1)
                while target.weekday() >= 5:
                    target += timedelta(days=1)
            return target
        
        elif horizon == PredictionHorizon.NEXT_DAY:
            # Next trading day at close
            target = now.replace(hour=16, minute=0, second=0, microsecond=0)
            target += timedelta(days=1)
            while target.weekday() >= 5:
                target += timedelta(days=1)
            return target
        
        return now + timedelta(hours=1)
    
    def _make_prediction(
        self, 
        symbol: str, 
        features: Dict[str, float],
        current_price: float,
        horizon: PredictionHorizon
    ) -> Prediction:
        """
        Make a prediction for a symbol.
        
        Uses weighted features from learning profile.
        """
        profile = self.profiles.get(symbol, StockLearningProfile(symbol=symbol))
        weights = profile.feature_weights or self._get_default_weights()
        
        # Calculate weighted score
        score = 0
        total_weight = 0
        signals_used = {}
        
        for feature_name, weight in weights.items():
            if feature_name in features:
                value = features[feature_name]
                signals_used[feature_name] = value
                
                # Normalize and weight the feature
                if feature_name == 'rsi_14':
                    # RSI: <30 bullish, >70 bearish
                    contribution = (50 - value) / 50 * weight
                elif feature_name == 'bb_position':
                    # BB: <0.2 bullish, >0.8 bearish
                    contribution = (0.5 - value) * 2 * weight
                elif feature_name in ['trend_strength', 'momentum_score', 'macd_hist']:
                    # Already scaled appropriately
                    contribution = value / 100 * weight
                elif feature_name == 'mean_reversion_score':
                    # High MR score + oversold = bullish
                    if features.get('rsi_14', 50) < 40:
                        contribution = value / 100 * weight
                    else:
                        contribution = -value / 100 * weight
                elif 'price_change' in feature_name:
                    # Recent momentum
                    contribution = value / 5 * weight  # Normalize ~5% moves
                else:
                    contribution = value * weight * 0.01
                
                score += contribution
                total_weight += abs(weight)
        
        # Normalize score to -1 to +1 range
        if total_weight > 0:
            score = score / total_weight
        score = max(-1, min(1, score))
        
        # Determine direction and confidence
        threshold = 0.1  # Minimum score to predict direction
        
        if score > threshold:
            direction = PredictionDirection.UP
            confidence = min(0.9, 0.5 + abs(score) * 0.4)
        elif score < -threshold:
            direction = PredictionDirection.DOWN
            confidence = min(0.9, 0.5 + abs(score) * 0.4)
        else:
            direction = PredictionDirection.FLAT
            confidence = 0.5
        
        # Adjust confidence based on historical accuracy
        if horizon == PredictionHorizon.ONE_HOUR:
            accuracy = profile.accuracy_1h
        elif horizon == PredictionHorizon.END_OF_DAY:
            accuracy = profile.accuracy_eod
        else:
            accuracy = profile.accuracy_next_day
        
        # Confidence is tempered by past accuracy
        confidence = confidence * (0.5 + accuracy * 0.5)
        
        # Expected move size based on volatility
        atr_pct = features.get('atr_pct', 1.5)
        if horizon == PredictionHorizon.ONE_HOUR:
            expected_move = atr_pct * 0.2  # Fraction of daily ATR
        elif horizon == PredictionHorizon.END_OF_DAY:
            expected_move = atr_pct * 0.5
        else:
            expected_move = atr_pct * 1.0
        
        if direction == PredictionDirection.DOWN:
            expected_move = -expected_move
        elif direction == PredictionDirection.FLAT:
            expected_move = 0
        
        # Create prediction
        pred_id = f"{symbol}_{horizon.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return Prediction(
            id=pred_id,
            symbol=symbol,
            horizon=horizon,
            prediction_time=datetime.now(),
            price_at_prediction=current_price,
            predicted_direction=direction,
            predicted_change_pct=expected_move,
            confidence=confidence,
            signals_used=signals_used,
            target_time=self._get_target_time(horizon)
        )
    
    def run_prediction_cycle(self) -> List[Prediction]:
        """
        Run one prediction cycle for all symbols and all horizons.
        
        Returns list of predictions made.
        """
        logger.info("=" * 50)
        logger.info("PREDICTION CYCLE STARTED")
        logger.info("=" * 50)
        
        predictions = []
        
        # Batch fetch data for all symbols
        bars_data = self.data_batcher.get_batch_bars(
            self.symbols,
            timeframe="5Min",
            limit=100  # ~8 hours of 5-min bars
        )
        
        # Get latest prices
        latest_prices = self.data_batcher.get_latest_prices(self.symbols)
        
        for symbol in self.symbols:
            if symbol not in bars_data:
                logger.warning(f"No data for {symbol}, skipping")
                continue
            
            df = bars_data[symbol]
            current_price = latest_prices.get(symbol, df['close'].iloc[-1])
            
            # Extract features
            features = self.feature_extractor.extract_all_features(df)
            if not features:
                logger.warning(f"Could not extract features for {symbol}")
                continue
            
            # Make predictions for each horizon
            for horizon in PredictionHorizon:
                pred = self._make_prediction(symbol, features, current_price, horizon)
                predictions.append(pred)
                
                # Save to database
                self.db.save_prediction(pred)
                
                logger.info(
                    f"  {symbol} [{horizon.value}]: "
                    f"{pred.predicted_direction.value.upper()} "
                    f"({pred.predicted_change_pct:+.2f}%) "
                    f"conf={pred.confidence:.2f}"
                )
        
        self.last_prediction_time = datetime.now()
        logger.info(f"Made {len(predictions)} predictions")
        
        return predictions
    
    def verify_predictions(self) -> int:
        """
        Check pending predictions against actual outcomes.
        
        Returns number of predictions verified.
        """
        pending = self.db.get_pending_predictions()
        
        if not pending:
            return 0
        
        logger.info(f"Verifying {len(pending)} pending predictions...")
        
        # Get current prices for all symbols in pending predictions
        symbols = list(set(p['symbol'] for p in pending))
        latest_prices = self.data_batcher.get_latest_prices(symbols)
        
        verified_count = 0
        
        for pred_dict in pending:
            symbol = pred_dict['symbol']
            
            if symbol not in latest_prices:
                continue
            
            actual_price = latest_prices[symbol]
            price_at_pred = pred_dict['price_at_prediction']
            actual_change_pct = (actual_price / price_at_pred - 1) * 100
            
            predicted_direction = pred_dict['predicted_direction']
            
            # Determine if prediction was correct
            flat_threshold = 0.3  # Less than 0.3% move = flat
            
            if abs(actual_change_pct) < flat_threshold:
                actual_direction = 'flat'
            elif actual_change_pct > 0:
                actual_direction = 'up'
            else:
                actual_direction = 'down'
            
            was_correct = (predicted_direction == actual_direction)
            
            # Update prediction in database
            self.db.update_prediction_outcome(
                pred_dict['id'],
                actual_price,
                actual_change_pct,
                was_correct
            )
            
            # Log feature performance for learning
            signals_used = json.loads(pred_dict['signals_used'])
            for feature_name, feature_value in signals_used.items():
                self.db.log_feature_performance(
                    symbol=symbol,
                    feature_name=feature_name,
                    feature_value=feature_value,
                    horizon=pred_dict['horizon'],
                    predicted_direction=predicted_direction,
                    was_correct=was_correct
                )
            
            # Update stock profile
            self._update_profile_from_result(
                symbol, 
                pred_dict['horizon'],
                was_correct,
                signals_used
            )
            
            verified_count += 1
            
            status = "✓" if was_correct else "✗"
            logger.info(
                f"  {status} {symbol} [{pred_dict['horizon']}]: "
                f"Predicted {predicted_direction}, "
                f"Actual {actual_direction} ({actual_change_pct:+.2f}%)"
            )
        
        logger.info(f"Verified {verified_count} predictions")
        return verified_count
    
    def _update_profile_from_result(
        self,
        symbol: str,
        horizon: str,
        was_correct: bool,
        signals_used: Dict[str, float]
    ):
        """Update stock learning profile based on prediction result."""
        profile = self.profiles.get(symbol)
        if profile is None:
            profile = StockLearningProfile(symbol=symbol)
        
        # Update counts
        profile.total_predictions += 1
        if was_correct:
            profile.total_correct += 1
        
        # Update horizon-specific accuracy (exponential moving average)
        alpha = 0.1  # Learning rate
        
        if horizon == PredictionHorizon.ONE_HOUR.value:
            profile.predictions_1h += 1
            profile.accuracy_1h = profile.accuracy_1h * (1 - alpha) + (1 if was_correct else 0) * alpha
        elif horizon == PredictionHorizon.END_OF_DAY.value:
            profile.predictions_eod += 1
            profile.accuracy_eod = profile.accuracy_eod * (1 - alpha) + (1 if was_correct else 0) * alpha
        else:
            profile.predictions_next_day += 1
            profile.accuracy_next_day = profile.accuracy_next_day * (1 - alpha) + (1 if was_correct else 0) * alpha
        
        # Update feature weights based on result
        weight_adjustment = 0.05 if was_correct else -0.05
        
        for feature_name in signals_used:
            if feature_name in profile.feature_weights:
                # Increase weight if prediction was correct, decrease if wrong
                profile.feature_weights[feature_name] *= (1 + weight_adjustment)
                # Keep weights bounded
                profile.feature_weights[feature_name] = max(0.1, min(3.0, profile.feature_weights[feature_name]))
        
        # Update minimum confidence threshold based on accuracy
        if profile.total_predictions >= 50:
            # Require higher confidence if accuracy is low
            profile.min_confidence_to_trade = max(0.6, 1.0 - profile.overall_accuracy)
        
        # Save updated profile
        self.profiles[symbol] = profile
        self.db.update_stock_profile(profile)
    
    def get_trading_readiness(self) -> Dict[str, Dict]:
        """
        Check which symbols are ready for actual trading.
        
        Returns readiness status for each symbol.
        """
        readiness = {}
        
        for symbol, profile in self.profiles.items():
            ready = (
                profile.total_predictions >= self.min_predictions_to_trade and
                profile.overall_accuracy >= self.min_accuracy_to_trade
            )
            
            readiness[symbol] = {
                'ready': ready,
                'predictions': profile.total_predictions,
                'accuracy': profile.overall_accuracy,
                'accuracy_1h': profile.accuracy_1h,
                'accuracy_eod': profile.accuracy_eod,
                'accuracy_next_day': profile.accuracy_next_day,
                'min_confidence': profile.min_confidence_to_trade,
                'reason': self._get_readiness_reason(profile)
            }
        
        return readiness
    
    def _get_readiness_reason(self, profile: StockLearningProfile) -> str:
        """Get human-readable reason for trading readiness status."""
        if profile.total_predictions < self.min_predictions_to_trade:
            return f"Need {self.min_predictions_to_trade - profile.total_predictions} more predictions"
        if profile.overall_accuracy < self.min_accuracy_to_trade:
            return f"Accuracy {profile.overall_accuracy:.1%} below {self.min_accuracy_to_trade:.1%} threshold"
        return "Ready to trade"
    
    def print_status(self):
        """Print current learning status."""
        print("\n" + "=" * 60)
        print("LEARNING TRADER STATUS")
        print("=" * 60)
        
        stats = self.db.get_prediction_stats()
        
        print(f"\nMode: {'LEARNING (no trades)' if self.learning_mode else 'TRADING'}")
        print(f"Symbols: {len(self.symbols)}")
        print(f"Last prediction: {self.last_prediction_time}")
        
        print("\n--- Prediction Stats by Horizon ---")
        for horizon, data in stats.items():
            acc = data.get('accuracy')
            acc_str = f"{acc:.1%}" if acc else "N/A"
            print(f"  {horizon}: {data.get('total', 0)} predictions, {acc_str} accuracy")
        
        print("\n--- Stock Readiness ---")
        readiness = self.get_trading_readiness()
        for symbol, status in readiness.items():
            ready_str = "✓ READY" if status['ready'] else "✗ Learning"
            print(f"  {symbol}: {ready_str} ({status['predictions']} preds, {status['accuracy']:.1%} acc)")
            if not status['ready']:
                print(f"         → {status['reason']}")
        
        print("=" * 60 + "\n")
    
    def run_loop(self, interval_minutes: int = 60):
        """
        Main loop: predict every hour, verify continuously.
        
        Args:
            interval_minutes: How often to make new predictions
        """
        self.is_running = True
        logger.info(f"Starting learning loop (interval: {interval_minutes} min)")
        
        while self.is_running:
            try:
                # Only run during market hours
                if not self._is_market_open():
                    logger.info("Market closed, waiting...")
                    time_module.sleep(60)
                    continue
                
                # Verify any pending predictions
                self.verify_predictions()
                
                # Check if it's time for new predictions
                should_predict = (
                    self.last_prediction_time is None or
                    (datetime.now() - self.last_prediction_time) >= timedelta(minutes=interval_minutes)
                )
                
                if should_predict:
                    self.run_prediction_cycle()
                    self.print_status()
                
                # Sleep before next check
                time_module.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Stopping learning loop...")
                self.is_running = False
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time_module.sleep(60)
        
        logger.info("Learning loop stopped")
    
    def stop(self):
        """Stop the learning loop."""
        self.is_running = False


def main():
    """Example usage."""
    
    # Stocks to monitor
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "META", "NVDA", "AMD", "NFLX", "SPY",
        "QQQ", "DIA", "IWM", "VTI", "ARKK"
    ]
    
    # Create learning trader
    trader = LearningTrader(
        symbols=symbols,
        learning_mode=True  # Start in learning mode
    )
    
    # Print initial status
    trader.print_status()
    
    # Run one prediction cycle manually
    predictions = trader.run_prediction_cycle()
    print(f"\nMade {len(predictions)} predictions")
    
    # To run continuously:
    # trader.run_loop(interval_minutes=60)


if __name__ == "__main__":
    main()

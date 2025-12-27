"""
Background Trading Service
===========================

Runs continuously in the background:
- Market data streaming/polling
- Prediction generation every hour
- Prediction verification as targets hit
- Weight updates (learning)
- Trade execution when conditions met

The user interaction (morning briefing) is SEPARATE from this.
This service runs all day regardless of user interaction.

Author: Claude AI  
Date: November 29, 2025
"""

import asyncio
import threading
import queue
import json
import time as time_module
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import signal
import sys

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from learning_trader import (
    LearningTrader,
    PredictionDatabase,
    FeatureExtractor,
    StockLearningProfile,
    Prediction,
    PredictionHorizon,
    PredictionDirection,
    MarketDataBatcher
)
from historical_trainer import HistoricalTrainer
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class ServiceState(Enum):
    """Background service states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


class TradingMode(Enum):
    """Trading execution mode."""
    LEARNING_ONLY = "learning_only"  # Just predict and learn, no trades
    PAPER_TRADING = "paper_trading"  # Execute on paper account
    LIVE_TRADING = "live_trading"    # Real money (requires confirmation)


@dataclass
class TradeSignal:
    """A signal to potentially execute a trade."""
    symbol: str
    action: str  # 'buy' or 'sell'
    confidence: float
    predicted_change_pct: float
    horizon: PredictionHorizon
    
    # Position sizing
    suggested_quantity: int
    suggested_position_pct: float
    
    # Risk management
    stop_loss_pct: float
    take_profit_pct: float
    
    # Context
    reasoning: str
    features: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Execution status
    executed: bool = False
    order_id: Optional[str] = None


@dataclass
class ServiceConfig:
    """Configuration for the background service. Values loaded from config.yaml."""
    # Symbols to monitor
    symbols: List[str] = field(default_factory=list)
    excluded_symbols: List[str] = field(default_factory=list)

    # Timing - loaded from config.yaml service section
    prediction_interval_minutes: int = field(default_factory=lambda: config.get('service.prediction_interval_minutes', 60))
    verification_interval_minutes: int = 5  # Check pending predictions every 5 min
    data_refresh_interval_seconds: int = field(default_factory=lambda: config.get('service.data_refresh_interval_seconds', 60))

    # Trading parameters - loaded from config.yaml
    trading_mode: TradingMode = field(default_factory=lambda: TradingMode.PAPER_TRADING if config.get('trading.mode', 'paper') == 'paper' else (TradingMode.LIVE_TRADING if config.get('trading.mode') == 'live' else TradingMode.LEARNING_ONLY))
    min_confidence_to_trade: float = field(default_factory=lambda: config.get('learning.min_confidence_to_trade', 0.65))
    min_accuracy_to_trade: float = field(default_factory=lambda: config.get('learning.min_accuracy_to_trade', 0.55))
    min_predictions_to_trade: int = field(default_factory=lambda: config.get('learning.min_predictions_to_trade', 100))

    # Position sizing - loaded from config.yaml service section
    max_position_pct: float = field(default_factory=lambda: config.get('service.max_position_pct', 0.10))
    max_portfolio_risk_pct: float = field(default_factory=lambda: config.get('service.max_portfolio_risk_pct', 0.25))
    max_daily_loss_pct: float = field(default_factory=lambda: config.get('service.max_daily_loss_pct', 0.02))

    # Risk management - loaded from config.yaml service section
    default_stop_loss_pct: float = field(default_factory=lambda: config.get('service.default_stop_loss_pct', 0.02))
    default_take_profit_pct: float = field(default_factory=lambda: config.get('service.default_take_profit_pct', 0.05))


class MarketDataManager:
    """
    Manages real-time and historical market data.
    
    Uses polling (not WebSocket) for simplicity and reliability.
    """
    
    def __init__(self, api_key: str, api_secret: str, symbols: List[str]):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret
        )
        
        # Cache for latest data
        self.latest_bars: Dict[str, Dict] = {}  # symbol -> latest bar data
        self.latest_prices: Dict[str, float] = {}  # symbol -> price
        self.bars_cache: Dict[str, Any] = {}  # symbol -> DataFrame of recent bars
        
        self.last_update: Optional[datetime] = None
        
        logger.info(f"MarketDataManager initialized for {len(symbols)} symbols")
    
    def refresh_data(self) -> bool:
        """
        Refresh market data for all symbols.
        
        Returns True if successful.
        """
        try:
            # Batch fetch latest quotes
            request = StockLatestQuoteRequest(symbol_or_symbols=self.symbols)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            for symbol in self.symbols:
                if symbol in quotes:
                    quote = quotes[symbol]
                    if quote.bid_price and quote.ask_price:
                        self.latest_prices[symbol] = (quote.bid_price + quote.ask_price) / 2
                    elif quote.ask_price:
                        self.latest_prices[symbol] = quote.ask_price
            
            self.last_update = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing market data: {e}")
            return False
    
    def get_bars(self, symbol: str, timeframe: str = "5Min", limit: int = 100) -> Optional[Any]:
        """Get recent bars for a symbol."""
        try:
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute)),
                limit=limit
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            if symbol in bars.data:
                import pandas as pd
                df = pd.DataFrame([{
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap
                } for bar in bars.data[symbol]])
                
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                    self.bars_cache[symbol] = df
                    return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}")
            return None
    
    def get_all_bars(self, timeframe: str = "5Min", limit: int = 100) -> Dict[str, Any]:
        """Get bars for all symbols (batch call)."""
        try:
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            request = StockBarsRequest(
                symbol_or_symbols=self.symbols,
                timeframe=tf_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute)),
                limit=limit
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            import pandas as pd
            result = {}
            
            for symbol in self.symbols:
                if symbol in bars.data:
                    df = pd.DataFrame([{
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'vwap': bar.vwap
                    } for bar in bars.data[symbol]])
                    
                    if not df.empty:
                        df.set_index('timestamp', inplace=True)
                        result[symbol] = df
                        self.bars_cache[symbol] = df
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting batch bars: {e}")
            return {}
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        return self.latest_prices.get(symbol)
    
    def update_symbols(self, symbols: List[str]):
        """Update the list of symbols to monitor."""
        self.symbols = symbols
        logger.info(f"Updated symbol list: {len(symbols)} symbols")


class TradeExecutor:
    """Handles trade execution."""
    
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper
        )
        self.paper = paper
        
        logger.info(f"TradeExecutor initialized ({'PAPER' if paper else 'LIVE'})")
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'last_equity': float(account.last_equity)
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return {
                pos.symbol: {
                    'quantity': float(pos.qty),
                    'avg_cost': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_pl_pct': float(pos.unrealized_plpc)
                }
                for pos in positions
            }
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def execute_trade(self, signal: TradeSignal) -> Optional[str]:
        """
        Execute a trade based on a signal.
        
        Returns order ID if successful.
        """
        try:
            if signal.action == 'buy':
                order_side = OrderSide.BUY
            else:
                order_side = OrderSide.SELL
            
            # Create market order
            order_request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=signal.suggested_quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            
            logger.info(f"Order submitted: {signal.action.upper()} {signal.suggested_quantity} {signal.symbol}")
            
            return order.id
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False


class BackgroundTradingService:
    """
    The main background service that runs all day.
    
    Handles:
    - Continuous market data monitoring
    - Hourly prediction generation
    - Prediction verification
    - Learning/weight updates
    - Trade execution (when enabled)
    """
    
    def __init__(
        self,
        config: ServiceConfig = None,
        api_key: str = None,
        api_secret: str = None,
        data_dir: str = "data"
    ):
        self.config = config or ServiceConfig()
        self.api_key = api_key or globals()['config'].get('alpaca.api_key')
        self.api_secret = api_secret or globals()['config'].get('alpaca.api_secret')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.state = ServiceState.STOPPED
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Components (initialized on start)
        self.data_manager: Optional[MarketDataManager] = None
        self.trade_executor: Optional[TradeExecutor] = None
        self.prediction_db: Optional[PredictionDatabase] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        
        # Stock profiles
        self.profiles: Dict[str, StockLearningProfile] = {}
        
        # Signal queue for potential trades
        self.signal_queue: queue.Queue = queue.Queue()
        
        # Event callbacks
        self.on_prediction: Optional[Callable] = None
        self.on_verification: Optional[Callable] = None
        self.on_trade_signal: Optional[Callable] = None
        self.on_trade_executed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Timing trackers
        self.last_prediction_time: Optional[datetime] = None
        self.last_verification_time: Optional[datetime] = None
        self.last_data_refresh: Optional[datetime] = None
        
        # Daily stats
        self.daily_stats = {
            'predictions_made': 0,
            'predictions_verified': 0,
            'correct_predictions': 0,
            'trades_executed': 0,
            'starting_equity': 0,
            'signals_generated': 0
        }
        
        # Threads
        self._threads: List[threading.Thread] = []
        
        logger.info("BackgroundTradingService created")
    
    def _init_components(self):
        """Initialize all components."""
        active_symbols = [s for s in self.config.symbols if s not in self.config.excluded_symbols]
        
        self.data_manager = MarketDataManager(
            self.api_key, 
            self.api_secret, 
            active_symbols
        )
        
        self.trade_executor = TradeExecutor(
            self.api_key,
            self.api_secret,
            paper=(self.config.trading_mode != TradingMode.LIVE_TRADING)
        )
        
        self.prediction_db = PredictionDatabase(str(self.data_dir / "predictions.db"))
        self.feature_extractor = FeatureExtractor()
        
        # Load profiles
        for symbol in active_symbols:
            profile = self.prediction_db.get_stock_profile(symbol)
            if profile:
                self.profiles[symbol] = profile
            else:
                self.profiles[symbol] = StockLearningProfile(
                    symbol=symbol,
                    feature_weights=self._get_default_weights()
                )
        
        # Get starting equity
        account = self.trade_executor.get_account()
        self.daily_stats['starting_equity'] = account.get('equity', 0)
        
        logger.info(f"Components initialized. Monitoring {len(active_symbols)} symbols")
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default feature weights."""
        return {
            'rsi_14': 1.0, 'rsi_7': 0.8,
            'macd_hist': 1.0, 'macd_accelerating': 0.8,
            'trend_strength': 1.0, 'momentum_score': 1.0,
            'bb_position': 1.0, 'volume_ratio': 0.5,
            'price_change_5': 1.0, 'price_change_10': 0.8,
            'price_vs_sma20': 1.0, 'mean_reversion_score': 1.0,
        }
    
    def _is_market_hours(self) -> bool:
        """Check if market is open."""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Market hours (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        return market_open <= now.time() <= market_close
    
    # =========================================================================
    # PREDICTION ENGINE
    # =========================================================================
    
    def _make_predictions(self):
        """Make predictions for all active symbols."""
        active_symbols = [s for s in self.config.symbols if s not in self.config.excluded_symbols]
        
        if not active_symbols:
            return
        
        logger.info(f"Making predictions for {len(active_symbols)} symbols...")
        
        # Batch fetch data
        bars_data = self.data_manager.get_all_bars(timeframe="5Min", limit=100)
        
        predictions_made = 0
        
        for symbol in active_symbols:
            if symbol not in bars_data:
                continue
            
            df = bars_data[symbol]
            current_price = self.data_manager.get_price(symbol) or df['close'].iloc[-1]
            
            # Extract features
            features = self.feature_extractor.extract_all_features(df)
            if not features:
                continue
            
            # Make prediction for each horizon
            for horizon in PredictionHorizon:
                prediction = self._generate_prediction(
                    symbol, features, current_price, horizon
                )
                
                if prediction:
                    # Save to database
                    self.prediction_db.save_prediction(prediction)
                    predictions_made += 1
                    
                    # Check if this generates a trade signal
                    self._evaluate_trade_signal(prediction, features)
                    
                    # Callback
                    if self.on_prediction:
                        self.on_prediction(prediction)
        
        self.daily_stats['predictions_made'] += predictions_made
        self.last_prediction_time = datetime.now()
        
        logger.info(f"Made {predictions_made} predictions")
    
    def _generate_prediction(
        self,
        symbol: str,
        features: Dict[str, float],
        current_price: float,
        horizon: PredictionHorizon
    ) -> Optional[Prediction]:
        """Generate a single prediction."""
        
        profile = self.profiles.get(symbol)
        if not profile:
            return None
        
        weights = profile.feature_weights
        
        # Calculate weighted score
        score = 0
        total_weight = 0
        signals_used = {}
        
        for feature_name, weight in weights.items():
            if feature_name not in features:
                continue
            
            value = features[feature_name]
            signals_used[feature_name] = value
            
            # Normalize and weight
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

        # Determine direction and confidence (threshold from config)
        threshold = config.get('learning.direction_threshold', 0.1)
        if score > threshold:
            direction = PredictionDirection.UP
            confidence = min(0.9, 0.5 + abs(score) * 0.4)
        elif score < -threshold:
            direction = PredictionDirection.DOWN
            confidence = min(0.9, 0.5 + abs(score) * 0.4)
        else:
            direction = PredictionDirection.FLAT
            confidence = 0.5
        
        # Adjust confidence by historical accuracy
        if horizon == PredictionHorizon.ONE_HOUR:
            accuracy = profile.accuracy_1h
        elif horizon == PredictionHorizon.END_OF_DAY:
            accuracy = profile.accuracy_eod
        else:
            accuracy = profile.accuracy_next_day
        
        confidence = confidence * (0.5 + accuracy * 0.5)
        
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
        
        # Target time
        now = datetime.now()
        if horizon == PredictionHorizon.ONE_HOUR:
            target_time = now + timedelta(hours=1)
        elif horizon == PredictionHorizon.END_OF_DAY:
            target_time = now.replace(hour=16, minute=0, second=0)
            if now.time() >= time(16, 0):
                target_time += timedelta(days=1)
        else:
            target_time = (now + timedelta(days=1)).replace(hour=16, minute=0, second=0)
        
        pred_id = f"{symbol}_{horizon.value}_{now.strftime('%Y%m%d%H%M%S')}"
        
        return Prediction(
            id=pred_id,
            symbol=symbol,
            horizon=horizon,
            prediction_time=now,
            price_at_prediction=current_price,
            predicted_direction=direction,
            predicted_change_pct=expected_move,
            confidence=confidence,
            signals_used=signals_used,
            target_time=target_time
        )
    
    # =========================================================================
    # VERIFICATION & LEARNING
    # =========================================================================
    
    def _verify_predictions(self):
        """Verify pending predictions against actual outcomes."""
        pending = self.prediction_db.get_pending_predictions()
        
        if not pending:
            return
        
        # Get current prices
        symbols = list(set(p['symbol'] for p in pending))
        self.data_manager.refresh_data()
        
        verified_count = 0
        correct_count = 0
        
        for pred_dict in pending:
            symbol = pred_dict['symbol']
            actual_price = self.data_manager.get_price(symbol)
            
            if actual_price is None:
                continue
            
            price_at_pred = pred_dict['price_at_prediction']
            actual_change_pct = (actual_price / price_at_pred - 1) * 100
            
            predicted_direction = pred_dict['predicted_direction']

            # Determine actual direction (threshold from config)
            flat_threshold = config.get('learning.flat_threshold', 0.3)
            if abs(actual_change_pct) < flat_threshold:
                actual_direction = 'flat'
            elif actual_change_pct > 0:
                actual_direction = 'up'
            else:
                actual_direction = 'down'
            
            was_correct = (predicted_direction == actual_direction)
            
            # Update database
            self.prediction_db.update_prediction_outcome(
                pred_dict['id'],
                actual_price,
                actual_change_pct,
                was_correct
            )
            
            # Update profile (learning)
            self._update_profile(
                symbol,
                pred_dict['horizon'],
                was_correct,
                json.loads(pred_dict['signals_used'])
            )
            
            verified_count += 1
            if was_correct:
                correct_count += 1
            
            # Callback
            if self.on_verification:
                self.on_verification({
                    'symbol': symbol,
                    'horizon': pred_dict['horizon'],
                    'predicted': predicted_direction,
                    'actual': actual_direction,
                    'was_correct': was_correct,
                    'actual_change_pct': actual_change_pct
                })
        
        self.daily_stats['predictions_verified'] += verified_count
        self.daily_stats['correct_predictions'] += correct_count
        self.last_verification_time = datetime.now()
        
        if verified_count > 0:
            logger.info(f"Verified {verified_count} predictions, {correct_count} correct ({correct_count/verified_count:.1%})")
    
    def _update_profile(
        self,
        symbol: str,
        horizon: str,
        was_correct: bool,
        signals_used: Dict[str, float]
    ):
        """Update stock profile based on prediction outcome."""
        profile = self.profiles.get(symbol)
        if not profile:
            return
        
        # Update counts
        profile.total_predictions += 1
        if was_correct:
            profile.total_correct += 1

        # Update horizon accuracy (EMA with config alpha)
        alpha = config.get('learning.ema_alpha', 0.1)
        if horizon == 'h':
            profile.predictions_1h += 1
            profile.accuracy_1h = profile.accuracy_1h * (1 - alpha) + (1 if was_correct else 0) * alpha
        elif horizon == 'eod':
            profile.predictions_eod += 1
            profile.accuracy_eod = profile.accuracy_eod * (1 - alpha) + (1 if was_correct else 0) * alpha
        else:
            profile.predictions_next_day += 1
            profile.accuracy_next_day = profile.accuracy_next_day * (1 - alpha) + (1 if was_correct else 0) * alpha

        # Update feature weights (adjustment from config)
        weight_adj = config.get('learning.weight_adjustment', 0.02)
        adjustment = weight_adj if was_correct else -weight_adj
        min_weight = config.get('learning.min_weight', 0.5)
        max_weight = config.get('learning.max_weight', 2.0)
        for feature_name in signals_used:
            if feature_name in profile.feature_weights:
                profile.feature_weights[feature_name] *= (1 + adjustment)
                profile.feature_weights[feature_name] = max(min_weight, min(max_weight, profile.feature_weights[feature_name]))
        
        # Save
        self.profiles[symbol] = profile
        self.prediction_db.update_stock_profile(profile)
    
    # =========================================================================
    # TRADE SIGNAL GENERATION
    # =========================================================================
    
    def _evaluate_trade_signal(self, prediction: Prediction, features: Dict[str, float]):
        """Evaluate if a prediction should generate a trade signal."""
        
        # Only generate signals in trading modes
        if self.config.trading_mode == TradingMode.LEARNING_ONLY:
            return
        
        profile = self.profiles.get(prediction.symbol)
        if not profile:
            return
        
        # Check if stock is ready for trading
        if profile.total_predictions < self.config.min_predictions_to_trade:
            return
        
        if profile.overall_accuracy < self.config.min_accuracy_to_trade:
            return
        
        # Check confidence threshold
        if prediction.confidence < self.config.min_confidence_to_trade:
            return
        
        # Only trade on directional predictions
        if prediction.predicted_direction == PredictionDirection.FLAT:
            return
        
        # Generate signal
        account = self.trade_executor.get_account()
        equity = account.get('equity', 0)
        
        if equity <= 0:
            return
        
        # Position sizing
        position_value = equity * self.config.max_position_pct * prediction.confidence
        current_price = prediction.price_at_prediction
        quantity = int(position_value / current_price)
        
        if quantity <= 0:
            return
        
        # Check existing position
        positions = self.trade_executor.get_positions()
        current_position = positions.get(prediction.symbol, {})
        
        action = 'buy' if prediction.predicted_direction == PredictionDirection.UP else 'sell'
        
        # Don't buy if already have position, don't sell if no position
        if action == 'buy' and current_position.get('quantity', 0) > 0:
            return
        if action == 'sell' and current_position.get('quantity', 0) <= 0:
            return
        
        signal = TradeSignal(
            symbol=prediction.symbol,
            action=action,
            confidence=prediction.confidence,
            predicted_change_pct=prediction.predicted_change_pct,
            horizon=prediction.horizon,
            suggested_quantity=quantity,
            suggested_position_pct=position_value / equity,
            stop_loss_pct=self.config.default_stop_loss_pct,
            take_profit_pct=self.config.default_take_profit_pct,
            reasoning=f"{prediction.horizon.value} prediction: {prediction.predicted_direction.value} with {prediction.confidence:.1%} confidence",
            features=features
        )
        
        self.signal_queue.put(signal)
        self.daily_stats['signals_generated'] += 1
        
        logger.info(f"Trade signal generated: {action.upper()} {quantity} {prediction.symbol}")
        
        if self.on_trade_signal:
            self.on_trade_signal(signal)
    
    def _process_trade_signals(self):
        """Process pending trade signals."""
        
        while not self.signal_queue.empty():
            try:
                signal = self.signal_queue.get_nowait()
                
                # Execute trade
                order_id = self.trade_executor.execute_trade(signal)
                
                if order_id:
                    signal.executed = True
                    signal.order_id = order_id
                    self.daily_stats['trades_executed'] += 1
                    
                    if self.on_trade_executed:
                        self.on_trade_executed(signal)
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing trade signal: {e}")
    
    # =========================================================================
    # MAIN LOOPS
    # =========================================================================
    
    def _data_loop(self):
        """Background loop for refreshing market data."""
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time_module.sleep(1)
                continue
            
            if self._is_market_hours():
                self.data_manager.refresh_data()
                self.last_data_refresh = datetime.now()
            
            time_module.sleep(self.config.data_refresh_interval_seconds)
    
    def _prediction_loop(self):
        """Background loop for making predictions."""
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time_module.sleep(1)
                continue
            
            if not self._is_market_hours():
                time_module.sleep(60)
                continue
            
            # Check if it's time for predictions
            should_predict = (
                self.last_prediction_time is None or
                (datetime.now() - self.last_prediction_time) >= 
                timedelta(minutes=self.config.prediction_interval_minutes)
            )
            
            if should_predict:
                try:
                    self._make_predictions()
                except Exception as e:
                    logger.error(f"Error in prediction loop: {e}")
                    if self.on_error:
                        self.on_error(e)
            
            time_module.sleep(60)  # Check every minute
    
    def _verification_loop(self):
        """Background loop for verifying predictions."""
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time_module.sleep(1)
                continue
            
            # Check if it's time to verify
            should_verify = (
                self.last_verification_time is None or
                (datetime.now() - self.last_verification_time) >=
                timedelta(minutes=self.config.verification_interval_minutes)
            )
            
            if should_verify:
                try:
                    self._verify_predictions()
                except Exception as e:
                    logger.error(f"Error in verification loop: {e}")
                    if self.on_error:
                        self.on_error(e)
            
            time_module.sleep(30)  # Check every 30 seconds
    
    def _trading_loop(self):
        """Background loop for processing trade signals."""
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time_module.sleep(1)
                continue
            
            if self.config.trading_mode == TradingMode.LEARNING_ONLY:
                time_module.sleep(10)
                continue
            
            if not self._is_market_hours():
                time_module.sleep(60)
                continue
            
            try:
                self._process_trade_signals()
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                if self.on_error:
                    self.on_error(e)
            
            time_module.sleep(5)  # Check every 5 seconds
    
    # =========================================================================
    # SERVICE CONTROL
    # =========================================================================
    
    def start(self):
        """Start the background service."""
        if self.state == ServiceState.RUNNING:
            logger.warning("Service already running")
            return
        
        logger.info("Starting background trading service...")
        self.state = ServiceState.STARTING
        
        # Initialize components
        self._init_components()
        
        # Clear events
        self._stop_event.clear()
        self._pause_event.clear()
        
        # Start threads
        self._threads = [
            threading.Thread(target=self._data_loop, name="DataLoop", daemon=True),
            threading.Thread(target=self._prediction_loop, name="PredictionLoop", daemon=True),
            threading.Thread(target=self._verification_loop, name="VerificationLoop", daemon=True),
            threading.Thread(target=self._trading_loop, name="TradingLoop", daemon=True),
        ]
        
        for thread in self._threads:
            thread.start()
        
        self.state = ServiceState.RUNNING
        logger.info("Background trading service started")
    
    def stop(self):
        """Stop the background service."""
        if self.state == ServiceState.STOPPED:
            return
        
        logger.info("Stopping background trading service...")
        self.state = ServiceState.STOPPING
        
        self._stop_event.set()
        
        # Wait for threads
        for thread in self._threads:
            thread.join(timeout=5)
        
        self._threads = []
        self.state = ServiceState.STOPPED
        logger.info("Background trading service stopped")
    
    def pause(self):
        """Pause the service (stops new predictions/trades but keeps running)."""
        self._pause_event.set()
        self.state = ServiceState.PAUSED
        logger.info("Service paused")
    
    def resume(self):
        """Resume the service."""
        self._pause_event.clear()
        self.state = ServiceState.RUNNING
        logger.info("Service resumed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        account = self.trade_executor.get_account() if self.trade_executor else {}
        
        current_equity = account.get('equity', 0)
        starting_equity = self.daily_stats['starting_equity']
        daily_pl = current_equity - starting_equity if starting_equity > 0 else 0
        daily_pl_pct = (daily_pl / starting_equity * 100) if starting_equity > 0 else 0
        
        verified = self.daily_stats['predictions_verified']
        correct = self.daily_stats['correct_predictions']
        accuracy = correct / verified if verified > 0 else 0
        
        return {
            'state': self.state.value,
            'trading_mode': self.config.trading_mode.value,
            'active_symbols': len([s for s in self.config.symbols if s not in self.config.excluded_symbols]),
            'market_open': self._is_market_hours(),
            'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'last_verification': self.last_verification_time.isoformat() if self.last_verification_time else None,
            'daily_stats': {
                'predictions_made': self.daily_stats['predictions_made'],
                'predictions_verified': verified,
                'accuracy': accuracy,
                'signals_generated': self.daily_stats['signals_generated'],
                'trades_executed': self.daily_stats['trades_executed'],
                'daily_pl': daily_pl,
                'daily_pl_pct': daily_pl_pct
            },
            'account': {
                'equity': current_equity,
                'cash': account.get('cash', 0),
                'buying_power': account.get('buying_power', 0)
            }
        }
    
    # =========================================================================
    # SYMBOL MANAGEMENT
    # =========================================================================
    
    def add_symbol(self, symbol: str, train_first: bool = True) -> Dict[str, Any]:
        """
        Add a symbol to monitoring.
        
        If train_first=True, will fast-train on historical data first.
        """
        if symbol in self.config.symbols:
            return {'success': False, 'reason': 'Symbol already being monitored'}
        
        result = {'symbol': symbol, 'success': True}
        
        if train_first:
            # Fast train
            trainer = HistoricalTrainer(
                symbols=[symbol],
                api_key=self.api_key,
                api_secret=self.api_secret,
                db_path=str(self.data_dir / "predictions.db")
            )
            
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
            
            train_result = trainer.train_on_historical(start_date, end_date)
            
            profile = trainer.profiles.get(symbol)
            if profile:
                self.profiles[symbol] = profile
                result['predictions'] = profile.total_predictions
                result['accuracy'] = profile.overall_accuracy
                result['ready_to_trade'] = (
                    profile.total_predictions >= self.config.min_predictions_to_trade and
                    profile.overall_accuracy >= self.config.min_accuracy_to_trade
                )
        
        # Add to config
        self.config.symbols.append(symbol)
        
        # Update data manager
        if self.data_manager:
            self.data_manager.update_symbols(
                [s for s in self.config.symbols if s not in self.config.excluded_symbols]
            )
        
        logger.info(f"Added symbol: {symbol}")
        return result
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from monitoring."""
        if symbol in self.config.symbols:
            self.config.symbols.remove(symbol)
        
        if self.data_manager:
            self.data_manager.update_symbols(
                [s for s in self.config.symbols if s not in self.config.excluded_symbols]
            )
        
        logger.info(f"Removed symbol: {symbol}")
    
    def exclude_symbol(self, symbol: str):
        """Exclude a symbol from trading today (still monitors)."""
        if symbol not in self.config.excluded_symbols:
            self.config.excluded_symbols.append(symbol)
        logger.info(f"Excluded symbol: {symbol}")
    
    def include_symbol(self, symbol: str):
        """Re-include a previously excluded symbol."""
        if symbol in self.config.excluded_symbols:
            self.config.excluded_symbols.remove(symbol)
        logger.info(f"Re-included symbol: {symbol}")
    
    def set_trading_mode(self, mode: TradingMode):
        """Change trading mode."""
        old_mode = self.config.trading_mode
        self.config.trading_mode = mode
        
        # Reinitialize executor if needed
        if self.trade_executor and mode == TradingMode.LIVE_TRADING:
            self.trade_executor = TradeExecutor(
                self.api_key, self.api_secret, paper=False
            )
        elif self.trade_executor and mode != TradingMode.LIVE_TRADING:
            self.trade_executor = TradeExecutor(
                self.api_key, self.api_secret, paper=True
            )
        
        logger.info(f"Trading mode changed: {old_mode.value} -> {mode.value}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_service(
    symbols: List[str] = None,
    trading_mode: TradingMode = None,  # None means use config.yaml
    **kwargs
) -> BackgroundTradingService:
    """Create a configured background trading service."""

    default_symbols = symbols or [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "META", "NVDA", "AMD", "SPY", "QQQ"
    ]

    # Create config - trading_mode is loaded from config.yaml if not specified
    service_config = ServiceConfig(
        symbols=default_symbols,
        **kwargs
    )

    # Override trading mode if explicitly provided
    if trading_mode is not None:
        service_config.trading_mode = trading_mode

    logger.info(f"Creating service with trading mode: {service_config.trading_mode.value}")

    return BackgroundTradingService(config=service_config)


def run_service():
    """Run the service (blocking)."""
    service = create_service()
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start service
    service.start()
    
    # Keep running
    while service.state == ServiceState.RUNNING:
        time_module.sleep(1)


if __name__ == "__main__":
    run_service()

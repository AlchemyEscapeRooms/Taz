"""
Trading Bot API Server (PRIMARY)
=================================

FastAPI backend that:
- Serves the dashboard
- Provides REST endpoints for bot data
- WebSocket for real-time updates
- Connects to the background trading service

Run with: uvicorn api_server:app --reload --port 8000

Author: Claude AI
Date: November 29, 2025

==============================================================================
API CONSOLIDATION NOTE
==============================================================================
There are TWO API servers in this project:

1. api_server.py (THIS FILE) - PRIMARY
   - Port: 8000
   - Features: Full BackgroundTradingService integration, stock management,
               backtest API, learning profiles, trade signals
   - Endpoints: /api/service/*, /api/portfolio, /api/predictions/*,
                /api/learning/*, /api/stocks/*, /api/backtest/*, /api/signals

2. web_api.py - ALTERNATIVE (Dashboard-focused)
   - Port: 8000 (CONFLICT - cannot run simultaneously!)
   - Features: Simpler API focused on alchemy_dashboard.html
   - Endpoints: /api/status, /api/brain, /api/pnl, /api/trades, /api/bot/control

RECOMMENDATION:
- Use api_server.py as the primary API server
- web_api.py endpoints could be merged into this file if needed
- Do NOT run both servers at the same time (port conflict)

TODO: Consider merging web_api.py endpoints into this file for consolidation.
==============================================================================
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Global event loop reference for thread-safe callbacks
main_event_loop: Optional[asyncio.AbstractEventLoop] = None
from background_service import (
    BackgroundTradingService,
    ServiceConfig,
    ServiceState,
    TradingMode
)
# Legacy prediction system (being phased out in favor of RL)
try:
    from learning_trader import PredictionDatabase, StockLearningProfile
    LEGACY_PREDICTION_AVAILABLE = True
except ImportError:
    LEGACY_PREDICTION_AVAILABLE = False
    PredictionDatabase = None
    StockLearningProfile = None

try:
    from historical_trainer import HistoricalTrainer
except ImportError:
    HistoricalTrainer = None
from simple_trader import (
    SimpleTrader,
    STRATEGIES,
    select_best_strategy,
    backtest_strategy,
    run_detailed_backtest,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger,
    calculate_volume_ratio,
    calculate_sma,
    get_portfolio_symbols
)
from utils.logger import get_logger
from utils.database import Database
from config import config

# Taz Trading System
try:
    from Taz.taz_trader import TazTrader
    from Taz.scanner.taz_scanner import TazScanner
    TAZ_AVAILABLE = True
except ImportError:
    TAZ_AVAILABLE = False
    TazTrader = None
    TazScanner = None

logger = get_logger(__name__)

# Database instance for storing backtest results
backtest_db = Database()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class StockCommand(BaseModel):
    symbol: str

class TradingModeCommand(BaseModel):
    mode: str  # 'learning', 'paper', 'live'

class BacktestRequest(BaseModel):
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float = 100000
    mode: str = "test"  # 'learn' or 'test'

class UserCommand(BaseModel):
    command: str

class BotCommand(BaseModel):
    action: str  # 'start', 'stop', 'pause'

class SettingsUpdate(BaseModel):
    min_confidence: Optional[float] = None
    max_position_size: Optional[float] = None
    max_daily_loss: Optional[float] = None
    stop_loss: Optional[float] = None
    max_drawdown: Optional[float] = None
    initial_capital: Optional[float] = None
    max_positions: Optional[int] = None
    max_trades_per_day: Optional[int] = None

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="Alchemy Trading Bot API",
    description="API for the learning trading bot",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
trading_service: Optional[BackgroundTradingService] = None
simple_trader: Optional[SimpleTrader] = None  # New simple trader instance
taz_trader: Optional[TazTrader] = None  # Taz aggressive trading system
websocket_clients: List[WebSocket] = []
data_dir = Path("data")

# KILL SWITCH - Emergency stop for all trading
kill_switch_active: bool = False
kill_switch_reason: Optional[str] = None
kill_switch_time: Optional[datetime] = None

# ============================================================================
# SERVICE MANAGEMENT
# ============================================================================

def get_simple_trader() -> SimpleTrader:
    """Get or create the SimpleTrader instance."""
    global simple_trader

    if simple_trader is None:
        # Determine paper mode from config
        trading_mode = config.get('trading.mode', 'paper')
        paper = trading_mode != 'live'

        # Get symbols from Alpaca portfolio (the stocks you actually own)
        symbols = get_portfolio_symbols(paper=paper)

        # Fallback to config if portfolio is empty
        if not symbols:
            logger.warning("No positions in Alpaca portfolio, falling back to config symbols")
            symbols = config.get('data.universe.initial_stocks',
                                ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "SPY", "QQQ"])

        simple_trader = SimpleTrader(
            symbols=symbols,
            paper=paper,
            calibration_days=config.get('simple_trader.calibration_days', 90),
            recalibrate_hours=config.get('simple_trader.recalibrate_hours', 24),
            position_size_pct=config.get('simple_trader.position_size_pct', 0.15),
            max_positions=config.get('simple_trader.max_positions', 15),
            min_hold_hours=config.get('simple_trader.min_hold_hours', 4)
        )

        # Load previous state
        simple_trader._load_state()

        logger.info(f"SimpleTrader initialized with {len(symbols)} symbols from portfolio ({'PAPER' if paper else 'LIVE'})")

    return simple_trader


def get_service() -> BackgroundTradingService:
    """Get or create the trading service."""
    global trading_service

    if trading_service is None:
        # Configuration - trading_mode now loaded from config.yaml automatically
        service_config = ServiceConfig(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "SPY", "QQQ"],
            # trading_mode is now set from config.yaml via ServiceConfig default
            prediction_interval_minutes=60,
            verification_interval_minutes=5,
        )

        logger.info(f"Trading mode from config: {service_config.trading_mode.value}")

        # Get API keys from environment (loaded from .env)
        api_key = os.environ.get('ALPACA_API_KEY') or config.get('alpaca.api_key')
        api_secret = os.environ.get('ALPACA_SECRET_KEY') or config.get('alpaca.api_secret')

        if not api_key or not api_secret:
            logger.error("Alpaca API keys not found! Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
            raise ValueError("Alpaca API keys not configured")

        trading_service = BackgroundTradingService(
            config=service_config,
            api_key=api_key,
            api_secret=api_secret,
            data_dir=str(data_dir)
        )
        
        # Set up callbacks for WebSocket broadcasts (thread-safe)
        def on_prediction_callback(p):
            if main_event_loop and main_event_loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_event('prediction', {
                    'symbol': p.symbol,
                    'horizon': p.horizon.value,
                    'direction': p.predicted_direction.value,
                    'confidence': p.confidence
                }), main_event_loop)

        def on_trade_signal_callback(s):
            if main_event_loop and main_event_loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_event('trade_signal', {
                    'symbol': s.symbol,
                    'action': s.action,
                    'confidence': s.confidence
                }), main_event_loop)

        trading_service.on_prediction = on_prediction_callback
        trading_service.on_trade_signal = on_trade_signal_callback
    
    return trading_service

async def broadcast_event(event_type: str, data: Dict):
    """Broadcast event to all connected WebSocket clients."""
    message = json.dumps({
        'type': event_type,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })
    
    disconnected = []
    for client in websocket_clients:
        try:
            await client.send_text(message)
        except:
            disconnected.append(client)
    
    # Clean up disconnected clients
    for client in disconnected:
        websocket_clients.remove(client)

# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()

    logger.info("API Server starting up...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SimpleTrader (the primary trading system)
    try:
        trader = get_simple_trader()
        logger.info(f"SimpleTrader initialized with {len(trader.symbols)} symbols from portfolio")
    except Exception as e:
        logger.error(f"Failed to initialize SimpleTrader: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global simple_trader
    if simple_trader and simple_trader.running:
        simple_trader.stop()
    logger.info("API Server shut down")

# ============================================================================
# STATIC FILES & DASHBOARD
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard."""
    # Try alchemy_dashboard.html first (the full-featured dashboard)
    dashboard_path = Path(__file__).parent / "alchemy_dashboard.html"

    if dashboard_path.exists():
        return FileResponse(dashboard_path)

    # Fallback to static folder
    static_path = Path(__file__).parent / "static" / "dashboard.html"
    if static_path.exists():
        return FileResponse(static_path)

    # Return embedded dashboard if no files exist
    return HTMLResponse(content=get_embedded_dashboard(), status_code=200)

# ============================================================================
# SERVICE CONTROL ENDPOINTS
# ============================================================================

@app.post("/api/service/start")
async def start_service():
    """Start the background trading service."""
    service = get_service()
    
    if service.state == ServiceState.RUNNING:
        return {"status": "already_running", "message": "Service is already running"}
    
    service.start()
    return {"status": "started", "message": "Trading service started"}

@app.post("/api/service/stop")
async def stop_service():
    """Stop the background trading service."""
    service = get_service()
    
    if service.state == ServiceState.STOPPED:
        return {"status": "already_stopped", "message": "Service is already stopped"}
    
    service.stop()
    return {"status": "stopped", "message": "Trading service stopped"}

@app.post("/api/service/pause")
async def pause_service():
    """Pause the trading service."""
    service = get_service()
    service.pause()
    return {"status": "paused", "message": "Trading service paused"}

@app.post("/api/service/resume")
async def resume_service():
    """Resume the trading service."""
    service = get_service()
    service.resume()
    return {"status": "resumed", "message": "Trading service resumed"}

@app.get("/api/service/status")
async def get_service_status():
    """Get current service status."""
    service = get_service()
    return service.get_status()

# ============================================================================
# BOT CONTROL ENDPOINT (compatibility with alchemy_dashboard.html)
# ============================================================================

@app.post("/api/bot/control")
async def control_bot(command: BotCommand):
    """Control the trading bot (start/stop/pause) - compatibility endpoint."""
    service = get_service()

    try:
        if command.action == 'start':
            if service.state != ServiceState.RUNNING:
                service.start()
            return {'success': True, 'status': 'running', 'state': service.state.value}

        elif command.action == 'stop':
            if service.state != ServiceState.STOPPED:
                service.stop()
            return {'success': True, 'status': 'stopped', 'state': service.state.value}

        elif command.action == 'pause':
            service.pause()
            return {'success': True, 'status': 'paused', 'state': service.state.value}

        elif command.action == 'resume':
            service.resume()
            return {'success': True, 'status': 'running', 'state': service.state.value}

        else:
            return {'success': False, 'error': f'Unknown action: {command.action}'}

    except Exception as e:
        logger.error(f"Error controlling bot: {e}")
        return {'success': False, 'error': str(e)}

# ============================================================================
# KILL SWITCH ENDPOINTS - EMERGENCY TRADING HALT
# ============================================================================

class KillSwitchRequest(BaseModel):
    reason: Optional[str] = "Manual activation"

@app.post("/api/killswitch/activate")
async def activate_kill_switch(request: KillSwitchRequest = None):
    """
    EMERGENCY: Immediately halt ALL trading activity.

    This will:
    1. Stop all trading immediately
    2. Optionally close all open positions (if close_positions=True)
    3. Prevent any new orders until deactivated

    Use this when:
    - Unusual market conditions
    - System malfunction detected
    - Manual override needed
    """
    global kill_switch_active, kill_switch_reason, kill_switch_time, simple_trader

    kill_switch_active = True
    kill_switch_reason = request.reason if request else "Manual activation"
    kill_switch_time = datetime.now()

    # Stop SimpleTrader if running
    if simple_trader and simple_trader.running:
        simple_trader.stop()

    # Stop background service if running
    if trading_service and trading_service.state != ServiceState.STOPPED:
        trading_service.stop()

    logger.critical(f"KILL SWITCH ACTIVATED: {kill_switch_reason}")

    # Broadcast to all websocket clients
    await broadcast_event('kill_switch', {
        'active': True,
        'reason': kill_switch_reason,
        'time': kill_switch_time.isoformat()
    })

    return {
        "success": True,
        "kill_switch_active": True,
        "reason": kill_switch_reason,
        "time": kill_switch_time.isoformat(),
        "message": "KILL SWITCH ACTIVATED - All trading halted"
    }

@app.post("/api/killswitch/deactivate")
async def deactivate_kill_switch():
    """
    Deactivate the kill switch and allow trading to resume.

    WARNING: Ensure market conditions are safe before deactivating.
    """
    global kill_switch_active, kill_switch_reason, kill_switch_time

    if not kill_switch_active:
        return {
            "success": False,
            "message": "Kill switch is not active"
        }

    previous_reason = kill_switch_reason
    previous_time = kill_switch_time

    kill_switch_active = False
    kill_switch_reason = None
    kill_switch_time = None

    logger.info(f"Kill switch deactivated. Was active since {previous_time} for: {previous_reason}")

    # Broadcast to all websocket clients
    await broadcast_event('kill_switch', {
        'active': False,
        'deactivated_at': datetime.now().isoformat()
    })

    return {
        "success": True,
        "kill_switch_active": False,
        "message": "Kill switch deactivated - Trading can resume",
        "was_active_since": previous_time.isoformat() if previous_time else None,
        "was_reason": previous_reason
    }

@app.get("/api/killswitch/status")
async def get_kill_switch_status():
    """Get current kill switch status."""
    return {
        "active": kill_switch_active,
        "reason": kill_switch_reason,
        "activated_at": kill_switch_time.isoformat() if kill_switch_time else None,
        "duration_minutes": (datetime.now() - kill_switch_time).total_seconds() / 60 if kill_switch_time else None
    }

@app.post("/api/killswitch/close-all-positions")
async def close_all_positions():
    """
    EMERGENCY: Close ALL open positions immediately.

    Use with caution - this will liquidate everything at market price.
    """
    global simple_trader

    if simple_trader is None:
        return {"success": False, "error": "SimpleTrader not initialized"}

    try:
        positions = simple_trader.get_current_positions()
        closed = []
        failed = []

        for symbol in list(positions.keys()):
            try:
                success = simple_trader.execute_sell(symbol, sell_reason="EMERGENCY LIQUIDATION - Kill Switch")
                if success:
                    closed.append(symbol)
                else:
                    failed.append(symbol)
            except Exception as e:
                failed.append(f"{symbol}: {str(e)}")

        logger.critical(f"Emergency liquidation: Closed {len(closed)}, Failed {len(failed)}")

        return {
            "success": True,
            "closed_positions": closed,
            "failed_positions": failed,
            "message": f"Closed {len(closed)} positions, {len(failed)} failed"
        }
    except Exception as e:
        logger.error(f"Emergency liquidation failed: {e}")
        return {"success": False, "error": str(e)}

# ============================================================================
# RL SHADOW & PROMOTION ENDPOINTS
# ============================================================================

@app.get("/api/rl/shadow-stats")
async def get_rl_shadow_stats():
    """Get RL shadow trading statistics."""
    try:
        from rl_shadow import get_rl_shadow
        shadow = get_rl_shadow()
        return shadow.get_stats()
    except Exception as e:
        logger.error(f"Error getting RL shadow stats: {e}")
        return {"error": str(e), "models_loaded": 0}

@app.get("/api/rl/promotion-check/{symbol}")
async def check_rl_promotion(symbol: str):
    """
    Check if an RL model is ready for promotion from shadow to live trading.

    This is a safety gate that requires:
    1. Walk-forward validation passed
    2. Minimum shadow trading period
    3. Positive simulated returns
    """
    try:
        from rl_system.walk_forward import WalkForwardValidator, PromotionGate
        from rl_shadow import get_rl_shadow

        shadow = get_rl_shadow()
        shadow_stats = shadow.get_stats()

        validator = WalkForwardValidator()
        validation_results = validator.load_results(symbol.upper())

        if validation_results is None:
            return {
                "symbol": symbol,
                "ready_for_promotion": False,
                "reason": "No walk-forward validation results found. Run validation first.",
                "next_step": "POST /api/rl/validate/{symbol}"
            }

        gate = PromotionGate()
        result = gate.check_promotion_ready(symbol.upper(), validation_results, shadow_stats)
        return result

    except Exception as e:
        logger.error(f"Error checking RL promotion: {e}")
        return {"error": str(e), "ready_for_promotion": False}

@app.get("/api/rl/validation-results/{symbol}")
async def get_rl_validation_results(symbol: str):
    """Get walk-forward validation results for a symbol."""
    try:
        from rl_system.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator()
        results = validator.load_results(symbol.upper())

        if results is None:
            return {"symbol": symbol, "validated": False, "message": "No validation results found"}
        return results

    except Exception as e:
        logger.error(f"Error getting validation results: {e}")
        return {"error": str(e)}

# ============================================================================
# CIRCUIT BREAKER & RISK CONTROL ENDPOINTS
# ============================================================================

@app.get("/api/risk/status")
async def get_risk_status():
    """Get current risk control status including circuit breakers."""
    try:
        trader = get_simple_trader()
        status = trader.get_status()
        risk = status.get('risk', {})

        return {
            "success": True,
            "circuit_breaker": {
                "active": risk.get('circuit_breaker_active', False),
                "reason": risk.get('circuit_breaker_reason')
            },
            "daily_pnl": risk.get('daily_pnl', 0),
            "daily_pnl_pct": risk.get('daily_pnl_pct', 0),
            "daily_loss_limit_pct": risk.get('daily_loss_limit', 5),
            "max_position_size_pct": risk.get('max_position_size', 10),
            "consecutive_losses": risk.get('consecutive_losses', 0),
            "consecutive_wins": risk.get('consecutive_wins', 0),
            "kill_switch_active": kill_switch_active
        }
    except Exception as e:
        logger.error(f"Error getting risk status: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/risk/reset-circuit-breaker")
async def reset_circuit_breaker():
    """Reset the circuit breaker to allow trading to resume."""
    try:
        trader = get_simple_trader()
        trader.reset_circuit_breaker()
        return {
            "success": True,
            "message": "Circuit breaker reset - trading allowed"
        }
    except Exception as e:
        logger.error(f"Error resetting circuit breaker: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/risk/reset-daily")
async def reset_daily_risk():
    """Reset daily risk tracking (typically done at market open)."""
    try:
        trader = get_simple_trader()
        trader.reset_daily_risk()
        return {
            "success": True,
            "message": "Daily risk tracking reset"
        }
    except Exception as e:
        logger.error(f"Error resetting daily risk: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/status")
async def get_status():
    """Get bot status from SimpleTrader."""
    try:
        trader = get_simple_trader()
        status = trader.get_status()
        account = status.get('account', {})

        # Determine actual state
        if kill_switch_active:
            state = 'killed'
        elif trader.running:
            state = 'running'
        else:
            state = 'ready'

        return {
            'state': state,
            'bot_active': trader.running and not kill_switch_active,
            'trading_mode': 'paper' if trader.paper else 'live',
            'symbols_count': len(trader.symbols),
            'predictions_today': 0,  # SimpleTrader doesn't make predictions
            'trades_today': status.get('stats', {}).get('trades_executed', 0),
            'uptime': 'N/A',
            'active_symbols': len(trader.symbols),
            'market_open': trader._is_market_hours(),
            'kill_switch': {
                'active': kill_switch_active,
                'reason': kill_switch_reason,
                'activated_at': kill_switch_time.isoformat() if kill_switch_time else None
            },
            'daily_stats': {
                'predictions_made': 0,
                'predictions_verified': 0,
                'accuracy': 0,
                'signals_generated': 0,
                'trades_executed': status.get('stats', {}).get('trades_executed', 0),
                'daily_pl': status.get('stats', {}).get('total_pnl', 0),
                'daily_pl_pct': 0
            },
            'account': account,
            'strategies': status.get('strategies', {})
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {'state': 'error', 'bot_active': False, 'error': str(e), 'kill_switch': {'active': kill_switch_active}}

# ============================================================================
# PORTFOLIO & ACCOUNT ENDPOINTS
# ============================================================================

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio positions from SimpleTrader."""
    try:
        trader = get_simple_trader()
        positions = trader.get_current_positions()
        account = trader.get_account()

        # Format positions for frontend
        formatted_positions = []
        for symbol, data in positions.items():
            formatted_positions.append({
                "symbol": symbol,
                "quantity": data['shares'],
                "avgCost": data['avg_cost'],
                "currentPrice": data['current_price'],
                "marketValue": data['shares'] * data['current_price'],
                "unrealizedPL": data['unrealized_pnl'],
                "unrealizedPLPct": (data['unrealized_pnl'] / (data['shares'] * data['avg_cost']) * 100) if data['shares'] > 0 else 0
            })

        return {
            "positions": formatted_positions,
            "account": {
                "equity": account.get('equity', 0),
                "cash": account.get('cash', 0),
                "buyingPower": account.get('buying_power', 0)
            }
        }
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return {"positions": [], "error": str(e)}

@app.get("/api/account")
async def get_account():
    """Get account information from SimpleTrader."""
    try:
        trader = get_simple_trader()
        return trader.get_account()
    except Exception as e:
        logger.error(f"Error getting account: {e}")
        return {"error": str(e)}

# ============================================================================
# PREDICTIONS & LEARNING ENDPOINTS
# ============================================================================

@app.get("/api/predictions/active")
async def get_active_predictions():
    """Get currently active (unverified) predictions."""
    if not LEGACY_PREDICTION_AVAILABLE:
        return {"predictions": [], "message": "Legacy prediction system disabled - using RL instead"}
    db = PredictionDatabase(str(data_dir / "predictions.db"))
    
    # Get predictions that haven't been verified yet
    pending = db.get_pending_predictions(before_time=datetime.now() + timedelta(days=1))
    
    # Format for frontend
    formatted = []
    for pred in pending[:50]:  # Limit to 50 most recent
        formatted.append({
            "id": pred['id'],
            "symbol": pred['symbol'],
            "horizon": pred['horizon'],
            "direction": pred['predicted_direction'],
            "predictedChange": pred['predicted_change_pct'],
            "confidence": pred['confidence'],
            "predictionTime": pred['prediction_time'],
            "targetTime": pred['target_time']
        })
    
    return {"predictions": formatted}

@app.get("/api/predictions/stats")
async def get_prediction_stats(days: int = 7):
    """Get prediction statistics."""
    if not LEGACY_PREDICTION_AVAILABLE:
        return {"period_days": days, "by_horizon": {}, "message": "Legacy prediction system disabled"}
    db = PredictionDatabase(str(data_dir / "predictions.db"))
    stats = db.get_prediction_stats(days=days)

    return {
        "period_days": days,
        "by_horizon": stats
    }

@app.get("/api/learning/stocks")
async def get_stock_learning_profiles():
    """Get learning profiles for all stocks."""
    if not LEGACY_PREDICTION_AVAILABLE:
        return {"profiles": [], "message": "Legacy prediction system disabled - using RL instead"}
    service = get_service()
    db = PredictionDatabase(str(data_dir / "predictions.db"))
    
    profiles = []
    # Get thresholds from config
    min_predictions = config.get('learning.min_predictions_to_trade', 100)
    min_accuracy = config.get('learning.min_accuracy_to_trade', 0.55)

    for symbol in service.config.symbols:
        profile = db.get_stock_profile(symbol)

        if profile:
            ready = (
                profile.total_predictions >= min_predictions and
                profile.overall_accuracy >= min_accuracy
            )
            
            profiles.append({
                "symbol": symbol,
                "totalPredictions": profile.total_predictions,
                "accuracy": profile.overall_accuracy,
                "accuracy1h": profile.accuracy_1h,
                "accuracyEod": profile.accuracy_eod,
                "accuracyNextDay": profile.accuracy_next_day,
                "ready": ready,
                "status": "ready" if ready else ("learning" if profile.total_predictions > 0 else "new"),
                "excluded": symbol in service.config.excluded_symbols
            })
        else:
            profiles.append({
                "symbol": symbol,
                "totalPredictions": 0,
                "accuracy": 0,
                "accuracy1h": 0.5,
                "accuracyEod": 0.5,
                "accuracyNextDay": 0.5,
                "ready": False,
                "status": "new",
                "excluded": symbol in service.config.excluded_symbols
            })
    
    return {"stocks": profiles}

# ============================================================================
# STOCK MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/api/stocks/add")
async def add_stock(command: StockCommand, background_tasks: BackgroundTasks):
    """Add a new stock to monitoring."""
    service = get_service()
    symbol = command.symbol.upper()
    
    if symbol in service.config.symbols:
        return {"success": False, "message": f"{symbol} is already being monitored"}
    
    # Add and train in background
    result = service.add_symbol(symbol, train_first=True)
    
    return {
        "success": result.get('success', False),
        "symbol": symbol,
        "predictions": result.get('predictions', 0),
        "accuracy": result.get('accuracy', 0),
        "readyToTrade": result.get('ready_to_trade', False),
        "message": f"Added {symbol} to watchlist"
    }

@app.post("/api/stocks/remove")
async def remove_stock(command: StockCommand):
    """Remove a stock from monitoring."""
    service = get_service()
    symbol = command.symbol.upper()
    
    if symbol not in service.config.symbols:
        return {"success": False, "message": f"{symbol} is not being monitored"}
    
    service.remove_symbol(symbol)
    return {"success": True, "message": f"Removed {symbol} from watchlist"}

@app.post("/api/stocks/exclude")
async def exclude_stock(command: StockCommand):
    """Exclude a stock from trading today."""
    service = get_service()
    symbol = command.symbol.upper()
    
    service.exclude_symbol(symbol)
    return {"success": True, "message": f"Excluded {symbol} from trading today"}

@app.post("/api/stocks/include")
async def include_stock(command: StockCommand):
    """Re-include a previously excluded stock."""
    service = get_service()
    symbol = command.symbol.upper()
    
    service.include_symbol(symbol)
    return {"success": True, "message": f"Re-included {symbol} for trading"}

# ============================================================================
# STOCK SCANNER ENDPOINTS
# ============================================================================

# Scanner state
_scanner_instance = None
_scan_results = []
_scan_in_progress = False

def get_scanner():
    """Get or create scanner instance."""
    global _scanner_instance
    if _scanner_instance is None:
        from scanner.stock_scanner import StockScanner
        from scanner.sp500_provider import SP500Provider
        _scanner_instance = StockScanner(simple_trader=simple_trader)
    return _scanner_instance

class ScanRequest(BaseModel):
    batch_size: int = 50
    fetch_news: bool = True
    symbols: Optional[str] = None  # Comma-separated symbols

class AutoAddRequest(BaseModel):
    min_score: float = 60.0
    max_additions: int = 10

@app.get("/api/scanner/status")
async def get_scanner_status():
    """Get scanner status including news budget."""
    scanner = get_scanner()
    status = scanner.get_status()
    status['scan_in_progress'] = _scan_in_progress

    # Add trader watchlist info if available
    if simple_trader:
        status['watchlist'] = simple_trader.get_watchlist_capacity()

    return status

@app.get("/api/scanner/results")
async def get_scan_results(limit: int = 20, min_score: float = 0.0):
    """Get latest scan results."""
    scanner = get_scanner()
    results = scanner.get_top_discoveries(limit=100)

    # Filter by min score
    filtered = [r for r in results if r.composite_score >= min_score]

    return {
        "count": len(filtered[:limit]),
        "total_scanned": len(results),
        "results": [r.to_dict() for r in filtered[:limit]]
    }

@app.post("/api/scanner/run")
async def run_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    """
    Trigger a stock scan.

    Runs in background to avoid timeout.
    """
    global _scan_in_progress, _scan_results

    if _scan_in_progress:
        raise HTTPException(status_code=409, detail="Scan already in progress")

    scanner = get_scanner()

    # Get symbols
    if request.symbols:
        symbols = [s.strip().upper() for s in request.symbols.split(',')]
    else:
        from scanner.sp500_provider import SP500Provider
        provider = SP500Provider()
        symbols = provider.get_symbols()[:request.batch_size]

    def run_scan_task():
        global _scan_in_progress, _scan_results
        _scan_in_progress = True
        try:
            results = scanner.scan_batch(
                symbols[:request.batch_size],
                fetch_news=request.fetch_news
            )
            _scan_results = results
        finally:
            _scan_in_progress = False

    background_tasks.add_task(run_scan_task)

    return {
        "success": True,
        "message": f"Scan started for {min(len(symbols), request.batch_size)} symbols",
        "symbols_count": min(len(symbols), request.batch_size)
    }

@app.post("/api/scanner/auto-add")
async def auto_add_stocks(request: AutoAddRequest):
    """Auto-add top scoring stocks to SimpleTrader."""
    global simple_trader

    if simple_trader is None:
        raise HTTPException(status_code=400, detail="SimpleTrader not initialized. Start it first.")

    scanner = get_scanner()
    scanner.simple_trader = simple_trader

    added = scanner.auto_add_to_trader(
        min_score=request.min_score,
        max_additions=request.max_additions
    )

    # Record additions to history
    for symbol in added:
        # Find score from results
        score = None
        for r in scanner.latest_results:
            if r.symbol == symbol:
                score = r.composite_score
                break
        record_scanner_add(symbol, score)

    return {
        "success": True,
        "added": added,
        "count": len(added),
        "message": f"Added {len(added)} stocks to trader" if added else "No stocks met criteria"
    }

@app.get("/api/scanner/sp500")
async def get_sp500_symbols():
    """Get the S&P 500 symbol list."""
    from scanner.sp500_provider import SP500Provider
    provider = SP500Provider()
    symbols = provider.get_symbols()
    return {
        "count": len(symbols),
        "symbols": symbols
    }

# Scanner history storage
_scanner_history = {
    'added': [],    # [{symbol, time, score}, ...]
    'removed': []   # [{symbol, time, reason}, ...]
}

def record_scanner_add(symbol: str, score: float = None):
    """Record a stock addition from scanner."""
    from datetime import datetime
    _scanner_history['added'].insert(0, {
        'symbol': symbol,
        'time': datetime.now().isoformat(),
        'score': score
    })
    # Keep only last 20
    _scanner_history['added'] = _scanner_history['added'][:20]

def record_scanner_remove(symbol: str, reason: str = None):
    """Record a stock removal."""
    from datetime import datetime
    _scanner_history['removed'].insert(0, {
        'symbol': symbol,
        'time': datetime.now().isoformat(),
        'reason': reason
    })
    # Keep only last 20
    _scanner_history['removed'] = _scanner_history['removed'][:20]

@app.get("/api/scanner/history")
async def get_scanner_history():
    """Get history of stocks added/removed by scanner."""
    return {
        "added": _scanner_history['added'][:10],
        "removed": _scanner_history['removed'][:10]
    }

@app.post("/api/scanner/add-stock")
async def scanner_add_stock(command: StockCommand):
    """Manually add a stock from scanner results."""
    global simple_trader

    if simple_trader is None:
        raise HTTPException(status_code=400, detail="SimpleTrader not initialized")

    result = simple_trader.add_symbol(command.symbol.upper())

    if result.get('success'):
        record_scanner_add(command.symbol.upper())

    return result

@app.post("/api/scanner/remove-stock")
async def scanner_remove_stock(command: StockCommand):
    """Remove a stock from the watchlist."""
    global simple_trader

    if simple_trader is None:
        raise HTTPException(status_code=400, detail="SimpleTrader not initialized")

    result = simple_trader.remove_symbol(command.symbol.upper())

    if result.get('success'):
        record_scanner_remove(command.symbol.upper(), "Manual removal")

    return result

# ============================================================================
# TRADING MODE ENDPOINTS
# ============================================================================

@app.post("/api/trading/mode")
async def set_trading_mode(command: TradingModeCommand):
    """Set the trading mode."""
    service = get_service()
    
    mode_map = {
        'learning': TradingMode.LEARNING_ONLY,
        'paper': TradingMode.PAPER_TRADING,
        'live': TradingMode.LIVE_TRADING
    }
    
    if command.mode not in mode_map:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {command.mode}")
    
    if command.mode == 'live':
        return {
            "success": False,
            "requiresConfirmation": True,
            "message": "Live trading requires confirmation. Use /api/trading/confirm-live"
        }
    
    service.set_trading_mode(mode_map[command.mode])
    return {"success": True, "mode": command.mode}

@app.post("/api/trading/confirm-live")
async def confirm_live_trading():
    """Confirm and enable live trading."""
    service = get_service()
    service.set_trading_mode(TradingMode.LIVE_TRADING)
    return {"success": True, "mode": "live", "warning": "LIVE TRADING ENABLED - Real money will be used"}

# ============================================================================
# SETTINGS ENDPOINTS
# ============================================================================

@app.get("/api/settings")
async def get_settings():
    """Get current bot settings."""
    try:
        return {
            "success": True,
            "settings": {
                "trading_mode": config.get('trading.mode', 'paper'),
                "min_confidence": config.get('learning.min_confidence_to_trade', 0.65),
                "max_position_size": config.get('trading.max_position_size', 0.1),
                "max_daily_loss": config.get('trading.max_daily_loss', 0.05),
                "stop_loss": config.get('trading.stop_loss_pct', 0.02),
                "max_drawdown": config.get('trading.max_drawdown', 0.15),
                "initial_capital": config.get('trading.initial_capital', 100000),
                "max_positions": config.get('trading.max_positions', 20),
                "max_trades_per_day": config.get('trading.max_trades_per_day', 1000),
                "watchlist": config.get('data.universe.initial_stocks', [])
            }
        }
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    """Update bot settings (saves to config.yaml)."""
    try:
        import yaml

        # Read current config
        config_path = Path(__file__).parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            current_config = yaml.safe_load(f)

        # Update settings that were provided
        if settings.min_confidence is not None:
            if 'learning' not in current_config:
                current_config['learning'] = {}
            current_config['learning']['min_confidence_to_trade'] = settings.min_confidence

        if settings.max_position_size is not None:
            if 'trading' not in current_config:
                current_config['trading'] = {}
            current_config['trading']['max_position_size'] = settings.max_position_size

        if settings.max_daily_loss is not None:
            current_config['trading']['max_daily_loss'] = settings.max_daily_loss

        if settings.stop_loss is not None:
            current_config['trading']['stop_loss_pct'] = settings.stop_loss

        if settings.max_drawdown is not None:
            current_config['trading']['max_drawdown'] = settings.max_drawdown

        if settings.initial_capital is not None:
            current_config['trading']['initial_capital'] = settings.initial_capital

        if settings.max_positions is not None:
            current_config['trading']['max_positions'] = settings.max_positions

        if settings.max_trades_per_day is not None:
            current_config['trading']['max_trades_per_day'] = settings.max_trades_per_day

        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)

        # Reload config
        config.reload()

        logger.info(f"Settings updated: {settings}")
        return {"success": True, "message": "Settings saved successfully"}

    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return {"success": False, "error": str(e)}

# ============================================================================
# BACKTESTING ENDPOINTS
# ============================================================================

@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Run a backtest on historical data with full trade simulation."""

    try:
        trainer = HistoricalTrainer(
            symbols=[request.symbol.upper()],
            db_path=str(data_dir / "backtest_temp.db") if request.mode == "test" else str(data_dir / "predictions.db")
        )

        # Set initial capital from request
        trainer.initial_capital = request.initial_capital
        trainer.cash = request.initial_capital

        results = trainer.train_on_historical(
            start_date=request.start_date,
            end_date=request.end_date,
            prediction_interval=1,
            verbose=False
        )

        profile = trainer.profiles.get(request.symbol.upper())
        trading = results.get('trading', {})

        if profile:
            # Save backtest results to database
            try:
                final_equity = trading.get('final_equity', request.initial_capital)
                total_return = trading.get('total_pnl_pct', 0)

                backtest_db.store_backtest_result(
                    strategy_name=f"{request.symbol.upper()} - {request.strategy}",
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=request.initial_capital,
                    final_capital=final_equity,
                    total_return=total_return,
                    sharpe_ratio=0,  # Not calculated in this backtest type
                    max_drawdown=trading.get('max_drawdown', 0),
                    win_rate=trading.get('win_rate', 0),
                    total_trades=trading.get('total_trades', 0),
                    parameters=json.dumps({
                        'symbol': request.symbol.upper(),
                        'strategy': request.strategy,
                        'mode': request.mode
                    }),
                    results=json.dumps({
                        'accuracy': profile.overall_accuracy,
                        'accuracy_1h': profile.accuracy_1h,
                        'accuracy_eod': profile.accuracy_eod,
                        'accuracy_next_day': profile.accuracy_next_day,
                        'total_predictions': profile.total_predictions,
                        'trading': trading
                    })
                )
                logger.info(f"Backtest results saved: {request.symbol.upper()} - {request.strategy}")
            except Exception as db_err:
                logger.warning(f"Failed to save backtest to database: {db_err}")

            return {
                "success": True,
                "symbol": request.symbol.upper(),
                "results": {
                    "totalPredictions": profile.total_predictions,
                    "accuracy": profile.overall_accuracy,
                    "accuracy1h": profile.accuracy_1h,
                    "accuracyEod": profile.accuracy_eod,
                    "accuracyNextDay": profile.accuracy_next_day,
                    "topFeatures": sorted(
                        profile.feature_weights.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5],
                    # Trading P&L results
                    "trading": {
                        "initialCapital": trading.get('initial_capital', request.initial_capital),
                        "finalEquity": trading.get('final_equity', request.initial_capital),
                        "totalPnl": trading.get('total_pnl', 0),
                        "totalPnlPct": trading.get('total_pnl_pct', 0),
                        "totalTrades": trading.get('total_trades', 0),
                        "winningTrades": trading.get('winning_trades', 0),
                        "losingTrades": trading.get('losing_trades', 0),
                        "winRate": trading.get('win_rate', 0),
                        "profitFactor": min(trading.get('profit_factor', 0), 999.99),  # Cap infinity for JSON
                        "avgWin": trading.get('avg_win', 0),
                        "avgLoss": trading.get('avg_loss', 0),
                        "maxDrawdown": trading.get('max_drawdown', 0),
                        "trades": trading.get('trades', [])
                    }
                }
            }
        else:
            return {"success": False, "error": "No results generated"}

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


# ============================================================================
# SIMPLE TRADER BACKTEST ENDPOINT (NEW - Uses simple_trader.py logic)
# ============================================================================

class SimpleBacktestRequest(BaseModel):
    symbol: str
    strategy: Optional[str] = None  # If None, auto-select best strategy
    lookback_days: int = 90
    initial_capital: float = 10000


@app.post("/api/backtest/simple")
async def run_simple_backtest(request: SimpleBacktestRequest):
    """
    Run a backtest using simple_trader.py strategy logic.
    This tests individual indicator strategies (RSI, MACD, Bollinger, etc.)
    """
    try:
        import pandas as pd
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        symbol = request.symbol.upper()

        # Get API keys
        api_key = os.environ.get('ALPACA_API_KEY') or config.get('alpaca.api_key')
        api_secret = os.environ.get('ALPACA_SECRET_KEY') or config.get('alpaca.api_secret')

        if not api_key or not api_secret:
            return {"success": False, "error": "Alpaca API keys not configured"}

        # Fetch historical data
        client = StockHistoricalDataClient(api_key, api_secret)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.lookback_days)

        bars_request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date
        )

        bars = client.get_stock_bars(bars_request)
        df = bars.df

        if df.empty:
            return {"success": False, "error": f"No data available for {symbol}"}

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')

        # Calculate all indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_diff'] = calculate_macd(df['close'])
        df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
        df['volume_ratio'] = calculate_volume_ratio(df['volume'])
        df['sma20'] = calculate_sma(df['close'], 20)
        df = df.dropna()

        if len(df) < 30:
            return {"success": False, "error": f"Not enough data for {symbol} (need at least 30 bars)"}

        # Test all strategies or just the requested one
        results = []

        if request.strategy and request.strategy in STRATEGIES:
            # Test specific strategy with trade logging
            strategy_class = STRATEGIES[request.strategy]
            strategy = strategy_class()
            result = backtest_strategy(df, strategy, request.initial_capital, symbol=symbol, log_trades=True)
            results.append(result)
        else:
            # Test all strategies (without logging to avoid spam)
            for name, strategy_class in STRATEGIES.items():
                strategy = strategy_class()
                result = backtest_strategy(df, strategy, request.initial_capital, symbol=symbol, log_trades=False)
                results.append(result)

        # Sort by return
        results.sort(key=lambda x: x['return_pct'], reverse=True)
        best = results[0]

        # Re-run the best strategy with trade logging enabled
        if not request.strategy:
            best_strategy_class = STRATEGIES[best['name']]
            best_strategy = best_strategy_class()
            backtest_strategy(df, best_strategy, request.initial_capital, symbol=symbol, log_trades=True)

        # Calculate final equity
        final_equity = request.initial_capital * (1 + best['return_pct'] / 100)

        # Save backtest results to database
        try:
            end_date = datetime.now()
            start_date_calc = end_date - timedelta(days=request.lookback_days)

            backtest_db.store_backtest_result(
                strategy_name=f"{symbol} - {best['name']} (SimpleTrader)",
                start_date=start_date_calc.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                initial_capital=request.initial_capital,
                final_capital=round(final_equity, 2),
                total_return=round(best['return_pct'], 2),
                sharpe_ratio=0,  # Not calculated in simple backtest
                max_drawdown=0,  # Not calculated in simple backtest
                win_rate=round(best['win_rate'], 1),
                total_trades=best['trades'],
                parameters=json.dumps({
                    'symbol': symbol,
                    'strategy': request.strategy or 'auto-select',
                    'lookback_days': request.lookback_days,
                    'bars_analyzed': len(df)
                }),
                results=json.dumps({
                    'best_strategy': best['name'],
                    'return_pct': round(best['return_pct'], 2),
                    'win_rate': round(best['win_rate'], 1),
                    'total_trades': best['trades'],
                    'winning_trades': best['winners'],
                    'all_strategies': [
                        {
                            'name': r['name'],
                            'return_pct': round(r['return_pct'], 2),
                            'win_rate': round(r['win_rate'], 1),
                            'trades': r['trades']
                        }
                        for r in results
                    ]
                })
            )
            logger.info(f"Simple backtest results saved: {symbol} - {best['name']}")
        except Exception as db_err:
            logger.warning(f"Failed to save simple backtest to database: {db_err}")

        return {
            "success": True,
            "symbol": symbol,
            "lookback_days": request.lookback_days,
            "bars_analyzed": len(df),
            "best_strategy": {
                "name": best['name'],
                "return_pct": round(best['return_pct'], 2),
                "win_rate": round(best['win_rate'], 1),
                "total_trades": best['trades'],
                "winning_trades": best['winners'],
                "initial_capital": request.initial_capital,
                "final_equity": round(final_equity, 2),
                "total_pnl": round(final_equity - request.initial_capital, 2)
            },
            "all_strategies": [
                {
                    "name": r['name'],
                    "return_pct": round(r['return_pct'], 2),
                    "win_rate": round(r['win_rate'], 1),
                    "trades": r['trades']
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"Simple backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


# ============================================================================
# DETAILED BACKTEST LOG ENDPOINT (Decision-by-decision analysis)
# ============================================================================

class DetailedBacktestRequest(BaseModel):
    symbol: str
    strategy: Optional[str] = None  # If None, auto-select best strategy
    lookback_days: int = 30  # Default to 30 days for detailed analysis
    initial_capital: float = 10000
    use_safety_rules: bool = True  # If True, use hold mode to avoid selling at a loss


@app.post("/api/backtest/detailed")
async def run_detailed_backtest_endpoint(request: DetailedBacktestRequest):
    """
    Run a detailed backtest with decision-by-decision logging.

    Returns every bar's decision (BUY, SELL, HOLD) with:
    - Layman's terms explanation of WHY the decision was made
    - All indicator values at that moment
    - Portfolio state at each point
    - Complete trade history with P&L

    This is designed for the Reports page to show users exactly
    what the bot would do and why.
    """
    try:
        import math

        symbol = request.symbol.upper()

        # Run the detailed backtest
        result = run_detailed_backtest(
            symbol=symbol,
            strategy_name=request.strategy,
            lookback_days=request.lookback_days,
            initial_capital=request.initial_capital,
            use_safety_rules=request.use_safety_rules
        )

        if 'error' in result:
            return {"success": False, "error": result['error']}

        # Handle infinity in profit factor (JSON doesn't support infinity)
        summary = result.get('summary', {})
        trades_summary = summary.get('trades', {})
        if trades_summary.get('profit_factor') == float('inf'):
            trades_summary['profit_factor'] = 999.99

        # Save to database
        try:
            end_date = datetime.now()
            start_date_calc = end_date - timedelta(days=request.lookback_days)

            backtest_db.store_backtest_result(
                strategy_name=f"{symbol} - {summary.get('strategy', 'Unknown')} (Detailed)",
                start_date=start_date_calc.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                initial_capital=request.initial_capital,
                final_capital=summary.get('capital', {}).get('final', request.initial_capital),
                total_return=summary.get('capital', {}).get('return_pct', 0),
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=trades_summary.get('win_rate', 0),
                total_trades=trades_summary.get('total', 0),
                parameters=json.dumps({
                    'symbol': symbol,
                    'strategy': request.strategy or 'auto-select',
                    'lookback_days': request.lookback_days,
                    'bars_analyzed': summary.get('period', {}).get('bars_analyzed', 0)
                }),
                results=json.dumps({
                    'strategy': summary.get('strategy'),
                    'buy_rule': summary.get('strategy_buy_rule'),
                    'sell_rule': summary.get('strategy_sell_rule'),
                    'return_pct': summary.get('capital', {}).get('return_pct', 0),
                    'decisions': summary.get('decisions', {})
                })
            )
            logger.info(f"Detailed backtest results saved: {symbol}")
        except Exception as db_err:
            logger.warning(f"Failed to save detailed backtest to database: {db_err}")

        return {
            "success": True,
            "summary": summary,
            "decision_log": result.get('decision_log', []),
            "trades": result.get('trades', []),
            "equity_curve": result.get('equity_curve', [])
        }

    except Exception as e:
        logger.error(f"Detailed backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


# ============================================================================
# SIMPLE TRADER CONTROL ENDPOINTS
# ============================================================================

@app.get("/api/simple-trader/status")
async def get_simple_trader_status():
    """Get SimpleTrader status including strategies per stock."""
    try:
        trader = get_simple_trader()
        status = trader.get_status()

        return {
            "success": True,
            "running": status['running'],
            "paper": status['paper'],
            "market_open": status['market_open'],
            "symbols": status['symbols'],
            "strategies": status['strategies'],
            "positions": status['positions'],
            "account": status['account'],
            "stats": status['stats']
        }
    except Exception as e:
        logger.error(f"Error getting simple trader status: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/simple-trader/start")
async def start_simple_trader(background_tasks: BackgroundTasks):
    """Start the SimpleTrader in background."""
    try:
        trader = get_simple_trader()

        if trader.running:
            return {"success": False, "message": "Taz is already running"}

        # Calibrate strategies if needed
        uncalibrated = [s for s in trader.symbols if s not in trader.stock_configs]
        if uncalibrated:
            logger.info(f"Calibrating {len(uncalibrated)} stocks...")
            for symbol in uncalibrated:
                trader.calibrate_if_needed(symbol)

        # Run in background thread
        check_interval = config.get('simple_trader.check_interval_seconds', 900)
        def run_trader():
            try:
                trader.run(check_interval_seconds=check_interval)
            except Exception as e:
                logger.error(f"Taz thread crashed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                trader.running = False

        import threading
        thread = threading.Thread(target=run_trader, daemon=True)
        thread.start()

        return {
            "success": True,
            "message": "Taz started",
            "strategies": {s: c.strategy.name for s, c in trader.stock_configs.items()}
        }
    except Exception as e:
        logger.error(f"Error starting simple trader: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/simple-trader/stop")
async def stop_simple_trader():
    """Stop the SimpleTrader."""
    try:
        trader = get_simple_trader()

        if not trader.running:
            return {"success": False, "message": "Taz is not running"}

        trader.stop()
        return {"success": True, "message": "Taz stopped"}
    except Exception as e:
        logger.error(f"Error stopping simple trader: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/simple-trader/calibrate")
async def calibrate_simple_trader(symbol: Optional[str] = None):
    """Calibrate strategies for all stocks or a specific stock."""
    try:
        trader = get_simple_trader()

        if symbol:
            # Calibrate single stock
            symbol = symbol.upper()
            result = select_best_strategy(symbol, trader.calibration_days)

            strategy_class = STRATEGIES.get(result['best_strategy'])
            if strategy_class:
                from simple_trader import StockConfig
                trader.stock_configs[symbol] = StockConfig(
                    symbol=symbol,
                    strategy=strategy_class(),
                    last_calibration=datetime.now()
                )
                trader._save_state()

            return {
                "success": True,
                "symbol": symbol,
                "best_strategy": result['best_strategy'],
                "return_pct": round(result['best_return'], 2),
                "win_rate": round(result['best_win_rate'], 1),
                "all_results": result['all_results']
            }
        else:
            # Calibrate all stocks
            trader.calibrate_all()
            return {
                "success": True,
                "message": f"Calibrated {len(trader.stock_configs)} stocks",
                "strategies": {s: c.strategy.name for s, c in trader.stock_configs.items()}
            }
    except Exception as e:
        logger.error(f"Error calibrating: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/simple-trader/force-buy")
async def force_buy(symbol: str, shares: int = 1):
    """Force a buy order for testing purposes."""
    try:
        trader = get_simple_trader()
        config = trader.stock_configs.get(symbol)
        strategy_name = config.strategy.name if config else "Manual Test"

        # Execute the buy
        success = trader.execute_buy(symbol.upper(), strategy_name)

        return {
            "success": success,
            "symbol": symbol.upper(),
            "strategy": strategy_name,
            "message": f"Buy order {'executed' if success else 'failed'} for {symbol}"
        }
    except Exception as e:
        logger.error(f"Error forcing buy: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/simple-trader/force-sell")
async def force_sell(symbol: str):
    """Force a sell order for testing purposes."""
    try:
        trader = get_simple_trader()

        # Execute the sell
        success = trader.execute_sell(symbol.upper(), sell_reason="Manual force sell")

        return {
            "success": success,
            "symbol": symbol.upper(),
            "message": f"Sell order {'executed' if success else 'failed'} for {symbol}"
        }
    except Exception as e:
        logger.error(f"Error forcing sell: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/simple-trader/signals")
async def get_simple_trader_signals():
    """Get current buy/sell signals from SimpleTrader."""
    try:
        trader = get_simple_trader()
        signals = []
        current_positions = trader.get_current_positions()

        # Get RL shadow (the kid with the coloring book - NO influence on trading)
        rl_shadow = None
        try:
            from rl_shadow import get_rl_shadow
            rl_shadow = get_rl_shadow()
        except Exception as e:
            logger.debug(f"RL Shadow not available: {e}")

        for symbol in trader.symbols:
            config = trader.stock_configs.get(symbol)
            if not config:
                continue

            df = trader.fetch_latest_data(symbol)
            if df is None or len(df) < 2:
                continue

            strategy = config.strategy
            i = len(df) - 1
            row = df.iloc[i]

            has_position = symbol in current_positions
            buy_signal = bool(strategy.check_buy(df, i))
            sell_signal = bool(strategy.check_sell(df, i))

            # Get position info for RL state
            position_pct = 0
            unrealized_pnl_pct = 0
            if has_position:
                pos = current_positions[symbol]
                total_equity = trader.get_account().get('equity', 100000)
                position_value = pos.get('shares', 0) * pos.get('current_price', 0)
                position_pct = position_value / total_equity if total_equity > 0 else 0
                avg_cost = pos.get('avg_cost', pos.get('current_price', 1))
                if avg_cost > 0:
                    unrealized_pnl_pct = (pos.get('current_price', avg_cost) - avg_cost) / avg_cost

            signal_info = {
                "symbol": symbol,
                "strategy": strategy.name,
                "price": float(row['close']),
                "has_position": bool(has_position),
                "indicators": {
                    "rsi": round(float(row['rsi']), 1),
                    "macd_diff": round(float(row['macd_diff']), 4),
                    "bb_position": round(float(row['bb_position']), 2),
                    "volume_ratio": round(float(row['volume_ratio']), 2)
                },
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "buy_rule": strategy.buy_rule,
                "sell_rule": strategy.sell_rule
            }

            # Determine action (SimpleTrader's decision - this is what matters)
            if not has_position and buy_signal:
                signal_info["action"] = "BUY"
                signal_info["confidence"] = 80
            elif has_position and sell_signal:
                signal_info["action"] = "SELL"
                signal_info["confidence"] = 80
            else:
                signal_info["action"] = "HOLD"
                signal_info["confidence"] = 50

            # RL Shadow recommendation (just watching, NO influence on action above)
            # Like a kid with a coloring book at work - we see what they drew but ignore it
            rl_rec = None
            if rl_shadow:
                try:
                    rl_rec = rl_shadow.get_recommendation(
                        symbol, df,
                        has_position=has_position,
                        position_pct=position_pct,
                        unrealized_pnl_pct=unrealized_pnl_pct
                    )
                except:
                    pass

            if rl_rec:
                signal_info["rl_shadow"] = {
                    "action": rl_rec['action'],
                    "q_values": rl_rec['q_values'],
                    "agrees": rl_rec['action'] == signal_info["action"]
                }
            else:
                signal_info["rl_shadow"] = None  # No model for this symbol

            signals.append(signal_info)

        # Sort by action priority (BUY/SELL first)
        signals.sort(key=lambda x: (0 if x['action'] in ['BUY', 'SELL'] else 1, x['symbol']))

        # Get RL stats
        rl_stats = None
        if rl_shadow:
            try:
                rl_stats = rl_shadow.get_stats()
            except:
                pass

        return {
            "success": True,
            "signals": signals,
            "market_open": bool(trader._is_market_hours()),
            "timestamp": datetime.now().isoformat(),
            "rl_shadow_stats": rl_stats
        }
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return {"success": False, "error": str(e), "signals": []}


@app.get("/api/simple-trader/strategies")
async def get_available_strategies():
    """Get list of available trading strategies."""
    strategies = []
    for name, strategy_class in STRATEGIES.items():
        strategy = strategy_class()
        strategies.append({
            "name": name,
            "buy_rule": strategy.buy_rule,
            "sell_rule": strategy.sell_rule
        })
    return {"strategies": strategies}


# ============================================================================
# MORNING BRIEFING ENDPOINT
# ============================================================================

@app.get("/api/briefing")
async def get_morning_briefing():
    """Get the morning briefing data."""
    service = get_service()

    # Account info
    account = {}
    if service.trade_executor:
        account = service.trade_executor.get_account()

    # Yesterday's stats (from legacy system if available, else from SimpleTrader)
    stats = {}
    total_preds = 0
    correct_preds = 0
    accuracy = 0
    best_performers = []

    if LEGACY_PREDICTION_AVAILABLE:
        db = PredictionDatabase(str(data_dir / "predictions.db"))
        stats = db.get_prediction_stats(days=1)
        total_preds = sum(s.get('total', 0) for s in stats.values())
        correct_preds = sum(s.get('correct', 0) for s in stats.values())
        accuracy = correct_preds / total_preds if total_preds > 0 else 0

        # Get best performers (stocks with highest accuracy)
        for symbol in service.config.symbols:
            profile = db.get_stock_profile(symbol)
            if profile and profile.total_predictions > 50:
                best_performers.append({
                    "symbol": symbol,
                    "accuracy": profile.overall_accuracy,
                    "predictions": profile.total_predictions
                })
        best_performers.sort(key=lambda x: x['accuracy'], reverse=True)
    else:
        # Use SimpleTrader stats instead
        try:
            trader = get_simple_trader()
            trader_stats = trader.stats
            total_trades = trader_stats.get('winning_trades', 0) + trader_stats.get('losing_trades', 0)
            accuracy = trader_stats.get('winning_trades', 0) / total_trades if total_trades > 0 else 0
        except:
            pass

    # Excluded stocks
    excluded = service.config.excluded_symbols

    return {
        "date": datetime.now().strftime("%A, %B %d, %Y"),
        "account": {
            "equity": account.get('equity', 0),
            "cash": account.get('cash', 0),
            "buyingPower": account.get('buying_power', 0)
        },
        "yesterdayPerformance": {
            "predictions": total_preds,
            "correct": correct_preds,
            "accuracy": accuracy,
            "byHorizon": stats
        },
        "bestPerformers": best_performers[:5],
        "monitoringCount": len(service.config.symbols),
        "excludedToday": excluded,
        "tradingMode": service.config.trading_mode.value,
        "serviceState": service.state.value if service.state else "unknown"
    }

# ============================================================================
# TRADE SIGNALS ENDPOINT
# ============================================================================

@app.get("/api/signals")
async def get_trade_signals():
    """Get current trade signals from SimpleTrader strategies."""
    try:
        trader = get_simple_trader()
        signals = []
        current_positions = trader.get_current_positions()

        for symbol in trader.symbols:
            config = trader.stock_configs.get(symbol)
            if not config:
                continue

            df = trader.fetch_latest_data(symbol)
            if df is None or len(df) < 2:
                continue

            strategy = config.strategy
            i = len(df) - 1
            latest = df.iloc[i]
            has_position = symbol in current_positions

            buy_signal = strategy.check_buy(df, i)
            sell_signal = strategy.check_sell(df, i)

            # Only include if there's an actionable signal
            if buy_signal or sell_signal:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL' if sell_signal and has_position else 'BUY' if buy_signal and not has_position else 'HOLD',
                    'direction': 'down' if sell_signal else 'up',
                    'confidence': 75 if (buy_signal or sell_signal) else 50,
                    'strategy': strategy.name,
                    'buy_rule': strategy.buy_rule,
                    'sell_rule': strategy.sell_rule,
                    'timestamp': datetime.now().isoformat(),
                    'indicators': {
                        'rsi': round(float(latest.get('rsi', 0)), 1),
                        'macd_diff': round(float(latest.get('macd_diff', 0)), 4),
                        'bb_position': round(float(latest.get('bb_position', 0)), 2)
                    }
                })

        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            "signals": signals[:10],
            "total": len(signals),
            "tradingMode": 'paper' if trader.paper else 'live',
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return {"signals": [], "total": 0, "error": str(e)}

# ============================================================================
# AI BRAIN STATUS ENDPOINTS (merged from web_api.py)
# ============================================================================

@app.get("/api/brain")
async def get_brain_status():
    """Get SimpleTrader strategy status - per-stock strategy assignments."""
    try:
        trader = get_simple_trader()
        stats = trader.stats
        current_positions = trader.get_current_positions()

        # Count strategy usage
        strategy_counts = {}
        for symbol, config in trader.stock_configs.items():
            strategy_name = config.strategy.name
            if strategy_name not in strategy_counts:
                strategy_counts[strategy_name] = 0
            strategy_counts[strategy_name] += 1

        # Format as "global weights" (really strategy distribution)
        global_weights = []
        for strategy_name, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            global_weights.append({
                'signal': strategy_name,
                'name': strategy_name,
                'description': f'Used by {count} stocks',
                'weight': count / len(trader.stock_configs) if trader.stock_configs else 0,
                'accuracy': 0,  # Would need backtest results
                'uses': count,
                'status': 'strong' if count >= 5 else 'normal'
            })

        # Per-stock strategies with position info
        per_stock = {}
        stock_assignments = []  # For easier display
        for symbol, config in trader.stock_configs.items():
            has_position = symbol in current_positions
            pos_info = current_positions.get(symbol, {})

            stock_data = {
                'signal': config.strategy.name,
                'name': config.strategy.name,
                'buy_rule': config.strategy.buy_rule,
                'sell_rule': config.strategy.sell_rule,
                'last_calibration': config.last_calibration.isoformat() if config.last_calibration else None,
                'use_safety_rules': config.use_safety_rules
            }
            per_stock[symbol] = [stock_data]

            # Also build flat list for display
            stock_assignments.append({
                'symbol': symbol,
                'strategy': config.strategy.name,
                'buy_rule': config.strategy.buy_rule,
                'sell_rule': config.strategy.sell_rule,
                'safety_rules': config.use_safety_rules,
                'has_position': has_position,
                'unrealized_pnl': pos_info.get('unrealized_pnl', 0) if has_position else None,
                'last_calibration': config.last_calibration.strftime('%m/%d %H:%M') if config.last_calibration else 'Never'
            })

        # Sort by symbol
        stock_assignments.sort(key=lambda x: x['symbol'])

        # Calculate win rate
        total_trades = stats.get('winning_trades', 0) + stats.get('losing_trades', 0)
        win_rate = (stats.get('winning_trades', 0) / total_trades * 100) if total_trades > 0 else 0

        return {
            'overall_accuracy': win_rate,
            'total_predictions': 0,  # SimpleTrader doesn't make predictions
            'correct': stats.get('winning_trades', 0),
            'wrong': stats.get('losing_trades', 0),
            'global_weights': global_weights,
            'per_stock_weights': per_stock,
            'stock_assignments': stock_assignments,  # New: easy-to-display list
            'learning_status': 'calibrated',
            'total_stocks': len(trader.stock_configs),
            'total_pnl': stats.get('total_pnl', 0)
        }

    except Exception as e:
        logger.error(f"Error getting brain status: {e}")
        return {
            'overall_accuracy': 0,
            'total_predictions': 0,
            'global_weights': [],
            'per_stock_weights': {},
            'stock_assignments': [],
            'error': str(e)
        }


@app.get("/api/rl-shadow")
async def get_rl_shadow_status():
    """Get RL Shadow status - the kid with the coloring book."""
    try:
        from rl_shadow import get_rl_shadow
        rl = get_rl_shadow()

        # Get current recommendations for all symbols with models
        trader = get_simple_trader()
        current_positions = trader.get_current_positions()

        recommendations = []
        for symbol in rl.get_available_symbols():
            # Get market data if available
            df = None
            try:
                df = trader.fetch_latest_data(symbol)
            except:
                pass

            if df is not None and len(df) >= 30:
                has_position = symbol in current_positions
                pos = current_positions.get(symbol, {})

                position_pct = 0
                unrealized_pnl_pct = 0
                if has_position:
                    total_equity = trader.get_account().get('equity', 100000)
                    position_value = pos.get('shares', 0) * pos.get('current_price', 0)
                    position_pct = position_value / total_equity if total_equity > 0 else 0
                    avg_cost = pos.get('avg_cost', 1)
                    if avg_cost > 0:
                        unrealized_pnl_pct = (pos.get('current_price', avg_cost) - avg_cost) / avg_cost

                rec = rl.get_recommendation(
                    symbol, df,
                    has_position=has_position,
                    position_pct=position_pct,
                    unrealized_pnl_pct=unrealized_pnl_pct
                )

                if rec:
                    rec['has_position'] = has_position
                    rec['current_price'] = float(df.iloc[-1]['close'])
                    recommendations.append(rec)

        stats = rl.get_stats()

        return {
            'success': True,
            'models_loaded': stats['models_loaded'],
            'available_symbols': stats['symbols'],
            'recommendations': recommendations,
            'shadow_stats': {
                'total_signals': stats['total_signals'],
                'agreements': stats['agreements'],
                'agreement_rate': stats['agreement_rate']
            },
            'message': "RL is watching and learning (no influence on trading)"
        }

    except Exception as e:
        logger.error(f"Error getting RL shadow status: {e}")
        return {
            'success': False,
            'models_loaded': 0,
            'available_symbols': [],
            'recommendations': [],
            'error': str(e)
        }

@app.get("/api/brain/details")
async def get_brain_details():
    """Get detailed brain information for the Brain tab."""
    try:
        from core.market_monitor import get_market_monitor
        monitor = get_market_monitor()

        # Get all the brain data
        stats = monitor.prediction_tracker.get_accuracy_stats(days=30)
        signal_perf = monitor.prediction_tracker.get_signal_performance(days=30)
        stock_weights_df = monitor.prediction_tracker.get_all_stock_weights()

        # Signal explanations
        signal_info = {
            'momentum_20d': {
                'name': '20-Day Momentum',
                'description': 'Measures the rate of price change over 20 trading days. Positive momentum suggests upward trend continuation.',
                'interpretation': 'Higher weight = bot trusts momentum signals more for predictions'
            },
            'rsi': {
                'name': 'RSI (Relative Strength Index)',
                'description': 'Oscillator measuring speed and magnitude of price changes. Values above 70 suggest overbought, below 30 oversold.',
                'interpretation': 'Higher weight = bot relies more on overbought/oversold signals'
            },
            'macd_signal': {
                'name': 'MACD Signal Line',
                'description': 'Trend-following momentum indicator showing relationship between two moving averages.',
                'interpretation': 'Higher weight = bot trusts MACD crossovers for entry/exit timing'
            },
            'volume_ratio': {
                'name': 'Volume Ratio',
                'description': 'Compares current volume to average volume. High ratios suggest significant market interest.',
                'interpretation': 'Higher weight = bot values volume confirmation for trades'
            },
            'price_vs_sma20': {
                'name': 'Price vs 20-SMA',
                'description': 'Relationship between current price and 20-day simple moving average.',
                'interpretation': 'Higher weight = bot uses moving average crossovers more heavily'
            },
            'bollinger_position': {
                'name': 'Bollinger Band Position',
                'description': 'Where price sits within Bollinger Bands (volatility measure).',
                'interpretation': 'Higher weight = bot trusts mean-reversion signals from bands'
            }
        }

        # Format detailed weights with full info
        detailed_weights = []
        for signal, weight in monitor.signal_weights.items():
            info = signal_info.get(signal, {})

            # Get accuracy
            acc = 0
            uses = 0
            if not signal_perf.empty:
                sig_row = signal_perf[signal_perf['signal'] == signal]
                if not sig_row.empty:
                    acc = float(sig_row.iloc[0]['accuracy'])
                    uses = int(sig_row.iloc[0]['uses'])

            detailed_weights.append({
                'signal': signal,
                'name': info.get('name', signal),
                'description': info.get('description', ''),
                'interpretation': info.get('interpretation', ''),
                'weight': float(weight),
                'accuracy': acc,
                'uses': uses,
                'status': 'strong' if weight > 1.1 else ('weak' if weight < 0.9 else 'normal')
            })

        # Format per-stock data
        stock_profiles = []
        if not stock_weights_df.empty:
            for symbol in stock_weights_df['symbol'].unique():
                symbol_data = stock_weights_df[stock_weights_df['symbol'] == symbol]
                total_samples = int(symbol_data['sample_size'].sum())

                signals = []
                for _, row in symbol_data.iterrows():
                    signals.append({
                        'signal': row['signal_name'],
                        'name': signal_info.get(row['signal_name'], {}).get('name', row['signal_name']),
                        'weight': float(row['weight']),
                        'accuracy': float(row['accuracy']) if row['accuracy'] else 0,
                        'uses': int(row['sample_size']) if row['sample_size'] else 0
                    })

                stock_profiles.append({
                    'symbol': symbol,
                    'total_predictions': total_samples,
                    'signals': sorted(signals, key=lambda x: x['weight'], reverse=True)
                })

        return {
            'summary': {
                'total_predictions': stats.get('total_predictions', 0),
                'overall_accuracy': stats.get('accuracy', 0),
                'correct': stats.get('correct', 0),
                'wrong': stats.get('wrong', 0)
            },
            'global_weights': sorted(detailed_weights, key=lambda x: x['weight'], reverse=True),
            'stock_profiles': sorted(stock_profiles, key=lambda x: x['total_predictions'], reverse=True),
            'learning_explanation': {
                'how_it_works': 'The bot tracks which signals lead to correct predictions. Signals that perform well get MORE weight (trusted more). Signals that perform poorly get LESS weight.',
                'weight_range': 'Weights range from 0.5 (minimum, rarely trusted) to 2.0 (maximum, highly trusted). Starting value is 1.0.',
                'per_stock_learning': 'Each stock develops its own signal preferences over time. MACD might work great for AAPL but poorly for TSLA - the bot learns this.'
            }
        }

    except Exception as e:
        logger.error(f"Error getting brain details: {e}")
        return {'error': str(e)}

# ============================================================================
# P&L AND TRADES ENDPOINTS (merged from web_api.py)
# ============================================================================

@app.get("/api/pnl")
async def get_pnl():
    """Get P&L data from SimpleTrader."""
    try:
        trader = get_simple_trader()
        status = trader.get_status()
        stats = status.get('stats', {})
        account = status.get('account', {})

        # Get current portfolio value from Alpaca
        current_value = account.get('equity', 100000)

        # Get SimpleTrader's tracked P&L
        total_pnl = stats.get('total_pnl', 0)

        # Calculate unrealized P&L from positions
        positions = status.get('positions', {})
        unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions.values())

        # Initial capital (rough estimate)
        initial_capital = 100000  # Default starting capital

        return {
            'today_pnl': total_pnl,  # SimpleTrader tracks cumulative
            'today_pnl_pct': (total_pnl / current_value * 100) if current_value > 0 else 0,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / initial_capital * 100) if initial_capital > 0 else 0,
            'unrealized_pnl': unrealized_pnl,
            'initial_capital': initial_capital,
            'current_value': current_value,
            'trades_executed': stats.get('trades_executed', 0),
            'winning_trades': stats.get('winning_trades', 0),
            'losing_trades': stats.get('losing_trades', 0),
            'sparkline': []  # Would need historical tracking to populate
        }

    except Exception as e:
        logger.error(f"Error getting P&L: {e}")
        return {
            'today_pnl': 0,
            'today_pnl_pct': 0,
            'total_pnl': 0,
            'initial_capital': 100000,
            'current_value': 100000,
            'sparkline': []
        }

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades."""
    try:
        from utils.trade_logger import get_trade_logger
        import math

        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            return {'trades': []}

        def safe_float(val):
            """Convert to float, handling NaN and None."""
            if val is None:
                return None
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            except (ValueError, TypeError):
                return None

        trades = []
        for _, row in df.head(limit).iterrows():
            trades.append({
                'id': row['trade_id'],
                'timestamp': str(row['timestamp']),
                'symbol': row['symbol'],
                'action': row['action'],
                'quantity': safe_float(row['quantity']),
                'price': safe_float(row['price']),
                'pnl': safe_float(row.get('realized_pnl')),
                'strategy': row.get('strategy_name', ''),
                'reason': row.get('primary_signal', '')
            })

        return {'trades': trades}

    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return {'trades': []}

@app.get("/api/yesterday")
async def get_yesterday_summary():
    """Get yesterday's trading summary."""
    try:
        import pandas as pd
        from utils.trade_logger import get_trade_logger

        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            return {
                'pnl': 0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'best_trade': None,
                'worst_trade': None
            }

        # Filter by date - use UTC and make timezone-naive for comparison
        cutoff = datetime.now() - timedelta(days=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        df = df[df['timestamp'] >= cutoff]

        # Calculate stats
        sells = df[df['realized_pnl'].notna()]

        if sells.empty:
            return {
                'pnl': 0,
                'trades': len(df),
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'best_trade': None,
                'worst_trade': None
            }

        wins = sells[sells['realized_pnl'] > 0]
        losses = sells[sells['realized_pnl'] < 0]

        best = sells.loc[sells['realized_pnl'].idxmax()] if not sells.empty else None
        worst = sells.loc[sells['realized_pnl'].idxmin()] if not sells.empty else None

        return {
            'pnl': float(sells['realized_pnl'].sum()),
            'trades': len(sells),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(sells) * 100) if len(sells) > 0 else 0,
            'best_trade': {
                'symbol': best['symbol'],
                'pnl': float(best['realized_pnl'])
            } if best is not None else None,
            'worst_trade': {
                'symbol': worst['symbol'],
                'pnl': float(worst['realized_pnl'])
            } if worst is not None else None
        }

    except Exception as e:
        logger.error(f"Error getting yesterday summary: {e}")
        return {'pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0}

# ============================================================================
# TRENDING AND PREDICTIONS ENDPOINTS (merged from web_api.py)
# ============================================================================

@app.get("/api/trending")
async def get_trending():
    """Get stocks trending upward with bot confidence."""
    try:
        import pandas as pd
        from core.market_monitor import get_market_monitor

        monitor = get_market_monitor()
        trending = []

        # Get recent predictions
        with monitor.prediction_tracker.db.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT symbol, predicted_direction, confidence, timestamp
                FROM ai_predictions
                WHERE timestamp >= datetime('now', '-1 day')
                ORDER BY timestamp DESC
            """, conn)

        if not df.empty:
            # Get latest prediction per symbol where direction is 'up'
            up_predictions = df[df['predicted_direction'] == 'up']

            for symbol in up_predictions['symbol'].unique():
                symbol_preds = up_predictions[up_predictions['symbol'] == symbol]
                latest = symbol_preds.iloc[0]

                # Get the hybrid weight confidence for this symbol
                weights = monitor.get_weights_for_symbol(symbol)
                avg_weight = sum(weights.values()) / len(weights) if weights else 1.0

                trending.append({
                    'symbol': symbol,
                    'direction': 'up',
                    'confidence': float(latest['confidence']),
                    'brain_confidence': min(100, float(latest['confidence']) * avg_weight),
                    'change_pct': 0  # Would need real-time data
                })

        # Sort by confidence
        trending.sort(key=lambda x: x['confidence'], reverse=True)

        return {'trending': trending[:5]}

    except Exception as e:
        logger.error(f"Error getting trending: {e}")
        return {'trending': []}

@app.get("/api/predicted-profit")
async def get_predicted_profit():
    """Get predicted profit based on open signals."""
    try:
        import pandas as pd
        from core.market_monitor import get_market_monitor

        monitor = get_market_monitor()

        # Get active high-confidence predictions
        with monitor.prediction_tracker.db.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT symbol, predicted_direction, confidence, predicted_change_pct
                FROM ai_predictions
                WHERE resolved = 0
                AND confidence >= 60
                ORDER BY confidence DESC
            """, conn)

        if df.empty:
            return {
                'predicted_low': 0,
                'predicted_high': 0,
                'open_signals': 0,
                'avg_confidence': 0
            }

        # Calculate predicted profit range
        avg_conf = float(df['confidence'].mean())
        num_signals = len(df)

        # Estimate based on confidence and typical trade size
        initial_capital_cfg = config.get('trading.initial_capital', 100000)
        # Handle "auto" or string values
        if isinstance(initial_capital_cfg, str):
            initial_capital = 100000  # Default if "auto"
        else:
            initial_capital = float(initial_capital_cfg)
        position_size = float(config.get('trading.max_position_size', 0.1))
        typical_position = initial_capital * position_size

        # Conservative and optimistic estimates
        avg_predicted_change = float(df['predicted_change_pct'].mean()) if 'predicted_change_pct' in df.columns else 2.0

        predicted_low = num_signals * typical_position * (avg_predicted_change * 0.5) / 100
        predicted_high = num_signals * typical_position * (avg_predicted_change * 1.5) / 100

        return {
            'predicted_low': round(predicted_low, 2),
            'predicted_high': round(predicted_high, 2),
            'open_signals': num_signals,
            'avg_confidence': round(avg_conf, 1)
        }

    except Exception as e:
        logger.error(f"Error getting predicted profit: {e}")
        return {'predicted_low': 0, 'predicted_high': 0, 'open_signals': 0}

# ============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    websocket_clients.append(websocket)
    
    logger.info(f"WebSocket client connected. Total clients: {len(websocket_clients)}")
    
    try:
        # Send initial status
        service = get_service()
        await websocket.send_json({
            "type": "connected",
            "data": service.get_status()
        })
        
        # Keep connection alive and listen for messages
        while True:
            try:
                # Wait for messages from client
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                # Handle client commands
                message = json.loads(data)
                
                if message.get('type') == 'ping':
                    await websocket.send_json({"type": "pong"})
                elif message.get('type') == 'get_status':
                    await websocket.send_json({
                        "type": "status",
                        "data": service.get_status()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)

# ============================================================================
# COMMAND ENDPOINT (for natural language commands)
# ============================================================================

@app.post("/api/command")
async def process_command(command: UserCommand):
    """Process a natural language command."""
    service = get_service()
    cmd = command.command.strip().upper()
    
    # Parse command
    if cmd in ['STATUS', 'STATS']:
        return {"response": "status", "data": service.get_status()}
    
    elif cmd == 'START':
        if service.state != ServiceState.RUNNING:
            service.start()
        return {"response": "Service started"}
    
    elif cmd == 'STOP':
        if service.state != ServiceState.STOPPED:
            service.stop()
        return {"response": "Service stopped"}
    
    elif cmd == 'PAUSE':
        service.pause()
        return {"response": "Service paused"}
    
    elif cmd == 'RESUME':
        service.resume()
        return {"response": "Service resumed"}
    
    elif 'SKIP' in cmd or 'EXCLUDE' in cmd:
        # Extract symbols
        import re
        symbols = re.findall(r'\b[A-Z]{1,5}\b', cmd)
        exclude_words = {'SKIP', 'EXCLUDE', 'AND', 'OR', 'THE', 'TODAY'}
        symbols = [s for s in symbols if s not in exclude_words]
        
        for symbol in symbols:
            service.exclude_symbol(symbol)
        
        return {"response": f"Excluded: {', '.join(symbols)}"}
    
    elif 'ADD' in cmd or 'WATCH' in cmd:
        import re
        symbols = re.findall(r'\b[A-Z]{1,5}\b', cmd)
        exclude_words = {'ADD', 'WATCH', 'AND', 'OR', 'THE', 'TO'}
        symbols = [s for s in symbols if s not in exclude_words]
        
        results = []
        for symbol in symbols:
            result = service.add_symbol(symbol, train_first=True)
            results.append(f"{symbol}: {'Added' if result.get('success') else 'Failed'}")
        
        return {"response": '\n'.join(results)}
    
    else:
        return {"response": f"Unknown command: {command.command}"}

# ============================================================================
# TAZ TRADING SYSTEM API
# ============================================================================

def get_taz_trader() -> Optional[TazTrader]:
    """Get or create the TazTrader instance."""
    global taz_trader

    if not TAZ_AVAILABLE:
        return None

    if taz_trader is None:
        taz_trader = TazTrader(
            initial_capital=1000,
            paper=True,  # Default to paper trading
            max_position_pct=0.40,
            max_positions=3,
            stop_loss_pct=0.05,
            take_profit_pct=0.03,
            trade_crypto=True,
            use_rl=True
        )

    return taz_trader


@app.get("/api/taz/status")
async def taz_status():
    """Get Taz trading system status."""
    if not TAZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="Taz trading system not available")

    trader = get_taz_trader()
    if not trader:
        raise HTTPException(status_code=503, detail="Failed to initialize Taz trader")

    return trader.get_status()


@app.get("/api/taz/account")
async def taz_account():
    """Get Taz account information."""
    if not TAZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="Taz trading system not available")

    trader = get_taz_trader()
    return trader.get_account()


@app.get("/api/taz/positions")
async def taz_positions():
    """Get current Taz positions."""
    if not TAZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="Taz trading system not available")

    trader = get_taz_trader()
    return {
        "positions": trader.get_current_positions(),
        "tracked": {
            s: {
                "symbol": p.symbol,
                "asset_type": p.asset_type,
                "entry_price": p.entry_price,
                "strategy": p.strategy,
                "entry_time": p.entry_time.isoformat()
            }
            for s, p in trader.positions.items()
        }
    }


@app.get("/api/taz/stats")
async def taz_stats():
    """Get Taz trading statistics."""
    if not TAZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="Taz trading system not available")

    trader = get_taz_trader()
    stats = trader.stats
    total_trades = max(stats.trades_executed, 1)

    return {
        "trades_executed": stats.trades_executed,
        "winning_trades": stats.winning_trades,
        "losing_trades": stats.losing_trades,
        "win_rate": (stats.winning_trades / total_trades) * 100,
        "total_profit": stats.total_profit,
        "biggest_win": stats.biggest_win,
        "biggest_loss": stats.biggest_loss,
        "best_streak": stats.best_streak,
        "worst_streak": stats.worst_streak
    }


@app.get("/api/taz/scanner")
async def taz_scanner_results():
    """Get latest Taz scanner results."""
    if not TAZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="Taz trading system not available")

    results_file = Path("Taz/data/taz_scanner_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)

    return {"opportunities": [], "scan_time": None}


@app.post("/api/taz/scan")
async def taz_run_scan():
    """Run a new Taz scan."""
    if not TAZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="Taz trading system not available")

    scanner = TazScanner(min_volatility=2.0, min_volume_ratio=1.0)
    opportunities = scanner.scan_all()

    return {
        "scan_time": datetime.now().isoformat(),
        "total_opportunities": len(opportunities),
        "buy_signals": len([o for o in opportunities if o.signal == 'BUY']),
        "top_5": [o.to_dict() for o in opportunities[:5]]
    }


class TazStartCommand(BaseModel):
    capital: float = 1000
    paper: bool = True
    crypto: bool = True
    rl: bool = True


@app.post("/api/taz/start")
async def taz_start(cmd: TazStartCommand, background_tasks: BackgroundTasks):
    """Start Taz trader in background."""
    global taz_trader

    if not TAZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="Taz trading system not available")

    # Create new trader with specified settings
    taz_trader = TazTrader(
        initial_capital=cmd.capital,
        paper=cmd.paper,
        trade_crypto=cmd.crypto,
        use_rl=cmd.rl
    )

    # Start in background
    import threading
    trader_thread = threading.Thread(target=taz_trader.run, daemon=True)
    trader_thread.start()

    return {
        "status": "started",
        "mode": "paper" if cmd.paper else "live",
        "capital": cmd.capital,
        "crypto": cmd.crypto,
        "rl": cmd.rl
    }


@app.post("/api/taz/stop")
async def taz_stop():
    """Stop the Taz trader."""
    global taz_trader

    if not TAZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="Taz trading system not available")

    if taz_trader and taz_trader.running:
        taz_trader.stop()
        return {"status": "stopped"}

    return {"status": "not_running"}


@app.get("/taz", response_class=HTMLResponse)
async def serve_taz_dashboard():
    """Serve the Taz dashboard."""
    dashboard_path = Path(__file__).parent / "Taz" / "taz_dashboard.html"

    if dashboard_path.exists():
        return FileResponse(dashboard_path)

    # Return a simple fallback
    return HTMLResponse(content="""
    <html>
    <head><title>Taz Dashboard</title></head>
    <body>
        <h1>Taz Dashboard Not Found</h1>
        <p>Please create Taz/taz_dashboard.html</p>
    </body>
    </html>
    """, status_code=200)


# ============================================================================
# EMBEDDED DASHBOARD (fallback if static file doesn't exist)
# ============================================================================

def get_embedded_dashboard():
    """Return the embedded dashboard HTML."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Alchemy Trading Bot</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: system-ui; background: #0d1117; color: #f0f6fc; padding: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 10px 0; }
        .title { font-size: 24px; font-weight: bold; margin-bottom: 20px; }
        .btn { padding: 12px 24px; border-radius: 8px; border: none; cursor: pointer; margin: 5px; }
        .btn-green { background: #238636; color: white; }
        .btn-red { background: #da3633; color: white; }
        .status { padding: 8px 16px; border-radius: 8px; display: inline-block; }
        .status-running { background: rgba(63,185,80,0.2); color: #3fb950; }
        .status-stopped { background: rgba(248,81,73,0.2); color: #f85149; }
        pre { background: #0d1117; padding: 15px; border-radius: 8px; overflow: auto; }
    </style>
</head>
<body>
    <div class="title">⚗️ Alchemy Trading Bot</div>
    
    <div class="card">
        <h3>Service Status</h3>
        <div id="status" class="status status-stopped">Loading...</div>
        <div style="margin-top: 15px;">
            <button class="btn btn-green" onclick="startService()">Start</button>
            <button class="btn btn-red" onclick="stopService()">Stop</button>
            <button class="btn" style="background:#30363d;color:white" onclick="refreshStatus()">Refresh</button>
        </div>
    </div>
    
    <div class="card">
        <h3>Status Details</h3>
        <pre id="details">Loading...</pre>
    </div>
    
    <script>
        async function refreshStatus() {
            try {
                const res = await fetch('/api/service/status');
                const data = await res.json();
                
                const statusEl = document.getElementById('status');
                statusEl.textContent = data.state || 'unknown';
                statusEl.className = 'status ' + (data.state === 'running' ? 'status-running' : 'status-stopped');
                
                document.getElementById('details').textContent = JSON.stringify(data, null, 2);
            } catch (e) {
                document.getElementById('status').textContent = 'Error: ' + e.message;
            }
        }
        
        async function startService() {
            await fetch('/api/service/start', {method: 'POST'});
            refreshStatus();
        }
        
        async function stopService() {
            await fetch('/api/service/stop', {method: 'POST'});
            refreshStatus();
        }
        
        refreshStatus();
        setInterval(refreshStatus, 5000);
    </script>
</body>
</html>
"""

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

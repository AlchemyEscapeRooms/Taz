"""
TAZ API Server
==============
Simple FastAPI backend for the Taz trading system.

Run with: python taz_api_server.py
Or: uvicorn taz_api_server:app --reload --port 8000
"""

import os
import sys
import json
import threading
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
load_dotenv()

# Add Taz to path
sys.path.insert(0, str(Path(__file__).parent / 'Taz'))

from Taz.taz_trader import TazTrader
from Taz.scanner.taz_scanner import TazScanner

# ============================================================================
# App Setup
# ============================================================================

app = FastAPI(
    title="Taz Trading API",
    description="API for the Tazmanian Devil aggressive crypto trading system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Taz folder for dashboard
app.mount("/Taz", StaticFiles(directory="Taz"), name="taz_static")

# ============================================================================
# Global State
# ============================================================================

taz_trader: Optional[TazTrader] = None
taz_scanner: Optional[TazScanner] = None
trader_thread: Optional[threading.Thread] = None


def get_trader() -> TazTrader:
    """Get or create TazTrader instance."""
    global taz_trader
    if taz_trader is None:
        taz_trader = TazTrader(
            initial_capital=1000,
            paper=True,
            max_position_pct=0.15,
            max_positions=3,
            stop_loss_pct=0.02,
            take_profit_pct=0.03,
            trade_crypto=True,
            trade_stocks=False,
            use_rl=True
        )
    return taz_trader


def get_scanner() -> TazScanner:
    """Get or create TazScanner instance."""
    global taz_scanner
    if taz_scanner is None:
        taz_scanner = TazScanner(min_volatility=2.0, min_volume_ratio=1.0)
    return taz_scanner


# ============================================================================
# Models
# ============================================================================

class TazStartCommand(BaseModel):
    capital: float = 1000
    paper: bool = True
    crypto: bool = True
    stocks: bool = False
    rl: bool = True


# ============================================================================
# Routes - Dashboard
# ============================================================================

@app.get("/")
async def root():
    """Redirect to Taz dashboard."""
    return HTMLResponse(content="""
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/Taz/taz_dashboard.html">
        </head>
        <body>
            <p>Redirecting to <a href="/Taz/taz_dashboard.html">Taz Dashboard</a>...</p>
        </body>
    </html>
    """)


@app.get("/dashboard")
async def dashboard():
    """Serve Taz dashboard."""
    return FileResponse("Taz/taz_dashboard.html")


# ============================================================================
# Routes - Status & Account
# ============================================================================

@app.get("/api/taz/status")
async def taz_status():
    """Get Taz trader status."""
    try:
        trader = get_trader()
        return trader.get_status()
    except Exception as e:
        return {
            "running": False,
            "error": str(e),
            "account": {"equity": 0, "cash": 0, "buying_power": 0},
            "positions": {},
            "stats": {"trades": 0, "win_rate": 0, "total_profit": 0}
        }


@app.get("/api/taz/account")
async def taz_account():
    """Get account info."""
    try:
        trader = get_trader()
        return trader.get_account()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/taz/positions")
async def taz_positions():
    """Get current positions."""
    try:
        trader = get_trader()
        return {
            "positions": trader.get_current_positions(),
            "tracked": {
                s: {
                    "symbol": p.symbol,
                    "asset_type": p.asset_type,
                    "entry_price": p.entry_price,
                    "strategy": p.strategy
                }
                for s, p in trader.positions.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/taz/stats")
async def taz_stats():
    """Get trading stats."""
    try:
        trader = get_trader()
        stats = trader.stats
        return {
            "trades_executed": stats.trades_executed,
            "winning_trades": stats.winning_trades,
            "losing_trades": stats.losing_trades,
            "total_profit": stats.total_profit,
            "biggest_win": stats.biggest_win,
            "biggest_loss": stats.biggest_loss,
            "best_streak": stats.best_streak,
            "worst_streak": stats.worst_streak,
            "win_rate": (stats.winning_trades / max(stats.trades_executed, 1)) * 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Routes - Scanner
# ============================================================================

@app.get("/api/taz/scanner")
async def taz_scanner_results():
    """Get latest scanner results."""
    try:
        scanner = get_scanner()
        results_file = Path("Taz/data/taz_scanner_results.json")

        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
            return data
        else:
            return {
                "scan_time": None,
                "total_opportunities": 0,
                "buy_signals": 0,
                "opportunities": []
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/taz/scan")
async def taz_run_scan():
    """Run a new scan."""
    try:
        scanner = get_scanner()

        # Run crypto scan (default for Taz)
        opportunities = scanner.scan_crypto()

        # Save results
        scanner.opportunities = opportunities
        scanner._save_results()

        buy_signals = [o for o in opportunities if o.signal == 'BUY']

        return {
            "status": "complete",
            "total_opportunities": len(opportunities),
            "buy_signals": len(buy_signals),
            "top_5": [o.to_dict() for o in opportunities[:5]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Routes - Trading Control
# ============================================================================

@app.post("/api/taz/start")
async def taz_start(cmd: TazStartCommand):
    """Start Taz trader."""
    global taz_trader, trader_thread

    try:
        # Create new trader
        taz_trader = TazTrader(
            initial_capital=cmd.capital,
            paper=cmd.paper,
            trade_crypto=cmd.crypto,
            trade_stocks=cmd.stocks,
            use_rl=cmd.rl,
            max_position_pct=0.15,
            stop_loss_pct=0.02
        )

        # Start in background thread
        trader_thread = threading.Thread(target=taz_trader.run, daemon=True)
        trader_thread.start()

        return {
            "status": "started",
            "mode": "paper" if cmd.paper else "live",
            "capital": cmd.capital,
            "crypto": cmd.crypto,
            "stocks": cmd.stocks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/taz/stop")
async def taz_stop():
    """Stop Taz trader."""
    global taz_trader

    try:
        if taz_trader and taz_trader.running:
            taz_trader.stop()
            return {"status": "stopped"}
        else:
            return {"status": "not_running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "trader_running": taz_trader.running if taz_trader else False
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("""
    ============================================================
                    TAZ API SERVER
    ============================================================
    Dashboard: http://localhost:8000
    API Docs:  http://localhost:8000/docs
    ============================================================
    """)

    uvicorn.run(app, host="0.0.0.0", port=8000)

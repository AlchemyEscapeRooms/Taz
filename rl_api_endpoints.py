"""
RL Trading System API Endpoints
================================
Endpoints for the RL (Reinforcement Learning) trading dashboard.
These are imported by api_server.py to extend the main FastAPI app.
"""

import json
import subprocess
import threading
import time as time_module
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

# Create router for RL endpoints
rl_router = APIRouter()

# Store for RL training processes
rl_training_processes = {}
rl_training_progress = {}
rl_shadow_process = None


@rl_router.get("/rl", response_class=HTMLResponse)
async def serve_rl_dashboard():
    """Serve the RL Trading Dashboard"""
    dashboard_path = Path(__file__).parent / "rl_dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>RL Dashboard not found</h1>", status_code=404)


@rl_router.get("/api/rl/models")
async def list_rl_models():
    """List all saved RL models"""
    models_dir = Path(__file__).parent / "rl_system" / "models"
    models = []

    if models_dir.exists():
        for config_file in models_dir.glob("*_best_config.json"):
            symbol = config_file.stem.replace("_best_config", "")
            weights_file = models_dir / f"{symbol}_best_model.weights.h5"

            if weights_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)

                    models.append({
                        "symbol": symbol,
                        "train_step": config_data.get("train_step", 0),
                        "epsilon": config_data.get("epsilon", 0),
                        "state_size": config_data.get("state_size", 0),
                        "config": config_data.get("config", {}),
                        "weights_file": str(weights_file),
                        "last_modified": weights_file.stat().st_mtime
                    })
                except Exception as e:
                    models.append({"symbol": symbol, "error": str(e)})

    return {"models": models}


@rl_router.delete("/api/rl/models/{symbol}")
async def delete_rl_model(symbol: str):
    """Delete a saved RL model"""
    models_dir = Path(__file__).parent / "rl_system" / "models"
    deleted = []

    for pattern in [f"{symbol}_best_model.weights.h5", f"{symbol}_best_target.weights.h5", f"{symbol}_best_config.json"]:
        file_path = models_dir / pattern
        if file_path.exists():
            file_path.unlink()
            deleted.append(pattern)

    if deleted:
        return {"success": True, "deleted": deleted}
    return {"success": False, "message": f"No model found for {symbol}"}


@rl_router.post("/api/rl/train")
async def start_rl_training(request: Request):
    """Start RL model training"""
    data = await request.json()
    symbol = data.get("symbol", "AAPL").upper()
    episodes = data.get("episodes", 100)
    days = data.get("days", 365)

    task_id = f"{symbol}_{int(time_module.time())}"
    rl_training_progress[task_id] = {
        "symbol": symbol, "episodes": episodes, "current_episode": 0,
        "status": "starting", "log": [], "started_at": time_module.time()
    }

    rl_system_dir = Path(__file__).parent / "rl_system"
    cmd = f"cd \"{rl_system_dir}\" && python rl_trader.py train --symbol {symbol} --episodes {episodes} --days {days}"

    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        rl_training_processes[task_id] = process
        rl_training_progress[task_id]["status"] = "running"
        rl_training_progress[task_id]["pid"] = process.pid

        def read_output():
            for line in iter(process.stdout.readline, ''):
                if line:
                    rl_training_progress[task_id]["log"].append(line.strip())
                    if "Episode" in line and "/" in line:
                        try:
                            parts = line.split("Episode")[1].split("/")
                            rl_training_progress[task_id]["current_episode"] = int(parts[0].strip())
                        except:
                            pass
            process.wait()
            rl_training_progress[task_id]["status"] = "completed" if process.returncode == 0 else "failed"

        threading.Thread(target=read_output, daemon=True).start()
        return {"success": True, "task_id": task_id, "message": f"Training started for {symbol}"}
    except Exception as e:
        rl_training_progress[task_id]["status"] = "failed"
        rl_training_progress[task_id]["error"] = str(e)
        return {"success": False, "error": str(e)}


@rl_router.get("/api/rl/progress/{task_id}")
async def get_rl_progress(task_id: str):
    """Get training progress for a task"""
    if task_id in rl_training_progress:
        progress = rl_training_progress[task_id].copy()
        progress["log"] = progress["log"][-50:]
        return progress
    return {"error": "Task not found"}


@rl_router.post("/api/rl/stop/{task_id}")
async def stop_rl_training(task_id: str):
    """Stop a running training task"""
    if task_id in rl_training_processes:
        try:
            rl_training_processes[task_id].terminate()
            rl_training_progress[task_id]["status"] = "stopped"
            return {"success": True, "message": "Training stopped"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {"success": False, "message": "Task not found"}


@rl_router.post("/api/rl/evaluate")
async def evaluate_rl_model(request: Request):
    """Evaluate an RL model"""
    data = await request.json()
    symbol = data.get("symbol", "AAPL").upper()
    days = data.get("days", 30)

    rl_system_dir = Path(__file__).parent / "rl_system"
    cmd = f"cd \"{rl_system_dir}\" && python rl_trader.py evaluate --symbol {symbol} --days {days}"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return {"success": True, "output": result.stdout, "error": result.stderr if result.returncode != 0 else None}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Evaluation timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@rl_router.post("/api/rl/compare")
async def compare_rl_strategies(request: Request):
    """Compare RL model vs SimpleTrader"""
    data = await request.json()
    symbol = data.get("symbol", "AAPL").upper()
    days = data.get("days", 30)

    rl_system_dir = Path(__file__).parent / "rl_system"
    cmd = f"cd \"{rl_system_dir}\" && python rl_trader.py compare --symbol {symbol} --days {days}"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        return {"success": True, "output": result.stdout, "error": result.stderr if result.returncode != 0 else None}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Comparison timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@rl_router.post("/api/rl/shadow/start")
async def start_shadow_trading(request: Request):
    """Start shadow trading mode"""
    global rl_shadow_process
    data = await request.json()
    symbol = data.get("symbol", "AAPL").upper()

    if rl_shadow_process is not None:
        return {"success": False, "message": "Shadow trading already running"}

    rl_system_dir = Path(__file__).parent / "rl_system"
    cmd = f"cd \"{rl_system_dir}\" && python rl_trader.py shadow --symbol {symbol}"

    try:
        rl_shadow_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return {"success": True, "message": f"Shadow trading started for {symbol}", "pid": rl_shadow_process.pid}
    except Exception as e:
        return {"success": False, "error": str(e)}


@rl_router.post("/api/rl/shadow/stop")
async def stop_shadow_trading():
    """Stop shadow trading mode"""
    global rl_shadow_process
    if rl_shadow_process is None:
        return {"success": False, "message": "Shadow trading not running"}
    try:
        rl_shadow_process.terminate()
        rl_shadow_process = None
        return {"success": True, "message": "Shadow trading stopped"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@rl_router.get("/api/rl/shadow/status")
async def get_shadow_status():
    """Get shadow trading status"""
    global rl_shadow_process
    if rl_shadow_process is None:
        return {"running": False}
    if rl_shadow_process.poll() is not None:
        rl_shadow_process = None
        return {"running": False}
    return {"running": True, "pid": rl_shadow_process.pid}

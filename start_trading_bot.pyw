#!/usr/bin/env python
"""
AI Trading Bot - Auto-Start Script

This script automatically starts the paper trading bot in the background.
Use .pyw extension on Windows for no console window.

Run this at startup to have the bot trading automatically every weekday.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Import and start the auto trader
from auto_trader import run_scheduler

if __name__ == "__main__":
    # This will:
    # 1. Set daily goals at 8:30 AM
    # 2. Start paper trading during market hours
    # 3. Generate end-of-day report at 4:05 PM
    run_scheduler()

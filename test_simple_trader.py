"""Test the simple trader calibration."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from simple_trader import SimpleTrader

trader = SimpleTrader(
    symbols=["AAPL", "TSLA", "NVDA", "SPY"],
    paper=True
)

trader.calibrate_all()

print()
print("STATUS:")
status = trader.get_status()
for symbol, strategy in status["strategies"].items():
    print(f"  {symbol}: {strategy}")

print()
print("Account:", status["account"])

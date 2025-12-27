"""Run simple trader for a short demo."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
from datetime import datetime
from simple_trader import SimpleTrader, STRATEGIES

print("="*60)
print("SIMPLE TRADER - DEMO RUN")
print("="*60)
print()

# Create trader with fewer stocks for demo
trader = SimpleTrader(
    symbols=["AAPL", "TSLA", "NVDA", "MSFT", "SPY"],
    paper=True,
    calibration_days=90,
    recalibrate_hours=24,
    position_size_pct=0.15,
    max_positions=15,
    min_hold_hours=4
)

# Calibrate
print("Step 1: Calibrating best strategy for each stock...")
print()
trader.calibrate_all()

# Show current status
print("\nStep 2: Current Status")
print("-"*60)
status = trader.get_status()

print(f"\nAccount:")
print(f"  Equity: ${status['account']['equity']:,.2f}")
print(f"  Cash: ${status['account']['cash']:,.2f}")
print(f"  Buying Power: ${status['account']['buying_power']:,.2f}")

print(f"\nStrategies Selected:")
for symbol, strategy in status['strategies'].items():
    print(f"  {symbol}: {strategy}")

print(f"\nCurrent Positions: {len(status['positions'])}")
for symbol, pos in status['positions'].items():
    print(f"  {symbol}: {pos['shares']} shares @ ${pos['avg_cost']:.2f} (P&L: ${pos['unrealized_pnl']:.2f})")

print(f"\nMarket Open: {status['market_open']}")

# Show current indicator values and signal status
print("\n" + "-"*60)
print("Step 3: Current Indicator Values & Signals")
print("-"*60)

current_positions = trader.get_current_positions()

for symbol in trader.symbols:
    config = trader.stock_configs.get(symbol)
    if not config:
        continue

    df = trader.fetch_latest_data(symbol)
    if df is None or len(df) < 2:
        print(f"\n{symbol}: No data available")
        continue

    strategy = config.strategy
    i = len(df) - 1
    row = df.iloc[i]
    prev_row = df.iloc[i-1]

    print(f"\n{symbol} - Strategy: {strategy.name}")
    print(f"  Last bar: {df.index[i]}")
    print(f"  Price: ${row['close']:.2f}")
    print(f"  RSI: {row['rsi']:.1f}")
    print(f"  MACD Diff: {row['macd_diff']:.3f} (prev: {prev_row['macd_diff']:.3f})")
    print(f"  Bollinger Position: {row['bb_position']:.2f} (0=low, 1=high)")
    print(f"  Volume Ratio: {row['volume_ratio']:.2f}x avg")

    has_position = symbol in current_positions
    buy_signal = strategy.check_buy(df, i)
    sell_signal = strategy.check_sell(df, i)

    print(f"  Has Position: {has_position}")
    print(f"  BUY Signal ({strategy.buy_rule}): {'YES!' if buy_signal else 'No'}")
    print(f"  SELL Signal ({strategy.sell_rule}): {'YES!' if sell_signal else 'No'}")

    if not has_position and buy_signal:
        print(f"  >>> WOULD BUY if market open")
    elif has_position and sell_signal:
        print(f"  >>> WOULD SELL if market open")

print("\n" + "="*60)
print("DEMO COMPLETE")
print("="*60)
print()
print("To run continuously: python simple_trader.py")
print("The bot will check every 15 minutes during market hours.")

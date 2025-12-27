"""Quick scanner test."""
from scanner.taz_scanner import TazScanner

scanner = TazScanner(min_volatility=0.0, min_volume_ratio=0.0)

# Scan just a few stocks
test_stocks = ['TSLA', 'NVDA', 'AMD', 'GME', 'COIN', 'MARA']
results = []

print("\nTAZ SCANNER TEST")
print("="*60)

for symbol in test_stocks:
    opp = scanner._analyze_stock(symbol)
    if opp:
        results.append(opp)
        print(f"{symbol:6} Score:{opp.score:5.1f} | Vol:{opp.volatility:4.1f}% | Signal:{opp.signal:5} | ${opp.current_price:.2f}")
    else:
        print(f"{symbol:6} - No data")

print()
print(f"Found {len(results)} opportunities")
print("="*60)

"""
Stock Discovery Scanner - Standalone Runner
============================================

Scans S&P 500 stocks for high-potential opportunities and auto-adds
top candidates to SimpleTrader.

Usage:
    python run_scanner.py                     # Single scan, show results
    python run_scanner.py --auto-add          # Auto-add to trader
    python run_scanner.py --continuous        # Run every 4 hours
    python run_scanner.py --batch 100         # Scan 100 stocks
    python run_scanner.py --min-score 80      # Set minimum score for auto-add
    python run_scanner.py --symbols AAPL,MSFT # Scan specific symbols
    python run_scanner.py --no-news           # Skip news (technicals only)

Author: Claude AI
Date: December 2024
"""

import argparse
import sys
import time
from datetime import datetime

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

from scanner.sp500_provider import SP500Provider
from scanner.stock_scanner import StockScanner
from simple_trader import SimpleTrader, get_portfolio_symbols
from config import config


def print_header(text: str):
    """Print a header line."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_results(results, limit: int = 20):
    """Print scan results in a table."""
    print(f"\n{'Rank':<5} {'Symbol':<8} {'Score':<8} {'Sent':<8} {'Tech':<8} {'Vol':<8} {'Mom':<8} {'News':<6}")
    print("-" * 75)

    for i, r in enumerate(results[:limit], 1):
        sent_str = f"{r.sentiment_score:+.2f}" if r.sentiment_score else "N/A"
        print(f"{i:<5} {r.symbol:<8} {r.composite_score:<8.1f} {sent_str:<8} "
              f"{r.technical_score:<8.1f} {r.volume_score:<8.1f} {r.momentum_score:<8.1f} {r.news_count:<6}")

        # Show top headline if available
        if r.news_headlines:
            headline = r.news_headlines[0][:55] + "..." if len(r.news_headlines[0]) > 55 else r.news_headlines[0]
            print(f"      -> {headline}")


def progress_callback(current: int, total: int, symbol: str):
    """Progress callback for scanning."""
    pct = current / total * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {pct:5.1f}% ({current}/{total}) {symbol:<8}", end="", flush=True)


def run_scan(args):
    """Run a single scan."""
    print_header(f"STOCK DISCOVERY SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize scanner
    paper = config.get('trading.mode', 'paper') != 'live'

    # Get SimpleTrader if auto-add is enabled
    simple_trader = None
    if args.auto_add:
        symbols = get_portfolio_symbols(paper=paper) or ['SPY']
        simple_trader = SimpleTrader(symbols=symbols, paper=paper)
        print(f"\nSimpleTrader initialized with {len(symbols)} existing symbols")
        print(f"Watchlist capacity: {simple_trader.get_watchlist_capacity()}")

    scanner = StockScanner(
        simple_trader=simple_trader,
        daily_news_budget=100,
        scan_batch_size=args.batch
    )

    # Get symbols to scan
    if args.symbols:
        symbols_to_scan = [s.strip().upper() for s in args.symbols.split(',')]
        print(f"\nScanning specified symbols: {', '.join(symbols_to_scan)}")
    else:
        provider = SP500Provider()
        symbols_to_scan = provider.get_symbols()
        print(f"\nLoaded {len(symbols_to_scan)} S&P 500 symbols")

    # Limit to batch size
    if len(symbols_to_scan) > args.batch:
        symbols_to_scan = symbols_to_scan[:args.batch]
        print(f"Scanning first {args.batch} symbols (use --batch to change)")

    # Show news budget
    status = scanner.get_status()
    print(f"News API budget: {status['news_budget_remaining']}/{status['news_budget_total']} remaining")

    # Run scan
    print(f"\nScanning {len(symbols_to_scan)} stocks...")
    fetch_news = not args.no_news

    results = scanner.scan_batch(
        symbols_to_scan,
        fetch_news=fetch_news,
        progress_callback=progress_callback
    )
    print()  # New line after progress bar

    # Show results
    print_header("TOP DISCOVERIES")
    print_results(results, limit=args.top)

    # Show summary
    print_header("SCAN SUMMARY")
    high_score = [r for r in results if r.composite_score >= 70]
    med_score = [r for r in results if 60 <= r.composite_score < 70]
    low_score = [r for r in results if r.composite_score < 60]

    print(f"High potential (70+):  {len(high_score)} stocks")
    print(f"Medium potential (60-70): {len(med_score)} stocks")
    print(f"Low potential (<60):   {len(low_score)} stocks")

    if high_score:
        print(f"\nHigh potential stocks: {', '.join(r.symbol for r in high_score[:10])}")

    # Auto-add if requested
    if args.auto_add:
        print_header("AUTO-ADD TO TRADER")
        added = scanner.auto_add_to_trader(
            min_score=args.min_score,
            max_additions=args.max_add
        )
        if added:
            print(f"Successfully added {len(added)} stocks: {', '.join(added)}")
            for symbol in added:
                result = next((r for r in results if r.symbol == symbol), None)
                if result:
                    print(f"  - {symbol}: Score {result.composite_score:.1f}, Strategy: {result.recommended_strategy}")
        else:
            print("No stocks met the criteria for auto-add.")
            print(f"(min_score: {args.min_score}, max_add: {args.max_add})")

    # Final status
    status = scanner.get_status()
    print(f"\nNews API budget remaining: {status['news_budget_remaining']}/{status['news_budget_total']}")


def main():
    parser = argparse.ArgumentParser(
        description='Stock Discovery Scanner - Find high-potential stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_scanner.py                     # Basic scan, show results
    python run_scanner.py --auto-add          # Auto-add top stocks to trader
    python run_scanner.py --batch 200         # Scan 200 stocks
    python run_scanner.py --continuous        # Run every 4 hours
    python run_scanner.py --symbols AAPL,NVDA # Scan specific symbols
    python run_scanner.py --no-news           # Skip news (save API budget)
        """
    )

    parser.add_argument(
        '--continuous', '-c',
        action='store_true',
        help='Run continuously every 4 hours'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=4,
        help='Hours between scans in continuous mode (default: 4)'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=50,
        help='Number of stocks to scan (default: 50)'
    )
    parser.add_argument(
        '--auto-add', '-a',
        action='store_true',
        help='Auto-add high-scoring stocks to SimpleTrader'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=60.0,
        help='Minimum score for auto-add (default: 60)'
    )
    parser.add_argument(
        '--max-add',
        type=int,
        default=3,
        help='Maximum stocks to auto-add per scan (default: 3)'
    )
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        default=None,
        help='Comma-separated symbols to scan (overrides S&P 500)'
    )
    parser.add_argument(
        '--top', '-t',
        type=int,
        default=20,
        help='Number of top results to display (default: 20)'
    )
    parser.add_argument(
        '--no-news',
        action='store_true',
        help='Skip news fetching (technicals only, saves API budget)'
    )

    args = parser.parse_args()

    if args.continuous:
        print(f"Running in continuous mode (every {args.interval} hours)")
        print("Press Ctrl+C to stop\n")

        while True:
            try:
                run_scan(args)
                print(f"\nNext scan in {args.interval} hours...")
                print("Press Ctrl+C to stop\n")
                time.sleep(args.interval * 60 * 60)
            except KeyboardInterrupt:
                print("\n\nStopping scanner...")
                break
    else:
        try:
            run_scan(args)
        except KeyboardInterrupt:
            print("\n\nScan interrupted.")
        except Exception as e:
            print(f"\nError: {e}")
            raise


if __name__ == '__main__':
    main()

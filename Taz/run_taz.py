#!/usr/bin/env python3
"""
RUN TAZ - Tazmanian Devil Trading System Launcher
==================================================

This script provides different modes to run Taz:
1. Full trader mode (scanner + RL + trading)
2. Scanner only mode (find opportunities)
3. Training mode (train RL agents)
4. Backtest mode (test strategies)

Usage:
    python run_taz.py                    # Full trading mode
    python run_taz.py --scan             # Scan only
    python run_taz.py --train TSLA       # Train RL on TSLA
    python run_taz.py --train-all        # Train on volatile stocks
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def run_trader(args):
    """Run the full Taz trader."""
    from taz_trader import TazTrader

    print("""
    ===========================================================
            TAZMANIAN DEVIL TRADING SYSTEM
    ===========================================================
         Purpose: Grow small accounts FAST
         Risk Level: AGGRESSIVE
    ===========================================================
    """)

    trader = TazTrader(
        initial_capital=args.capital,
        paper=not args.live,
        max_position_pct=args.position_size,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        check_interval=args.interval,
        trade_crypto=not args.no_crypto,
        use_rl=not args.no_rl
    )

    trader.run()


def run_scanner(args):
    """Run scanner only mode."""
    from scanner.taz_scanner import TazScanner

    print("\n[TAZ SCANNER] Starting volatility hunt...\n")

    scanner = TazScanner(
        min_volatility=args.min_volatility,
        min_volume_ratio=args.min_volume
    )

    if args.crypto_only:
        opportunities = scanner.scan_crypto()
    elif args.stocks_only:
        opportunities = scanner.scan_stocks()
    else:
        opportunities = scanner.scan_all()

    scanner.print_summary()

    # Show detailed top opportunities
    print("\n" + "="*70)
    print("DETAILED ANALYSIS - TOP 5 OPPORTUNITIES")
    print("="*70)

    for i, opp in enumerate(scanner.get_top_opportunities(5), 1):
        print(f"""
{i}. {opp.symbol} ({opp.asset_type.upper()})
   Score: {opp.score:.1f}/100
   Signal: {opp.signal} | Strategy: {opp.strategy}

   Price: ${opp.current_price:.4f}
   1h Change: {opp.price_change_1h:+.2f}%
   24h Change: {opp.price_change_24h:+.2f}%

   Volatility: {opp.volatility:.1f}%
   Volume Ratio: {opp.volume_ratio:.1f}x average
   RSI: {opp.rsi:.0f}
   MACD: {opp.macd_signal}
   Bollinger: {opp.bollinger_position:.0%} position
        """)


def run_training(args):
    """Train RL agent on specified symbols."""
    from rl.taz_rl_agent import TazRLTrainer

    trainer = TazRLTrainer()

    if args.train_all:
        # Train on multiple volatile symbols
        volatile_symbols = [
            'TSLA', 'NVDA', 'AMD', 'PLTR', 'COIN',
            'GME', 'AMC', 'MARA', 'RIOT'
        ]
        print(f"\n[TAZ TRAINING] Training on {len(volatile_symbols)} volatile stocks...")

        results = trainer.train_multi_symbol(
            symbols=volatile_symbols,
            episodes_per_symbol=args.episodes,
            asset_type='stock'
        )

        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        for symbol, result in results.items():
            if 'error' in result:
                print(f"{symbol}: ERROR - {result['error']}")
            else:
                print(f"{symbol}: Return={result['best_return']:.2%}, Trades={result['avg_trades']:.1f}")

    elif args.train_crypto:
        # Train on crypto
        crypto_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD']
        print(f"\n[TAZ TRAINING] Training on {len(crypto_symbols)} crypto pairs...")

        for symbol in crypto_symbols:
            try:
                result = trainer.train(
                    symbol=symbol,
                    episodes=args.episodes,
                    days=args.days,
                    asset_type='crypto'
                )
                print(f"{symbol}: Best Return = {result['best_return']:.2%}")
            except Exception as e:
                print(f"{symbol}: Error - {e}")

    else:
        # Train on single symbol
        symbol = args.symbol or 'TSLA'
        asset_type = 'crypto' if '/' in symbol else 'stock'

        print(f"\n[TAZ TRAINING] Training on {symbol}...")
        print(f"Episodes: {args.episodes}, Days: {args.days}")

        result = trainer.train(
            symbol=symbol,
            episodes=args.episodes,
            days=args.days,
            asset_type=asset_type
        )

        print(f"\n[TAZ] Training complete!")
        print(f"Best Return: {result['best_return']:.2%}")
        print(f"Avg Return: {result['avg_return']:.2%}")


def run_status():
    """Show current Taz status."""
    from taz_trader import TazTrader

    trader = TazTrader(paper=True)
    status = trader.get_status()

    print("""
    ===========================================================
                    TAZ STATUS REPORT
    ===========================================================
    """)

    account = status['account']
    print(f"  Account Equity: ${account.get('equity', 0):,.2f}")
    print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
    print(f"  Market Open: {'Yes' if status['market_open'] else 'No'}")
    print(f"  Crypto Enabled: {'Yes' if status['crypto_enabled'] else 'No'}")

    positions = status.get('positions', {})
    if positions:
        print(f"\n  Current Positions ({len(positions)}):")
        for symbol, pos in positions.items():
            pnl = pos.get('unrealized_pnl', 0)
            pnl_pct = pos.get('pnl_pct', 0)
            print(f"    {symbol}: ${pos.get('current_price', 0):.2f} ({pnl_pct:+.1f}%) P&L: ${pnl:+.2f}")
    else:
        print("\n  No open positions")

    stats = status.get('stats', {})
    print(f"\n  Stats:")
    print(f"    Trades: {stats.get('trades', 0)}")
    print(f"    Win Rate: {stats.get('win_rate', 0):.1f}%")
    print(f"    Total Profit: ${stats.get('total_profit', 0):,.2f}")


def run_backtest(args):
    """Run walk-forward backtesting - simulates live trading with NO future peeking."""
    from rl.taz_walk_forward import TazWalkForwardValidator, TazWalkForwardConfig

    print("""
    ===========================================================
    |         TAZ WALK-FORWARD BACKTEST                       |
    |                                                         |
    |  Simulates LIVE trading - agent only sees past data     |
    |  NO future peeking - tests on truly unseen data         |
    ===========================================================
    """)

    config = TazWalkForwardConfig(
        train_window_days=args.train_days,
        test_window_days=args.test_days,
        min_folds=args.min_folds
    )

    validator = TazWalkForwardValidator(config)

    if args.backtest_all:
        # Backtest multiple volatile symbols
        symbols = ['TSLA', 'NVDA', 'AMD', 'COIN', 'GME', 'MARA']
        print(f"\n[TAZ] Running walk-forward on {len(symbols)} symbols...")

        results = {}
        for symbol in symbols:
            try:
                result = validator.validate_symbol(
                    symbol=symbol,
                    asset_type='stock',
                    episodes_per_fold=args.episodes
                )
                results[symbol] = result
            except Exception as e:
                print(f"[TAZ] {symbol} failed: {e}")
                results[symbol] = {'passed': False, 'reason': str(e)}

        # Summary
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*60)
        passed = [s for s, r in results.items() if r.get('passed', False)]
        failed = [s for s, r in results.items() if not r.get('passed', False)]

        print(f"PASSED ({len(passed)}): {', '.join(passed) if passed else 'None'}")
        print(f"FAILED ({len(failed)}): {', '.join(failed) if failed else 'None'}")

    else:
        # Single symbol backtest
        symbol = args.backtest if isinstance(args.backtest, str) else 'TSLA'
        asset_type = 'crypto' if '/' in symbol or args.crypto else 'stock'

        print(f"\n[TAZ] Walk-forward validation for {symbol}...")
        print(f"      Train window: {args.train_days} days")
        print(f"      Test window:  {args.test_days} days")
        print(f"      Episodes/fold: {args.episodes}")
        print(f"\n      Agent will ONLY see data up to each test point.")
        print(f"      Just like live trading - no future peeking!\n")

        result = validator.validate_symbol(
            symbol=symbol,
            asset_type=asset_type,
            episodes_per_fold=args.episodes
        )

        if result['passed']:
            print(f"\n[PASS] {symbol} is validated for aggressive trading!")
        else:
            print(f"\n[FAIL] {symbol} needs more work: {result['reason']}")


def main():
    parser = argparse.ArgumentParser(
        description='Taz - Tazmanian Devil Aggressive Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_taz.py                     # Run full trader
  python run_taz.py --scan              # Scan for opportunities
  python run_taz.py --train TSLA        # Train RL on TSLA
  python run_taz.py --train-all         # Train on all volatile stocks
  python run_taz.py --backtest TSLA     # Walk-forward backtest (like live!)
  python run_taz.py --backtest-all      # Backtest all volatile stocks
  python run_taz.py --status            # Show status

Risk Warning:
  Taz is designed for AGGRESSIVE growth. High risk = high reward potential.
  Only use with money you can afford to lose.
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--scan', action='store_true', help='Scanner only mode')
    mode_group.add_argument('--train', type=str, nargs='?', const='TSLA', metavar='SYMBOL',
                           help='Train RL on symbol (default: TSLA)')
    mode_group.add_argument('--train-all', action='store_true', help='Train on all volatile stocks')
    mode_group.add_argument('--train-crypto', action='store_true', help='Train on crypto pairs')
    mode_group.add_argument('--backtest', type=str, nargs='?', const='TSLA', metavar='SYMBOL',
                           help='Walk-forward backtest (simulates live trading)')
    mode_group.add_argument('--backtest-all', action='store_true', help='Backtest all volatile stocks')
    mode_group.add_argument('--status', action='store_true', help='Show current status')

    # Trading parameters
    parser.add_argument('--capital', type=float, default=1000, help='Initial capital (default: 1000)')
    parser.add_argument('--live', action='store_true', help='Live trading (default: paper)')
    parser.add_argument('--position-size', type=float, default=0.40, help='Max position size (default: 0.40)')
    parser.add_argument('--max-positions', type=int, default=3, help='Max positions (default: 3)')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='Stop loss % (default: 0.05)')
    parser.add_argument('--take-profit', type=float, default=0.03, help='Take profit % (default: 0.03)')
    parser.add_argument('--interval', type=int, default=30, help='Check interval seconds (default: 30)')
    parser.add_argument('--no-crypto', action='store_true', help='Disable crypto trading')
    parser.add_argument('--no-rl', action='store_true', help='Disable RL agent')

    # Scanner parameters
    parser.add_argument('--min-volatility', type=float, default=2.0, help='Min volatility % (default: 2.0)')
    parser.add_argument('--min-volume', type=float, default=1.0, help='Min volume ratio (default: 1.0)')
    parser.add_argument('--stocks-only', action='store_true', help='Scan stocks only')
    parser.add_argument('--crypto-only', action='store_true', help='Scan crypto only')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=200, help='Training episodes (default: 200)')
    parser.add_argument('--days', type=int, default=60, help='Days of data for training (default: 60)')

    # Backtesting parameters
    parser.add_argument('--train-days', type=int, default=30, help='Walk-forward train window (default: 30)')
    parser.add_argument('--test-days', type=int, default=7, help='Walk-forward test window (default: 7)')
    parser.add_argument('--min-folds', type=int, default=4, help='Minimum folds for validation (default: 4)')
    parser.add_argument('--crypto', action='store_true', help='Treat symbol as crypto')

    args = parser.parse_args()

    # Assign symbol for training if provided
    args.symbol = args.train if isinstance(args.train, str) else None

    # Route to appropriate mode
    if args.status:
        run_status()
    elif args.scan:
        run_scanner(args)
    elif args.train or args.train_all or args.train_crypto:
        run_training(args)
    elif args.backtest or args.backtest_all:
        run_backtest(args)
    else:
        run_trader(args)


if __name__ == '__main__':
    main()

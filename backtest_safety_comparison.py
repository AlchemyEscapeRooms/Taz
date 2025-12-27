"""
TSLA Safety Rules Comparison Backtest
=====================================
Compares three configurations:
1. Safety OFF (aggressive - sell immediately on signal)
2. Safety ON with 24-hour hold period
3. Safety ON with 4-hour hold period

Using hourly bars from 01/01/2023 to yesterday.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv
load_dotenv()

# Import from simple_trader
from simple_trader import (
    calculate_rsi, calculate_macd, calculate_bollinger,
    calculate_volume_ratio, calculate_sma, STRATEGIES, backtest_strategy
)

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')


def fetch_tsla_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch TSLA hourly data for the specified date range."""
    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    print(f"Fetching TSLA hourly data from {start_date.date()} to {end_date.date()}...")

    # Try to fetch all at once first (Alpaca usually handles this well for hourly data)
    request = StockBarsRequest(
        symbol_or_symbols="TSLA",
        timeframe=TimeFrame.Hour,
        start=start_date,
        end=end_date
    )

    try:
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            raise ValueError("No data returned!")

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs("TSLA", level='symbol')

        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        print(f"Total bars fetched: {len(df)}")
        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        raise


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'], df['macd_diff'] = calculate_macd(df['close'])
    df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
    df['volume_ratio'] = calculate_volume_ratio(df['volume'])
    df['sma20'] = calculate_sma(df['close'], 20)
    df = df.dropna()
    return df


def run_backtest_with_config(
    df: pd.DataFrame,
    strategy_name: str,
    use_safety_rules: bool,
    min_hold_hours: int,
    stop_loss_pct: float,
    initial_capital: float = 10000
) -> Dict[str, Any]:
    """
    Run a backtest with specified configuration.
    """
    strategy_class = STRATEGIES.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy = strategy_class()

    # Run backtest
    result = backtest_strategy(
        df=df,
        strategy=strategy,
        initial_capital=initial_capital,
        symbol="TSLA",
        log_trades=False,
        min_hold_hours=min_hold_hours,
        stop_loss_pct=stop_loss_pct,
        use_safety_rules=use_safety_rules
    )

    return result


def find_best_strategy_for_config(
    df: pd.DataFrame,
    use_safety_rules: bool,
    min_hold_hours: int,
    stop_loss_pct: float = 0.10,
    initial_capital: float = 10000
) -> Dict[str, Any]:
    """
    Test all strategies and find the best one for given config.
    """
    results = []

    for name, strategy_class in STRATEGIES.items():
        strategy = strategy_class()
        result = backtest_strategy(
            df=df,
            strategy=strategy,
            initial_capital=initial_capital,
            symbol="TSLA",
            log_trades=False,
            min_hold_hours=min_hold_hours,
            stop_loss_pct=stop_loss_pct,
            use_safety_rules=use_safety_rules
        )
        results.append(result)

    # Sort by return
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    return results[0], results  # Return best and all results


def run_comparison():
    """Run the three-way comparison."""

    # Date range: 01/01/2023 to yesterday
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now() - timedelta(days=1)

    print("=" * 70)
    print("TSLA SAFETY RULES COMPARISON BACKTEST")
    print("=" * 70)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Timeframe: Hourly bars")
    print("=" * 70)

    # Fetch data
    df = fetch_tsla_data(start_date, end_date)
    df = add_indicators(df)

    print(f"\nData ready: {len(df)} bars with indicators")
    print(f"Date range in data: {df.index[0]} to {df.index[-1]}")

    # Configuration for the three tests
    configs = [
        {
            "name": "SAFETY OFF (Aggressive)",
            "use_safety_rules": False,
            "min_hold_hours": 0,  # Not used when safety is off
            "stop_loss_pct": 0.10
        },
        {
            "name": "SAFETY ON (24-hour hold)",
            "use_safety_rules": True,
            "min_hold_hours": 24,
            "stop_loss_pct": 0.10
        },
        {
            "name": "SAFETY ON (4-hour hold)",
            "use_safety_rules": True,
            "min_hold_hours": 4,
            "stop_loss_pct": 0.10
        }
    ]

    all_results = []

    print("\n" + "=" * 70)
    print("RUNNING BACKTESTS...")
    print("=" * 70)

    for config in configs:
        print(f"\n--- {config['name']} ---")

        best_result, all_strategy_results = find_best_strategy_for_config(
            df=df,
            use_safety_rules=config['use_safety_rules'],
            min_hold_hours=config['min_hold_hours'],
            stop_loss_pct=config['stop_loss_pct']
        )

        # Store results
        config_result = {
            "config_name": config['name'],
            "use_safety_rules": config['use_safety_rules'],
            "min_hold_hours": config['min_hold_hours'],
            "stop_loss_pct": config['stop_loss_pct'],
            "best_strategy": best_result['name'],
            "return_pct": best_result['return_pct'],
            "win_rate": best_result['win_rate'],
            "total_trades": best_result['trades'],
            "winners": best_result['winners'],
            "all_strategies": all_strategy_results
        }
        all_results.append(config_result)

        print(f"Best strategy: {best_result['name']}")
        print(f"Return: {best_result['return_pct']:+.2f}%")
        print(f"Win rate: {best_result['win_rate']:.1f}%")
        print(f"Trades: {best_result['trades']} ({best_result['winners']} winners)")

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Configuration':<30} {'Strategy':<20} {'Return':>10} {'Win Rate':>10} {'Trades':>8}")
    print("-" * 78)

    for r in all_results:
        print(f"{r['config_name']:<30} {r['best_strategy']:<20} {r['return_pct']:>+9.2f}% {r['win_rate']:>9.1f}% {r['total_trades']:>8}")

    # Detailed breakdown
    print("\n" + "=" * 70)
    print("DETAILED BREAKDOWN BY STRATEGY")
    print("=" * 70)

    for r in all_results:
        print(f"\n--- {r['config_name']} ---")
        print(f"{'Strategy':<20} {'Return':>10} {'Win Rate':>10} {'Trades':>8} {'Winners':>8}")
        print("-" * 56)

        # Sort by return for display
        sorted_strategies = sorted(r['all_strategies'], key=lambda x: x['return_pct'], reverse=True)
        for s in sorted_strategies[:5]:  # Top 5 strategies
            print(f"{s['name']:<20} {s['return_pct']:>+9.2f}% {s['win_rate']:>9.1f}% {s['trades']:>8} {s['winners']:>8}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best_config = max(all_results, key=lambda x: x['return_pct'])
    worst_config = min(all_results, key=lambda x: x['return_pct'])

    print(f"\nBest performing: {best_config['config_name']}")
    print(f"  - Return: {best_config['return_pct']:+.2f}%")
    print(f"  - Strategy: {best_config['best_strategy']}")
    print(f"  - Win Rate: {best_config['win_rate']:.1f}%")

    print(f"\nWorst performing: {worst_config['config_name']}")
    print(f"  - Return: {worst_config['return_pct']:+.2f}%")
    print(f"  - Strategy: {worst_config['best_strategy']}")
    print(f"  - Win Rate: {worst_config['win_rate']:.1f}%")

    # Compare hold periods specifically
    safety_on_24h = next(r for r in all_results if r['min_hold_hours'] == 24)
    safety_on_4h = next(r for r in all_results if r['min_hold_hours'] == 4)

    print(f"\n4-hour vs 24-hour hold comparison:")
    print(f"  24-hour hold: {safety_on_24h['return_pct']:+.2f}% return, {safety_on_24h['win_rate']:.1f}% win rate")
    print(f"  4-hour hold:  {safety_on_4h['return_pct']:+.2f}% return, {safety_on_4h['win_rate']:.1f}% win rate")
    diff = safety_on_4h['return_pct'] - safety_on_24h['return_pct']
    print(f"  Difference: {diff:+.2f}% (4-hour {'outperforms' if diff > 0 else 'underperforms'})")

    return all_results


if __name__ == "__main__":
    results = run_comparison()

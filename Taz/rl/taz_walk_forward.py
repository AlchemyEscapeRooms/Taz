"""
TAZ Walk-Forward Backtesting System
====================================
Aggressive walk-forward validation designed for fast-profit strategies.

Key differences from standard walk-forward:
- Shorter training windows (recent data matters more)
- More lenient criteria (aggressive trading = more variance)
- Focus on profit velocity, not just returns
- Integrated with Taz RL agent

Usage:
    python taz_walk_forward.py --symbol TSLA --folds 10
    python taz_walk_forward.py --symbol BTC/USD --crypto
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TazWalkForward')


@dataclass
class TazWalkForwardConfig:
    """Configuration optimized for aggressive trading."""
    # Shorter windows for aggressive trading
    train_window_days: int = 30       # 30 days training (not 180)
    test_window_days: int = 7         # 7 days testing (not 30)

    # Minimum folds
    min_folds: int = 4                # At least 4 folds

    # More lenient criteria for aggressive trading
    min_test_return: float = -0.05    # Allow up to 5% loss (volatile)
    min_avg_return: float = 0.02      # Average must be 2%+ profitable
    min_test_win_rate: float = 0.35   # Lower win rate OK if wins are big
    min_profit_factor: float = 1.2    # Wins > losses by 20%

    # Pass requirements
    min_pass_rate: float = 0.5        # 50% of folds must be profitable
    min_test_trades: int = 3          # At least 3 trades per fold


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""
    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Training metrics
    train_return: float
    train_trades: int
    train_win_rate: float

    # Testing metrics (out of sample)
    test_return: float
    test_trades: int
    test_win_rate: float
    test_profit_factor: float
    test_biggest_win: float
    test_biggest_loss: float

    # Validation
    passed: bool
    fail_reason: str = ""


class TazWalkForwardValidator:
    """
    Walk-forward validation for Taz aggressive trading.

    Validates that the RL agent can profit on unseen data,
    not just memorize historical patterns.
    """

    def __init__(self, config: TazWalkForwardConfig = None):
        self.config = config or TazWalkForwardConfig()
        self.results: List[FoldResult] = []
        self.results_dir = Path(__file__).parent / "taz_validation_results"
        self.results_dir.mkdir(exist_ok=True)

        # Alpaca clients
        self.stock_client = None
        self.crypto_client = None
        self._init_clients()

    def _init_clients(self):
        """Initialize data clients."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient

            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')

            if api_key and secret_key:
                self.stock_client = StockHistoricalDataClient(api_key, secret_key)
                self.crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
        except Exception as e:
            logger.error(f"Failed to init clients: {e}")

    def fetch_data(self, symbol: str, days: int = 120, asset_type: str = 'stock') -> pd.DataFrame:
        """Fetch historical data for validation."""
        from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = datetime.now()
        start = end - timedelta(days=days)

        if asset_type == 'crypto':
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )
            bars = self.crypto_client.get_crypto_bars(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )
            bars = self.stock_client.get_stock_bars(request)

        df = bars.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')

        return df[['open', 'high', 'low', 'close', 'volume']].copy()

    def create_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create train/test splits for walk-forward validation."""
        folds = []

        # Convert days to hours (hourly data)
        train_bars = self.config.train_window_days * 24
        test_bars = self.config.test_window_days * 24
        fold_size = train_bars + test_bars

        total_bars = len(df)

        # Need minimum data
        if total_bars < fold_size * 2:
            logger.warning(f"Not enough data: {total_bars} bars, need {fold_size * 2}")
            return []

        start_idx = 0
        while start_idx + fold_size <= total_bars:
            train_end = start_idx + train_bars
            test_end = train_end + test_bars

            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            folds.append((train_df, test_df))

            # Move forward by test window (non-overlapping tests)
            start_idx += test_bars

        logger.info(f"Created {len(folds)} walk-forward folds")
        return folds

    def validate_symbol(
        self,
        symbol: str,
        asset_type: str = 'stock',
        episodes_per_fold: int = 50
    ) -> Dict:
        """
        Run full walk-forward validation on a symbol.

        Args:
            symbol: Stock or crypto symbol
            asset_type: 'stock' or 'crypto'
            episodes_per_fold: Training episodes per fold

        Returns:
            Validation summary
        """
        from rl_system.taz_rl_agent import TazRLTrainer, TazTradingEnvironment, TazRLConfig

        logger.info(f"\n{'='*60}")
        logger.info(f"TAZ WALK-FORWARD VALIDATION: {symbol}")
        logger.info(f"{'='*60}")

        # Fetch data - need enough for multiple folds
        days_needed = (self.config.train_window_days + self.config.test_window_days) * (self.config.min_folds + 2)
        days_needed = min(days_needed, 90)  # Cap at 90 days due to API limits
        df = self.fetch_data(symbol, days=days_needed, asset_type=asset_type)

        if len(df) < 500:
            return {
                'symbol': symbol,
                'passed': False,
                'reason': f'Insufficient data: {len(df)} bars'
            }

        # Create folds
        folds = self.create_folds(df)

        if len(folds) < self.config.min_folds:
            return {
                'symbol': symbol,
                'passed': False,
                'reason': f'Need {self.config.min_folds} folds, got {len(folds)}'
            }

        self.results = []
        config = TazRLConfig()

        for i, (train_df, test_df) in enumerate(folds):
            logger.info(f"\n--- Fold {i+1}/{len(folds)} ---")

            try:
                # Train on this fold
                train_env = TazTradingEnvironment(train_df, config)
                from rl_system.taz_rl_agent import TazDQNAgent
                agent = TazDQNAgent(train_env.state_size, config.action_size, config)

                # Training loop
                for episode in range(episodes_per_fold):
                    state = train_env.reset()
                    while True:
                        action = agent.act(state)
                        next_state, reward, done, info = train_env.step(action)
                        agent.remember(state, action, reward, next_state, done)
                        agent.replay()
                        state = next_state
                        if done:
                            break

                train_stats = train_env.get_episode_stats()

                # Test on unseen data
                test_env = TazTradingEnvironment(test_df, config)
                state = test_env.reset()

                while True:
                    action = agent.act(state, training=False)
                    next_state, reward, done, info = test_env.step(action)
                    state = next_state
                    if done:
                        break

                test_stats = test_env.get_episode_stats()

                # Calculate profit factor
                wins = sum(t.get('pnl', 0) for t in test_stats['trades'] if t.get('pnl', 0) > 0)
                losses = abs(sum(t.get('pnl', 0) for t in test_stats['trades'] if t.get('pnl', 0) < 0))
                profit_factor = wins / losses if losses > 0 else float('inf') if wins > 0 else 0

                # Biggest win/loss
                pnls = [t.get('pnl', 0) for t in test_stats['trades'] if t.get('pnl') is not None]
                biggest_win = max(pnls) if pnls else 0
                biggest_loss = min(pnls) if pnls else 0

                # Check if fold passes
                passed, fail_reason = self._check_fold(test_stats, profit_factor)

                result = FoldResult(
                    fold_num=i + 1,
                    train_start=str(train_df.index[0]),
                    train_end=str(train_df.index[-1]),
                    test_start=str(test_df.index[0]),
                    test_end=str(test_df.index[-1]),
                    train_return=train_stats['total_return'],
                    train_trades=train_stats['total_trades'],
                    train_win_rate=train_stats['win_rate'],
                    test_return=test_stats['total_return'],
                    test_trades=test_stats['total_trades'],
                    test_win_rate=test_stats['win_rate'],
                    test_profit_factor=profit_factor,
                    test_biggest_win=biggest_win,
                    test_biggest_loss=biggest_loss,
                    passed=passed,
                    fail_reason=fail_reason
                )

                self.results.append(result)

                status = "PASS" if passed else "FAIL"
                logger.info(f"Fold {i+1}: Test Return={test_stats['total_return']:.2%}, "
                           f"Trades={test_stats['total_trades']}, "
                           f"WinRate={test_stats['win_rate']:.1%}, "
                           f"PF={profit_factor:.2f} [{status}]")

            except Exception as e:
                logger.error(f"Fold {i+1} failed: {e}")
                import traceback
                traceback.print_exc()

        # Calculate summary
        summary = self._calculate_summary(symbol, asset_type)
        self._save_results(symbol, summary)
        self._print_summary(summary)

        return summary

    def _check_fold(self, test_stats: Dict, profit_factor: float) -> Tuple[bool, str]:
        """Check if a fold passes criteria."""

        # Check minimum trades
        if test_stats['total_trades'] < self.config.min_test_trades:
            return False, f"Too few trades: {test_stats['total_trades']}"

        # Check return (lenient - allows some loss)
        if test_stats['total_return'] < self.config.min_test_return:
            return False, f"Return too low: {test_stats['total_return']:.2%}"

        # Check win rate
        if test_stats['win_rate'] < self.config.min_test_win_rate:
            return False, f"Win rate too low: {test_stats['win_rate']:.1%}"

        # Check profit factor
        if profit_factor < self.config.min_profit_factor:
            return False, f"Profit factor too low: {profit_factor:.2f}"

        return True, ""

    def _calculate_summary(self, symbol: str, asset_type: str) -> Dict:
        """Calculate overall validation summary."""
        if not self.results:
            return {
                'symbol': symbol,
                'asset_type': asset_type,
                'passed': False,
                'reason': 'No folds completed'
            }

        passed_folds = [r for r in self.results if r.passed]
        pass_rate = len(passed_folds) / len(self.results)

        test_returns = [r.test_return for r in self.results]
        avg_return = np.mean(test_returns)

        # Overall pass criteria
        overall_passed = (
            pass_rate >= self.config.min_pass_rate and
            avg_return >= self.config.min_avg_return
        )

        reason = ""
        if not overall_passed:
            if pass_rate < self.config.min_pass_rate:
                reason = f"Pass rate {pass_rate:.0%} < {self.config.min_pass_rate:.0%}"
            elif avg_return < self.config.min_avg_return:
                reason = f"Avg return {avg_return:.2%} < {self.config.min_avg_return:.1%}"

        return {
            'symbol': symbol,
            'asset_type': asset_type,
            'passed': overall_passed,
            'reason': reason if not overall_passed else 'All criteria met',

            # Fold statistics
            'total_folds': len(self.results),
            'passed_folds': len(passed_folds),
            'pass_rate': pass_rate,

            # Return statistics
            'avg_test_return': avg_return,
            'std_test_return': np.std(test_returns),
            'min_test_return': min(test_returns),
            'max_test_return': max(test_returns),
            'total_cumulative_return': sum(test_returns),

            # Trade statistics
            'avg_trades_per_fold': np.mean([r.test_trades for r in self.results]),
            'avg_win_rate': np.mean([r.test_win_rate for r in self.results]),
            'avg_profit_factor': np.mean([r.test_profit_factor for r in self.results if r.test_profit_factor != float('inf')]),

            # Details
            'fold_results': [asdict(r) for r in self.results],
            'validated_at': datetime.now().isoformat(),
            'config': asdict(self.config)
        }

    def _save_results(self, symbol: str, summary: Dict):
        """Save validation results."""
        safe_symbol = symbol.replace('/', '_')
        filepath = self.results_dir / f"taz_{safe_symbol}_validation.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

    def _print_summary(self, summary: Dict):
        """Print validation summary."""
        status = "PASSED" if summary.get('passed', False) else "FAILED"

        # Handle case where no folds completed
        if 'passed_folds' not in summary:
            print(f"""
================================================================
         TAZ WALK-FORWARD VALIDATION RESULTS
================================================================
  Symbol: {summary.get('symbol', 'N/A'):12}
  Status: {status:12}
  Reason: {summary.get('reason', 'Unknown')[:45]:45}
================================================================
            """)
            return

        print(f"""
================================================================
         TAZ WALK-FORWARD VALIDATION RESULTS
================================================================
  Symbol: {summary['symbol']:12}  Type: {summary['asset_type']:10}
  Status: {status:12}
  Reason: {summary['reason'][:45]:45}
----------------------------------------------------------------
  Folds: {summary['passed_folds']}/{summary['total_folds']} passed ({summary['pass_rate']:.0%})

  Average Test Return:   {summary['avg_test_return']:+.2%}
  Best Fold Return:      {summary['max_test_return']:+.2%}
  Worst Fold Return:     {summary['min_test_return']:+.2%}
  Cumulative Return:     {summary['total_cumulative_return']:+.2%}

  Avg Trades/Fold:       {summary['avg_trades_per_fold']:.1f}
  Avg Win Rate:          {summary['avg_win_rate']:.1%}
  Avg Profit Factor:     {summary['avg_profit_factor']:.2f}
================================================================
        """)


def main():
    """Run walk-forward validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Taz Walk-Forward Validation')
    parser.add_argument('--symbol', default='TSLA', help='Symbol to validate')
    parser.add_argument('--crypto', action='store_true', help='Symbol is crypto')
    parser.add_argument('--train-days', type=int, default=30, help='Training window days')
    parser.add_argument('--test-days', type=int, default=7, help='Testing window days')
    parser.add_argument('--episodes', type=int, default=50, help='Training episodes per fold')

    args = parser.parse_args()

    config = TazWalkForwardConfig(
        train_window_days=args.train_days,
        test_window_days=args.test_days
    )

    validator = TazWalkForwardValidator(config)

    asset_type = 'crypto' if args.crypto else 'stock'

    result = validator.validate_symbol(
        symbol=args.symbol,
        asset_type=asset_type,
        episodes_per_fold=args.episodes
    )

    if result['passed']:
        print(f"\n✅ {args.symbol} PASSED walk-forward validation!")
        print("   Model is likely to perform on unseen data.")
    else:
        print(f"\n❌ {args.symbol} FAILED walk-forward validation.")
        print(f"   Reason: {result['reason']}")
        print("   Consider more training or different parameters.")


if __name__ == '__main__':
    main()

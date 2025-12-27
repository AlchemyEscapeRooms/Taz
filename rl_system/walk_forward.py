"""
Walk-Forward Validation for RL Trading System
==============================================
Prevents overfitting by:
1. Training on rolling windows
2. Validating on out-of-sample forward data
3. Only promoting models that pass validation

This is the "adult supervision" for the RL kid.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger('WalkForward')


@dataclass
class ValidationResult:
    """Result from a single walk-forward fold."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_return: float
    test_return: float
    train_sharpe: float
    test_sharpe: float
    train_win_rate: float
    test_win_rate: float
    test_trades: int
    passed: bool


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    train_window_days: int = 180    # 6 months training
    test_window_days: int = 30      # 1 month testing
    min_folds: int = 6              # At least 6 folds for significance
    min_test_return: float = 0.0    # Must be profitable in test
    min_test_sharpe: float = 0.5    # Minimum Sharpe ratio
    min_test_win_rate: float = 0.45 # At least 45% win rate
    min_pass_rate: float = 0.6      # 60% of folds must pass
    min_test_trades: int = 5        # Minimum trades per fold


class WalkForwardValidator:
    """
    Walk-forward validation prevents overfitting by:
    - Training on historical windows
    - Testing on unseen forward data
    - Requiring consistent performance across multiple folds
    """

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.results: List[ValidationResult] = []
        self.results_dir = Path(__file__).parent / "validation_results"
        self.results_dir.mkdir(exist_ok=True)

    def create_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create train/test splits for walk-forward validation.

        Returns list of (train_df, test_df) tuples.
        """
        folds = []
        total_days = len(df)
        fold_size = self.config.train_window_days + self.config.test_window_days

        start_idx = 0
        while start_idx + fold_size <= total_days:
            train_end = start_idx + self.config.train_window_days
            test_end = train_end + self.config.test_window_days

            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            folds.append((train_df, test_df))

            # Move forward by test window size (non-overlapping test sets)
            start_idx += self.config.test_window_days

        logger.info(f"Created {len(folds)} walk-forward folds")
        return folds

    def validate_model(
        self,
        symbol: str,
        df: pd.DataFrame,
        train_func,  # Function that trains and returns agent
        evaluate_func  # Function that evaluates agent on data
    ) -> Dict:
        """
        Run full walk-forward validation.

        Args:
            symbol: Stock symbol
            df: Full historical DataFrame
            train_func: Function(train_df) -> trained_agent
            evaluate_func: Function(agent, test_df) -> metrics dict

        Returns:
            Validation summary with pass/fail status
        """
        self.results = []
        folds = self.create_folds(df)

        if len(folds) < self.config.min_folds:
            logger.warning(f"Not enough data for {self.config.min_folds} folds, got {len(folds)}")
            return {
                'symbol': symbol,
                'passed': False,
                'reason': f"Insufficient data for {self.config.min_folds} folds",
                'folds_available': len(folds)
            }

        for i, (train_df, test_df) in enumerate(folds):
            try:
                # Train on this fold
                agent = train_func(train_df)

                # Evaluate on train data
                train_metrics = evaluate_func(agent, train_df)

                # Evaluate on test data (out-of-sample)
                test_metrics = evaluate_func(agent, test_df)

                # Check if this fold passes
                passed = self._check_fold_passed(test_metrics)

                result = ValidationResult(
                    fold=i,
                    train_start=str(train_df.index[0]) if hasattr(train_df.index[0], 'isoformat') else str(train_df.index[0]),
                    train_end=str(train_df.index[-1]) if hasattr(train_df.index[-1], 'isoformat') else str(train_df.index[-1]),
                    test_start=str(test_df.index[0]) if hasattr(test_df.index[0], 'isoformat') else str(test_df.index[0]),
                    test_end=str(test_df.index[-1]) if hasattr(test_df.index[-1], 'isoformat') else str(test_df.index[-1]),
                    train_return=train_metrics.get('total_return', 0),
                    test_return=test_metrics.get('total_return', 0),
                    train_sharpe=train_metrics.get('sharpe_ratio', 0),
                    test_sharpe=test_metrics.get('sharpe_ratio', 0),
                    train_win_rate=train_metrics.get('win_rate', 0),
                    test_win_rate=test_metrics.get('win_rate', 0),
                    test_trades=test_metrics.get('total_trades', 0),
                    passed=passed
                )

                self.results.append(result)
                logger.info(f"Fold {i}: Test return={test_metrics.get('total_return', 0):.2%}, "
                           f"Sharpe={test_metrics.get('sharpe_ratio', 0):.2f}, "
                           f"Passed={passed}")

            except Exception as e:
                logger.error(f"Fold {i} failed: {e}")
                continue

        # Calculate overall results
        summary = self._calculate_summary(symbol)
        self._save_results(symbol, summary)

        return summary

    def _check_fold_passed(self, metrics: Dict) -> bool:
        """Check if a single fold passes the criteria."""
        return (
            metrics.get('total_return', -1) >= self.config.min_test_return and
            metrics.get('sharpe_ratio', -1) >= self.config.min_test_sharpe and
            metrics.get('win_rate', 0) >= self.config.min_test_win_rate and
            metrics.get('total_trades', 0) >= self.config.min_test_trades
        )

    def _calculate_summary(self, symbol: str) -> Dict:
        """Calculate overall validation summary."""
        if not self.results:
            return {
                'symbol': symbol,
                'passed': False,
                'reason': 'No valid folds completed'
            }

        passed_folds = sum(1 for r in self.results if r.passed)
        total_folds = len(self.results)
        pass_rate = passed_folds / total_folds

        # Aggregate metrics across folds
        test_returns = [r.test_return for r in self.results]
        test_sharpes = [r.test_sharpe for r in self.results]
        test_win_rates = [r.test_win_rate for r in self.results]

        overall_passed = pass_rate >= self.config.min_pass_rate

        reason = ""
        if not overall_passed:
            reason = f"Only {pass_rate:.0%} of folds passed (need {self.config.min_pass_rate:.0%})"

        return {
            'symbol': symbol,
            'passed': overall_passed,
            'reason': reason if not overall_passed else 'All criteria met',
            'total_folds': total_folds,
            'passed_folds': passed_folds,
            'pass_rate': pass_rate,
            'avg_test_return': np.mean(test_returns),
            'std_test_return': np.std(test_returns),
            'avg_test_sharpe': np.mean(test_sharpes),
            'avg_test_win_rate': np.mean(test_win_rates),
            'min_test_return': min(test_returns),
            'max_test_return': max(test_returns),
            'fold_results': [asdict(r) for r in self.results],
            'validated_at': datetime.now().isoformat(),
            'config': asdict(self.config)
        }

    def _save_results(self, symbol: str, summary: Dict):
        """Save validation results to disk."""
        filepath = self.results_dir / f"{symbol}_validation.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved validation results to {filepath}")

    def load_results(self, symbol: str) -> Optional[Dict]:
        """Load previous validation results."""
        filepath = self.results_dir / f"{symbol}_validation.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return None


class PromotionGate:
    """
    Gate that controls whether an RL model can be promoted from
    shadow trading to live trading.

    Requirements for promotion:
    1. Pass walk-forward validation
    2. Meet minimum shadow trading period
    3. Shadow trades show positive expected value
    """

    def __init__(self):
        self.min_shadow_days = 30           # Minimum shadow period
        self.min_shadow_trades = 100        # Minimum shadow trades
        self.min_shadow_agreement = 0.4     # Min agreement with live
        self.min_shadow_return = 0.0        # Positive simulated return
        self.promotion_log = Path(__file__).parent / "promotion_log.json"

    def check_promotion_ready(
        self,
        symbol: str,
        validation_results: Dict,
        shadow_stats: Dict
    ) -> Dict:
        """
        Check if model is ready for promotion to live trading.

        Args:
            symbol: Stock symbol
            validation_results: Results from walk-forward validation
            shadow_stats: Statistics from shadow trading period

        Returns:
            Dict with ready status and reasons
        """
        checks = []
        all_passed = True

        # Check 1: Walk-forward validation passed
        wf_passed = validation_results.get('passed', False)
        checks.append({
            'name': 'Walk-Forward Validation',
            'passed': wf_passed,
            'detail': validation_results.get('reason', 'Not validated')
        })
        if not wf_passed:
            all_passed = False

        # Check 2: Minimum shadow trading period
        shadow_days = shadow_stats.get('shadow_days', 0)
        days_passed = shadow_days >= self.min_shadow_days
        checks.append({
            'name': 'Minimum Shadow Period',
            'passed': days_passed,
            'detail': f"{shadow_days} days (need {self.min_shadow_days})"
        })
        if not days_passed:
            all_passed = False

        # Check 3: Minimum shadow trades
        shadow_trades = shadow_stats.get('total_signals', 0)
        trades_passed = shadow_trades >= self.min_shadow_trades
        checks.append({
            'name': 'Minimum Shadow Trades',
            'passed': trades_passed,
            'detail': f"{shadow_trades} trades (need {self.min_shadow_trades})"
        })
        if not trades_passed:
            all_passed = False

        # Check 4: Agreement rate with live trader
        agreement_rate = shadow_stats.get('agreement_rate', 0)
        agreement_passed = agreement_rate >= self.min_shadow_agreement
        checks.append({
            'name': 'Agreement Rate',
            'passed': agreement_passed,
            'detail': f"{agreement_rate:.1%} (need {self.min_shadow_agreement:.1%})"
        })
        if not agreement_passed:
            all_passed = False

        # Check 5: Simulated shadow return positive
        shadow_return = shadow_stats.get('simulated_return', 0)
        return_passed = shadow_return >= self.min_shadow_return
        checks.append({
            'name': 'Shadow Return',
            'passed': return_passed,
            'detail': f"{shadow_return:.2%} (need >= {self.min_shadow_return:.1%})"
        })
        if not return_passed:
            all_passed = False

        result = {
            'symbol': symbol,
            'ready_for_promotion': all_passed,
            'checks': checks,
            'checked_at': datetime.now().isoformat()
        }

        # Log the check
        self._log_check(result)

        return result

    def _log_check(self, result: Dict):
        """Log promotion check to disk."""
        log = []
        if self.promotion_log.exists():
            with open(self.promotion_log, 'r') as f:
                log = json.load(f)

        log.append(result)

        # Keep last 100 checks
        log = log[-100:]

        with open(self.promotion_log, 'w') as f:
            json.dump(log, f, indent=2)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return float(np.sqrt(252) * excess_returns.mean() / returns.std())

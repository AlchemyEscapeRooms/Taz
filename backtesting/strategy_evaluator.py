"""Strategy evaluation and selection based on performance."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from backtesting.backtest_engine import BacktestEngine
from backtesting.strategies import STRATEGY_REGISTRY, DEFAULT_PARAMS
from utils.logger import get_logger
from utils.database import Database

logger = get_logger(__name__)


class StrategyEvaluator:
    """Evaluates and selects best trading strategies."""

    def __init__(self):
        self.db = Database()
        self.backtest_engine = BacktestEngine()
        self.strategy_performance = {}

    def evaluate_all_strategies(
        self,
        market_data: pd.DataFrame,
        symbols: List[str] = None
    ) -> pd.DataFrame:
        """Evaluate all available strategies."""

        logger.info("Evaluating all trading strategies")

        results = []

        for strategy_name, strategy_func in STRATEGY_REGISTRY.items():
            logger.info(f"Testing {strategy_name} strategy")

            params = DEFAULT_PARAMS.get(strategy_name, {})

            try:
                perf = self.backtest_engine.run_backtest(
                    market_data,
                    strategy_func,
                    params
                )

                perf['strategy_name'] = strategy_name
                perf['tested_at'] = datetime.now()

                results.append(perf)

                # Store in database
                self.db.store_backtest_result(
                    strategy_name=strategy_name,
                    start_date=market_data.index[0],
                    end_date=market_data.index[-1],
                    initial_capital=perf['initial_capital'],
                    final_capital=perf['final_capital'],
                    total_return=perf['total_return'],
                    sharpe_ratio=perf['sharpe_ratio'],
                    max_drawdown=perf['max_drawdown'],
                    win_rate=perf['win_rate'],
                    total_trades=perf['total_trades'],
                    parameters=str(params),
                    results=str(perf)
                )

            except Exception as e:
                logger.error(f"Error testing {strategy_name}: {e}")
                continue

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Rank strategies
            results_df = self._rank_strategies(results_df)

            logger.info("\n=== Strategy Performance Summary ===")
            # Format with proper column alignment
            header = f"{'Strategy':<20} {'Return %':>10} {'Sharpe':>10} {'Win Rate %':>12} {'Max DD %':>10}"
            logger.info(header)
            logger.info("-" * len(header))
            for _, row in results_df.iterrows():
                logger.info(f"{row['strategy_name']:<20} {row['total_return']:>10.2f} {row['sharpe_ratio']:>10.2f} {row['win_rate']:>12.2f} {row['max_drawdown']:>10.2f}")

            self.strategy_performance = results_df.to_dict('records')

        return results_df

    def _rank_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank strategies based on multiple metrics."""

        # Normalize metrics to 0-1 scale
        df['return_score'] = (df['total_return'] - df['total_return'].min()) / (df['total_return'].max() - df['total_return'].min() + 1e-10)
        df['sharpe_score'] = (df['sharpe_ratio'] - df['sharpe_ratio'].min()) / (df['sharpe_ratio'].max() - df['sharpe_ratio'].min() + 1e-10)
        df['drawdown_score'] = 1 - ((df['max_drawdown'].abs() - df['max_drawdown'].abs().min()) / (df['max_drawdown'].abs().max() - df['max_drawdown'].abs().min() + 1e-10))
        df['winrate_score'] = (df['win_rate'] - df['win_rate'].min()) / (df['win_rate'].max() - df['win_rate'].min() + 1e-10)

        # Weighted composite score
        df['composite_score'] = (
            df['return_score'] * 0.3 +
            df['sharpe_score'] * 0.35 +
            df['drawdown_score'] * 0.20 +
            df['winrate_score'] * 0.15
        )

        df = df.sort_values('composite_score', ascending=False)

        return df

    def select_best_strategy(
        self,
        market_condition: str = None
    ) -> tuple:
        """Select the best strategy based on current market conditions."""

        if not self.strategy_performance:
            logger.warning("No strategy performance data available")
            return None, None

        df = pd.DataFrame(self.strategy_performance)

        # If market condition is specified, filter strategies
        if market_condition == 'trending':
            # Prefer trend following and momentum
            suitable = df[df['strategy_name'].isin(['trend_following', 'momentum', 'breakout'])]
        elif market_condition == 'ranging':
            # Prefer mean reversion
            suitable = df[df['strategy_name'].isin(['mean_reversion', 'rsi', 'pairs_trading'])]
        elif market_condition == 'volatile':
            # Prefer strategies with good risk management
            suitable = df[df['max_drawdown'].abs() < df['max_drawdown'].abs().median()]
        else:
            # Use all strategies
            suitable = df

        if suitable.empty:
            suitable = df

        # Get best strategy
        best = suitable.sort_values('composite_score', ascending=False).iloc[0]

        strategy_name = best['strategy_name']
        params = DEFAULT_PARAMS.get(strategy_name, {})

        logger.info(f"Selected strategy: {strategy_name} (Score: {best['composite_score']:.3f})")

        return strategy_name, params

    def adaptive_strategy_selection(
        self,
        recent_performance: pd.DataFrame
    ) -> tuple:
        """Adaptively select strategy based on recent performance."""

        # Analyze recent market conditions
        if len(recent_performance) < 20:
            return self.select_best_strategy()

        # Calculate market regime indicators
        returns = recent_performance['close'].pct_change()
        volatility = returns.std()
        trend_strength = abs(recent_performance['close'].iloc[-1] / recent_performance['close'].iloc[-20] - 1)

        # Determine market condition
        if trend_strength > 0.05 and volatility < returns.mean():
            market_condition = 'trending'
        elif volatility > returns.std() * 1.5:
            market_condition = 'volatile'
        else:
            market_condition = 'ranging'

        logger.info(f"Detected market condition: {market_condition}")

        # Select appropriate strategy
        strategy_name, params = self.select_best_strategy(market_condition)

        # Log the adaptation
        self.db.log_learning(
            learning_type="strategy_selection",
            description=f"Switched to {strategy_name} strategy",
            previous_behavior="Previous strategy",
            new_behavior=f"{strategy_name} for {market_condition} market",
            trigger_event=f"market_condition_{market_condition}",
            expected_improvement=0.1
        )

        return strategy_name, params

    def monte_carlo_simulation(
        self,
        strategy_func,
        params: Dict[str, Any],
        market_data: pd.DataFrame,
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for strategy robustness testing."""

        logger.info(f"Running {n_simulations} Monte Carlo simulations")

        results = []

        for i in range(n_simulations):
            # Randomly shuffle returns while preserving distribution
            simulated_data = market_data.copy()

            # Bootstrap sampling
            sample_indices = np.random.choice(len(market_data), size=len(market_data), replace=True)
            simulated_data = market_data.iloc[sample_indices].reset_index(drop=True)

            # Run backtest
            perf = self.backtest_engine.run_backtest(
                simulated_data,
                strategy_func,
                params
            )

            results.append(perf['total_return'])

        results = np.array(results)

        # Calculate statistics
        monte_carlo_results = {
            'mean_return': np.mean(results),
            'median_return': np.median(results),
            'std_return': np.std(results),
            'min_return': np.min(results),
            'max_return': np.max(results),
            'percentile_5': np.percentile(results, 5),
            'percentile_95': np.percentile(results, 95),
            'probability_positive': np.sum(results > 0) / n_simulations,
            'value_at_risk_95': np.percentile(results, 5)
        }

        logger.info(f"Monte Carlo results: Mean return = {monte_carlo_results['mean_return']:.2f}%")
        logger.info(f"95% confidence interval: [{monte_carlo_results['percentile_5']:.2f}%, {monte_carlo_results['percentile_95']:.2f}%]")

        return monte_carlo_results

    def walk_forward_analysis(
        self,
        strategy_func,
        param_grid: Dict[str, List[Any]],
        market_data: pd.DataFrame,
        train_period: int = 252,
        test_period: int = 63
    ) -> Dict[str, Any]:
        """Perform walk-forward analysis."""

        logger.info("Running walk-forward analysis")

        results = []
        optimal_params_history = []

        i = 0
        while i + train_period + test_period < len(market_data):
            # Training period
            train_data = market_data.iloc[i:i+train_period]

            # Optimize parameters on training data
            best_params, _ = self.backtest_engine.optimize_parameters(
                train_data,
                strategy_func,
                param_grid
            )

            optimal_params_history.append(best_params)

            # Test period
            test_data = market_data.iloc[i+train_period:i+train_period+test_period]

            # Run backtest with optimized parameters
            perf = self.backtest_engine.run_backtest(
                test_data,
                strategy_func,
                best_params
            )

            results.append(perf)

            # Move forward
            i += test_period

        # Aggregate results
        total_return = sum([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        max_dd = min([r['max_drawdown'] for r in results])

        walk_forward_results = {
            'total_return': total_return,
            'avg_sharpe': avg_sharpe,
            'max_drawdown': max_dd,
            'n_periods': len(results),
            'optimal_params_history': optimal_params_history,
            'detailed_results': results
        }

        logger.info(f"Walk-forward analysis complete: Total return = {total_return:.2f}%")

        return walk_forward_results

    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""

        if not self.strategy_performance:
            return "No strategy performance data available"

        df = pd.DataFrame(self.strategy_performance)

        report = []
        report.append("=" * 80)
        report.append("STRATEGY PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append(f"Total strategies tested: {len(df)}")
        report.append(f"Average return: {df['total_return'].mean():.2f}%")
        report.append(f"Average Sharpe ratio: {df['sharpe_ratio'].mean():.2f}")
        report.append(f"Average win rate: {df['win_rate'].mean():.2f}%")
        report.append("")

        # Top performers
        report.append("TOP PERFORMING STRATEGIES")
        top_strategies = df.nlargest(3, 'composite_score')

        for idx, row in top_strategies.iterrows():
            report.append(f"\n{row['strategy_name'].upper()}")
            report.append(f"  Total Return: {row['total_return']:.2f}%")
            report.append(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}")
            report.append(f"  Max Drawdown: {row['max_drawdown']:.2f}%")
            report.append(f"  Win Rate: {row['win_rate']:.2f}%")
            report.append(f"  Total Trades: {row['total_trades']}")
            report.append(f"  Composite Score: {row['composite_score']:.3f}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

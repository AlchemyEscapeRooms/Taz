#!/usr/bin/env python3
"""
Example Usage Scripts for AI Trading Bot

This file demonstrates various ways to use the trading bot.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_trading_bot.core.trading_bot import TradingBot
from ai_trading_bot.core.personality_profiles import create_custom_profile, PERSONALITY_PROFILES
from ai_trading_bot.backtesting import BacktestEngine, StrategyEvaluator
from ai_trading_bot.backtesting.strategies import (
    momentum_strategy, mean_reversion_strategy, trend_following_strategy
)
from ai_trading_bot.data import MarketDataCollector, NewsCollector, SentimentAnalyzer
from ai_trading_bot.utils.database import Database
import pandas as pd


def example_1_simple_backtest():
    """Example 1: Simple backtest on a single stock."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Backtest")
    print("=" * 80 + "\n")

    # Get historical data
    collector = MarketDataCollector()
    df = collector.get_historical_data('AAPL', start_date='2022-01-01', end_date='2023-12-31')

    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(
        df,
        momentum_strategy,
        {'lookback': 20, 'threshold': 0.02, 'position_size': 0.1}
    )

    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")


def example_2_compare_strategies():
    """Example 2: Compare multiple strategies."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Compare Multiple Strategies")
    print("=" * 80 + "\n")

    collector = MarketDataCollector()
    df = collector.get_historical_data('SPY', start_date='2021-01-01')

    engine = BacktestEngine(initial_capital=100000)

    strategies = {
        'Momentum': (momentum_strategy, {'lookback': 20, 'threshold': 0.02, 'position_size': 0.1}),
        'Mean Reversion': (mean_reversion_strategy, {'period': 20, 'std_dev': 2, 'position_size': 0.1}),
        'Trend Following': (trend_following_strategy, {'short_period': 20, 'long_period': 50, 'position_size': 0.1})
    }

    results_df = engine.compare_strategies(df, strategies)

    print("\nStrategy Comparison:")
    print(results_df[['strategy_name', 'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']].to_string(index=False))


def example_3_sentiment_analysis():
    """Example 3: Analyze news sentiment for stocks."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: News Sentiment Analysis")
    print("=" * 80 + "\n")

    news_collector = NewsCollector()
    sentiment_analyzer = SentimentAnalyzer()

    symbols = ['AAPL', 'TSLA', 'MSFT']

    for symbol in symbols:
        # Collect news
        articles = news_collector.get_stock_news(symbol, lookback_hours=24)

        if articles:
            # Analyze sentiment
            sentiment = sentiment_analyzer.analyze_multiple_articles(articles)

            print(f"{symbol}:")
            print(f"  Sentiment: {sentiment['sentiment_label']} ({sentiment['overall_sentiment']:.3f})")
            print(f"  Articles: {sentiment['total_articles']}")
            print(f"  Confidence: {sentiment['confidence']:.2f}")
            print()


def example_4_paper_trading():
    """Example 4: Run paper trading with custom profile."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Paper Trading with Custom Profile")
    print("=" * 80 + "\n")

    # Create custom profile
    my_profile = create_custom_profile(
        name="My Conservative Strategy",
        base_profile="conservative_income",
        max_position_size=0.08,
        preferred_strategies=["mean_reversion", "pairs_trading"],
        max_daily_trades=10
    )

    print(f"Created custom profile: {my_profile.name}")
    print(f"Risk Tolerance: {my_profile.risk_tolerance}")
    print(f"Trading Style: {my_profile.trading_style}")
    print(f"Max Position Size: {my_profile.max_position_size * 100}%")
    print(f"Preferred Strategies: {', '.join(my_profile.preferred_strategies)}")
    print()
    print("To start trading with this profile:")
    print("bot = TradingBot(initial_capital=50000, personality=my_profile, mode='paper')")
    print("bot.start()")


def example_5_parameter_optimization():
    """Example 5: Optimize strategy parameters."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Parameter Optimization")
    print("=" * 80 + "\n")

    collector = MarketDataCollector()
    df = collector.get_historical_data('QQQ', start_date='2022-01-01')

    engine = BacktestEngine(initial_capital=100000)

    # Define parameter grid
    param_grid = {
        'lookback': [10, 20, 30],
        'threshold': [0.01, 0.02, 0.03],
        'position_size': [0.05, 0.10, 0.15]
    }

    print("Optimizing momentum strategy parameters...")
    best_params, best_performance = engine.optimize_parameters(
        df,
        momentum_strategy,
        param_grid,
        optimization_metric='sharpe_ratio'
    )

    print(f"\nBest Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print(f"\nBest Performance:")
    print(f"  Total Return: {best_performance.get('total_return', 0):.2f}%")
    print(f"  Sharpe Ratio: {best_performance.get('sharpe_ratio', 0):.2f}")
    print(f"  Win Rate: {best_performance.get('win_rate', 0):.2f}%")


def example_6_learning_analysis():
    """Example 6: Analyze what the bot has learned."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Learning Analysis")
    print("=" * 80 + "\n")

    db = Database()

    # Get learning history
    learning_df = db.get_learning_history(days=30)

    if not learning_df.empty:
        print(f"Total Learning Events: {len(learning_df)}")
        print(f"\nRecent Learning Events:")

        for idx, row in learning_df.head(10).iterrows():
            print(f"\n{row['timestamp']}")
            print(f"  Type: {row['learning_type']}")
            print(f"  Description: {row['description']}")
            print(f"  Previous Behavior: {row['previous_behavior']}")
            print(f"  New Behavior: {row['new_behavior']}")
            print(f"  Expected Improvement: {row['expected_improvement']:.2%}")
    else:
        print("No learning events yet. Run the bot for a few days to see learning.")


def example_7_prediction_tracking():
    """Example 7: Track prediction accuracy."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Prediction Accuracy Tracking")
    print("=" * 80 + "\n")

    db = Database()

    # Get prediction performance
    perf_df = db.get_prediction_performance(days=30)

    if not perf_df.empty:
        print("Prediction Performance by Symbol:")
        print(perf_df.to_string(index=False))

        print(f"\nOverall Statistics:")
        print(f"  Average Accuracy: {perf_df['avg_accuracy'].mean():.2%}")
        print(f"  Average Confidence: {perf_df['avg_confidence'].mean():.2%}")
        print(f"  Average Profit Impact: ${perf_df['avg_profit_impact'].mean():.2f}")
    else:
        print("No predictions yet. Run the bot to generate predictions.")


def example_8_portfolio_analysis():
    """Example 8: Portfolio diversification analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Portfolio Analysis")
    print("=" * 80 + "\n")

    collector = MarketDataCollector()

    # Get correlation matrix for common stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    print(f"Calculating correlations for {len(symbols)} stocks...")
    correlation_matrix = collector.calculate_correlations(symbols, period=60)

    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(2))

    # Find pairs with high correlation
    print("\nHighly Correlated Pairs (>0.7):")
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                print(f"  {correlation_matrix.index[i]} - {correlation_matrix.columns[j]}: {corr:.2f}")


def example_9_all_personalities():
    """Example 9: Show all personality profiles."""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: All Personality Profiles")
    print("=" * 80 + "\n")

    for name, profile in PERSONALITY_PROFILES.items():
        print(f"\n{profile.name}")
        print("  " + "=" * 60)
        print(f"  Description: {profile.description}")
        print(f"  Risk Tolerance: {profile.risk_tolerance}")
        print(f"  Trading Style: {profile.trading_style}")
        print(f"  Max Position Size: {profile.max_position_size * 100}%")
        print(f"  Max Daily Trades: {profile.max_daily_trades}")
        print(f"  Stop Loss: {profile.stop_loss_pct * 100}%")
        print(f"  Preferred Strategies: {', '.join(profile.preferred_strategies)}")
        print(f"  ML Weight: {profile.ml_prediction_weight * 100}%")
        print(f"  News Weight: {profile.news_sentiment_weight * 100}%")
        print(f"  Technical Weight: {profile.technical_analysis_weight * 100}%")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("AI TRADING BOT - EXAMPLE USAGE")
    print("=" * 80)

    examples = [
        ("1", "Simple Backtest", example_1_simple_backtest),
        ("2", "Compare Strategies", example_2_compare_strategies),
        ("3", "Sentiment Analysis", example_3_sentiment_analysis),
        ("4", "Paper Trading Setup", example_4_paper_trading),
        ("5", "Parameter Optimization", example_5_parameter_optimization),
        ("6", "Learning Analysis", example_6_learning_analysis),
        ("7", "Prediction Tracking", example_7_prediction_tracking),
        ("8", "Portfolio Analysis", example_8_portfolio_analysis),
        ("9", "All Personalities", example_9_all_personalities),
    ]

    print("\nAvailable Examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")

    print("\nEnter example number (1-9), 'all' for all examples, or 'q' to quit:")
    choice = input("> ").strip().lower()

    if choice == 'q':
        return
    elif choice == 'all':
        for _, _, func in examples:
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
            print("\n" + "-" * 80 + "\n")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                examples[idx][2]()
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")


if __name__ == '__main__':
    main()

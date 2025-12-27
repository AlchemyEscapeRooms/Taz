"""Main AI Trading Bot orchestrator - integrates all components."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, time as dt_time, timedelta
import schedule
import time

from ml_models import PredictionModel, EnsemblePredictor, ModelTrainer, FeatureEngineer
from backtesting import BacktestEngine, StrategyEvaluator
from data import MarketDataCollector, NewsCollector, SentimentAnalyzer
from portfolio import PortfolioManager, PositionTracker, RiskManager
from utils.logger import get_logger
from utils.database import Database
from utils.trade_logger import TradeReason, get_trade_logger
from config import config
from core.personality_profiles import PersonalityProfile, get_profile
from core.order_executor import OrderExecutor, OrderResult
from core.market_monitor import MarketMonitor, get_market_monitor

logger = get_logger(__name__)


class TradingBot:
    """AI Trading Bot - Main orchestrator integrating all components."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        personality: str = "balanced_growth",
        mode: str = "paper"
    ):
        logger.info("=" * 80)
        logger.info("INITIALIZING AI TRADING BOT")
        logger.info("=" * 80)

        self.initial_capital = initial_capital
        self.mode = mode  # paper or live
        self.personality = get_profile(personality)

        logger.info(f"Personality: {self.personality.name}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Initial Capital: ${initial_capital:,.2f}")

        # Initialize components
        self.db = Database()
        self.market_data = MarketDataCollector()
        self.news_collector = NewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.portfolio = PortfolioManager(initial_capital)
        self.model_trainer = ModelTrainer()
        self.strategy_evaluator = StrategyEvaluator()
        self.feature_engineer = FeatureEngineer()

        # Order execution and trade logging
        self.order_executor = OrderExecutor(mode=mode)
        self.trade_logger = get_trade_logger()

        # AI Learning System - shared across all personality profiles
        self.market_monitor = get_market_monitor()
        logger.info("AI Learning System initialized - signal weights will adapt based on accuracy")

        # ML models
        self.prediction_models = {}
        self.current_strategy = None
        self.current_strategy_params = {}

        # Trading state
        self.is_trading = False
        self.market_open = False
        self.daily_trades_count = 0
        self.last_rebalance = None

        # Performance tracking
        self.daily_performance = {
            'trades': 0,
            'profit': 0.0,
            'loss': 0.0,
            'predictions_made': 0,
            'predictions_correct': 0
        }

        logger.info("AI Trading Bot initialized successfully")

    def start(self):
        """Start the trading bot."""
        logger.info("Starting AI Trading Bot...")

        # Initial setup
        self._run_startup_sequence()

        # Schedule daily tasks
        self._schedule_tasks()

        # Main trading loop
        self.is_trading = True

        logger.info("Trading Bot is now running")
        logger.info("Press Ctrl+C to stop")

        try:
            while self.is_trading:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping Trading Bot...")
            self.stop()

    def stop(self):
        """Stop the trading bot."""
        self.is_trading = False
        logger.info("Trading Bot stopped")

    def _schedule_tasks(self):
        """Schedule daily tasks."""
        # Pre-market analysis
        schedule.every().day.at("08:00").do(self.pre_market_analysis)

        # Market open preparation
        schedule.every().day.at("09:25").do(self.prepare_for_market_open)

        # Active trading cycle - runs every 1 minute during market hours
        schedule.every(1).minutes.do(self.run_trading_cycle)

        # Mid-day review
        schedule.every().day.at("12:00").do(self.mid_day_review)

        # End of day analysis
        schedule.every().day.at("16:05").do(self.end_of_day_analysis)

        # After hours learning
        schedule.every().day.at("17:00").do(self.after_hours_learning)

        # Weekly model retraining
        schedule.every().saturday.at("02:00").do(self.retrain_models)

        # Weekly review
        schedule.every().sunday.at("10:00").do(self.weekly_review)

    def _run_startup_sequence(self):
        """Run initial startup sequence."""
        logger.info("Running startup sequence...")

        # Load historical data
        symbols = config.get('data.universe.initial_stocks', ['SPY', 'QQQ'])
        logger.info(f"Loading data for {len(symbols)} symbols")

        # Screen for additional stocks
        screened = self.market_data.screen_stocks()
        symbols.extend(screened[:20])  # Add top 20 screened stocks

        # Train initial models with combined data from multiple symbols
        logger.info("Training initial ML models...")
        training_dfs = []
        for symbol in symbols[:10]:  # Get data from first 10 symbols
            # Get 2 years of data for better training
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            df = self.market_data.get_historical_data(symbol, start_date=start_date)
            if not df.empty and len(df) > 100:
                df['symbol'] = symbol  # Tag with symbol for reference
                training_dfs.append(df)
                logger.info(f"Loaded {len(df)} samples from {symbol}")

        if training_dfs:
            # Combine all data for training
            combined_df = pd.concat(training_dfs, ignore_index=True)
            logger.info(f"Training with {len(combined_df)} total samples from {len(training_dfs)} symbols")
            self.model_trainer.train_prediction_models(combined_df)

        # Evaluate strategies and select best from personality's preferred
        logger.info("Evaluating trading strategies...")
        if symbols:
            df = self.market_data.get_historical_data(symbols[0])
            if not df.empty:
                results_df = self.strategy_evaluator.evaluate_all_strategies(df)
                self.current_strategy = self._select_best_from_results(results_df)
            else:
                self.current_strategy = self.personality.preferred_strategies[0]
        else:
            self.current_strategy = self.personality.preferred_strategies[0]
        logger.info(f"Selected initial strategy: {self.current_strategy}")

        logger.info("Startup sequence complete")

    def _select_best_from_results(self, results_df) -> str:
        """
        Select the best strategy from evaluation results, filtered by personality preferences.

        Args:
            results_df: DataFrame with strategy evaluation results from evaluate_all_strategies()

        Returns:
            Name of the best performing strategy from preferred list
        """
        preferred = self.personality.preferred_strategies

        if results_df is None or results_df.empty:
            logger.warning("No evaluation results, using first preferred strategy")
            return preferred[0]

        # Filter to only preferred strategies
        preferred_results = results_df[results_df['strategy_name'].isin(preferred)]

        if preferred_results.empty:
            logger.warning(f"None of preferred strategies {preferred} found in results, using first")
            return preferred[0]

        # Use the composite_score already calculated by strategy_evaluator
        if 'composite_score' in preferred_results.columns:
            best_row = preferred_results.loc[preferred_results['composite_score'].idxmax()]
        else:
            # Fallback: use sharpe ratio
            best_row = preferred_results.loc[preferred_results['sharpe_ratio'].idxmax()]

        best_strategy = best_row['strategy_name']

        logger.info(f"\n=== Strategy Selection (from {len(preferred)} preferred) ===")
        for _, row in preferred_results.sort_values('composite_score' if 'composite_score' in preferred_results.columns else 'sharpe_ratio', ascending=False).iterrows():
            marker = " <-- SELECTED" if row['strategy_name'] == best_strategy else ""
            logger.info(f"  {row['strategy_name']}: {row['total_return']:.2f}% return, {row['sharpe_ratio']:.2f} Sharpe{marker}")

        return best_strategy

    def pre_market_analysis(self):
        """Pre-market analysis and planning."""
        logger.info("=" * 80)
        logger.info("PRE-MARKET ANALYSIS")
        logger.info("=" * 80)

        # Reset daily counters
        self.daily_trades_count = 0
        self.portfolio.risk_manager.reset_daily(self.portfolio.cash)
        self.daily_performance = {
            'trades': 0,
            'profit': 0.0,
            'loss': 0.0,
            'predictions_made': 0,
            'predictions_correct': 0
        }

        # Get market sentiment
        market_news = self.news_collector.get_market_news()
        market_sentiment = self.sentiment_analyzer.analyze_multiple_articles(market_news)

        logger.info(f"Market Sentiment: {market_sentiment['sentiment_label']} ({market_sentiment['overall_sentiment']:.3f})")

        # Analyze symbols in portfolio
        positions = self.portfolio.position_tracker.get_all_positions()

        for symbol in positions.keys():
            # Get news and sentiment
            news = self.news_collector.get_stock_news(symbol)
            if news:
                sentiment = self.sentiment_analyzer.analyze_multiple_articles(news)
                logger.info(f"{symbol} Sentiment: {sentiment['sentiment_label']} ({sentiment['overall_sentiment']:.3f})")

        # Re-evaluate and potentially switch strategy based on recent performance
        previous_strategy = self.current_strategy
        self.current_strategy = self._select_best_strategy(lookback_days=30)

        if self.current_strategy != previous_strategy:
            logger.info(f"Strategy switch: {previous_strategy} -> {self.current_strategy}")
        else:
            logger.info(f"Continuing with strategy: {self.current_strategy}")

        # Make predictions for watchlist
        predictions = self._generate_daily_predictions()

        # Generate morning report
        self._generate_morning_report(predictions, market_sentiment)

        logger.info("Pre-market analysis complete")

    def _generate_daily_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions for potential trades using AI learning system."""
        predictions = []

        # Get watchlist
        symbols = self._get_watchlist()

        for symbol in symbols:
            try:
                # Get data
                df = self.market_data.get_historical_data(symbol)
                if df.empty:
                    continue

                # Use AI Learning System for predictions (applies to ALL personality profiles)
                ai_prediction = self.market_monitor._analyze_and_predict(symbol, df)

                if ai_prediction:
                    prediction = {
                        'symbol': symbol,
                        'predicted_return': ai_prediction.predicted_change_pct / 100,
                        'confidence': ai_prediction.confidence / 100,
                        'direction': ai_prediction.predicted_direction,
                        'current_price': ai_prediction.entry_price,
                        'target_price': ai_prediction.target_price,
                        'signals': ai_prediction.signals,
                        'reasoning': ai_prediction.reasoning,
                        'timestamp': datetime.now()
                    }

                    predictions.append(prediction)

                    # Track the prediction for learning
                    if ai_prediction.confidence >= 60:
                        self.market_monitor.prediction_tracker.add_prediction(ai_prediction)

                    self.daily_performance['predictions_made'] += 1

                # Also use ML model if available (ensemble approach)
                df_featured = self.feature_engineer.engineer_features(df)

                if 'price_predictor' in self.model_trainer.models:
                    ensemble = self.model_trainer.models['price_predictor']

                    # Prepare features
                    feature_cols = ensemble.models[0].feature_cols if ensemble.models else []
                    if feature_cols and all(col in df_featured.columns for col in feature_cols):
                        X = df_featured[feature_cols].iloc[-1:].values

                        # Make prediction
                        pred, confidence = ensemble.predict_with_confidence(X)

                        # Store ML prediction for comparison
                        self.db.store_prediction(
                            symbol=symbol,
                            prediction_type='price_prediction',
                            predicted_value=pred[0],
                            predicted_direction='up' if pred[0] > 0 else 'down',
                            confidence=confidence[0],
                            model_version="ensemble_v1"
                        )

            except Exception as e:
                logger.error(f"Error generating prediction for {symbol}: {e}")
                continue

        # Sort by confidence * predicted return
        predictions.sort(key=lambda x: abs(x['predicted_return']) * x['confidence'], reverse=True)

        # Log current signal weights being used
        logger.info("Current AI Signal Weights:")
        for signal, weight in self.market_monitor.signal_weights.items():
            logger.info(f"  {signal}: {weight:.3f}")

        return predictions

    def _get_watchlist(self) -> List[str]:
        """Get list of symbols to watch - combines Alpaca positions with configured stocks."""
        symbols = set()

        # 1. Get Alpaca positions if in paper/live mode
        if self.mode in ['paper', 'live']:
            try:
                alpaca_positions = self.order_executor.get_positions()
                if alpaca_positions:
                    symbols.update(alpaca_positions.keys())
                    logger.debug(f"Added {len(alpaca_positions)} symbols from Alpaca positions")
            except Exception as e:
                logger.warning(f"Could not get Alpaca positions: {e}")

        # 2. Add configured stocks from config.yaml
        configured_stocks = config.get('data.universe.initial_stocks', [])
        if isinstance(configured_stocks, list):
            symbols.update(configured_stocks)

        # 3. Current internal positions (if any)
        internal_positions = self.portfolio.position_tracker.get_all_positions().keys()
        symbols.update(internal_positions)

        # 4. Add top screened stocks (limit to avoid too many)
        try:
            screened = self.market_data.screen_stocks()
            symbols.update(screened[:20])
        except Exception as e:
            logger.debug(f"Stock screening unavailable: {e}")

        return list(symbols)

    def _generate_morning_report(
        self,
        predictions: List[Dict[str, Any]],
        market_sentiment: Dict[str, Any]
    ):
        """Generate morning preview report."""
        logger.info("\n" + "=" * 80)
        logger.info("MORNING PREVIEW - " + datetime.now().strftime("%Y-%m-%d"))
        logger.info("=" * 80)

        # Portfolio summary
        current_prices = self._get_current_prices()
        portfolio_value = self.portfolio.get_portfolio_value(current_prices)
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100

        logger.info(f"\nPORTFOLIO SUMMARY")
        logger.info(f"  Total Value: ${portfolio_value:,.2f}")
        logger.info(f"  Cash: ${self.portfolio.cash:,.2f}")
        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Open Positions: {len(self.portfolio.position_tracker.get_all_positions())}")

        # Market sentiment
        logger.info(f"\nMARKET SENTIMENT: {market_sentiment['sentiment_label'].upper()}")
        logger.info(f"  Score: {market_sentiment['overall_sentiment']:.3f}")
        logger.info(f"  Articles Analyzed: {market_sentiment['total_articles']}")

        # Top predictions
        logger.info(f"\nTOP PREDICTIONS FOR TODAY")
        for i, pred in enumerate(predictions[:5], 1):
            logger.info(f"  {i}. {pred['symbol']}: {pred['direction'].upper()} "
                       f"(Confidence: {pred['confidence']:.2f}, "
                       f"Expected Return: {pred['predicted_return']*100:.2f}%)")

        # Today's focus
        logger.info(f"\nTODAY'S FOCUS")
        logger.info(f"  Strategy: {self.current_strategy}")
        logger.info(f"  Max Trades: {self.personality.max_daily_trades}")
        logger.info(f"  Risk Level: {self.personality.risk_tolerance}")

        logger.info("=" * 80 + "\n")

    def prepare_for_market_open(self):
        """Prepare for market open."""
        logger.info("Preparing for market open...")
        self.market_open = True
        self.daily_trades_count = 0

    def run_trading_cycle(self):
        """
        Run a single trading cycle - scans for signals and executes trades.
        This runs every 1 minute during market hours.
        """
        # Check if market is open (9:30 AM - 4:00 PM ET on weekdays)
        now = datetime.now()
        market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Skip weekends
        if now.weekday() >= 5:
            return

        # Skip outside market hours
        if now < market_start or now > market_end:
            return

        # Ensure market_open flag is set
        if not self.market_open:
            self.market_open = True

        # Check daily trade limit
        if self.daily_trades_count >= self.personality.max_daily_trades:
            logger.debug(f"Daily trade limit reached ({self.daily_trades_count}/{self.personality.max_daily_trades})")
            return

        logger.info(f"Running trading cycle at {now.strftime('%H:%M:%S')}...")

        # Get the current strategy
        from backtesting.strategies import (
            momentum_strategy, mean_reversion_strategy, trend_following_strategy,
            breakout_strategy, rsi_strategy, macd_strategy, ai_prediction_strategy,
            DEFAULT_PARAMS
        )

        strategy_map = {
            'momentum': momentum_strategy,
            'mean_reversion': mean_reversion_strategy,
            'trend_following': trend_following_strategy,
            'breakout': breakout_strategy,
            'rsi': rsi_strategy,
            'macd': macd_strategy,
            'ai_prediction': ai_prediction_strategy,
        }

        strategy_func = strategy_map.get(self.current_strategy, trend_following_strategy)
        strategy_params = DEFAULT_PARAMS.get(self.current_strategy, {}).copy()
        
        # For live trading, ensure state-based logic is used for trend_following
        if self.current_strategy == 'trend_following':
            strategy_params['use_crossover_only'] = False

        # Get watchlist
        symbols = self._get_watchlist()
        if not symbols:
            symbols = config.get('data.universe.initial_stocks', ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA'])

        trades_this_cycle = 0
        symbols_analyzed = 0

        for symbol in symbols[:10]:  # Limit to 10 symbols per cycle
            try:
                # Skip if daily limit reached
                if self.daily_trades_count >= self.personality.max_daily_trades:
                    break

                # Get historical data for strategy
                df = self.market_data.get_historical_data(symbol)
                if df.empty:
                    logger.debug(f"No data for {symbol}")
                    continue
                    
                if len(df) < 50:
                    logger.debug(f"Insufficient data for {symbol}: {len(df)} bars (need 50+)")
                    continue
                
                symbols_analyzed += 1

                # Create a mock engine for position checking
                # Includes max_position_size from risk manager so strategies use correct limits
                class MockEngine:
                    def __init__(self, positions, cash, max_position_size):
                        self.open_positions = positions
                        self.capital = cash
                        self.max_position_size = max_position_size

                positions = self.portfolio.position_tracker.get_all_positions()
                max_pos_size = self.portfolio.risk_manager.max_position_size
                mock_engine = MockEngine(positions, self.portfolio.cash, max_pos_size)

                # Generate signals from primary strategy
                signals = strategy_func(df, mock_engine, strategy_params)
                
                # Also try AI prediction strategy if not already using it
                if self.current_strategy != 'ai_prediction':
                    ai_params = DEFAULT_PARAMS.get('ai_prediction', {'min_confidence': 70, 'position_size': 0.1})
                    ai_signals = ai_prediction_strategy(df, mock_engine, ai_params)
                    
                    # Add high-confidence AI signals
                    for sig in ai_signals:
                        if sig.get('reason', {}).get('signal_value', 0) >= 80:  # Only 80%+ confidence
                            signals.append(sig)
                            logger.info(f"AI signal added for {symbol}: {sig.get('action')} "
                                       f"(confidence: {sig.get('reason', {}).get('signal_value', 0):.1f}%)")

                # Log signal generation
                if signals:
                    logger.info(f"Generated {len(signals)} signal(s) for {symbol}")
                
                # Process signals
                for signal in signals:
                    if signal.get('symbol') == symbol or signal.get('symbol') == 'DEFAULT':
                        signal['symbol'] = symbol
                        
                        # Log before attempting trade
                        action = signal.get('action', 'unknown')
                        reason = signal.get('reason', {})
                        logger.info(f"Processing signal: {action.upper()} {symbol} - "
                                   f"{reason.get('primary_signal', 'unknown')} "
                                   f"(value: {reason.get('signal_value', 'N/A')})")
                        
                        result = self.process_trading_signal(symbol, signal)

                        if result and result.success:
                            trades_this_cycle += 1
                            logger.info(f"Trade executed for {symbol}: {signal.get('action')} "
                                       f"(Strategy: {self.current_strategy})")
                        elif result:
                            logger.warning(f"Trade failed for {symbol}: {result.error_message}")

            except Exception as e:
                logger.error(f"Error in trading cycle for {symbol}: {e}")
                continue

        if trades_this_cycle > 0:
            logger.info(f"Trading cycle complete: {trades_this_cycle} trades executed from {symbols_analyzed} symbols analyzed")
        else:
            logger.info(f"Trading cycle complete: no trades executed ({symbols_analyzed} symbols analyzed)")

    def mid_day_review(self):
        """Mid-day performance review."""
        logger.info("Running mid-day review...")

        current_prices = self._get_current_prices()
        portfolio_value = self.portfolio.get_portfolio_value(current_prices)

        logger.info(f"Mid-day Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"Trades Today: {self.daily_trades_count}")
        logger.info(f"Daily P&L: ${self.portfolio.risk_manager.daily_pl:,.2f}")

    def end_of_day_analysis(self):
        """End of day analysis and reporting."""
        logger.info("=" * 80)
        logger.info("END OF DAY ANALYSIS")
        logger.info("=" * 80)

        self.market_open = False

        # Calculate daily performance
        current_prices = self._get_current_prices()
        portfolio_value = self.portfolio.get_portfolio_value(current_prices)
        daily_pl = self.portfolio.risk_manager.daily_pl

        # Get trades
        trades_today = self.db.get_trades_history(days=1)

        # Calculate metrics
        if not trades_today.empty:
            winning_trades = len(trades_today[trades_today['profit_loss'] > 0])
            losing_trades = len(trades_today[trades_today['profit_loss'] < 0])
            win_rate = winning_trades / len(trades_today) * 100 if len(trades_today) > 0 else 0
        else:
            winning_trades = losing_trades = 0
            win_rate = 0

        # Generate report
        logger.info(f"\nDAILY PERFORMANCE - {datetime.now().strftime('%Y-%m-%d')}")
        logger.info(f"  Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"  Daily P&L: ${daily_pl:,.2f}")
        logger.info(f"  Total Trades: {len(trades_today) if not trades_today.empty else 0}")
        logger.info(f"  Winning Trades: {winning_trades}")
        logger.info(f"  Losing Trades: {losing_trades}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Predictions Made: {self.daily_performance['predictions_made']}")

        # Store performance
        self.db.store_performance_metrics(
            period='daily',
            metrics={
                'date': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'daily_pl': daily_pl,
                'total_trades': len(trades_today) if not trades_today.empty else 0,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate
            }
        )

        # Save portfolio snapshot
        self.portfolio.record_snapshot(current_prices)

        logger.info("=" * 80)

    def after_hours_learning(self):
        """After hours: learn from today's performance."""
        logger.info("Running after-hours learning...")

        # Evaluate predictions
        self._evaluate_todays_predictions()

        # Run AI Learning System to adjust signal weights
        logger.info("AI Learning System updating signal weights...")
        self.market_monitor._learn_from_history()

        # Log updated weights
        logger.info("Updated AI Signal Weights:")
        for signal, weight in self.market_monitor.signal_weights.items():
            logger.info(f"  {signal}: {weight:.3f}")

        # Get accuracy stats
        accuracy = self.market_monitor.prediction_tracker.get_accuracy_stats(days=7)
        if accuracy['total_predictions'] > 0:
            logger.info(f"AI Prediction Accuracy (7 days): {accuracy['accuracy']:.1f}% "
                       f"({accuracy['total_predictions']} predictions)")

        # Analyze what worked and what didn't
        trades_today = self.db.get_trades_history(days=1)

        if not trades_today.empty:
            # Analyze winning vs losing patterns
            winning = trades_today[trades_today['profit_loss'] > 0]
            losing = trades_today[trades_today['profit_loss'] < 0]

            if len(winning) > 0 and len(losing) > 0:
                # Compare patterns
                avg_win = winning['profit_loss'].mean()
                avg_loss = losing['profit_loss'].mean()

                logger.info(f"Average Win: ${avg_win:.2f}")
                logger.info(f"Average Loss: ${avg_loss:.2f}")

                # Log learning
                self.db.log_learning(
                    learning_type="daily_analysis",
                    description=f"Analyzed {len(trades_today)} trades",
                    previous_behavior="",
                    new_behavior=f"Win rate: {len(winning)/len(trades_today)*100:.1f}%",
                    trigger_event="end_of_day",
                    expected_improvement=0.0
                )

        logger.info("After-hours learning complete")

    def _evaluate_todays_predictions(self):
        """Evaluate accuracy of today's predictions."""
        from datetime import timedelta

        # Get predictions from earlier today that haven't been evaluated
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, symbol, predicted_value, predicted_direction, timestamp
                FROM predictions
                WHERE timestamp >= ? AND evaluation_timestamp IS NULL
                AND timestamp < datetime('now', '-1 hours')
            """, (today_start,))

            predictions = cursor.fetchall()

        if not predictions:
            return

        logger.info(f"Evaluating {len(predictions)} predictions from today")

        for pred in predictions:
            try:
                # Get historical data to see what actually happened
                df = self.market_data.get_historical_data(
                    pred['symbol'],
                    start_date=(datetime.fromisoformat(pred['timestamp']) - timedelta(days=1)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )

                if df.empty or len(df) < 2:
                    continue

                # Find the price at prediction time and current price
                pred_price = df.iloc[0]['close']  # Price at prediction time
                current_price = df.iloc[-1]['close']  # Current price

                # Calculate actual return
                actual_return = (current_price - pred_price) / pred_price
                actual_direction = 'up' if actual_return > 0 else 'down'

                # Evaluate prediction
                self.db.evaluate_prediction(
                    prediction_id=pred['id'],
                    actual_value=actual_return,
                    actual_direction=actual_direction,
                    profit_impact=0.0  # Would need to track if we traded on this
                )

                # Update daily performance
                if pred['predicted_direction'] == actual_direction:
                    self.daily_performance['predictions_correct'] += 1

            except Exception as e:
                logger.error(f"Error evaluating prediction {pred['id']}: {e}")
                continue

        accuracy_rate = (self.daily_performance['predictions_correct'] /
                         self.daily_performance['predictions_made'] * 100
                         if self.daily_performance['predictions_made'] > 0 else 0)

        logger.info(f"Prediction accuracy today: {accuracy_rate:.1f}%")

    def retrain_models(self):
        """Retrain ML models weekly with combined data from multiple symbols."""
        logger.info("Retraining models...")

        symbols = self._get_watchlist()

        # Collect data from multiple symbols
        training_dfs = []
        for symbol in symbols[:15]:  # Get data from 15 symbols
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            df = self.market_data.get_historical_data(symbol, start_date=start_date)
            if not df.empty and len(df) > 100:
                df['symbol'] = symbol
                training_dfs.append(df)

        if training_dfs:
            combined_df = pd.concat(training_dfs, ignore_index=True)
            logger.info(f"Retraining with {len(combined_df)} samples from {len(training_dfs)} symbols")
            self.model_trainer.retrain_models(combined_df)

        logger.info("Model retraining complete")

    def weekly_review(self):
        """Weekly performance review."""
        logger.info("=" * 80)
        logger.info("WEEKLY REVIEW")
        logger.info("=" * 80)

        # Get week performance
        trades_week = self.db.get_trades_history(days=7)

        if not trades_week.empty:
            total_pl = trades_week['profit_loss'].sum()
            win_rate = len(trades_week[trades_week['profit_loss'] > 0]) / len(trades_week) * 100

            logger.info(f"Weekly P&L: ${total_pl:,.2f}")
            logger.info(f"Weekly Win Rate: {win_rate:.1f}%")
            logger.info(f"Total Trades: {len(trades_week)}")

        logger.info("=" * 80)

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all positions."""
        prices = {}
        for symbol in self.portfolio.position_tracker.get_all_positions().keys():
            quote = self.market_data.get_real_time_quote(symbol)
            prices[symbol] = quote.get('price', 0)
        return prices

    def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        reason: Dict[str, Any] = None,
        order_type: str = "market",
        limit_price: float = None
    ) -> OrderResult:
        """
        Execute a trade with full logging and reasoning.

        Args:
            symbol: Stock symbol to trade
            action: "buy" or "sell"
            quantity: Number of shares
            reason: Dict with reasoning (primary_signal, explanation, etc.)
            order_type: "market" or "limit"
            limit_price: Price for limit orders

        Returns:
            OrderResult with execution details
        """
        # Check if we can trade
        if not self.is_trading:
            logger.warning("Trading bot is not running")
            return None

        if not self.market_open:
            logger.warning("Market is not open")
            return None

        # Check daily trade limit
        if self.daily_trades_count >= self.personality.max_daily_trades:
            logger.warning(f"Daily trade limit reached ({self.personality.max_daily_trades})")
            return None

        # Get current market snapshot
        quote = self.market_data.get_real_time_quote(symbol)
        current_price = quote.get('price', 0)

        if current_price <= 0:
            logger.error(f"Could not get price for {symbol}")
            return None

        market_snapshot = {
            'price': current_price,
            'bid': quote.get('bid', current_price),
            'ask': quote.get('ask', current_price),
            'volume': quote.get('volume', 0)
        }

        # Build TradeReason from dict
        if reason:
            trade_reason = TradeReason(
                primary_signal=reason.get('primary_signal', 'manual'),
                signal_value=reason.get('signal_value', 0),
                threshold=reason.get('threshold', 0),
                direction=reason.get('direction', 'n/a'),
                supporting_indicators=reason.get('supporting_indicators', {}),
                confirmations=reason.get('confirmations', []),
                explanation=reason.get('explanation', '')
            )
        else:
            trade_reason = TradeReason(
                primary_signal='manual_trade',
                signal_value=0,
                threshold=0,
                direction='n/a',
                explanation='Manual trade without detailed reasoning'
            )

        # Check with risk manager
        current_prices = self._get_current_prices()
        current_prices[symbol] = current_price
        portfolio_value = self.portfolio.get_portfolio_value(current_prices)

        if action.lower() == "buy":
            trade_value = quantity * current_price
            max_allowed = self.portfolio.risk_manager.get_max_trade_value(self.portfolio.cash)

            # If trade exceeds limit, reduce quantity to fit within risk limits
            if trade_value > max_allowed:
                old_quantity = quantity
                quantity = max_allowed / current_price
                trade_value = quantity * current_price
                logger.info(f"Position size adjusted from {old_quantity:.2f} to {quantity:.2f} shares "
                           f"(${trade_value:.2f}) to fit within risk limits")

            if not self.portfolio.risk_manager.can_trade(trade_value, 'long'):
                logger.warning(f"Trade rejected by risk manager: {symbol}")
                return None

        # Execute the order
        result = self.order_executor.execute_order(
            symbol=symbol,
            side=action.lower(),
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            strategy_name=self.current_strategy,
            strategy_params=self.current_strategy_params,
            reason=trade_reason,
            market_snapshot=market_snapshot,
            portfolio_value=portfolio_value
        )

        if result.success:
            # Update portfolio tracking
            if action.lower() == "buy":
                self.portfolio.position_tracker.add_position(
                    symbol=symbol,
                    quantity=result.filled_quantity,
                    price=result.filled_price,
                    order_id=result.order_id
                )
                self.portfolio.cash -= result.filled_quantity * result.filled_price

                # Track prediction for AI Learning System
                self._track_live_prediction(symbol, result.filled_price, reason)

            else:
                closed_lots = self.portfolio.position_tracker.remove_position(
                    symbol=symbol,
                    quantity=result.filled_quantity,
                    price=result.filled_price
                )
                # Add proceeds to cash
                for lot in closed_lots:
                    self.portfolio.cash += lot['proceeds']

                    # Update risk manager P&L
                    self.portfolio.risk_manager.record_trade_result(lot['profit_loss'])

                # Resolve prediction for AI Learning System
                self._resolve_live_prediction(symbol, result.filled_price)

            # Update counters
            self.daily_trades_count += 1
            self.daily_performance['trades'] += 1

            if action.lower() == "sell":
                # Record profit/loss
                for lot in closed_lots:
                    if lot['profit_loss'] > 0:
                        self.daily_performance['profit'] += lot['profit_loss']
                    else:
                        self.daily_performance['loss'] += abs(lot['profit_loss'])

            logger.info(f"Trade executed: {action.upper()} {result.filled_quantity} {symbol} @ ${result.filled_price:.2f}")

        return result

    def _track_live_prediction(self, symbol: str, entry_price: float, reason: Dict = None):
        """Track a prediction for AI Learning System when entering a live/paper position."""
        try:
            from core.market_monitor import Prediction

            # Get current data for signal analysis
            df = self.market_data.get_historical_data(symbol)

            # Build signals dict from current indicators
            signals_dict = {}
            if reason:
                signals_dict['primary_signal'] = reason.get('primary_signal', 'unknown')
                signals_dict['signal_value'] = reason.get('signal_value', 0)
                if 'supporting_indicators' in reason:
                    signals_dict.update(reason['supporting_indicators'])

            # Calculate additional indicators
            if not df.empty and len(df) >= 20:
                # RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, 0.0001)
                rsi = 100 - (100 / (1 + rs))
                signals_dict['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

                # MACD
                ema12 = df['close'].ewm(span=12).mean()
                ema26 = df['close'].ewm(span=26).mean()
                macd = ema12 - ema26
                signals_dict['macd'] = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0

            # Create prediction
            prediction = Prediction(
                prediction_id=self.market_monitor.prediction_tracker.generate_prediction_id(),
                timestamp=datetime.now(),
                symbol=symbol,
                predicted_direction='up',  # Buying = predicting up
                confidence=70.0,
                predicted_change_pct=2.0,
                timeframe=self.mode,  # 'paper' or 'live'
                target_price=entry_price * 1.02,
                entry_price=entry_price,
                signals=signals_dict,
                reasoning=f"Live trade entry: {reason.get('primary_signal', 'strategy') if reason else 'manual'}"
            )

            # Store in market monitor's active predictions
            self.market_monitor.prediction_tracker.add_prediction(prediction)

            # Also store mapping by symbol for resolution
            if not hasattr(self, '_live_predictions'):
                self._live_predictions = {}
            self._live_predictions[symbol] = prediction.prediction_id

            logger.info(f"AI Learning: Tracking prediction {prediction.prediction_id} for {symbol}")

        except Exception as e:
            logger.warning(f"Error tracking live prediction: {e}")

    def _resolve_live_prediction(self, symbol: str, exit_price: float):
        """Resolve a prediction when closing a live/paper position."""
        try:
            if not hasattr(self, '_live_predictions') or symbol not in self._live_predictions:
                return

            prediction_id = self._live_predictions[symbol]

            # Resolve the prediction
            self.market_monitor.prediction_tracker.resolve_prediction(prediction_id, exit_price)

            # Remove from tracking
            del self._live_predictions[symbol]

            logger.info(f"AI Learning: Resolved prediction {prediction_id} for {symbol}")

        except Exception as e:
            logger.warning(f"Error resolving live prediction: {e}")

    def process_trading_signal(
        self,
        symbol: str,
        signal: Dict[str, Any]
    ) -> Optional[OrderResult]:
        """
        Process a trading signal from a strategy.

        Args:
            symbol: Stock symbol
            signal: Signal dict with action, quantity, price, reason

        Returns:
            OrderResult if trade executed, None otherwise
        """
        action = signal.get('action')
        quantity = signal.get('quantity', 0)
        reason = signal.get('reason', {})

        if not action or quantity <= 0:
            return None

        return self.execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            reason=reason
        )

    def run_live_strategy(self, strategy_func, strategy_params: Dict = None):
        """
        Run a strategy in live mode, generating and executing signals.

        Args:
            strategy_func: Strategy function from backtesting.strategies
            strategy_params: Parameters for the strategy
        """
        from backtesting.strategies import DEFAULT_PARAMS

        strategy_name = strategy_func.__name__
        self.current_strategy = strategy_name
        self.current_strategy_params = strategy_params or DEFAULT_PARAMS.get(strategy_name, {})

        logger.info(f"Running live strategy: {strategy_name}")

        # Get watchlist
        symbols = self._get_watchlist()

        for symbol in symbols:
            try:
                # Get historical data for strategy
                df = self.market_data.get_historical_data(symbol, period='60d')
                if df.empty or len(df) < 50:
                    continue

                # Create a mock engine for position checking
                # Includes max_position_size from risk manager so strategies use correct limits
                class MockEngine:
                    def __init__(self, positions, cash, max_position_size):
                        self.open_positions = positions
                        self.capital = cash
                        self.max_position_size = max_position_size

                positions = self.portfolio.position_tracker.get_all_positions()
                max_pos_size = self.portfolio.risk_manager.max_position_size
                mock_engine = MockEngine(positions, self.portfolio.cash, max_pos_size)

                # Generate signals
                signals = strategy_func(df, mock_engine, self.current_strategy_params)

                # Process signals
                for signal in signals:
                    if signal.get('symbol') == symbol or signal.get('symbol') == 'DEFAULT':
                        signal['symbol'] = symbol  # Ensure symbol is set
                        result = self.process_trading_signal(symbol, signal)

                        if result and result.success:
                            logger.info(f"Signal executed for {symbol}: {signal.get('action')}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

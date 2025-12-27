"""Model training and retraining orchestration."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from datetime import datetime, timedelta

from ml_models.prediction_model import PredictionModel, EnsemblePredictor, LSTMPredictor
from ml_models.feature_engineering import FeatureEngineer
from utils.logger import get_logger
from utils.database import Database
from config import config

logger = get_logger(__name__)


class ModelTrainer:
    """Manages model training, retraining, and performance tracking."""

    def __init__(self):
        self.db = Database()
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.retrain_schedule = {}
        self.performance_threshold = 0.55

    def train_prediction_models(
        self,
        df: pd.DataFrame,
        target_col: str = 'future_return',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> EnsemblePredictor:
        """Train ensemble of prediction models."""

        logger.info("Starting model training process")

        # Engineer features
        df_featured = self.feature_engineer.engineer_features(df)

        # Create target variable (future return)
        prediction_horizon = config.get('ml.models.price_prediction.prediction_horizon', 5)
        df_featured['future_return'] = df_featured['close'].pct_change(prediction_horizon).shift(-prediction_horizon)

        # Remove NaN values
        df_featured = df_featured.dropna()

        # Select features
        feature_cols = self.feature_engineer.select_features(
            df_featured,
            target='future_return',
            method='correlation',
            n_features=30
        )

        # Prepare data
        X = df_featured[feature_cols].values
        y = df_featured['future_return'].values

        # Time series split to prevent data leakage
        total_size = len(X)
        train_size = int(total_size * (1 - test_size - val_size))
        val_size_int = int(total_size * val_size)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size_int]
        y_val = y[train_size:train_size + val_size_int]
        X_test = X[train_size + val_size_int:]
        y_test = y[train_size + val_size_int:]

        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Create ensemble
        ensemble = EnsemblePredictor()

        # Add multiple model types
        model_types = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']

        for model_type in model_types:
            model = PredictionModel(
                model_type=model_type,
                model_name=f"price_predictor_{model_type}",
                version="1.0"
            )
            ensemble.add_model(model)

        # Train all models
        metrics = ensemble.train_all(X_train, y_train, X_val, y_val, feature_cols)

        # Evaluate on test set
        test_predictions = ensemble.predict(X_test)
        test_score = np.corrcoef(y_test, test_predictions)[0, 1]
        direction_accuracy = np.mean(np.sign(y_test) == np.sign(test_predictions))

        logger.info(f"Ensemble test correlation: {test_score:.4f}")
        logger.info(f"Direction accuracy: {direction_accuracy:.4f}")

        # Store model performance
        self.db.store_performance_metrics(
            period="training",
            metrics={
                'test_correlation': test_score,
                'direction_accuracy': direction_accuracy,
                'timestamp': datetime.now().isoformat()
            }
        )

        # Save the ensemble
        self.models['price_predictor'] = ensemble

        return ensemble

    def should_retrain(self, model_name: str) -> bool:
        """Determine if a model needs retraining."""

        # Check last retrain date
        if model_name in self.retrain_schedule:
            last_retrain = self.retrain_schedule[model_name]
            retrain_freq = config.get('ml.models.price_prediction.retrain_frequency', 'weekly')

            if retrain_freq == 'daily':
                retrain_delta = timedelta(days=1)
            elif retrain_freq == 'weekly':
                retrain_delta = timedelta(days=7)
            elif retrain_freq == 'monthly':
                retrain_delta = timedelta(days=30)
            else:
                retrain_delta = timedelta(days=7)

            if datetime.now() - last_retrain < retrain_delta:
                return False

        # Check model performance
        perf_df = self.db.get_prediction_performance(days=7)

        if not perf_df.empty:
            avg_accuracy = perf_df['avg_accuracy'].mean()

            if avg_accuracy < self.performance_threshold:
                logger.info(f"Model {model_name} accuracy ({avg_accuracy:.3f}) below threshold ({self.performance_threshold})")
                return True

        # Check by schedule
        return True

    def retrain_models(self, market_data: pd.DataFrame):
        """Retrain models based on new data and performance."""

        logger.info("Checking if models need retraining")

        if self.should_retrain('price_predictor'):
            logger.info("Retraining price prediction models")

            # Retrain
            new_ensemble = self.train_prediction_models(market_data)

            # Update
            self.models['price_predictor'] = new_ensemble
            self.retrain_schedule['price_predictor'] = datetime.now()

            # Log the learning
            self.db.log_learning(
                learning_type="model_retrain",
                description="Retrained price prediction models with new data",
                previous_behavior="Old model version",
                new_behavior="New model version with updated weights",
                trigger_event="scheduled_retrain",
                expected_improvement=0.05
            )

            logger.info("Model retraining complete")
        else:
            logger.info("Models are current - no retraining needed")

    def train_strategy_specific_models(
        self,
        df: pd.DataFrame,
        strategy_type: str
    ) -> PredictionModel:
        """Train models optimized for specific trading strategies."""

        logger.info(f"Training model for {strategy_type} strategy")

        # Engineer features
        df_featured = self.feature_engineer.engineer_features(df)

        # Strategy-specific target
        if strategy_type == "momentum":
            # Predict momentum continuation
            df_featured['target'] = (
                (df_featured['close'].shift(-5) > df_featured['close']).astype(int) * 2 - 1
            )
        elif strategy_type == "mean_reversion":
            # Predict reversal from extremes
            df_featured['price_zscore'] = (
                (df_featured['close'] - df_featured['close'].rolling(20).mean()) /
                df_featured['close'].rolling(20).std()
            )
            df_featured['target'] = -df_featured['price_zscore']
        elif strategy_type == "breakout":
            # Predict breakout continuation
            df_featured['target'] = (
                (df_featured['close'] > df_featured['close'].rolling(20).max()).astype(int) * 2 - 1
            )
        else:
            # Default: predict returns
            df_featured['target'] = df_featured['close'].pct_change(5).shift(-5)

        df_featured = df_featured.dropna()

        # Select appropriate features
        feature_cols = self.feature_engineer.select_features(
            df_featured,
            target='target',
            n_features=25
        )

        X = df_featured[feature_cols].values
        y = df_featured['target'].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Train model
        model = PredictionModel(
            model_type="xgboost",
            model_name=f"{strategy_type}_predictor",
            version="1.0"
        )

        model.train(X_train, y_train, X_test, y_test, feature_cols)

        logger.info(f"{strategy_type} model training complete")

        return model

    def evaluate_model_impact(self) -> Dict[str, Any]:
        """Evaluate how model predictions impact actual trading performance."""

        # Get recent predictions and trades
        predictions_df = self.db.get_prediction_performance(days=30)
        trades_df = self.db.get_trades_history(days=30)

        if predictions_df.empty or trades_df.empty:
            return {}

        # Analyze correlation between prediction confidence and trade success
        evaluation = {
            'avg_prediction_accuracy': predictions_df['avg_accuracy'].mean(),
            'avg_profit_impact': predictions_df['avg_profit_impact'].mean(),
            'total_predictions': len(predictions_df),
            'trade_win_rate': len(trades_df[trades_df['profit_loss'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0
        }

        logger.info(f"Model impact evaluation: {evaluation}")

        return evaluation

    def adaptive_feature_selection(self, df: pd.DataFrame) -> List[str]:
        """Adaptively select features based on recent performance."""

        # Engineer all features
        df_featured = self.feature_engineer.engineer_features(df)

        # Get recent performance data
        perf_df = self.db.get_prediction_performance(days=14)

        if perf_df.empty:
            # Default feature selection
            return self.feature_engineer.select_features(
                df_featured,
                target='close',
                n_features=30
            )

        # Analyze which feature sets performed best
        # This is a simplified version - could be more sophisticated
        if perf_df['avg_accuracy'].mean() < 0.5:
            # Try different feature selection method
            logger.info("Switching to variance-based feature selection")

            self.db.log_learning(
                learning_type="feature_selection",
                description="Switched feature selection method due to poor performance",
                previous_behavior="correlation-based selection",
                new_behavior="variance-based selection",
                trigger_event="low_prediction_accuracy",
                expected_improvement=0.05
            )

            return self.feature_engineer.select_features(
                df_featured,
                target='close',
                method='variance',
                n_features=30
            )

        return self.feature_engineer.select_features(
            df_featured,
            target='close',
            n_features=30
        )

    def cross_validate_models(
        self,
        df: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """Perform time series cross-validation."""

        logger.info(f"Running {n_splits}-fold cross-validation")

        # Engineer features
        df_featured = self.feature_engineer.engineer_features(df)
        df_featured['target'] = df_featured['close'].pct_change(5).shift(-5)
        df_featured = df_featured.dropna()

        feature_cols = self.feature_engineer.select_features(
            df_featured,
            target='target',
            n_features=30
        )

        X = df_featured[feature_cols].values
        y = df_featured['target'].values

        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = {
            'xgboost': [],
            'lightgbm': [],
            'random_forest': []
        }

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_type in results.keys():
                model = PredictionModel(model_type=model_type)
                model.train(X_train, y_train, feature_names=feature_cols)

                predictions = model.predict(X_test)
                accuracy = np.mean(np.sign(predictions) == np.sign(y_test))
                results[model_type].append(accuracy)

        # Log results
        for model_type, accuracies in results.items():
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            logger.info(f"{model_type}: {avg_acc:.4f} ± {std_acc:.4f}")

        return results

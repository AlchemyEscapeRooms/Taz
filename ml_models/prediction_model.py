"""ML prediction models with self-learning capabilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger
from utils.database import Database

logger = get_logger(__name__)


class PredictionModel:
    """Base prediction model with self-learning capabilities."""

    def __init__(
        self,
        model_type: str = "xgboost",
        model_name: str = "default",
        version: str = "1.0"
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.version = version
        self.model = None
        self.feature_cols = []
        self.performance_history = []
        self.db = Database()

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                min_data_in_bin=1,      # Allow small datasets
                min_data_in_leaf=1,     # Allow small datasets
                n_jobs=1,               # Disable parallel to avoid Windows subprocess issues
                verbose=-1              # Suppress warnings
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            self.model = LinearRegression()

        logger.info(f"Initialized {self.model_type} model")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None
    ) -> Dict[str, float]:
        """Train the model and return performance metrics."""

        if feature_names:
            self.feature_cols = feature_names

        logger.info(f"Training {self.model_type} with {len(X_train)} samples")

        # Train the model
        if X_val is not None and y_val is not None:
            if self.model_type == "xgboost":
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif self.model_type == "lightgbm":
                # LightGBM 4.x uses callbacks for verbose control
                from lightgbm import early_stopping, log_evaluation
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[log_evaluation(period=-1)]  # Suppress logging
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        metrics = {'train_r2': train_score}

        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_r2'] = val_score

            # Calculate additional metrics
            y_pred = self.model.predict(X_val)
            mae = np.mean(np.abs(y_val - y_pred))
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

            metrics['mae'] = mae
            metrics['rmse'] = rmse

            # Directional accuracy
            direction_correct = np.sum(np.sign(y_pred) == np.sign(y_val))
            metrics['direction_accuracy'] = direction_correct / len(y_val)

        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence estimates."""
        predictions = self.predict(X)

        # For tree-based models, use prediction variance
        if self.model_type in ["random_forest", "xgboost", "lightgbm"]:
            if hasattr(self.model, 'estimators_'):
                # Get predictions from all trees
                tree_predictions = np.array([
                    tree.predict(X) for tree in self.model.estimators_
                ])
                confidence = 1 - (np.std(tree_predictions, axis=0) / (np.abs(predictions) + 1e-10))
            else:
                # Simple confidence based on prediction magnitude
                confidence = np.ones_like(predictions) * 0.7
        else:
            confidence = np.ones_like(predictions) * 0.6

        confidence = np.clip(confidence, 0, 1)
        return predictions, confidence

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance
            })
            return df.sort_values('importance', ascending=False)
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()

    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = f"models/{self.model_name}_{self.version}_{datetime.now().strftime('%Y%m%d')}.pkl"

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'version': self.version,
            'feature_cols': self.feature_cols,
            'performance_history': self.performance_history
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.model_name = model_data['model_name']
        self.version = model_data['version']
        self.feature_cols = model_data['feature_cols']
        self.performance_history = model_data.get('performance_history', [])

        logger.info(f"Model loaded from {path}")

    def update_from_results(self, actual_results: pd.DataFrame):
        """Self-learning: Update model based on actual trading results."""

        # Analyze prediction accuracy
        prediction_accuracy = self.db.get_prediction_performance(days=30)

        if prediction_accuracy.empty:
            logger.info("No prediction data available for learning yet")
            return

        avg_accuracy = prediction_accuracy['avg_accuracy'].mean()
        avg_profit_impact = prediction_accuracy['avg_profit_impact'].mean()

        logger.info(f"Current model performance: Accuracy={avg_accuracy:.3f}, Profit Impact={avg_profit_impact:.2f}")

        # If performance is below threshold, trigger retraining
        if avg_accuracy < 0.5 or avg_profit_impact < 0:
            logger.warning("Model performance below threshold - marking for retraining")

            self.db.log_learning(
                learning_type="model_performance",
                description="Model accuracy dropped below threshold",
                previous_behavior=f"Accuracy: {avg_accuracy:.3f}",
                new_behavior="Triggering model retraining",
                trigger_event="low_accuracy",
                expected_improvement=0.1
            )

            return True  # Signal that retraining is needed

        return False


class LSTMPredictor:
    """LSTM model for time series prediction."""

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 30,
        model_name: str = "lstm_predictor"
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_name = model_name
        self.model = None
        self.feature_cols = []

        self._build_model()

    def _build_model(self):
        """Build LSTM architecture."""
        self.model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        logger.info("LSTM model built")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Train the LSTM model."""

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        metrics = {
            'train_loss': float(history.history['loss'][-1]),
            'train_mae': float(history.history['mae'][-1])
        }

        if validation_data:
            metrics['val_loss'] = float(history.history['val_loss'][-1])
            metrics['val_mae'] = float(history.history['val_mae'][-1])

        logger.info(f"LSTM training complete. Metrics: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def save(self, path: str = None):
        """Save LSTM model."""
        if path is None:
            path = f"models/{self.model_name}_{datetime.now().strftime('%Y%m%d')}.h5"

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"LSTM model saved to {path}")

    def load(self, path: str):
        """Load LSTM model."""
        self.model = keras.models.load_model(path)
        logger.info(f"LSTM model loaded from {path}")


class EnsemblePredictor:
    """Ensemble of multiple models for robust predictions."""

    def __init__(self, models: List[PredictionModel] = None):
        self.models = models or []
        self.weights = None
        self.db = Database()

    def add_model(self, model: PredictionModel):
        """Add a model to the ensemble."""
        self.models.append(model)
        logger.info(f"Added {model.model_type} to ensemble")

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """Train all models in the ensemble."""

        all_metrics = {}

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_type}")
            metrics = model.train(X_train, y_train, X_val, y_val, feature_names)
            all_metrics[model.model_type] = metrics

        # Calculate ensemble weights based on validation performance
        if X_val is not None and y_val is not None:
            self._calculate_weights(X_val, y_val)

        return all_metrics

    def _calculate_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calculate optimal weights for each model based on performance."""

        predictions = []
        errors = []

        for model in self.models:
            pred = model.predict(X_val)
            predictions.append(pred)

            # Calculate inverse of error as weight
            error = np.mean(np.abs(y_val - pred))
            errors.append(error)

        # Inverse error weighting
        errors = np.array(errors)
        inv_errors = 1 / (errors + 1e-10)
        self.weights = inv_errors / inv_errors.sum()

        logger.info(f"Ensemble weights: {dict(zip([m.model_type for m in self.models], self.weights))}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""

        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted average
        if self.weights is not None:
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble predictions with confidence based on model agreement."""

        predictions = []
        confidences = []

        for model in self.models:
            if hasattr(model, 'predict_with_confidence'):
                pred, conf = model.predict_with_confidence(X)
            else:
                pred = model.predict(X)
                conf = np.ones_like(pred) * 0.6

            predictions.append(pred)
            confidences.append(conf)

        predictions = np.array(predictions)
        confidences = np.array(confidences)

        # Weighted average prediction
        if self.weights is not None:
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        # Confidence based on model agreement and individual confidences
        model_agreement = 1 - (np.std(predictions, axis=0) / (np.abs(ensemble_pred) + 1e-10))
        avg_confidence = np.mean(confidences, axis=0)

        # Combined confidence
        ensemble_confidence = (model_agreement + avg_confidence) / 2
        ensemble_confidence = np.clip(ensemble_confidence, 0, 1)

        return ensemble_pred, ensemble_confidence

    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual model predictions for analysis."""

        contributions = {}
        for model in self.models:
            pred = model.predict(X)
            contributions[model.model_type] = pred

        return contributions

    def evaluate_and_adapt(self, actual_results: pd.DataFrame):
        """Evaluate ensemble performance and adapt weights."""

        # Get recent prediction performance
        perf_df = self.db.get_prediction_performance(days=7)

        if perf_df.empty:
            return

        # Adjust weights based on recent performance
        model_scores = {}
        for model in self.models:
            model_perf = perf_df[perf_df['prediction_type'] == model.model_type]
            if not model_perf.empty:
                score = model_perf['avg_accuracy'].mean() * model_perf['avg_profit_impact'].mean()
                model_scores[model.model_type] = max(score, 0.01)
            else:
                model_scores[model.model_type] = 0.5

        # Update weights
        scores = np.array([model_scores.get(m.model_type, 0.5) for m in self.models])
        self.weights = scores / scores.sum()

        logger.info(f"Updated ensemble weights based on performance: {dict(zip([m.model_type for m in self.models], self.weights))}")

        self.db.log_learning(
            learning_type="ensemble_reweighting",
            description="Adjusted ensemble weights based on recent performance",
            previous_behavior=str(model_scores),
            new_behavior=str(dict(zip([m.model_type for m in self.models], self.weights))),
            trigger_event="periodic_evaluation",
            expected_improvement=0.05
        )

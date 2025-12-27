"""Feature engineering for ML models."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import talib as ta

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Generate features for machine learning models."""

    def __init__(self, technical_indicators: List[str] = None):
        self.technical_indicators = technical_indicators or [
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'RSI_14', 'MACD', 'BB_20',
            'ATR_14', 'ADX_14', 'OBV', 'VWAP'
        ]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features from OHLCV data."""
        df = df.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add technical indicators
        df = self._add_technical_indicators(df)

        # Add price patterns
        df = self._add_price_patterns(df)

        # Add volume features
        df = self._add_volume_features(df)

        # Add momentum features
        df = self._add_momentum_features(df)

        # Add volatility features
        df = self._add_volatility_features(df)

        # Add statistical features
        df = self._add_statistical_features(df)

        # Drop NaN values from initial rows
        df = df.dropna()

        logger.info(f"Generated {len(df.columns)} features")
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""

        # Moving Averages
        if 'SMA_20' in self.technical_indicators:
            df['SMA_20'] = ta.SMA(df['close'], timeperiod=20)
        if 'SMA_50' in self.technical_indicators:
            df['SMA_50'] = ta.SMA(df['close'], timeperiod=50)
        if 'SMA_200' in self.technical_indicators:
            df['SMA_200'] = ta.SMA(df['close'], timeperiod=200)

        if 'EMA_12' in self.technical_indicators:
            df['EMA_12'] = ta.EMA(df['close'], timeperiod=12)
        if 'EMA_26' in self.technical_indicators:
            df['EMA_26'] = ta.EMA(df['close'], timeperiod=26)

        # RSI
        if 'RSI_14' in self.technical_indicators:
            df['RSI_14'] = ta.RSI(df['close'], timeperiod=14)

        # MACD
        if 'MACD' in self.technical_indicators:
            macd, signal, hist = ta.MACD(df['close'])
            df['MACD'] = macd
            df['MACD_signal'] = signal
            df['MACD_hist'] = hist

        # Bollinger Bands
        if 'BB_20' in self.technical_indicators:
            upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20)
            df['BB_upper'] = upper
            df['BB_middle'] = middle
            df['BB_lower'] = lower
            df['BB_width'] = (upper - lower) / middle

        # ATR
        if 'ATR_14' in self.technical_indicators:
            df['ATR_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # ADX
        if 'ADX_14' in self.technical_indicators:
            df['ADX_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # OBV
        if 'OBV' in self.technical_indicators:
            df['OBV'] = ta.OBV(df['close'], df['volume'])

        # VWAP
        if 'VWAP' in self.technical_indicators:
            df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # Stochastic
        df['STOCH_k'], df['STOCH_d'] = ta.STOCH(df['high'], df['low'], df['close'])

        # CCI
        df['CCI'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        # Williams %R
        df['WILLR'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features."""

        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_10d'] = df['close'].pct_change(10)

        # High-Low range
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']

        # Close position within range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Gap features
        df['gap'] = df['open'] / df['close'].shift(1) - 1

        # Candlestick patterns (simplified)
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        # Trend direction
        df['trend_5d'] = np.where(df['close'] > df['close'].shift(5), 1, -1)
        df['trend_10d'] = np.where(df['close'] > df['close'].shift(10), 1, -1)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""

        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)

        # Volume trend
        df['volume_trend'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()

        # Price-Volume relationship
        df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features."""

        # Rate of Change
        df['ROC_5'] = ta.ROC(df['close'], timeperiod=5)
        df['ROC_10'] = ta.ROC(df['close'], timeperiod=10)

        # Momentum
        df['MOM_5'] = ta.MOM(df['close'], timeperiod=5)
        df['MOM_10'] = ta.MOM(df['close'], timeperiod=10)

        # Moving Average crossovers
        df['MA_cross_5_20'] = np.where(
            df['close'].rolling(5).mean() > df['close'].rolling(20).mean(), 1, -1
        )

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""

        # Historical volatility
        df['volatility_5d'] = df['price_change'].rolling(window=5).std()
        df['volatility_10d'] = df['price_change'].rolling(window=10).std()
        df['volatility_20d'] = df['price_change'].rolling(window=20).std()

        # Volatility ratio
        df['volatility_ratio'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-10)

        # True Range
        df['true_range'] = ta.TRANGE(df['high'], df['low'], df['close'])
        df['tr_ratio'] = df['true_range'] / df['close']

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""

        # Z-score
        df['price_zscore_20'] = (
            (df['close'] - df['close'].rolling(20).mean()) /
            (df['close'].rolling(20).std() + 1e-10)
        )

        # Percentile rank
        df['price_percentile_20'] = (
            df['close'].rolling(20).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x)
            )
        )

        # Distance from moving averages
        df['dist_from_sma20'] = (df['close'] - df['SMA_20']) / df['SMA_20']
        df['dist_from_sma50'] = (df['close'] - df['SMA_50']) / df['SMA_50']

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target: str,
        method: str = 'correlation',
        n_features: int = 30
    ) -> List[str]:
        """Select most important features."""

        # Exclude non-feature columns (including 'symbol' which contains string ticker names)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol', target]
        # Only include numeric columns to avoid string-to-float conversion errors
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if method == 'correlation':
            # Select features based on correlation with target
            correlations = df[feature_cols].corrwith(df[target]).abs()
            selected = correlations.nlargest(n_features).index.tolist()

        elif method == 'variance':
            # Select features with highest variance
            variances = df[feature_cols].var()
            selected = variances.nlargest(n_features).index.tolist()

        else:
            # Default: use all features
            selected = feature_cols[:n_features]

        logger.info(f"Selected {len(selected)} features using {method} method")
        return selected

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        sequence_length: int = 60,
        prediction_horizon: int = 1
    ) -> tuple:
        """Create sequences for time series prediction."""

        X, y = [], []

        for i in range(sequence_length, len(df) - prediction_horizon + 1):
            # Features: past sequence_length days
            X.append(df[feature_cols].iloc[i-sequence_length:i].values)

            # Target: future return after prediction_horizon days
            y.append(df[target_col].iloc[i + prediction_horizon - 1])

        return np.array(X), np.array(y)

    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Normalize features to 0-1 range."""
        df = df.copy()

        for col in feature_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)

        return df

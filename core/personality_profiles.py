"""Personality profiles for different trading styles and risk tolerances."""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class PersonalityProfile:
    """Trading personality profile."""

    name: str
    description: str
    risk_tolerance: str  # conservative, moderate, aggressive
    trading_style: str  # day_trader, swing_trader, long_term_investor
    max_position_size: float
    max_portfolio_risk: float
    preferred_strategies: list
    min_confidence_threshold: float
    max_daily_trades: int
    stop_loss_pct: float
    take_profit_targets: list
    diversification_level: str  # high, medium, low
    news_sentiment_weight: float
    ml_prediction_weight: float
    technical_analysis_weight: float
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"{self.name} - {self.description}"


# Pre-defined personality profiles
PERSONALITY_PROFILES = {
    "conservative_income": PersonalityProfile(
        name="Conservative Income Seeker",
        description="Focus on stable, dividend-paying stocks with low volatility",
        risk_tolerance="conservative",
        trading_style="long_term_investor",
        max_position_size=0.05,  # 5% per position
        max_portfolio_risk=0.01,  # 1% max risk
        preferred_strategies=["mean_reversion", "pairs_trading"],
        min_confidence_threshold=0.75,
        max_daily_trades=5,
        stop_loss_pct=0.03,  # 3%
        take_profit_targets=[0.05, 0.10],  # 5%, 10%
        diversification_level="high",
        news_sentiment_weight=0.2,
        ml_prediction_weight=0.3,
        technical_analysis_weight=0.5,
        parameters={
            "prefer_large_cap": True,
            "min_dividend_yield": 0.02,
            "max_beta": 1.0
        }
    ),

    "balanced_growth": PersonalityProfile(
        name="Balanced Growth",
        description="Balanced approach seeking steady growth with moderate risk",
        risk_tolerance="moderate",
        trading_style="swing_trader",
        max_position_size=0.10,  # 10% per position
        max_portfolio_risk=0.02,  # 2% max risk
        preferred_strategies=["trend_following", "ml_hybrid", "momentum"],
        min_confidence_threshold=0.65,
        max_daily_trades=20,
        stop_loss_pct=0.02,  # 2%
        take_profit_targets=[0.02, 0.05, 0.10],  # 2%, 5%, 10%
        diversification_level="medium",
        news_sentiment_weight=0.3,
        ml_prediction_weight=0.4,
        technical_analysis_weight=0.3,
        parameters={
            "rebalance_frequency": "weekly",
            "sector_diversification": True
        }
    ),

    "aggressive_growth": PersonalityProfile(
        name="Aggressive Growth",
        description="High risk, high reward strategy for maximum capital appreciation",
        risk_tolerance="aggressive",
        trading_style="day_trader",
        max_position_size=0.15,  # 15% per position
        max_portfolio_risk=0.03,  # 3% max risk
        preferred_strategies=["momentum", "breakout", "ml_hybrid"],
        min_confidence_threshold=0.55,
        max_daily_trades=100,
        stop_loss_pct=0.015,  # 1.5%
        take_profit_targets=[0.02, 0.04, 0.08, 0.15],  # 2%, 4%, 8%, 15%
        diversification_level="low",
        news_sentiment_weight=0.4,
        ml_prediction_weight=0.4,
        technical_analysis_weight=0.2,
        parameters={
            "prefer_high_volatility": True,
            "momentum_threshold": 0.03,
            "quick_exits": True
        }
    ),

    "day_trader_scalper": PersonalityProfile(
        name="Day Trader Scalper",
        description="High-frequency trading with quick entries and exits for small profits",
        risk_tolerance="moderate",
        trading_style="day_trader",
        max_position_size=0.08,  # 8% per position
        max_portfolio_risk=0.015,  # 1.5% max risk
        preferred_strategies=["rsi", "macd", "breakout"],
        min_confidence_threshold=0.60,
        max_daily_trades=1000,  # High-frequency
        stop_loss_pct=0.005,  # 0.5% tight stop
        take_profit_targets=[0.005, 0.01, 0.015],  # 0.5%, 1%, 1.5% - quick profits
        diversification_level="low",
        news_sentiment_weight=0.2,
        ml_prediction_weight=0.3,
        technical_analysis_weight=0.5,
        parameters={
            "holding_time_max": 3600,  # 1 hour max
            "prefer_liquid_stocks": True,
            "min_volume": 5000000
        }
    ),

    "value_investor": PersonalityProfile(
        name="Value Investor",
        description="Long-term value investing focusing on fundamentals",
        risk_tolerance="conservative",
        trading_style="long_term_investor",
        max_position_size=0.12,  # 12% per position
        max_portfolio_risk=0.02,  # 2% max risk
        preferred_strategies=["mean_reversion", "pairs_trading"],
        min_confidence_threshold=0.70,
        max_daily_trades=3,
        stop_loss_pct=0.10,  # 10% - wider stops for long-term
        take_profit_targets=[0.20, 0.50, 1.00],  # 20%, 50%, 100% - long-term targets
        diversification_level="high",
        news_sentiment_weight=0.3,
        ml_prediction_weight=0.2,
        technical_analysis_weight=0.5,
        parameters={
            "focus_fundamentals": True,
            "min_pe_ratio": 0,
            "max_pe_ratio": 15,
            "prefer_undervalued": True,
            "min_holding_days": 90
        }
    ),

    "momentum_trader": PersonalityProfile(
        name="Momentum Trader",
        description="Ride trending stocks for maximum momentum profits",
        risk_tolerance="aggressive",
        trading_style="swing_trader",
        max_position_size=0.12,  # 12% per position
        max_portfolio_risk=0.025,  # 2.5% max risk
        preferred_strategies=["momentum", "trend_following", "breakout"],
        min_confidence_threshold=0.60,
        max_daily_trades=30,
        stop_loss_pct=0.02,  # 2%
        take_profit_targets=[0.05, 0.10, 0.20],  # 5%, 10%, 20%
        diversification_level="medium",
        news_sentiment_weight=0.4,
        ml_prediction_weight=0.3,
        technical_analysis_weight=0.3,
        parameters={
            "trend_strength_min": 0.05,
            "follow_market_leaders": True,
            "volume_confirmation": True
        }
    ),

    "ai_optimized": PersonalityProfile(
        name="AI Optimized",
        description="Fully ML-driven with minimal human bias, adapts based on performance",
        risk_tolerance="moderate",
        trading_style="swing_trader",
        max_position_size=0.10,  # 10% per position
        max_portfolio_risk=0.02,  # 2% max risk
        preferred_strategies=["ml_hybrid"],  # Primarily ML-based
        min_confidence_threshold=0.65,
        max_daily_trades=50,
        stop_loss_pct=0.02,  # 2%
        take_profit_targets=[0.03, 0.06, 0.12],  # Dynamic based on ML
        diversification_level="medium",
        news_sentiment_weight=0.35,
        ml_prediction_weight=0.50,  # Heavy ML weight
        technical_analysis_weight=0.15,
        parameters={
            "adaptive_learning": True,
            "self_optimize": True,
            "auto_adjust_risk": True,
            "continuous_learning": True
        }
    ),

    "ai_adaptive": PersonalityProfile(
        name="AI Adaptive Trader",
        description="Uses AI Learning System to continuously adapt signal weights based on prediction accuracy",
        risk_tolerance="moderate",
        trading_style="swing_trader",
        max_position_size=0.10,  # 10% per position
        max_portfolio_risk=0.02,  # 2% max risk
        preferred_strategies=["trend_following", "momentum", "rsi", "macd", "breakout", "ai_prediction", "mean_reversion", "pairs_trading", "ml_hybrid"],  # All strategies
        min_confidence_threshold=0.60,  # 60% confidence threshold
        max_daily_trades=30,
        stop_loss_pct=0.02,  # 2%
        take_profit_targets=[0.02, 0.05, 0.10],  # 2%, 5%, 10%
        diversification_level="medium",
        news_sentiment_weight=0.25,
        ml_prediction_weight=0.35,
        technical_analysis_weight=0.40,
        parameters={
            "use_ai_learning_system": True,  # Enable AI Learning System
            "adaptive_signal_weights": True,  # Adjust weights based on accuracy
            "prediction_tracking": True,  # Track all predictions
            "learn_from_trades": True,  # Learn from trade outcomes
            "accuracy_threshold": 0.60,  # 60% accuracy threshold for weight adjustments
            "weight_adjustment_rate": 0.05,  # 5% weight adjustment per learning cycle
            "min_predictions_for_learning": 10,  # Minimum predictions before adjusting weights
            "relearn_frequency_hours": 1  # Re-evaluate weights every hour
        }
    )
}


def get_profile(name: str) -> PersonalityProfile:
    """Get a personality profile by name."""
    return PERSONALITY_PROFILES.get(name, PERSONALITY_PROFILES["balanced_growth"])


def list_profiles() -> list:
    """List all available personality profiles."""
    return list(PERSONALITY_PROFILES.keys())


def create_custom_profile(
    name: str,
    base_profile: str = "balanced_growth",
    **overrides
) -> PersonalityProfile:
    """Create a custom profile based on an existing one."""
    base = PERSONALITY_PROFILES.get(base_profile, PERSONALITY_PROFILES["balanced_growth"])

    # Create a copy and apply overrides
    custom = PersonalityProfile(
        name=name,
        description=overrides.get('description', base.description),
        risk_tolerance=overrides.get('risk_tolerance', base.risk_tolerance),
        trading_style=overrides.get('trading_style', base.trading_style),
        max_position_size=overrides.get('max_position_size', base.max_position_size),
        max_portfolio_risk=overrides.get('max_portfolio_risk', base.max_portfolio_risk),
        preferred_strategies=overrides.get('preferred_strategies', base.preferred_strategies),
        min_confidence_threshold=overrides.get('min_confidence_threshold', base.min_confidence_threshold),
        max_daily_trades=overrides.get('max_daily_trades', base.max_daily_trades),
        stop_loss_pct=overrides.get('stop_loss_pct', base.stop_loss_pct),
        take_profit_targets=overrides.get('take_profit_targets', base.take_profit_targets),
        diversification_level=overrides.get('diversification_level', base.diversification_level),
        news_sentiment_weight=overrides.get('news_sentiment_weight', base.news_sentiment_weight),
        ml_prediction_weight=overrides.get('ml_prediction_weight', base.ml_prediction_weight),
        technical_analysis_weight=overrides.get('technical_analysis_weight', base.technical_analysis_weight),
        parameters=overrides.get('parameters', base.parameters.copy())
    )

    return custom

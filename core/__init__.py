"""Core trading bot modules."""

from core.personality_profiles import PersonalityProfile, PERSONALITY_PROFILES
from core.order_executor import OrderExecutor, OrderResult, OrderStatus

# Lazy import to avoid circular imports
def get_trading_bot():
    from core.trading_bot import TradingBot
    return TradingBot

__all__ = [
    'PersonalityProfile',
    'PERSONALITY_PROFILES',
    'get_trading_bot',
    'OrderExecutor',
    'OrderResult',
    'OrderStatus'
]

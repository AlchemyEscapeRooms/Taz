"""Pre-built trading strategies for backtesting."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import talib as ta

from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


def momentum_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Momentum-based trading strategy."""

    signals = []

    if len(data) < params.get('lookback', 20):
        logger.debug(f"Momentum: Insufficient data ({len(data)} bars, need {params.get('lookback', 20)})")
        return signals

    # Get parameters
    lookback = params.get('lookback', 20)
    threshold = params.get('threshold', 0.02)

    # Calculate momentum
    current_price = data['close'].iloc[-1]
    past_price = data['close'].iloc[-lookback]
    momentum = (current_price - past_price) / past_price

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Calculate additional indicators for context
    sma_20 = data['close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_price
    volatility = data['close'].pct_change().tail(20).std() if len(data) >= 20 else 0

    logger.debug(f"Momentum [{symbol}]: momentum={momentum:.2%}, threshold={threshold:.2%}, price=${current_price:.2f}")

    # Generate signals
    if momentum > threshold and symbol not in engine.open_positions:
        # Buy signal - use risk manager limits if available
        max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
        position_size = engine.capital * max_position_pct
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'momentum',
                'signal_value': momentum,
                'threshold': threshold,
                'direction': 'above',
                'supporting_indicators': {
                    'lookback_price': past_price,
                    'sma_20': sma_20,
                    'volatility': volatility
                },
                'explanation': f"BUY: {lookback}-day momentum ({momentum:.2%}) exceeded threshold ({threshold:.2%}). "
                              f"Price rose from ${past_price:.2f} to ${current_price:.2f}."
            }
        })

    elif momentum < -threshold and symbol in engine.open_positions:
        # Sell signal
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'momentum',
                'signal_value': momentum,
                'threshold': -threshold,
                'direction': 'below',
                'supporting_indicators': {
                    'lookback_price': past_price,
                    'sma_20': sma_20,
                    'volatility': volatility
                },
                'explanation': f"SELL: {lookback}-day momentum ({momentum:.2%}) fell below threshold ({-threshold:.2%}). "
                              f"Price dropped from ${past_price:.2f} to ${current_price:.2f}."
            }
        })

    return signals


def mean_reversion_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Mean reversion strategy using Bollinger Bands."""

    signals = []

    if len(data) < params.get('period', 20):
        return signals

    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2)

    # Calculate Bollinger Bands
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    current_price = data['close'].iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    current_sma = sma.iloc[-1]
    current_std = std.iloc[-1]

    # Calculate z-score (distance from mean in std devs)
    z_score = (current_price - current_sma) / current_std if current_std > 0 else 0

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Buy when price touches lower band
    if current_price <= current_lower and symbol not in engine.open_positions:
        max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
        position_size = engine.capital * max_position_pct
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'bollinger_lower_band',
                'signal_value': current_price,
                'threshold': current_lower,
                'direction': 'below',
                'supporting_indicators': {
                    'sma': current_sma,
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'z_score': z_score,
                    'std_dev_multiplier': std_dev
                },
                'explanation': f"BUY: Price ${current_price:.2f} touched lower Bollinger Band ${current_lower:.2f}. "
                              f"Z-score: {z_score:.2f} (oversold). Expecting mean reversion to SMA ${current_sma:.2f}."
            }
        })

    # Sell when price reaches SMA or upper band
    elif symbol in engine.open_positions:
        if current_price >= current_sma:
            signals.append({
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'reason': {
                    'primary_signal': 'bollinger_mean_reversion',
                    'signal_value': current_price,
                    'threshold': current_sma,
                    'direction': 'above',
                    'supporting_indicators': {
                        'sma': current_sma,
                        'upper_band': current_upper,
                        'lower_band': current_lower,
                        'z_score': z_score
                    },
                    'explanation': f"SELL: Price ${current_price:.2f} reverted to SMA ${current_sma:.2f}. "
                                  f"Mean reversion target reached. Z-score: {z_score:.2f}."
                }
            })

    return signals


def trend_following_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """
    Trend following using moving average state (not just crossovers).
    
    This strategy uses STATE-BASED logic:
    - BUY when short MA > long MA (uptrend) AND we don't have a position
    - SELL when short MA < long MA (downtrend) AND we have a position
    
    For backtesting, set use_crossover_only=True in params to use event-based logic.
    For live trading, state-based logic ensures we don't miss opportunities.
    """

    signals = []

    short_period = params.get('short_period', 20)
    long_period = params.get('long_period', 50)
    use_crossover_only = params.get('use_crossover_only', False)  # False = state-based (live), True = event-based (backtest)
    min_trend_strength = params.get('min_trend_strength', 0.5)  # Minimum % difference between MAs

    if len(data) < long_period:
        return signals

    # Calculate moving averages
    short_ma = data['close'].rolling(window=short_period).mean()
    long_ma = data['close'].rolling(window=long_period).mean()

    # Get current and previous values
    current_short = short_ma.iloc[-1]
    current_long = long_ma.iloc[-1]
    prev_short = short_ma.iloc[-2]
    prev_long = long_ma.iloc[-2]

    current_price = data['close'].iloc[-1]
    ma_diff = current_short - current_long
    ma_diff_pct = (ma_diff / current_long) * 100 if current_long > 0 else 0

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Detect crossover events (for logging)
    bullish_crossover = prev_short <= prev_long and current_short > current_long
    bearish_crossover = prev_short >= prev_long and current_short < current_long

    # STATE-BASED LOGIC (for live trading)
    if not use_crossover_only:
        # BUY: We're in an uptrend (short MA > long MA) with sufficient strength
        if current_short > current_long and ma_diff_pct >= min_trend_strength:
            if symbol not in engine.open_positions:
                max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
                position_size = engine.capital * max_position_pct
                quantity = position_size / current_price

                signal_type = 'ma_crossover_bullish' if bullish_crossover else 'uptrend_state'
                explanation = (f"BUY: Golden Cross detected. " if bullish_crossover else f"BUY: Uptrend confirmed. ")
                explanation += (f"{short_period}-day SMA (${current_short:.2f}) is above "
                               f"{long_period}-day SMA (${current_long:.2f}) by {ma_diff_pct:.2f}%.")

                signals.append({
                    'action': 'buy',
                    'symbol': symbol,
                    'price': current_price,
                    'quantity': quantity,
                    'reason': {
                        'primary_signal': signal_type,
                        'signal_value': ma_diff_pct,
                        'threshold': min_trend_strength,
                        'direction': 'above',
                        'supporting_indicators': {
                            f'sma_{short_period}': current_short,
                            f'sma_{long_period}': current_long,
                            'ma_spread_pct': ma_diff_pct,
                            'is_crossover': bullish_crossover
                        },
                        'explanation': explanation
                    }
                })

        # SELL: We're in a downtrend (short MA < long MA)
        elif current_short < current_long:
            if symbol in engine.open_positions:
                signal_type = 'ma_crossover_bearish' if bearish_crossover else 'downtrend_state'
                explanation = (f"SELL: Death Cross detected. " if bearish_crossover else f"SELL: Downtrend detected. ")
                explanation += (f"{short_period}-day SMA (${current_short:.2f}) is below "
                               f"{long_period}-day SMA (${current_long:.2f}) by {abs(ma_diff_pct):.2f}%.")

                signals.append({
                    'action': 'sell',
                    'symbol': symbol,
                    'price': current_price,
                    'reason': {
                        'primary_signal': signal_type,
                        'signal_value': ma_diff_pct,
                        'threshold': 0,
                        'direction': 'below',
                        'supporting_indicators': {
                            f'sma_{short_period}': current_short,
                            f'sma_{long_period}': current_long,
                            'ma_spread_pct': ma_diff_pct,
                            'is_crossover': bearish_crossover
                        },
                        'explanation': explanation
                    }
                })

    # EVENT-BASED LOGIC (for backtesting - original behavior)
    else:
        # Bullish crossover
        if bullish_crossover:
            if symbol not in engine.open_positions:
                max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
                position_size = engine.capital * max_position_pct
                quantity = position_size / current_price

                signals.append({
                    'action': 'buy',
                    'symbol': symbol,
                    'price': current_price,
                    'quantity': quantity,
                    'reason': {
                        'primary_signal': 'ma_crossover_bullish',
                        'signal_value': ma_diff,
                        'threshold': 0,
                        'direction': 'above',
                        'supporting_indicators': {
                            f'sma_{short_period}': current_short,
                            f'sma_{long_period}': current_long,
                            'prev_short_ma': prev_short,
                            'prev_long_ma': prev_long,
                            'ma_spread_pct': ma_diff_pct
                        },
                        'explanation': f"BUY: Golden Cross detected. {short_period}-day SMA (${current_short:.2f}) "
                                      f"crossed above {long_period}-day SMA (${current_long:.2f}). Bullish trend confirmed."
                    }
                })

        # Bearish crossover
        elif bearish_crossover:
            if symbol in engine.open_positions:
                signals.append({
                    'action': 'sell',
                    'symbol': symbol,
                    'price': current_price,
                    'reason': {
                        'primary_signal': 'ma_crossover_bearish',
                        'signal_value': ma_diff,
                        'threshold': 0,
                        'direction': 'below',
                        'supporting_indicators': {
                            f'sma_{short_period}': current_short,
                            f'sma_{long_period}': current_long,
                            'prev_short_ma': prev_short,
                            'prev_long_ma': prev_long,
                            'ma_spread_pct': ma_diff_pct
                        },
                        'explanation': f"SELL: Death Cross detected. {short_period}-day SMA (${current_short:.2f}) "
                                      f"crossed below {long_period}-day SMA (${current_long:.2f}). Bearish trend confirmed."
                    }
                })

    return signals


def breakout_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Breakout strategy based on price channels."""

    signals = []

    lookback = params.get('lookback', 20)

    if len(data) < lookback:
        return signals

    # Calculate high/low channels
    high_channel = data['high'].rolling(window=lookback).max()
    low_channel = data['low'].rolling(window=lookback).min()

    current_price = data['close'].iloc[-1]
    current_high_channel = high_channel.iloc[-2]  # Previous period to avoid looking ahead
    current_low_channel = low_channel.iloc[-2]
    channel_width = current_high_channel - current_low_channel
    channel_width_pct = (channel_width / current_low_channel) * 100 if current_low_channel > 0 else 0

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Breakout above resistance
    if current_price > current_high_channel and symbol not in engine.open_positions:
        max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
        position_size = engine.capital * max_position_pct
        quantity = position_size / current_price
        breakout_strength = ((current_price - current_high_channel) / current_high_channel) * 100

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'channel_breakout_up',
                'signal_value': current_price,
                'threshold': current_high_channel,
                'direction': 'above',
                'supporting_indicators': {
                    'resistance_level': current_high_channel,
                    'support_level': current_low_channel,
                    'channel_width': channel_width,
                    'channel_width_pct': channel_width_pct,
                    'breakout_strength_pct': breakout_strength,
                    'lookback_period': lookback
                },
                'explanation': f"BUY: Price ${current_price:.2f} broke above {lookback}-day resistance ${current_high_channel:.2f}. "
                              f"Breakout strength: {breakout_strength:.2f}%. Channel width: {channel_width_pct:.1f}%."
            }
        })

    # Break below support
    elif current_price < current_low_channel and symbol in engine.open_positions:
        breakdown_strength = ((current_low_channel - current_price) / current_low_channel) * 100

        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'channel_breakdown',
                'signal_value': current_price,
                'threshold': current_low_channel,
                'direction': 'below',
                'supporting_indicators': {
                    'resistance_level': current_high_channel,
                    'support_level': current_low_channel,
                    'channel_width': channel_width,
                    'breakdown_strength_pct': breakdown_strength
                },
                'explanation': f"SELL: Price ${current_price:.2f} broke below {lookback}-day support ${current_low_channel:.2f}. "
                              f"Breakdown strength: {breakdown_strength:.2f}%. Exiting to prevent further losses."
            }
        })

    return signals


def rsi_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """RSI-based strategy for overbought/oversold conditions."""

    signals = []

    period = params.get('period', 14)
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)

    if len(data) < period + 1:
        return signals

    # Calculate RSI
    rsi = ta.RSI(data['close'].values, timeperiod=period)

    current_rsi = rsi[-1]
    prev_rsi = rsi[-2] if len(rsi) > 1 else current_rsi
    current_price = data['close'].iloc[-1]

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Buy when oversold
    if current_rsi < oversold and symbol not in engine.open_positions:
        max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
        position_size = engine.capital * max_position_pct
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'rsi_oversold',
                'signal_value': current_rsi,
                'threshold': oversold,
                'direction': 'below',
                'supporting_indicators': {
                    'rsi_period': period,
                    'prev_rsi': prev_rsi,
                    'rsi_change': current_rsi - prev_rsi,
                    'oversold_threshold': oversold,
                    'overbought_threshold': overbought
                },
                'explanation': f"BUY: RSI({period}) = {current_rsi:.1f} is below oversold threshold ({oversold}). "
                              f"Asset is oversold, expecting bounce. Previous RSI: {prev_rsi:.1f}."
            }
        })

    # Sell when overbought
    elif current_rsi > overbought and symbol in engine.open_positions:
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'rsi_overbought',
                'signal_value': current_rsi,
                'threshold': overbought,
                'direction': 'above',
                'supporting_indicators': {
                    'rsi_period': period,
                    'prev_rsi': prev_rsi,
                    'rsi_change': current_rsi - prev_rsi
                },
                'explanation': f"SELL: RSI({period}) = {current_rsi:.1f} is above overbought threshold ({overbought}). "
                              f"Asset is overbought, expecting pullback. Taking profits."
            }
        })

    return signals


def macd_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """MACD crossover strategy."""

    signals = []

    fast = params.get('fast', 12)
    slow = params.get('slow', 26)
    signal_period = params.get('signal', 9)

    if len(data) < slow + signal_period:
        return signals

    # Calculate MACD
    macd, macd_signal, macd_hist = ta.MACD(
        data['close'].values,
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal_period
    )

    current_macd = macd[-1]
    current_signal = macd_signal[-1]
    current_hist = macd_hist[-1]
    prev_macd = macd[-2]
    prev_signal = macd_signal[-2]
    prev_hist = macd_hist[-2]

    current_price = data['close'].iloc[-1]
    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Bullish crossover
    if prev_macd <= prev_signal and current_macd > current_signal:
        if symbol not in engine.open_positions:
            max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
            position_size = engine.capital * max_position_pct
            quantity = position_size / current_price

            signals.append({
                'action': 'buy',
                'symbol': symbol,
                'price': current_price,
                'quantity': quantity,
                'reason': {
                    'primary_signal': 'macd_bullish_crossover',
                    'signal_value': current_macd,
                    'threshold': current_signal,
                    'direction': 'above',
                    'supporting_indicators': {
                        'macd_line': current_macd,
                        'signal_line': current_signal,
                        'histogram': current_hist,
                        'prev_macd': prev_macd,
                        'prev_signal': prev_signal,
                        'histogram_change': current_hist - prev_hist,
                        'fast_period': fast,
                        'slow_period': slow,
                        'signal_period': signal_period
                    },
                    'explanation': f"BUY: MACD bullish crossover. MACD ({current_macd:.4f}) crossed above "
                                  f"Signal line ({current_signal:.4f}). Histogram: {current_hist:.4f}. "
                                  f"Momentum shifting bullish."
                }
            })

    # Bearish crossover
    elif prev_macd >= prev_signal and current_macd < current_signal:
        if symbol in engine.open_positions:
            signals.append({
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'reason': {
                    'primary_signal': 'macd_bearish_crossover',
                    'signal_value': current_macd,
                    'threshold': current_signal,
                    'direction': 'below',
                    'supporting_indicators': {
                        'macd_line': current_macd,
                        'signal_line': current_signal,
                        'histogram': current_hist,
                        'prev_macd': prev_macd,
                        'prev_signal': prev_signal,
                        'histogram_change': current_hist - prev_hist
                    },
                    'explanation': f"SELL: MACD bearish crossover. MACD ({current_macd:.4f}) crossed below "
                                  f"Signal line ({current_signal:.4f}). Histogram: {current_hist:.4f}. "
                                  f"Momentum shifting bearish."
                }
            })

    return signals


def pairs_trading_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Pairs trading / statistical arbitrage strategy."""

    signals = []

    # This is a simplified version - full implementation would need two correlated assets

    lookback = params.get('lookback', 20)
    entry_z = params.get('entry_z', 2.0)
    exit_z = params.get('exit_z', 0.5)

    if len(data) < lookback:
        return signals

    # Calculate z-score
    sma = data['close'].rolling(window=lookback).mean()
    std = data['close'].rolling(window=lookback).std()
    z_score = (data['close'] - sma) / std

    current_z = z_score.iloc[-1]
    prev_z = z_score.iloc[-2] if len(z_score) > 1 else current_z
    current_price = data['close'].iloc[-1]
    current_sma = sma.iloc[-1]
    current_std = std.iloc[-1]

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Buy when significantly below mean
    if current_z < -entry_z and symbol not in engine.open_positions:
        max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
        position_size = engine.capital * max_position_pct
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'z_score_oversold',
                'signal_value': current_z,
                'threshold': -entry_z,
                'direction': 'below',
                'supporting_indicators': {
                    'z_score': current_z,
                    'prev_z_score': prev_z,
                    'sma': current_sma,
                    'std_dev': current_std,
                    'entry_threshold': entry_z,
                    'exit_threshold': exit_z,
                    'lookback': lookback
                },
                'explanation': f"BUY: Z-score ({current_z:.2f}) below -{entry_z} threshold. "
                              f"Price ${current_price:.2f} is {abs(current_z):.1f} std devs below {lookback}-day mean ${current_sma:.2f}. "
                              f"Expecting mean reversion."
            }
        })

    # Sell when reverting to mean
    elif abs(current_z) < exit_z and symbol in engine.open_positions:
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'z_score_mean_reversion',
                'signal_value': current_z,
                'threshold': exit_z,
                'direction': 'near_zero',
                'supporting_indicators': {
                    'z_score': current_z,
                    'prev_z_score': prev_z,
                    'sma': current_sma
                },
                'explanation': f"SELL: Z-score ({current_z:.2f}) reverted within ±{exit_z} of mean. "
                              f"Mean reversion target achieved. Taking profits."
            }
        })

    # Also sell if goes too far in opposite direction
    elif current_z > entry_z and symbol in engine.open_positions:
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'z_score_overbought',
                'signal_value': current_z,
                'threshold': entry_z,
                'direction': 'above',
                'supporting_indicators': {
                    'z_score': current_z,
                    'prev_z_score': prev_z,
                    'sma': current_sma
                },
                'explanation': f"SELL: Z-score ({current_z:.2f}) exceeded +{entry_z} threshold. "
                              f"Price overextended above mean. Exiting position."
            }
        })

    return signals


def ml_hybrid_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Hybrid strategy combining ML predictions with technical indicators."""

    signals = []

    # This would integrate with the ML models
    # Simplified version using technical indicators

    if len(data) < 50:
        return signals

    # Multiple confirmation signals
    confirmations = 0
    confirmation_details = []
    target_confirmations = params.get('min_confirmations', 3)

    current_price = data['close'].iloc[-1]
    current_volume = data['volume'].iloc[-1]

    # RSI confirmation
    rsi = ta.RSI(data['close'].values, timeperiod=14)
    current_rsi = rsi[-1]
    if current_rsi < 30:
        confirmations += 1
        confirmation_details.append(f"RSI oversold ({current_rsi:.1f})")
    elif current_rsi > 70:
        confirmations -= 1
        confirmation_details.append(f"RSI overbought ({current_rsi:.1f})")

    # MACD confirmation
    macd, macd_signal, hist = ta.MACD(data['close'].values)
    current_macd = macd[-1]
    current_macd_signal = macd_signal[-1]
    if macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
        confirmations += 1
        confirmation_details.append("MACD bullish crossover")
    elif macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
        confirmations -= 1
        confirmation_details.append("MACD bearish crossover")

    # Moving average confirmation
    sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
    sma_50 = data['close'].rolling(window=50).mean().iloc[-1]
    if sma_20 > sma_50:
        confirmations += 1
        confirmation_details.append(f"SMA20 > SMA50 (bullish trend)")
    else:
        confirmations -= 1
        confirmation_details.append(f"SMA20 < SMA50 (bearish trend)")

    # Volume confirmation
    vol_avg = data['volume'].rolling(window=20).mean().iloc[-1]
    volume_ratio = current_volume / vol_avg if vol_avg > 0 else 1
    if current_volume > vol_avg * 1.5:
        confirmations += 1
        confirmation_details.append(f"High volume ({volume_ratio:.1f}x avg)")

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Buy signal
    if confirmations >= target_confirmations and symbol not in engine.open_positions:
        max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))
        position_size = engine.capital * max_position_pct
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'multi_indicator_bullish',
                'signal_value': confirmations,
                'threshold': target_confirmations,
                'direction': 'above',
                'supporting_indicators': {
                    'confirmations': confirmations,
                    'target_confirmations': target_confirmations,
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'macd_signal': current_macd_signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'volume_ratio': volume_ratio
                },
                'confirmations': confirmation_details,
                'explanation': f"BUY: {confirmations} bullish confirmations (threshold: {target_confirmations}). "
                              f"Signals: {', '.join(confirmation_details)}."
            }
        })

    # Sell signal
    elif confirmations <= -target_confirmations and symbol in engine.open_positions:
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'multi_indicator_bearish',
                'signal_value': confirmations,
                'threshold': -target_confirmations,
                'direction': 'below',
                'supporting_indicators': {
                    'confirmations': confirmations,
                    'target_confirmations': target_confirmations,
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'macd_signal': current_macd_signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50
                },
                'confirmations': confirmation_details,
                'explanation': f"SELL: {abs(confirmations)} bearish confirmations (threshold: {target_confirmations}). "
                              f"Signals: {', '.join(confirmation_details)}."
            }
        })

    return signals


def ai_prediction_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """
    AI-based trading strategy using MarketMonitor predictions.
    
    This strategy generates signals based on the AI Learning System's predictions
    and technical indicator analysis. It uses a confidence threshold to filter
    high-quality signals.

    For live trading, this integrates with the MarketMonitor's prediction system.
    For backtesting, it calculates its own signals using similar logic.
    """

    signals = []

    min_confidence = params.get('min_confidence', 70)  # Minimum confidence to trade
    position_size_pct = params.get('position_size', 0.1)

    if len(data) < 50:
        logger.debug(f"AI Adaptive: Insufficient data ({len(data)} bars, need 50)")
        return signals

    current_price = data['close'].iloc[-1]
    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Load scoring weights from config (with fallbacks)
    score_weights = {
        'trend_strong': config.get('strategy_scoring.trend_strong_bullish', 20),
        'trend_mild': config.get('strategy_scoring.trend_mild_bullish', 10),
        'rsi_oversold': config.get('strategy_scoring.rsi_oversold', 25),
        'rsi_low': config.get('strategy_scoring.rsi_low', 10),
        'rsi_overbought': config.get('strategy_scoring.rsi_overbought', -25),
        'rsi_high': config.get('strategy_scoring.rsi_high', -10),
        'macd_bullish_cross': config.get('strategy_scoring.macd_bullish_cross', 20),
        'macd_bullish': config.get('strategy_scoring.macd_bullish', 10),
        'volume_high': config.get('strategy_scoring.volume_high', 15),
        'volume_confirm': config.get('strategy_scoring.volume_confirmation', 10),
    }

    # Calculate technical indicators for scoring
    score = 0
    reasons = []

    # 1. Momentum (20-day)
    momentum = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
    if momentum > 2:
        score += score_weights['trend_strong']
        reasons.append(f"Strong momentum: {momentum:.1f}%")
    elif momentum > 0:
        score += score_weights['trend_mild']
        reasons.append(f"Positive momentum: {momentum:.1f}%")
    elif momentum < -2:
        score -= score_weights['trend_strong']
        reasons.append(f"Weak momentum: {momentum:.1f}%")
    else:
        score -= score_weights['trend_mild']
        reasons.append(f"Negative momentum: {momentum:.1f}%")
    
    # 2. RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    if rsi < 30:
        score += score_weights['rsi_oversold']
        reasons.append(f"RSI oversold: {rsi:.1f}")
    elif rsi < 45:
        score += score_weights['rsi_low']
        reasons.append(f"RSI low: {rsi:.1f}")
    elif rsi > 70:
        score += score_weights['rsi_overbought']  # Negative value
        reasons.append(f"RSI overbought: {rsi:.1f}")
    elif rsi > 55:
        score += score_weights['rsi_high']  # Negative value
        reasons.append(f"RSI high: {rsi:.1f}")

    # 3. MACD
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd - macd_signal

    if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
        score += score_weights['macd_bullish_cross']
        reasons.append("MACD bullish crossover")
    elif macd_hist.iloc[-1] > 0:
        score += score_weights['macd_bullish']
        reasons.append("MACD positive")
    elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
        score -= score_weights['macd_bullish_cross']  # Use negative of bullish
        reasons.append("MACD bearish crossover")
    elif macd_hist.iloc[-1] < 0:
        score -= score_weights['macd_bullish']  # Use negative of bullish
        reasons.append("MACD negative")

    # 4. Trend (SMA 20 vs SMA 50)
    sma20 = data['close'].rolling(20).mean().iloc[-1]
    sma50 = data['close'].rolling(50).mean().iloc[-1]
    trend_strength = ((sma20 - sma50) / sma50) * 100

    if sma20 > sma50:
        score += score_weights['volume_high']  # Reusing volume_high as trend weight
        reasons.append(f"Uptrend: SMA20 {trend_strength:.1f}% above SMA50")
    else:
        score -= score_weights['volume_high']
        reasons.append(f"Downtrend: SMA20 {abs(trend_strength):.1f}% below SMA50")

    # 5. Volume confirmation
    vol_avg = data['volume'].rolling(20).mean().iloc[-1]
    vol_ratio = data['volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1
    if vol_ratio > 1.5:
        score += score_weights['volume_confirm']
        reasons.append(f"High volume: {vol_ratio:.1f}x average")

    # Convert score to confidence (0-100)
    # Max possible score is around 90, min is around -90
    confidence = min(100, max(0, (score + 90) / 1.8))

    # Determine direction
    direction = 'up' if score > 0 else 'down'

    # Log the analysis
    logger.debug(f"AI Adaptive [{symbol}]: score={score}, confidence={confidence:.1f}%, "
                 f"direction={direction}, reasons={len(reasons)}")

    # Generate signals based on confidence threshold
    if confidence >= min_confidence and direction == 'up':
        if symbol not in engine.open_positions:
            logger.info(f"AI Adaptive BUY signal for {symbol}: confidence={confidence:.1f}%, "
                       f"price=${current_price:.2f}, reasons: {', '.join(reasons[:3])}")
            max_position_pct = getattr(engine, 'max_position_size', position_size_pct)
            position_size = engine.capital * max_position_pct
            quantity = position_size / current_price
            
            signals.append({
                'action': 'buy',
                'symbol': symbol,
                'price': current_price,
                'quantity': quantity,
                'reason': {
                    'primary_signal': 'ai_prediction_bullish',
                    'signal_value': confidence,
                    'threshold': min_confidence,
                    'direction': 'above',
                    'supporting_indicators': {
                        'ai_score': score,
                        'confidence': confidence,
                        'momentum': momentum,
                        'rsi': rsi,
                        'macd_histogram': macd_hist.iloc[-1],
                        'trend_strength': trend_strength,
                        'volume_ratio': vol_ratio
                    },
                    'explanation': f"BUY: AI prediction confidence {confidence:.1f}% (threshold: {min_confidence}%). "
                                  f"Signals: {'; '.join(reasons)}."
                }
            })
    
    elif confidence >= min_confidence and direction == 'down':
        if symbol in engine.open_positions:
            signals.append({
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'reason': {
                    'primary_signal': 'ai_prediction_bearish',
                    'signal_value': confidence,
                    'threshold': min_confidence,
                    'direction': 'below',
                    'supporting_indicators': {
                        'ai_score': score,
                        'confidence': confidence,
                        'momentum': momentum,
                        'rsi': rsi,
                        'macd_histogram': macd_hist.iloc[-1],
                        'trend_strength': trend_strength
                    },
                    'explanation': f"SELL: AI prediction bearish with {confidence:.1f}% confidence. "
                                  f"Signals: {'; '.join(reasons)}."
                }
            })
    
    return signals


# Strategy registry
STRATEGY_REGISTRY = {
    'momentum': momentum_strategy,
    'mean_reversion': mean_reversion_strategy,
    'trend_following': trend_following_strategy,
    'breakout': breakout_strategy,
    'rsi': rsi_strategy,
    'macd': macd_strategy,
    'pairs_trading': pairs_trading_strategy,
    'ml_hybrid': ml_hybrid_strategy,
    'ai_prediction': ai_prediction_strategy
}


# Default parameters for each strategy
# position_size: Fraction of capital per trade
# NOTE: For live trading, strategies should use engine.max_position_size from the risk manager
# For backtesting without risk manager, position_size param is used as fallback
DEFAULT_PARAMS = {
    'momentum': {
        'lookback': 20,
        'threshold': 0.02,
        'position_size': 0.1  # 10% - matches risk manager default
    },
    'mean_reversion': {
        'period': 20,
        'std_dev': 2,
        'position_size': 0.1
    },
    'trend_following': {
        'short_period': 20,
        'long_period': 50,
        'position_size': 0.1,
        'use_crossover_only': False,  # State-based for live trading
        'min_trend_strength': 0.5     # Minimum 0.5% spread between MAs
    },
    'breakout': {
        'lookback': 20,
        'position_size': 0.1
    },
    'rsi': {
        'period': 14,
        'oversold': 30,
        'overbought': 70,
        'position_size': 0.1
    },
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9,
        'position_size': 0.1
    },
    'pairs_trading': {
        'lookback': 20,
        'entry_z': 2.0,
        'exit_z': 0.5,
        'position_size': 0.1
    },
    'ml_hybrid': {
        'min_confirmations': 3,
        'position_size': 0.1
    },
    'ai_prediction': {
        'min_confidence': 60,  # Lowered from 70 to allow more signals
        'position_size': 0.1
    }
}


def get_position_size(engine, params: Dict[str, Any], current_price: float) -> tuple:
    """
    Helper function to get the correct position size respecting risk limits.

    Strategies should use this instead of calculating position size directly.
    This ensures consistency between strategy position sizing and risk manager limits.

    Args:
        engine: The trading engine (mock or real) with capital and max_position_size
        params: Strategy parameters (may contain position_size as fallback)
        current_price: Current price of the asset

    Returns:
        tuple: (position_value, quantity)
    """
    # Use engine's max_position_size if available (from risk manager)
    # Otherwise fall back to params, then default of 0.1 (10%)
    max_position_pct = getattr(engine, 'max_position_size', params.get('position_size', 0.1))

    position_value = engine.capital * max_position_pct
    quantity = position_value / current_price if current_price > 0 else 0

    return position_value, quantity

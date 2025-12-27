"""
Simple Trader - Strategy Selector Based Trading
================================================

Before trading each stock:
1. Run strategy selector to find best indicator
2. Use ONLY that indicator's buy/sell thresholds
3. Execute when threshold is hit

No weighted scoring. No AI. No complex combinations.
One stock, one indicator, clear rules.

Author: Claude AI
Date: December 2024
"""

import os
import sys
import time as time_module
import threading
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Fallback for older Python

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv
load_dotenv()

# Import trade logger for logging all trades
from utils.trade_logger import get_trade_logger, TradeReason

# Import risk manager for risk controls
from portfolio.risk_manager import RiskManager

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')


# =============================================================================
# PORTFOLIO SYMBOL FETCHING
# =============================================================================

def get_portfolio_symbols(paper: bool = True) -> List[str]:
    """
    Get symbols from current Alpaca portfolio positions.

    This ensures the bot only monitors/trades stocks you actually own.

    Args:
        paper: If True, use paper trading account. If False, use live account.

    Returns:
        List of stock symbols currently in the portfolio.
    """
    try:
        client = TradingClient(API_KEY, API_SECRET, paper=paper)
        positions = client.get_all_positions()
        symbols = [pos.symbol for pos in positions]

        if not symbols:
            print("[WARNING] No positions found in Alpaca portfolio!")
            print("          The bot needs stocks in your portfolio to monitor.")

        return symbols
    except Exception as e:
        print(f"[ERROR] Failed to get portfolio symbols: {e}")
        return []


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series):
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    diff = macd - signal
    return macd, signal, diff


def calculate_bollinger(prices: pd.Series, period: int = 20, std_dev: float = 2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    position = (prices - lower) / (upper - lower)
    return upper, lower, position


def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    avg_volume = volume.rolling(window=period).mean()
    return volume / avg_volume


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period).mean()


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

@dataclass
class Strategy:
    """A trading strategy with clear buy/sell rules."""
    name: str
    buy_rule: str
    sell_rule: str

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        """Check if buy condition is met at index i."""
        raise NotImplementedError

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        """Check if sell condition is met at index i."""
        raise NotImplementedError


class RSI_30_70(Strategy):
    def __init__(self):
        super().__init__("RSI 30/70", "RSI < 30", "RSI > 70")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] < 30

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] > 70


class RSI_40_70(Strategy):
    def __init__(self):
        super().__init__("RSI 40/70", "RSI < 40", "RSI > 70")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] < 40

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] > 70


class MACDCross(Strategy):
    def __init__(self):
        super().__init__("MACD Cross", "MACD crosses above signal", "MACD crosses below signal")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        return df.iloc[i]['macd_diff'] > 0 and df.iloc[i-1]['macd_diff'] <= 0

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        return df.iloc[i]['macd_diff'] < 0 and df.iloc[i-1]['macd_diff'] >= 0


class Bollinger_20_80(Strategy):
    def __init__(self):
        super().__init__("Bollinger 20/80", "Price at lower 20% of bands", "Price at upper 20% of bands")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] < 0.2

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] > 0.8


class BollingerMeanRev(Strategy):
    def __init__(self):
        super().__init__("Bollinger Mean Rev", "Price at lower 20% of bands", "Price returns to middle")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] < 0.2

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] > 0.5


class VolumeSpike(Strategy):
    def __init__(self):
        super().__init__("Volume Spike", "High volume + price drop", "High volume + price rise")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        high_vol = df.iloc[i]['volume_ratio'] > 1.5
        price_down = df.iloc[i]['close'] < df.iloc[i-1]['close']
        return high_vol and price_down

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        high_vol = df.iloc[i]['volume_ratio'] > 1.5
        price_up = df.iloc[i]['close'] > df.iloc[i-1]['close']
        return high_vol and price_up


class MeanRev2Pct(Strategy):
    def __init__(self):
        super().__init__("Mean Rev 2%", "Price 2%+ below SMA20", "Price returns to SMA20")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['close'] < df.iloc[i]['sma20'] * 0.98

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['close'] >= df.iloc[i]['sma20']


# === AGGRESSIVE STRATEGIES (Lower thresholds = more trades) ===

class RSI_40_60(Strategy):
    """More aggressive RSI - triggers more often"""
    def __init__(self):
        super().__init__("RSI 40/60", "RSI < 40", "RSI > 60")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] < 40

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] > 60


class RSI_45_55(Strategy):
    """Very aggressive RSI - triggers frequently"""
    def __init__(self):
        super().__init__("RSI 45/55", "RSI < 45", "RSI > 55")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] < 45

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] > 55


class Bollinger_30_70(Strategy):
    """More aggressive Bollinger bands"""
    def __init__(self):
        super().__init__("Bollinger 30/70", "Price at lower 30% of bands", "Price at upper 30% of bands")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] < 0.3

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] > 0.7


class MeanRev1Pct(Strategy):
    """More aggressive mean reversion - 1% instead of 2%"""
    def __init__(self):
        super().__init__("Mean Rev 1%", "Price 1%+ below SMA20", "Price returns to SMA20")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['close'] < df.iloc[i]['sma20'] * 0.99

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['close'] >= df.iloc[i]['sma20']


class MomentumSimple(Strategy):
    """Simple momentum - buy on uptick, sell on downtick"""
    def __init__(self):
        super().__init__("Momentum Simple", "MACD diff > 0", "MACD diff < 0")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['macd_diff'] > 0

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['macd_diff'] < 0


# Map strategy names to classes
STRATEGIES = {
    "RSI 30/70": RSI_30_70,
    "RSI 40/70": RSI_40_70,
    "RSI 40/60": RSI_40_60,
    "RSI 45/55": RSI_45_55,
    "MACD Cross": MACDCross,
    "Bollinger 20/80": Bollinger_20_80,
    "Bollinger 30/70": Bollinger_30_70,
    "Bollinger Mean Rev": BollingerMeanRev,
    "Volume Spike": VolumeSpike,
    "Mean Rev 2%": MeanRev2Pct,
    "Mean Rev 1%": MeanRev1Pct,
    "Momentum Simple": MomentumSimple,
}


# =============================================================================
# STRATEGY SELECTOR
# =============================================================================

def backtest_strategy(df: pd.DataFrame, strategy: Strategy, initial_capital: float = 10000,
                       symbol: str = 'UNKNOWN', log_trades: bool = False,
                       min_hold_hours: int = 24, stop_loss_pct: float = 0.10,
                       use_safety_rules: bool = True) -> dict:
    """
    Run backtest with given strategy.

    If use_safety_rules=True (default):
    - Normal: sell on signal when profitable
    - If sell signal but at loss: enter 24h hold mode
    - During hold: sell immediately if recovers to profit
    - During hold: sell after 24h if signal present
    - Stop loss (10%) always active

    If use_safety_rules=False (old rules):
    - Simply sell on signal, regardless of profit/loss
    """
    cash = initial_capital
    shares = 0
    position_price = 0
    position_time = None
    trades = []
    trade_logger = None
    in_hold_mode = False  # Track if we're in 24h hold mode
    hold_start_time = None

    if log_trades:
        try:
            trade_logger = get_trade_logger()
        except:
            pass

    for i in range(len(df)):
        price = df.iloc[i]['close']
        timestamp = df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now()
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # Buy
        if strategy.check_buy(df, i) and shares == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cash -= shares * price
                position_price = price
                position_time = timestamp
                in_hold_mode = False  # Reset hold mode on new position
                trades.append({'type': 'BUY', 'price': price, 'shares': shares, 'timestamp': timestamp})

                # Log backtest trade
                if trade_logger:
                    try:
                        reason = TradeReason(
                            primary_signal=strategy.name,
                            signal_value=0,
                            threshold=0,
                            direction='buy_signal',
                            explanation=f"BACKTEST BUY: {strategy.buy_rule}"
                        )
                        trade_logger.log_trade(
                            symbol=symbol,
                            action='BUY',
                            quantity=shares,
                            price=price,
                            strategy_name=strategy.name,
                            strategy_params={'buy_rule': strategy.buy_rule, 'sell_rule': strategy.sell_rule},
                            reason=reason,
                            mode='backtest',
                            portfolio_value_before=cash + shares * price,
                            timestamp=timestamp if isinstance(timestamp, datetime) else None
                        )
                    except:
                        pass

        # Sell logic
        elif shares > 0:
            pnl_pct = ((price - position_price) / position_price) * 100
            should_sell = False
            sell_reason = ""
            hours_held = 0  # Initialize for logging

            if not use_safety_rules:
                # OLD RULES: Simply sell on signal
                if strategy.check_sell(df, i):
                    should_sell = True
                    sell_reason = f"SIGNAL: {pnl_pct:+.1f}%"
            else:
                # NEW RULES: Safety rules with hold mode

                # Calculate hours held (for hold mode timing)
                hours_held = 0
                if hold_start_time:
                    try:
                        ts = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp
                        ht = hold_start_time.replace(tzinfo=None) if hasattr(hold_start_time, 'tzinfo') and hold_start_time.tzinfo else hold_start_time
                        hours_held = (ts - ht).total_seconds() / 3600
                    except:
                        hours_held = 24

                # Stop loss always active
                if pnl_pct <= -stop_loss_pct * 100:
                    should_sell = True
                    sell_reason = f"STOP-LOSS: {pnl_pct:.1f}%"

                # If in hold mode
                elif in_hold_mode:
                    if pnl_pct > 0:
                        # Recovered to profitable during hold - sell immediately
                        should_sell = True
                        sell_reason = f"RECOVERED: +{pnl_pct:.1f}%"
                    elif hours_held >= min_hold_hours and strategy.check_sell(df, i):
                        # 24h passed and sell signal - sell at loss
                        should_sell = True
                        sell_reason = f"HOLD EXPIRED: {pnl_pct:.1f}% after {hours_held:.0f}h"

                # Normal trading (not in hold mode)
                else:
                    if strategy.check_sell(df, i):
                        if pnl_pct > 0:
                            # Profitable + sell signal = sell normally
                            should_sell = True
                            sell_reason = f"PROFIT: +{pnl_pct:.1f}%"
                        else:
                            # At a loss + sell signal = ENTER hold mode
                            in_hold_mode = True
                            hold_start_time = timestamp
            # end safety rules check

            if should_sell:
                proceeds = shares * price
                pnl = proceeds - (shares * position_price)
                cash += proceeds
                trades.append({
                    'type': 'SELL',
                    'price': price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'shares': shares,
                    'timestamp': timestamp,
                    'reason': sell_reason,
                    'hours_held': hours_held
                })

                # Log backtest trade
                if trade_logger:
                    try:
                        reason = TradeReason(
                            primary_signal=strategy.name,
                            signal_value=0,
                            threshold=0,
                            direction='sell_signal',
                            explanation=f"BACKTEST SELL: {sell_reason}"
                        )
                        trade_logger.log_trade(
                            symbol=symbol,
                            action='SELL',
                            quantity=shares,
                            price=price,
                            strategy_name=strategy.name,
                            strategy_params={'buy_rule': strategy.buy_rule, 'sell_rule': strategy.sell_rule},
                            reason=reason,
                            mode='backtest',
                            portfolio_value_before=cash,
                            realized_pnl=pnl,
                            timestamp=timestamp if isinstance(timestamp, datetime) else None
                        )
                    except:
                        pass

                shares = 0
                position_time = None
                in_hold_mode = False  # Reset hold mode after selling

    # Liquidate at end
    if shares > 0:
        final_price = df.iloc[-1]['close']
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        pnl_pct = ((final_price - position_price) / position_price) * 100
        cash += proceeds
        trades.append({'type': 'SELL', 'price': final_price, 'pnl': pnl, 'pnl_pct': pnl_pct, 'reason': 'END_OF_BACKTEST'})

    sell_trades = [t for t in trades if t['type'] == 'SELL']
    winners = [t for t in sell_trades if t.get('pnl', 0) > 0]
    total_pnl = sum(t.get('pnl', 0) for t in sell_trades)

    return {
        'name': strategy.name,
        'trades': len(sell_trades),
        'winners': len(winners),
        'win_rate': len(winners) / len(sell_trades) * 100 if sell_trades else 0,
        'return_pct': (cash - initial_capital) / initial_capital * 100
    }


def generate_decision_explanation(
    decision: str,
    strategy: Strategy,
    row: pd.Series,
    prev_row: pd.Series = None,
    position_info: dict = None
) -> str:
    """
    Generate a layman's terms explanation for why a trading decision was made.

    Args:
        decision: 'BUY', 'SELL', or 'HOLD'
        strategy: The strategy being used
        row: Current bar data with indicator values
        prev_row: Previous bar data (for comparisons)
        position_info: Info about current position if held

    Returns:
        Human-readable explanation of the decision
    """
    price = row['close']
    rsi = row.get('rsi', 50)
    macd_diff = row.get('macd_diff', 0)
    bb_position = row.get('bb_position', 0.5)
    volume_ratio = row.get('volume_ratio', 1.0)
    sma20 = row.get('sma20', price)

    # Price change from previous bar
    price_change_pct = 0
    if prev_row is not None:
        prev_price = prev_row['close']
        price_change_pct = ((price - prev_price) / prev_price) * 100

    strategy_name = strategy.name.lower()

    if decision == 'BUY':
        # RSI-based strategies
        if 'rsi' in strategy_name:
            if rsi < 30:
                return f"BUY: The stock appears oversold. RSI is at {rsi:.1f}, which is below 30 - historically this means the stock has dropped too much too fast and may bounce back. Think of it like a rubber band stretched too far."
            elif rsi < 40:
                return f"BUY: The stock is showing signs of being undervalued. RSI at {rsi:.1f} suggests selling pressure is easing and buyers may step in soon."
            elif rsi < 45:
                return f"BUY: RSI at {rsi:.1f} indicates the stock has momentum shifting. This strategy catches early buy signals before the crowd."

        # MACD strategies
        elif 'macd' in strategy_name:
            if 'cross' in strategy_name:
                return f"BUY: The MACD line just crossed above its signal line. This is like two moving averages giving a 'thumbs up' - short-term momentum is now stronger than the longer-term trend, suggesting upward movement."
            else:
                return f"BUY: MACD difference is positive ({macd_diff:.4f}), meaning short-term momentum exceeds long-term. The stock has positive momentum."

        # Bollinger strategies
        elif 'bollinger' in strategy_name:
            pct_in_band = bb_position * 100
            return f"BUY: Price is at the bottom {pct_in_band:.0f}% of its normal trading range (Bollinger Bands). Stocks typically bounce back toward the middle. It's like buying when something is 'on sale' relative to its recent prices."

        # Volume spike strategy
        elif 'volume' in strategy_name:
            return f"BUY: High trading volume ({volume_ratio:.1f}x normal) combined with a price drop ({price_change_pct:.2f}%). Heavy selling often exhausts itself, creating buying opportunities. Think of it as 'panic selling' that may be overdone."

        # Mean reversion strategies
        elif 'mean' in strategy_name:
            pct_below_sma = ((sma20 - price) / sma20) * 100
            return f"BUY: Price is {pct_below_sma:.1f}% below the 20-day average. Stocks tend to return to their average price over time. This is like buying something below its 'fair value'."

        return f"BUY: Strategy '{strategy.name}' triggered a buy signal. {strategy.buy_rule}."

    elif decision == 'SELL':
        pnl_pct = 0
        pnl_text = ""
        if position_info:
            entry_price = position_info.get('entry_price', price)
            pnl_pct = ((price - entry_price) / entry_price) * 100
            pnl_text = f" This trade {'made' if pnl_pct > 0 else 'lost'} {abs(pnl_pct):.1f}%."

        # RSI strategies
        if 'rsi' in strategy_name:
            if rsi > 70:
                return f"SELL: The stock appears overbought. RSI at {rsi:.1f} (above 70) means it has risen too quickly and may be due for a pullback. Taking profits before potential decline.{pnl_text}"
            elif rsi > 60:
                return f"SELL: RSI at {rsi:.1f} shows the stock is getting expensive. Exiting before the momentum fades.{pnl_text}"
            elif rsi > 55:
                return f"SELL: RSI at {rsi:.1f} hit the target threshold. This strategy takes profits quickly to lock in gains.{pnl_text}"

        # MACD strategies
        elif 'macd' in strategy_name:
            if 'cross' in strategy_name:
                return f"SELL: MACD crossed below its signal line - the 'thumbs up' turned into a 'thumbs down'. Short-term momentum is now weaker than the longer-term trend, suggesting the upward move is ending.{pnl_text}"
            else:
                return f"SELL: MACD difference turned negative ({macd_diff:.4f}), indicating momentum has shifted downward. Time to exit.{pnl_text}"

        # Bollinger strategies
        elif 'bollinger' in strategy_name:
            pct_in_band = bb_position * 100
            if 'mean' in strategy_name:
                return f"SELL: Price returned to the middle of its trading range ({pct_in_band:.0f}%). The 'discount' is gone - taking profits at fair value.{pnl_text}"
            else:
                return f"SELL: Price reached the upper {100-pct_in_band:.0f}% of its trading range. The stock may be stretched too high and could pull back.{pnl_text}"

        # Volume spike strategy
        elif 'volume' in strategy_name:
            return f"SELL: High volume ({volume_ratio:.1f}x normal) with price rising ({price_change_pct:.2f}%). Heavy buying often signals a peak. Selling into strength.{pnl_text}"

        # Mean reversion strategies
        elif 'mean' in strategy_name:
            return f"SELL: Price returned to or above the 20-day average (${sma20:.2f}). The 'undervalued' condition is resolved.{pnl_text}"

        return f"SELL: Strategy '{strategy.name}' triggered a sell signal. {strategy.sell_rule}.{pnl_text}"

    else:  # HOLD
        if position_info:
            entry_price = position_info.get('entry_price', price)
            pnl_pct = ((price - entry_price) / entry_price) * 100
            holding_time = position_info.get('holding_time', 'unknown')

            # Explain why we're holding with position
            if 'rsi' in strategy_name:
                return f"HOLD (in position): RSI at {rsi:.1f} - sell threshold not reached yet. Current P&L: {pnl_pct:+.1f}%. Waiting for RSI to signal exit ({strategy.sell_rule})."
            elif 'macd' in strategy_name:
                return f"HOLD (in position): MACD not signaling exit yet. MACD diff: {macd_diff:.4f}. Current P&L: {pnl_pct:+.1f}%. Still in the trade."
            elif 'bollinger' in strategy_name:
                return f"HOLD (in position): Price at {bb_position*100:.0f}% of Bollinger range. Current P&L: {pnl_pct:+.1f}%. Waiting for exit signal."
            elif 'volume' in strategy_name:
                return f"HOLD (in position): No high-volume price rise detected. Current P&L: {pnl_pct:+.1f}%. Waiting for selling opportunity."
            elif 'mean' in strategy_name:
                return f"HOLD (in position): Price still below SMA20 (${sma20:.2f}). Current P&L: {pnl_pct:+.1f}%. Waiting for price to reach average."

            return f"HOLD (in position): No sell signal triggered. {strategy.sell_rule}. Current P&L: {pnl_pct:+.1f}%."
        else:
            # Explain why we're not buying
            if 'rsi' in strategy_name:
                status = "overbought" if rsi > 70 else "neutral" if rsi > 30 else "oversold"
                return f"HOLD (no position): RSI at {rsi:.1f} ({status}) - buy threshold not met. Need {strategy.buy_rule} to buy."
            elif 'macd' in strategy_name:
                status = "bullish" if macd_diff > 0 else "bearish"
                return f"HOLD (no position): MACD is {status} (diff: {macd_diff:.4f}). Waiting for crossover signal to buy."
            elif 'bollinger' in strategy_name:
                zone = "upper" if bb_position > 0.7 else "middle" if bb_position > 0.3 else "lower"
                return f"HOLD (no position): Price in {zone} zone of Bollinger Bands ({bb_position*100:.0f}%). Need price in lower zone to buy."
            elif 'volume' in strategy_name:
                return f"HOLD (no position): Volume ratio {volume_ratio:.1f}x and price change {price_change_pct:+.2f}%. No buy signal triggered."
            elif 'mean' in strategy_name:
                pct_from_sma = ((price - sma20) / sma20) * 100
                return f"HOLD (no position): Price is {pct_from_sma:+.1f}% from SMA20. Need {strategy.buy_rule} to buy."

            return f"HOLD (no position): No buy signal triggered. Need: {strategy.buy_rule}."


def run_detailed_backtest(
    symbol: str,
    strategy_name: str = None,
    lookback_days: int = 90,
    initial_capital: float = 10000,
    min_hold_hours: int = 24,
    stop_loss_pct: float = 0.10,
    use_safety_rules: bool = True
) -> dict:
    """
    Run a detailed backtest that logs every bar's decision with explanations.

    This is designed for the Reports page to show the user exactly what the bot
    would have done at each point in time and WHY.

    Args:
        symbol: Stock symbol to backtest
        strategy_name: Specific strategy to use (if None, uses best strategy)
        lookback_days: Days of historical data to test
        initial_capital: Starting capital

    Returns:
        Dict containing:
        - summary: Overall results
        - decision_log: List of every bar's decision with explanation
        - trades: List of actual trades executed
    """
    # Fetch historical data
    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start_date,
        end=end_date
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if df.empty:
        return {'error': f'No data available for {symbol}'}

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level='symbol')

    # Calculate all indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'], df['macd_diff'] = calculate_macd(df['close'])
    df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
    df['volume_ratio'] = calculate_volume_ratio(df['volume'])
    df['sma20'] = calculate_sma(df['close'], 20)
    df = df.dropna()

    if len(df) < 30:
        return {'error': f'Not enough data for {symbol} (need at least 30 bars)'}

    # Select strategy
    if strategy_name and strategy_name in STRATEGIES:
        strategy = STRATEGIES[strategy_name]()
    else:
        # Auto-select best strategy
        best_result = select_best_strategy(symbol, lookback_days)
        strategy_name = best_result['best_strategy']
        strategy = STRATEGIES[strategy_name]()

    # Run detailed backtest
    cash = initial_capital
    shares = 0
    position_price = 0
    position_time = None
    in_hold_mode = False  # Track if we're in 24h hold mode
    hold_start_time = None

    decision_log = []
    trades = []
    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        timestamp = df.index[i]
        if hasattr(timestamp, 'to_pydatetime'):
            timestamp = timestamp.to_pydatetime()

        prev_row = df.iloc[i-1] if i > 0 else None
        price = row['close']

        # Calculate current portfolio value
        portfolio_value = cash + (shares * price)
        equity_curve.append({
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'value': portfolio_value
        })

        # Build position info for explanation
        position_info = None
        if shares > 0:
            holding_time = (timestamp - position_time).total_seconds() / 3600 if position_time else 0
            position_info = {
                'entry_price': position_price,
                'shares': shares,
                'holding_time': f"{holding_time:.1f} hours",
                'current_pnl': (price - position_price) * shares,
                'current_pnl_pct': ((price - position_price) / position_price) * 100 if position_price else 0
            }

        # Determine decision
        buy_signal = strategy.check_buy(df, i)
        sell_signal = strategy.check_sell(df, i)

        decision = 'HOLD'
        trade_info = None

        # Process BUY
        if buy_signal and shares == 0:
            decision = 'BUY'
            shares_to_buy = int(cash * 0.95 / price)
            if shares_to_buy > 0:
                shares = shares_to_buy
                cash -= shares * price
                position_price = price
                position_time = timestamp
                in_hold_mode = False  # Reset hold mode on new position
                hold_start_time = None

                trade_info = {
                    'action': 'BUY',
                    'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    'price': price,
                    'shares': shares,
                    'value': shares * price,
                    'reason': strategy.buy_rule
                }
                trades.append(trade_info)

        # Process SELL - with safety rules
        elif shares > 0:
            pnl_pct = ((price - position_price) / position_price) * 100
            should_sell = False
            sell_reason = ""
            hours_held = 0

            if not use_safety_rules:
                # OLD RULES: Simply sell on signal
                if sell_signal:
                    should_sell = True
                    sell_reason = f"SIGNAL: {pnl_pct:+.1f}%"
            else:
                # SAFETY RULES: Hold mode logic

                # Calculate hours in hold mode
                if hold_start_time:
                    try:
                        ts = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp
                        ht = hold_start_time.replace(tzinfo=None) if hasattr(hold_start_time, 'tzinfo') and hold_start_time.tzinfo else hold_start_time
                        hours_held = (ts - ht).total_seconds() / 3600
                    except:
                        hours_held = min_hold_hours  # Assume enough time passed on error

                # Stop loss always active
                if pnl_pct <= -stop_loss_pct * 100:
                    should_sell = True
                    sell_reason = f"STOP-LOSS: {pnl_pct:.1f}%"

                # If in hold mode
                elif in_hold_mode:
                    if pnl_pct > 0:
                        # Recovered to profitable during hold - sell immediately
                        should_sell = True
                        sell_reason = f"RECOVERED: +{pnl_pct:.1f}% after {hours_held:.0f}h hold"
                    elif hours_held >= min_hold_hours and sell_signal:
                        # 24h passed and sell signal - sell at loss
                        should_sell = True
                        sell_reason = f"HOLD EXPIRED: {pnl_pct:.1f}% after {hours_held:.0f}h"

                # Normal trading (not in hold mode)
                else:
                    if sell_signal:
                        if pnl_pct > 0:
                            # Profitable + sell signal = sell normally
                            should_sell = True
                            sell_reason = f"PROFIT: +{pnl_pct:.1f}%"
                        else:
                            # At a loss + sell signal = ENTER hold mode (don't sell yet)
                            in_hold_mode = True
                            hold_start_time = timestamp
                            decision = 'HOLD'  # Override to HOLD

            if should_sell:
                decision = 'SELL'
                proceeds = shares * price
                pnl = proceeds - (shares * position_price)

                trade_info = {
                    'action': 'SELL',
                    'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    'price': price,
                    'shares': shares,
                    'value': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': sell_reason,
                    'entry_price': position_price,
                    'holding_time': position_info['holding_time'] if position_info else 'unknown'
                }
                trades.append(trade_info)

                cash += proceeds
                shares = 0
                position_price = 0
                position_time = None
                in_hold_mode = False
                hold_start_time = None

        # Generate explanation
        explanation = generate_decision_explanation(
            decision, strategy, row, prev_row, position_info
        )

        # Create decision log entry
        log_entry = {
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'decision': decision,
            'price': round(price, 2),
            'explanation': explanation,
            'indicators': {
                'rsi': round(float(row['rsi']), 2),
                'macd_diff': round(float(row['macd_diff']), 4),
                'bb_position': round(float(row['bb_position']), 2),
                'volume_ratio': round(float(row['volume_ratio']), 2),
                'sma20': round(float(row['sma20']), 2)
            },
            'portfolio': {
                'cash': round(cash, 2),
                'shares': shares,
                'position_value': round(shares * price, 2),
                'total_value': round(cash + shares * price, 2)
            }
        }

        if trade_info:
            log_entry['trade'] = trade_info

        if position_info:
            log_entry['position'] = {
                'entry_price': round(position_info['entry_price'], 2),
                'current_pnl': round(position_info['current_pnl'], 2),
                'current_pnl_pct': round(position_info['current_pnl_pct'], 2),
                'holding_time': position_info['holding_time']
            }

        decision_log.append(log_entry)

    # Close any remaining position at end
    if shares > 0:
        final_price = df.iloc[-1]['close']
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        pnl_pct = ((final_price - position_price) / position_price) * 100
        cash += proceeds

        trades.append({
            'action': 'SELL (END)',
            'timestamp': decision_log[-1]['timestamp'],
            'price': final_price,
            'shares': shares,
            'value': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': 'End of backtest period - forced liquidation',
            'entry_price': position_price
        })

    # Calculate summary stats
    sell_trades = [t for t in trades if 'pnl' in t]
    winning_trades = [t for t in sell_trades if t['pnl'] > 0]
    losing_trades = [t for t in sell_trades if t['pnl'] < 0]

    total_pnl = sum(t['pnl'] for t in sell_trades)
    total_wins = sum(t['pnl'] for t in winning_trades)
    total_losses = abs(sum(t['pnl'] for t in losing_trades))

    # Decision counts
    buy_count = sum(1 for d in decision_log if d['decision'] == 'BUY')
    sell_count = sum(1 for d in decision_log if d['decision'] == 'SELL')
    hold_count = sum(1 for d in decision_log if d['decision'] == 'HOLD')

    summary = {
        'symbol': symbol,
        'strategy': strategy.name,
        'strategy_buy_rule': strategy.buy_rule,
        'strategy_sell_rule': strategy.sell_rule,
        'use_safety_rules': use_safety_rules,
        'safety_settings': {
            'min_hold_hours': min_hold_hours,
            'stop_loss_pct': stop_loss_pct
        },
        'period': {
            'start': decision_log[0]['timestamp'] if decision_log else None,
            'end': decision_log[-1]['timestamp'] if decision_log else None,
            'bars_analyzed': len(decision_log)
        },
        'capital': {
            'initial': initial_capital,
            'final': round(cash, 2),
            'return_pct': round(((cash - initial_capital) / initial_capital) * 100, 2),
            'total_pnl': round(total_pnl, 2)
        },
        'trades': {
            'total': len(sell_trades),
            'winners': len(winning_trades),
            'losers': len(losing_trades),
            'win_rate': round(len(winning_trades) / len(sell_trades) * 100, 1) if sell_trades else 0,
            'profit_factor': round(total_wins / total_losses, 2) if total_losses > 0 else float('inf'),
            'avg_win': round(total_wins / len(winning_trades), 2) if winning_trades else 0,
            'avg_loss': round(total_losses / len(losing_trades), 2) if losing_trades else 0
        },
        'decisions': {
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'action_rate': round((buy_count + sell_count) / len(decision_log) * 100, 1) if decision_log else 0
        }
    }

    return {
        'success': True,
        'summary': summary,
        'decision_log': decision_log,
        'trades': trades,
        'equity_curve': equity_curve
    }


def select_best_strategy(symbol: str, lookback_days: int = 90) -> dict:
    """
    Test all strategies on a stock and return the best one.
    Tests each strategy with both safety rules ON and OFF to find optimal config.
    """
    # Fetch data
    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start_date,
        end=end_date
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level='symbol')

    # Calculate all indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'], df['macd_diff'] = calculate_macd(df['close'])
    df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
    df['volume_ratio'] = calculate_volume_ratio(df['volume'])
    df['sma20'] = calculate_sma(df['close'], 20)
    df = df.dropna()

    # Test all strategies with BOTH safety rules ON and OFF
    results = []
    for name, strategy_class in STRATEGIES.items():
        strategy = strategy_class()

        # Test with safety rules ON
        result_safe = backtest_strategy(df, strategy, use_safety_rules=True)
        result_safe['use_safety_rules'] = True
        result_safe['config_name'] = f"{result_safe['name']} (safe)"
        results.append(result_safe)

        # Test with safety rules OFF (old rules)
        result_old = backtest_strategy(df, strategy, use_safety_rules=False)
        result_old['use_safety_rules'] = False
        result_old['config_name'] = f"{result_old['name']} (aggressive)"
        results.append(result_old)

    # Sort by return
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    best = results[0]

    return {
        'symbol': symbol,
        'best_strategy': best['name'],
        'best_return': best['return_pct'],
        'best_win_rate': best['win_rate'],
        'best_trades': best['trades'],
        'use_safety_rules': best['use_safety_rules'],
        'config_name': best['config_name'],
        'all_results': results
    }


# =============================================================================
# SIMPLE TRADER
# =============================================================================

@dataclass
class Position:
    """Current position in a stock."""
    symbol: str
    shares: int
    entry_price: float
    entry_time: datetime
    strategy_name: str


@dataclass
class StockConfig:
    """Configuration for trading a single stock."""
    symbol: str
    strategy: Strategy
    last_calibration: datetime = None
    use_safety_rules: bool = True  # False = old rules (better for volatile stocks like TSLA)


class SimpleTrader:
    """
    Simple trading bot that:
    1. Calibrates best strategy per stock
    2. Monitors indicators
    3. Buys/sells when thresholds are hit
    """

    def __init__(
        self,
        symbols: List[str],
        paper: bool = True,
        calibration_days: int = 90,
        recalibrate_hours: int = 24,
        position_size_pct: float = 0.15,  # 15% of portfolio per position
        max_positions: int = 15,
        min_hold_hours: int = 24,  # Minimum hold time before selling at a loss
        stop_loss_pct: float = 0.10  # Emergency stop-loss (10% = sell if down 10%)
    ):
        self.symbols = symbols
        self.paper = paper
        self.calibration_days = calibration_days
        self.recalibrate_hours = recalibrate_hours
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.min_hold_hours = min_hold_hours
        self.stop_loss_pct = stop_loss_pct

        # Alpaca clients
        self.trading_client = TradingClient(API_KEY, API_SECRET, paper=paper)
        self.data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

        # Stock configs (strategy per stock)
        self.stock_configs: Dict[str, StockConfig] = {}

        # Current positions
        self.positions: Dict[str, Position] = {}

        # Track positions in hold mode (symbol -> datetime when hold started)
        self.hold_mode_positions: Dict[str, datetime] = {}

        # State
        self.running = False
        self._stop_event = threading.Event()

        # Stats
        self.stats = {
            'trades_executed': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'consecutive_losses': 0,
            'consecutive_wins': 0
        }

        # Risk Manager - controls position sizing and daily loss limits
        self.risk_manager = RiskManager(initial_capital=100000.0)

        # Circuit breaker settings
        self.max_consecutive_losses = 5  # Stop trading after 5 consecutive losses
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None

        # Data directory for saving state
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        print(f"SimpleTrader initialized for {len(symbols)} symbols ({'PAPER' if paper else 'LIVE'})")
        print(f"Risk controls: Max daily loss {self.risk_manager.max_daily_loss*100:.0f}%, Max position {self.risk_manager.max_position_size*100:.0f}%")

    def calibrate_all(self):
        """Calibrate best strategy for all stocks."""
        print("\n" + "="*60)
        print("CALIBRATING STRATEGIES")
        print("="*60 + "\n")

        for symbol in self.symbols:
            try:
                result = select_best_strategy(symbol, self.calibration_days)

                strategy_class = STRATEGIES.get(result['best_strategy'])
                if strategy_class:
                    use_safety = result.get('use_safety_rules', True)
                    self.stock_configs[symbol] = StockConfig(
                        symbol=symbol,
                        strategy=strategy_class(),
                        last_calibration=datetime.now(),
                        use_safety_rules=use_safety
                    )

                    rules_label = "safe" if use_safety else "aggressive"
                    print(f"{symbol}: {result['best_strategy']} ({rules_label}) "
                          f"(+{result['best_return']:.1f}%, {result['best_win_rate']:.0f}% win rate)")

            except Exception as e:
                print(f"{symbol}: Error - {e}")

        print("\n" + "="*60)
        self._save_state()

    def calibrate_if_needed(self, symbol: str):
        """Recalibrate a stock if enough time has passed."""
        config = self.stock_configs.get(symbol)

        if config is None or config.last_calibration is None:
            needs_calibration = True
        else:
            hours_since = (datetime.now() - config.last_calibration).total_seconds() / 3600
            needs_calibration = hours_since >= self.recalibrate_hours

        if needs_calibration:
            try:
                result = select_best_strategy(symbol, self.calibration_days)
                strategy_class = STRATEGIES.get(result['best_strategy'])
                if strategy_class:
                    use_safety = result.get('use_safety_rules', True)
                    self.stock_configs[symbol] = StockConfig(
                        symbol=symbol,
                        strategy=strategy_class(),
                        last_calibration=datetime.now(),
                        use_safety_rules=use_safety
                    )
                    rules_label = "safe" if use_safety else "aggressive"
                    print(f"[RECALIBRATED] {symbol}: Now using {result['best_strategy']} ({rules_label})")
            except Exception as e:
                print(f"[ERROR] Calibration failed for {symbol}: {e}")

    def get_account(self) -> dict:
        """Get account info."""
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power)
            }
        except Exception as e:
            print(f"Error getting account: {e}")
            return {'equity': 0, 'cash': 0, 'buying_power': 0}

    def get_current_positions(self) -> Dict[str, dict]:
        """Get current positions from Alpaca."""
        try:
            positions = self.trading_client.get_all_positions()
            return {
                pos.symbol: {
                    'shares': int(float(pos.qty)),
                    'avg_cost': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pnl': float(pos.unrealized_pl)
                }
                for pos in positions
            }
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {}

    def add_symbol(self, symbol: str, calibrate: bool = True) -> Dict[str, Any]:
        """
        Dynamically add a symbol to the trader's watchlist.

        Args:
            symbol: Stock ticker to add
            calibrate: If True, immediately find best strategy

        Returns:
            dict with status and calibration results
        """
        symbol = symbol.upper()

        if symbol in self.symbols:
            return {'success': False, 'message': f'{symbol} already tracked'}

        # Check watchlist capacity (allow 2x max_positions for watchlist)
        max_watchlist = self.max_positions * 2
        if len(self.symbols) >= max_watchlist:
            return {'success': False, 'message': f'Max watchlist size ({max_watchlist}) reached'}

        self.symbols.append(symbol)
        print(f"[SCANNER] Added {symbol} to watchlist")

        strategy_name = None
        if calibrate:
            try:
                self.calibrate_if_needed(symbol)
                config = self.stock_configs.get(symbol)
                if config:
                    strategy_name = config.strategy.name
                    print(f"[SCANNER] {symbol} calibrated: {strategy_name}")
            except Exception as e:
                print(f"[WARNING] Calibration failed for {symbol}: {e}")

        self._save_state()

        return {
            'success': True,
            'symbol': symbol,
            'strategy': strategy_name,
            'message': f'Added {symbol} to watchlist'
        }

    def remove_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Remove a symbol from watchlist.

        Note: Won't remove if there's an open position.

        Args:
            symbol: Stock ticker to remove

        Returns:
            dict with status
        """
        symbol = symbol.upper()

        if symbol not in self.symbols:
            return {'success': False, 'message': f'{symbol} not in watchlist'}

        # Check if we have an open position
        current_positions = self.get_current_positions()
        if symbol in current_positions:
            return {'success': False, 'message': f'Cannot remove {symbol} - has open position'}

        self.symbols.remove(symbol)
        if symbol in self.stock_configs:
            del self.stock_configs[symbol]

        self._save_state()
        print(f"[SCANNER] Removed {symbol} from watchlist")

        return {'success': True, 'symbol': symbol, 'message': f'Removed {symbol} from watchlist'}

    def get_watchlist_capacity(self) -> Dict[str, int]:
        """Get current watchlist capacity info."""
        max_watchlist = self.max_positions * 2
        return {
            'current': len(self.symbols),
            'max': max_watchlist,
            'available': max_watchlist - len(self.symbols)
        }

    def fetch_latest_data(self, symbol: str, bars: int = 50) -> Optional[pd.DataFrame]:
        """Fetch latest bar data for a symbol."""
        try:
            # Use a date range to get recent data (works even when market is closed)
            end = datetime.now()
            start = end - timedelta(days=7)  # Get last 7 days of hourly data

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )

            result = self.data_client.get_stock_bars(request)
            df = result.df

            if df.empty:
                print(f"No data returned for {symbol}")
                return None

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')

            if df.empty or 'close' not in df.columns:
                print(f"No valid data for {symbol}")
                return None

            # Calculate indicators
            df['rsi'] = calculate_rsi(df['close'])
            df['macd'], df['macd_signal'], df['macd_diff'] = calculate_macd(df['close'])
            df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
            df['volume_ratio'] = calculate_volume_ratio(df['volume'])
            df['sma20'] = calculate_sma(df['close'], 20)

            return df.dropna()

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def execute_buy(self, symbol: str, strategy_name: str) -> bool:
        """Execute a buy order."""
        try:
            # Check circuit breaker
            if self.circuit_breaker_active:
                print(f"[BLOCKED] Circuit breaker active: {self.circuit_breaker_reason}")
                return False

            account = self.get_account()
            equity = account['equity']

            # Update risk manager with current capital
            self.risk_manager.daily_start_capital = equity

            # Check daily loss limit via risk manager
            if not self.risk_manager.check_daily_loss_limit():
                print(f"[BLOCKED] Daily loss limit reached - no new trades allowed")
                self.circuit_breaker_active = True
                self.circuit_breaker_reason = "Daily loss limit reached"
                return False

            # Check if we have room for more positions
            current_positions = self.get_current_positions()
            if len(current_positions) >= self.max_positions:
                print(f"[SKIP] Max positions ({self.max_positions}) reached")
                return False

            # Already have position?
            if symbol in current_positions:
                print(f"[SKIP] Already have position in {symbol}")
                return False

            # Calculate position size using risk manager
            position_value = self.risk_manager.get_max_trade_value(equity)

            # Get current price
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(quote_request)

            if symbol not in quotes:
                return False

            price = quotes[symbol].ask_price or quotes[symbol].bid_price
            shares = int(position_value / price)

            if shares <= 0:
                return False

            # Execute order
            order = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            result = self.trading_client.submit_order(order)

            # Track position
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                entry_price=price,
                entry_time=datetime.now(),
                strategy_name=strategy_name
            )

            print(f"[BUY] {shares} {symbol} @ ${price:.2f} using {strategy_name}")
            self.stats['trades_executed'] += 1
            self._save_state()

            # Log the trade
            try:
                trade_logger = get_trade_logger()
                config = self.stock_configs.get(symbol)
                strategy = config.strategy if config else None

                reason = TradeReason(
                    primary_signal=strategy_name,
                    signal_value=0,
                    threshold=0,
                    direction='buy_signal',
                    explanation=f"BUY: {strategy.buy_rule if strategy else 'Manual'}"
                )

                trade_logger.log_trade(
                    symbol=symbol,
                    action='BUY',
                    quantity=shares,
                    price=price,
                    strategy_name=strategy_name,
                    strategy_params={'buy_rule': strategy.buy_rule if strategy else '', 'sell_rule': strategy.sell_rule if strategy else ''},
                    reason=reason,
                    mode='paper' if self.paper else 'live',
                    portfolio_value_before=equity
                )
            except Exception as log_err:
                print(f"[WARN] Trade logging failed: {log_err}")

            return True

        except Exception as e:
            print(f"[ERROR] Buy failed for {symbol}: {e}")
            return False

    def execute_sell(self, symbol: str, sell_reason: str = None) -> bool:
        """Execute a sell order.

        Args:
            symbol: Stock symbol to sell
            sell_reason: The reason for selling (e.g., "STOP-LOSS: -11.9%", "PROFIT: +5.2%")
        """
        try:
            current_positions = self.get_current_positions()

            if symbol not in current_positions:
                print(f"[SKIP] No position in {symbol} to sell")
                return False

            pos = current_positions[symbol]
            shares = pos['shares']

            # Execute order
            order = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )

            result = self.trading_client.submit_order(order)

            # Calculate P&L
            pnl = pos['unrealized_pnl']
            self.stats['total_pnl'] += pnl

            # Update risk manager with trade result
            self.risk_manager.record_trade_result(pnl)

            if pnl > 0:
                self.stats['winning_trades'] += 1
                self.stats['consecutive_wins'] += 1
                self.stats['consecutive_losses'] = 0
                result_str = f"WIN +${pnl:.2f}"
            else:
                self.stats['losing_trades'] += 1
                self.stats['consecutive_losses'] += 1
                self.stats['consecutive_wins'] = 0
                result_str = f"LOSS ${pnl:.2f}"

                # Check for circuit breaker - consecutive losses
                if self.stats['consecutive_losses'] >= self.max_consecutive_losses:
                    self.circuit_breaker_active = True
                    self.circuit_breaker_reason = f"{self.stats['consecutive_losses']} consecutive losses"
                    print(f"[CIRCUIT BREAKER] Activated: {self.circuit_breaker_reason}")

            # Remove from tracking
            if symbol in self.positions:
                del self.positions[symbol]

            print(f"[SELL] {shares} {symbol} @ ${pos['current_price']:.2f} - {result_str}")
            self.stats['trades_executed'] += 1
            self._save_state()

            # Log the trade
            try:
                trade_logger = get_trade_logger()
                config = self.stock_configs.get(symbol)
                strategy = config.strategy if config else None
                strategy_name = strategy.name if strategy else 'Unknown'

                # Use provided sell_reason, or fall back to strategy's sell rule
                if sell_reason:
                    explanation = f"SELL: {sell_reason} - {result_str}"
                else:
                    explanation = f"SELL: {strategy.sell_rule if strategy else 'Manual'} - {result_str}"

                reason = TradeReason(
                    primary_signal=strategy_name,
                    signal_value=0,
                    threshold=0,
                    direction='sell_signal',
                    explanation=explanation
                )

                account = self.get_account()
                trade_logger.log_trade(
                    symbol=symbol,
                    action='SELL',
                    quantity=shares,
                    price=pos['current_price'],
                    strategy_name=strategy_name,
                    strategy_params={'buy_rule': strategy.buy_rule if strategy else '', 'sell_rule': strategy.sell_rule if strategy else ''},
                    reason=reason,
                    mode='paper' if self.paper else 'live',
                    portfolio_value_before=account['equity'],
                    realized_pnl=pnl
                )
            except Exception as log_err:
                print(f"[WARN] Trade logging failed: {log_err}")

            return True

        except Exception as e:
            print(f"[ERROR] Sell failed for {symbol}: {e}")
            return False

    def check_signals(self):
        """Check all stocks for buy/sell signals."""
        current_positions = self.get_current_positions()

        for symbol in self.symbols:
            # Make sure we have a strategy
            self.calibrate_if_needed(symbol)

            config = self.stock_configs.get(symbol)
            if not config:
                continue

            # Get latest data
            df = self.fetch_latest_data(symbol)
            if df is None or len(df) < 2:
                continue

            strategy = config.strategy
            i = len(df) - 1  # Latest bar

            # Check for signals
            has_position = symbol in current_positions

            if not has_position:
                # Check for buy signal
                if strategy.check_buy(df, i):
                    print(f"\n[SIGNAL] {symbol} - BUY triggered by {strategy.name}")
                    self.execute_buy(symbol, strategy.name)
            else:
                # Check for sell signal - use per-stock rules
                # Fallback: if position exists in Alpaca but not tracked internally, sync it
                if symbol not in self.positions:
                    alpaca_pos = current_positions.get(symbol, {})
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        shares=int(alpaca_pos.get('shares', 0)),
                        entry_price=float(alpaca_pos.get('avg_cost', alpaca_pos.get('current_price', df.iloc[i]['close']))),
                        entry_time=datetime.now(),
                        strategy_name=strategy.name
                    )
                    print(f"[SYNC] Created tracking for existing position: {symbol}")

                position = self.positions[symbol]
                current_price = df.iloc[i]['close']
                entry_price = position.entry_price
                pnl_pct = ((current_price - entry_price) / entry_price) * 100

                # Get per-stock safety rules setting
                use_safety = config.use_safety_rules

                should_sell = False
                sell_reason = ""

                if not use_safety:
                    # AGGRESSIVE RULES: Simply sell on signal
                    if strategy.check_sell(df, i):
                        should_sell = True
                        sell_reason = f"SIGNAL: {pnl_pct:+.1f}%"
                else:
                    # SAFETY RULES: Hold mode logic
                    in_hold_mode = symbol in self.hold_mode_positions

                    # Calculate hours in hold mode (not hours held overall)
                    hours_in_hold = 0
                    if in_hold_mode:
                        hold_start = self.hold_mode_positions[symbol]
                        hours_in_hold = (datetime.now() - hold_start).total_seconds() / 3600

                    # Stop loss always active
                    if pnl_pct <= -self.stop_loss_pct * 100:
                        should_sell = True
                        sell_reason = f"STOP-LOSS: {pnl_pct:.1f}% exceeds -{self.stop_loss_pct*100:.0f}% limit"

                    # If in hold mode
                    elif in_hold_mode:
                        if pnl_pct > 0:
                            # Recovered to profitable during hold - sell immediately
                            should_sell = True
                            sell_reason = f"RECOVERED: +{pnl_pct:.1f}% after {hours_in_hold:.1f}h hold"
                        elif hours_in_hold >= self.min_hold_hours and strategy.check_sell(df, i):
                            # 24h passed and sell signal - sell at loss
                            should_sell = True
                            sell_reason = f"HOLD EXPIRED: {pnl_pct:.1f}% after {hours_in_hold:.1f}h"
                        else:
                            remaining = self.min_hold_hours - hours_in_hold
                            if strategy.check_sell(df, i):
                                print(f"[HOLD] {symbol} - At loss ({pnl_pct:.1f}%), waiting {remaining:.1f}h more")

                    # Normal trading (not in hold mode)
                    else:
                        if strategy.check_sell(df, i):
                            if pnl_pct > 0:
                                # Profitable + sell signal = sell normally
                                should_sell = True
                                sell_reason = f"PROFIT: +{pnl_pct:.1f}%"
                            else:
                                # At a loss + sell signal = ENTER hold mode
                                self.hold_mode_positions[symbol] = datetime.now()
                                print(f"[HOLD MODE] {symbol} - Entering 24h hold at loss ({pnl_pct:.1f}%)")

                if should_sell:
                    print(f"\n[SIGNAL] {symbol} - SELL: {sell_reason}")
                    self.execute_sell(symbol, sell_reason=sell_reason)
                    # Clear hold mode if we were in it
                    if symbol in self.hold_mode_positions:
                        del self.hold_mode_positions[symbol]

    def _is_market_hours(self) -> bool:
        """Check if market is open (Eastern Time)."""
        et = ZoneInfo('America/New_York')
        now = datetime.now(et)

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)

        return market_open <= now.time() <= market_close

    def _save_state(self):
        """Save current state to file."""
        state = {
            'positions': {
                s: {
                    'shares': p.shares,
                    'entry_price': p.entry_price,
                    'entry_time': p.entry_time.isoformat(),
                    'strategy_name': p.strategy_name
                }
                for s, p in self.positions.items()
            },
            'stock_configs': {
                s: {
                    'strategy': c.strategy.name,
                    'last_calibration': c.last_calibration.isoformat() if c.last_calibration else None,
                    'use_safety_rules': c.use_safety_rules
                }
                for s, c in self.stock_configs.items()
            },
            'stats': self.stats
        }

        with open(self.data_dir / "simple_trader_state.json", 'w') as f:
            json.dump(state, f, indent=2)

    def _sync_positions_from_alpaca(self):
        """Sync internal position tracking with actual Alpaca positions."""
        try:
            alpaca_positions = self.get_current_positions()
            synced_count = 0

            for symbol, pos_data in alpaca_positions.items():
                if symbol not in self.positions:
                    # Create position from Alpaca data
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        shares=int(pos_data.get('shares', 0)),
                        entry_price=float(pos_data.get('avg_cost', pos_data.get('current_price', 0))),
                        entry_time=datetime.now(),  # Unknown, use now
                        strategy_name=self.stock_configs.get(symbol, StockConfig(strategy=MomentumSimple())).strategy.name if symbol in self.stock_configs else "Unknown"
                    )
                    synced_count += 1

            if synced_count > 0:
                print(f"[SYNC] Synced {synced_count} positions from Alpaca")

        except Exception as e:
            print(f"[ERROR] Failed to sync positions from Alpaca: {e}")

    def _load_state(self):
        """Load state from file."""
        state_file = self.data_dir / "simple_trader_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Restore stock configs
            for symbol, config in state.get('stock_configs', {}).items():
                strategy_class = STRATEGIES.get(config['strategy'])
                if strategy_class:
                    self.stock_configs[symbol] = StockConfig(
                        symbol=symbol,
                        strategy=strategy_class(),
                        last_calibration=datetime.fromisoformat(config['last_calibration']) if config['last_calibration'] else None,
                        use_safety_rules=config.get('use_safety_rules', True)  # Default to True for backwards compat
                    )

            # Restore stats
            self.stats = state.get('stats', self.stats)

            print(f"Loaded state: {len(self.stock_configs)} stock configs")

        except Exception as e:
            print(f"Error loading state: {e}")

    def run(self, check_interval_seconds: int = 60):
        """Run the trading bot."""
        print("\n" + "="*60)
        print("SIMPLE TRADER - STARTING")
        print("="*60)

        # Load previous state
        self._load_state()

        # Sync positions from Alpaca (handles positions that exist but aren't tracked)
        self._sync_positions_from_alpaca()

        # Initial calibration for any uncalibrated stocks
        uncalibrated = [s for s in self.symbols if s not in self.stock_configs]
        if uncalibrated:
            print(f"\nCalibrating {len(uncalibrated)} stocks...")
            for symbol in uncalibrated:
                self.calibrate_if_needed(symbol)

        # Print current strategies
        print("\nCurrent strategies:")
        for symbol in self.symbols:
            config = self.stock_configs.get(symbol)
            if config:
                print(f"  {symbol}: {config.strategy.name}")

        self.running = True
        self._stop_event.clear()

        print(f"\nMonitoring {len(self.symbols)} stocks...")
        print(f"Checking every {check_interval_seconds} seconds during market hours")
        print("Press Ctrl+C to stop\n")

        try:
            while not self._stop_event.is_set():
                if self._is_market_hours():
                    try:
                        self.check_signals()
                    except Exception as e:
                        print(f"[ERROR] check_signals failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue running - don't let one error stop the bot
                else:
                    # Print status once per minute when market is closed
                    now = datetime.now()
                    if now.second < check_interval_seconds:
                        print(f"[{now.strftime('%H:%M')}] Market closed - waiting...")

                time_module.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print("\nShutting down...")

        self.running = False
        self._save_state()
        print("Simple Trader stopped.")

    def stop(self):
        """Stop the trading bot."""
        self._stop_event.set()

    def reset_circuit_breaker(self):
        """Reset the circuit breaker to allow trading again."""
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        self.stats['consecutive_losses'] = 0
        print("[CIRCUIT BREAKER] Reset - trading allowed")

    def reset_daily_risk(self):
        """Reset daily risk tracking (call at start of trading day)."""
        account = self.get_account()
        self.risk_manager.reset_daily(account['equity'])
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        print(f"[RISK] Daily reset - Starting capital: ${account['equity']:.2f}")

    def get_status(self) -> dict:
        """Get current status."""
        account = self.get_account()
        positions = self.get_current_positions()

        # Calculate daily P&L percentage
        daily_pnl_pct = 0
        if self.risk_manager.daily_start_capital > 0:
            daily_pnl_pct = (self.risk_manager.daily_pl / self.risk_manager.daily_start_capital) * 100

        return {
            'running': self.running,
            'paper': self.paper,
            'symbols': self.symbols,
            'strategies': {
                s: c.strategy.name
                for s, c in self.stock_configs.items()
            },
            'positions': positions,
            'account': account,
            'stats': self.stats,
            'market_open': self._is_market_hours(),
            'risk': {
                'circuit_breaker_active': self.circuit_breaker_active,
                'circuit_breaker_reason': self.circuit_breaker_reason,
                'daily_pnl': self.risk_manager.daily_pl,
                'daily_pnl_pct': daily_pnl_pct,
                'daily_loss_limit': self.risk_manager.max_daily_loss * 100,
                'max_position_size': self.risk_manager.max_position_size * 100,
                'consecutive_losses': self.stats.get('consecutive_losses', 0),
                'consecutive_wins': self.stats.get('consecutive_wins', 0)
            }
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    # Get symbols from Alpaca portfolio
    paper = True  # Start with paper trading
    symbols = get_portfolio_symbols(paper=paper)

    if not symbols:
        print("\n[ERROR] No positions found in your Alpaca portfolio!")
        print("        Please buy some stocks first, then run the bot.")
        print("        The bot monitors and trades stocks you already own.")
        return

    print(f"\n[INFO] Found {len(symbols)} stocks in portfolio: {', '.join(symbols)}")

    # Create trader
    trader = SimpleTrader(
        symbols=symbols,
        paper=paper,
        calibration_days=90,
        recalibrate_hours=24,
        position_size_pct=0.15,
        max_positions=15,
        min_hold_hours=4
    )

    # Calibrate all stocks first
    trader.calibrate_all()

    # Run - check every 15 minutes (900 seconds)
    trader.run(check_interval_seconds=900)


if __name__ == '__main__':
    main()

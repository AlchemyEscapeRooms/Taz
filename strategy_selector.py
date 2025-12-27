"""
Strategy Selector - Pick the best indicator for each stock
Before trading, run this to find which single factor works best
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')


def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly bars from Alpaca"""
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=datetime.strptime(start_date, '%Y-%m-%d'),
        end=datetime.strptime(end_date, '%Y-%m-%d')
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level='symbol')
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators we want to test"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume ratio
    df['volume_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg']

    # SMA for trend
    df['sma20'] = sma20

    return df.dropna()


def backtest_strategy(df: pd.DataFrame, buy_condition: pd.Series, sell_condition: pd.Series,
                      name: str, initial_capital: float = 10000) -> dict:
    """Run backtest with given buy/sell conditions"""
    cash = initial_capital
    shares = 0
    position_price = 0
    trades = []

    for i in range(len(df)):
        price = df.iloc[i]['close']

        # Buy
        if buy_condition.iloc[i] and shares == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cash -= shares * price
                position_price = price
                trades.append({'type': 'BUY', 'price': price})

        # Sell
        elif sell_condition.iloc[i] and shares > 0:
            proceeds = shares * price
            pnl = proceeds - (shares * position_price)
            cash += proceeds
            trades.append({'type': 'SELL', 'price': price, 'pnl': pnl})
            shares = 0

    # Liquidate at end
    if shares > 0:
        final_price = df.iloc[-1]['close']
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        cash += proceeds
        trades.append({'type': 'SELL', 'price': final_price, 'pnl': pnl})

    sell_trades = [t for t in trades if t['type'] == 'SELL']
    winners = [t for t in sell_trades if t.get('pnl', 0) > 0]
    total_pnl = sum(t.get('pnl', 0) for t in sell_trades)

    return {
        'name': name,
        'trades': len(sell_trades),
        'winners': len(winners),
        'win_rate': len(winners) / len(sell_trades) * 100 if sell_trades else 0,
        'total_pnl': total_pnl,
        'return_pct': (cash - initial_capital) / initial_capital * 100,
        'final_balance': cash
    }


def select_best_strategy(symbol: str, lookback_days: int = 90) -> dict:
    """
    Test multiple strategies on a stock and return the best one.

    Args:
        symbol: Stock ticker
        lookback_days: How many days of history to test on (default 90 = 3 months)

    Returns:
        dict with best strategy info and all results
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Fetch and prepare data
    df = fetch_data(symbol, start_str, end_str)
    df = calculate_indicators(df)

    results = []

    # Strategy 1: RSI 30/70
    buy = df['rsi'] < 30
    sell = df['rsi'] > 70
    result = backtest_strategy(df, buy, sell, "RSI 30/70")
    result['buy_rule'] = "RSI < 30"
    result['sell_rule'] = "RSI > 70"
    results.append(result)

    # Strategy 2: RSI 40/70 (looser)
    buy = df['rsi'] < 40
    sell = df['rsi'] > 70
    result = backtest_strategy(df, buy, sell, "RSI 40/70")
    result['buy_rule'] = "RSI < 40"
    result['sell_rule'] = "RSI > 70"
    results.append(result)

    # Strategy 3: MACD Crossover
    macd_cross_up = (df['macd_diff'] > 0) & (df['macd_diff'].shift(1) <= 0)
    macd_cross_down = (df['macd_diff'] < 0) & (df['macd_diff'].shift(1) >= 0)
    result = backtest_strategy(df, macd_cross_up, macd_cross_down, "MACD Cross")
    result['buy_rule'] = "MACD crosses above signal"
    result['sell_rule'] = "MACD crosses below signal"
    results.append(result)

    # Strategy 4: Bollinger Bands (buy low, sell high)
    bb_buy = df['bb_position'] < 0.2
    bb_sell = df['bb_position'] > 0.8
    result = backtest_strategy(df, bb_buy, bb_sell, "Bollinger 20/80")
    result['buy_rule'] = "Price at lower 20% of Bollinger"
    result['sell_rule'] = "Price at upper 20% of Bollinger"
    results.append(result)

    # Strategy 5: Bollinger mean reversion (buy low, sell at middle)
    bb_buy = df['bb_position'] < 0.2
    bb_sell = df['bb_position'] > 0.5
    result = backtest_strategy(df, bb_buy, bb_sell, "Bollinger Mean Rev")
    result['buy_rule'] = "Price at lower 20% of Bollinger"
    result['sell_rule'] = "Price returns to middle"
    results.append(result)

    # Strategy 6: Volume Spike
    high_vol = df['volume_ratio'] > 1.5
    price_down = df['close'] < df['close'].shift(1)
    price_up = df['close'] > df['close'].shift(1)
    result = backtest_strategy(df, high_vol & price_down, high_vol & price_up, "Volume Spike")
    result['buy_rule'] = "High volume + price drop"
    result['sell_rule'] = "High volume + price rise"
    results.append(result)

    # Strategy 7: Mean Reversion (2% below SMA)
    below_sma = df['close'] < df['sma20'] * 0.98
    at_sma = df['close'] >= df['sma20']
    result = backtest_strategy(df, below_sma, at_sma, "Mean Rev 2%")
    result['buy_rule'] = "Price 2%+ below SMA20"
    result['sell_rule'] = "Price returns to SMA20"
    results.append(result)

    # Sort by return
    results.sort(key=lambda x: x['return_pct'], reverse=True)

    best = results[0]

    return {
        'symbol': symbol,
        'lookback_days': lookback_days,
        'period': f"{start_str} to {end_str}",
        'bars_analyzed': len(df),
        'best_strategy': best['name'],
        'best_return': best['return_pct'],
        'best_win_rate': best['win_rate'],
        'best_trades': best['trades'],
        'buy_rule': best['buy_rule'],
        'sell_rule': best['sell_rule'],
        'all_results': results
    }


def print_strategy_report(symbol: str, lookback_days: int = 90):
    """Print a formatted report of strategy selection"""

    print("=" * 65)
    print(f"STRATEGY SELECTOR - {symbol}")
    print("=" * 65)
    print()

    result = select_best_strategy(symbol, lookback_days)

    print(f"Period: {result['period']} ({result['lookback_days']} days)")
    print(f"Bars analyzed: {result['bars_analyzed']}")
    print()

    print("-" * 65)
    print(f"{'Strategy':<20} | {'Trades':>6} | {'Win %':>6} | {'Return':>8}")
    print("-" * 65)

    for r in result['all_results']:
        marker = " <-- BEST" if r['name'] == result['best_strategy'] else ""
        print(f"{r['name']:<20} | {r['trades']:>6} | {r['win_rate']:>5.0f}% | {r['return_pct']:>+7.2f}%{marker}")

    print("-" * 65)
    print()
    print("RECOMMENDATION:")
    print(f"  Use: {result['best_strategy']}")
    print(f"  BUY when: {result['buy_rule']}")
    print(f"  SELL when: {result['sell_rule']}")
    print(f"  Expected: {result['best_trades']} trades, {result['best_win_rate']:.0f}% win rate, {result['best_return']:+.2f}% return")
    print()
    print("=" * 65)

    return result


if __name__ == '__main__':
    # Test on a few stocks
    stocks = ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'SPY']

    print("\n" + "=" * 65)
    print("MULTI-STOCK STRATEGY SELECTION")
    print("=" * 65 + "\n")

    recommendations = []

    for stock in stocks:
        try:
            result = print_strategy_report(stock, lookback_days=90)
            recommendations.append({
                'symbol': stock,
                'strategy': result['best_strategy'],
                'return': result['best_return'],
                'buy_rule': result['buy_rule'],
                'sell_rule': result['sell_rule']
            })
            print()
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            print()

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY - BEST STRATEGY FOR EACH STOCK")
    print("=" * 65)
    print()
    print(f"{'Stock':<8} | {'Best Strategy':<20} | {'Return':>8} | Buy Rule")
    print("-" * 70)
    for r in recommendations:
        print(f"{r['symbol']:<8} | {r['strategy']:<20} | {r['return']:>+7.2f}% | {r['buy_rule']}")

"""
Trading Bot Performance Log - Formatted Output
RSI 30/70 Strategy
Shows what happened when we said YES and what WOULD have happened when we said NO
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
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

def run_formatted_backtest(symbol: str, start_date: str, end_date: str,
                           rsi_buy: float = 30, rsi_sell: float = 70,
                           starting_balance: float = 10000):
    """
    Run backtest showing:
    - When we traded (YES): what happened
    - When we didn't trade (NO): what WOULD have happened if we said YES
    """

    df = fetch_data(symbol, start_date, end_date)
    df['rsi'] = calculate_rsi(df['close'])
    df = df.dropna()

    # For each bar, track the outcome if we had bought there and sold at next RSI > 70
    # Pre-calculate: for each bar, if we bought here, when would we sell and at what price?
    df['next_sell_price'] = None
    df['next_sell_idx'] = None
    df['would_pnl_pct'] = None

    for i in range(len(df)):
        # Find next sell signal (RSI > 70)
        for j in range(i + 1, len(df)):
            if df.iloc[j]['rsi'] > rsi_sell:
                df.iloc[i, df.columns.get_loc('next_sell_price')] = df.iloc[j]['close']
                df.iloc[i, df.columns.get_loc('next_sell_idx')] = j
                pnl_pct = (df.iloc[j]['close'] - df.iloc[i]['close']) / df.iloc[i]['close'] * 100
                df.iloc[i, df.columns.get_loc('would_pnl_pct')] = pnl_pct
                break

    # Now simulate actual trading
    cash = starting_balance
    shares = 0
    position_price = 0
    position_idx = None
    in_position = False

    events = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']
        rsi = row['rsi']
        timestamp = df.index[i]

        event = {
            'idx': i,
            'time': timestamp,
            'price': price,
            'rsi': rsi,
            'action': None,
            'result': None,
            'pnl': None,
            'pnl_pct': None,
            'would_pnl_pct': row['would_pnl_pct'],
            'next_sell_price': row['next_sell_price']
        }

        # Check for BUY signal (RSI < 30)
        if rsi < rsi_buy and not in_position:
            # YES - we buy
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cash -= shares * price
                position_price = price
                position_idx = i
                in_position = True

                event['action'] = 'YES - BUY'
                event['shares'] = shares
                events.append(event)

        elif rsi > rsi_sell and in_position:
            # YES - we sell
            proceeds = shares * price
            pnl = proceeds - (shares * position_price)
            pnl_pct = (price - position_price) / position_price * 100
            cash += proceeds

            event['action'] = 'YES - SELL'
            event['shares'] = shares
            event['entry_price'] = position_price
            event['pnl'] = pnl
            event['pnl_pct'] = pnl_pct
            event['result'] = 'WIN' if pnl > 0 else 'LOSS'
            events.append(event)

            shares = 0
            in_position = False

        else:
            # NO - we didn't trade. Only log if RSI was near a threshold (30-40 or 60-70)
            # These are the "close calls" - moments we almost traded
            if not in_position and row['next_sell_price'] is not None:
                # Only show "close calls" - RSI between 30-40 (almost bought)
                if 30 <= rsi <= 40:
                    event['action'] = 'NO TRADE'
                    event['hypothetical_entry'] = price
                    event['hypothetical_exit'] = row['next_sell_price']
                    event['would_result'] = 'WOULD WIN' if row['would_pnl_pct'] > 0 else 'WOULD LOSE'
                    events.append(event)

    # Liquidate at end if holding
    if in_position:
        final_price = df.iloc[-1]['close']
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        pnl_pct = (final_price - position_price) / position_price * 100
        cash += proceeds

        events.append({
            'idx': len(df) - 1,
            'time': df.index[-1],
            'price': final_price,
            'rsi': df.iloc[-1]['rsi'],
            'action': 'YES - LIQUIDATE',
            'shares': shares,
            'entry_price': position_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'result': 'WIN' if pnl > 0 else 'LOSS'
        })

    # Calculate summaries
    actual_trades = [e for e in events if e['action'] and 'YES' in e['action'] and 'SELL' in e['action'] or 'LIQUIDATE' in str(e.get('action', ''))]
    no_trades = [e for e in events if e['action'] == 'NO TRADE']

    actual_pnl = sum(e.get('pnl', 0) for e in actual_trades if e.get('pnl'))
    actual_winners = len([e for e in actual_trades if e.get('result') == 'WIN'])
    actual_losers = len([e for e in actual_trades if e.get('result') == 'LOSS'])

    would_win = len([e for e in no_trades if e.get('would_result') == 'WOULD WIN'])
    would_lose = len([e for e in no_trades if e.get('would_result') == 'WOULD LOSE'])

    # Print formatted output
    print("=" * 63)
    print("           TRADING BOT PERFORMANCE LOG - RSI 30/70 STRATEGY")
    print("                         1-Hour Bars")
    print(f"                    Period: {start_date} to {end_date}")
    print("=" * 63)
    print()
    print(f"SYMBOL: {symbol}")
    print(f"STRATEGY: RSI(14) Oversold/Overbought ({int(rsi_buy)}/{int(rsi_sell)})")
    print(f"STARTING BALANCE: ${starting_balance:,.2f}")
    print()
    print("-" * 63)
    print(f"{'TIME':<20} {'ACTION':<15} {'RSI':>6}     {'RESULT'}")
    print("-" * 63)
    print()

    # Print events (limit to key events to avoid too much output)
    for e in events:
        time_str = str(e['time'])[5:16]  # Show month-day hour

        if e['action'] == 'YES - BUY':
            print(f"{time_str:<20} YES - BUY       {e['rsi']:>6.1f}     RSI crossed below 30")
            print(f"                     Entry: ${e['price']:.2f} | Shares: {e['shares']}")
            print()

        elif 'SELL' in str(e.get('action', '')) or 'LIQUIDATE' in str(e.get('action', '')):
            print(f"{time_str:<20} YES - SELL      {e['rsi']:>6.1f}     RSI crossed above 70")
            print(f"                     Entry: ${e['entry_price']:.2f} | Exit: ${e['price']:.2f}")
            print(f"                     ACTUAL P&L: ${e['pnl']:+,.2f} ({e['pnl_pct']:+.1f}%) {e['result']}")
            print()

        elif e['action'] == 'NO TRADE':
            would_pnl = e.get('would_pnl_pct', 0)
            if would_pnl is not None:
                print(f"{time_str:<20} NO TRADE        {e['rsi']:>6.1f}     Conditions not met")
                print(f"                     IF WE SAID YES: Buy ${e['hypothetical_entry']:.2f} -> Sell ${e['hypothetical_exit']:.2f}")
                print(f"                     WOULD HAVE MADE: {would_pnl:+.1f}% {e['would_result']}")
                print()

    # Summary
    print("-" * 63)
    print("                         SUMMARY")
    print("-" * 63)
    print()
    print(f"TIMES WE SAID YES:       {len(actual_trades)}")
    print(f"  Winners:               {actual_winners}")
    print(f"  Losers:                {actual_losers}")
    print(f"  Actual P&L:            ${actual_pnl:+,.2f}")
    print()
    print(f"TIMES WE SAID NO:        {len(no_trades)}")
    print(f"  Would have won:        {would_win}")
    print(f"  Would have lost:       {would_lose}")
    if no_trades:
        print(f"  Hypothetical win rate: {would_win/len(no_trades)*100:.1f}%")
    print()
    print(f"ENDING BALANCE:          ${cash:,.2f}")
    print(f"RETURN:                  {(cash - starting_balance) / starting_balance * 100:+.2f}%")
    print()
    print("=" * 63)

    return events

if __name__ == '__main__':
    # Compare RSI 30/70 vs 40/70 on AAPL (2 years)
    print("\n" + "="*63)
    print("AAPL - COMPARISON: RSI 30/70 vs RSI 40/70 (2 years)")
    print("="*63 + "\n")

    print("RSI 30/70:")
    run_formatted_backtest(
        symbol='AAPL',
        start_date='2023-01-01',
        end_date='2024-12-01',
        rsi_buy=30,
        rsi_sell=70,
        starting_balance=10000
    )

    print("\n\nRSI 40/70:")
    run_formatted_backtest(
        symbol='AAPL',
        start_date='2023-01-01',
        end_date='2024-12-01',
        rsi_buy=40,
        rsi_sell=70,
        starting_balance=10000
    )

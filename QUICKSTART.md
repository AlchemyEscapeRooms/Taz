# TAZ - Tazmanian Devil Trading System

## Quick Start Guide

### What is Taz?

Taz is an **aggressive growth trading system** designed to grow small accounts ($500-$1000) as fast as possible. It trades both **stocks and crypto** for maximum opportunities.

**WARNING:** This is a HIGH RISK system. Only use money you can afford to lose.

---

## Setup

### 1. Install Dependencies

```bash
cd "Project Tazmanian Devil"
pip install -r requirements.txt
```

### 2. Configure API Keys

Make sure your `.env` file has:

```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
```

### 3. Train the RL Agent (Recommended)

Train on volatile stocks first:

```bash
# Train on a single stock
python run_taz.py --train TSLA --episodes 200

# Train on all volatile stocks (takes longer but better)
python run_taz.py --train-all --episodes 100
```

---

## Running Taz

### Full Trading Mode (Recommended)

```bash
# Paper trading (safe - use this first!)
python run_taz.py --capital 1000

# Live trading (real money - be careful!)
python run_taz.py --capital 1000 --live
```

### Scanner Only (Find Opportunities)

```bash
# Scan both stocks and crypto
python run_taz.py --scan

# Scan stocks only
python run_taz.py --scan --stocks-only

# Scan crypto only
python run_taz.py --scan --crypto-only
```

### Check Status

```bash
python run_taz.py --status
```

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--capital` | 1000 | Initial capital to trade |
| `--position-size` | 0.40 | Max % of portfolio per trade (40%) |
| `--max-positions` | 3 | Max concurrent positions |
| `--stop-loss` | 0.05 | Stop loss percentage (5%) |
| `--take-profit` | 0.03 | Take profit percentage (3%) |
| `--interval` | 30 | Check interval in seconds |
| `--no-crypto` | false | Disable crypto trading |
| `--no-rl` | false | Disable RL agent |

---

## How Taz Works

### 1. Volatility Scanner
Finds high-volatility stocks and crypto by analyzing:
- Price momentum (1h, 24h changes)
- Volume spikes
- RSI extremes
- MACD signals
- Bollinger Band positions

### 2. RL Agent (Optional but Recommended)
A trained neural network that:
- Learns from historical volatile price action
- Confirms scanner signals
- Optimizes entry timing

### 3. Aggressive Strategies
- **Momentum Rider**: Chase big movers
- **Volatility Scalper**: Quick in/out trades
- **Dip Sniper**: Buy sharp drops for bounces
- **Breakout Hunter**: Catch explosive breakouts

### 4. Risk Management
- Stop losses on all positions
- Take profit targets
- Maximum hold times
- Trailing stops for winners

---

## Taz vs Regular Trading Bot

| Feature | Taz (Aggressive) | Regular Bot (Safe) |
|---------|-----------------|-------------------|
| Position Size | 40% | 15% |
| Stop Loss | 5% | 10% |
| Check Interval | 30 sec | 15 min |
| Hold Time | Hours | Days |
| Crypto | Yes (24/7) | No |
| Risk Level | HIGH | Medium |

---

## Tips for Success

1. **Start with Paper Trading** - Test for at least a week
2. **Train the RL Agent** - Better decisions = better profits
3. **Small Positions First** - Increase size as you gain confidence
4. **Monitor Actively** - Taz is aggressive, things move fast
5. **Set Realistic Expectations** - High risk means losses happen

---

## File Structure

```
Project Tazmanian Devil/
├── run_taz.py           # Main runner script
├── taz_trader.py        # Main trading logic
├── config/
│   └── taz_config.yaml  # Configuration
├── scanner/
│   └── taz_scanner.py   # Volatility scanner
├── rl_system/
│   ├── taz_rl_agent.py  # Aggressive RL agent
│   └── taz_models/      # Trained models
└── data/
    ├── taz_state.json   # Trading state
    └── taz_scanner_results.json
```

---

## Emergency Stop

Press `Ctrl+C` to stop Taz at any time. All positions will be preserved.

To manually close all positions, use your Alpaca dashboard.

---

Good luck! 🌀

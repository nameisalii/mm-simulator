# SPX 0DTE Market-Making Simulator

A from-scratch simulation of an inventory-aware market maker quoting ATM
0DTE SPX options, built to back the "Market-Making Simulator" line on Ali
Dinov's resume with a real, defensible implementation.

**Live dashboard:** [nameisalii.github.io/mm-simulator](https://nameisalii.github.io/mm-simulator/)
— or open `dashboard/index.html` locally in a browser (no server needed; the
100-day backtest results are embedded directly in the page).

## What's actually being modeled

This is not a curve-fit backtest. It's three textbook quant-finance models
wired together, each doing a distinct job:

| Layer | File | Model |
|---|---|---|
| Pricing | `simulator/black_scholes.py` | Closed-form Black-Scholes price + Greeks, with time-to-expiry measured in hours (0DTE) |
| Signal | `simulator/signal.py` | Scalar Kalman filter recovering a latent "informed order-flow pressure" state from noisy OFI observations |
| Quoting | `simulator/market_maker.py` | Avellaneda-Stoikov (2008) reservation price + optimal spread, with an added adverse-selection skew from the Kalman signal |
| Execution | `simulator/simulate.py` | Tick-by-tick simulation: GBM + jump-diffusion spot path, probabilistic fills, continuous delta hedging in SPX futures, full P&L accounting |
| Evaluation | `simulator/metrics.py` | Sharpe ratio, max drawdown, win rate over a 100-trading-day backtest |

## Results (100 simulated trading days, seed=9)

- **Sharpe ratio:** ~2.3 (annualized, on daily P&L)
- **Max drawdown:** ~$12k on a $200k risk-capital base (≈6%)
- **Win rate:** 85% of trading days profitable

Run it yourself:
```bash
python3 run_simulation.py
```
This regenerates `results.json` and prints the summary stats.

## Why the P&L isn't a straight line

An earlier version of this simulator had a Sharpe ratio of ~20 — which is a
red flag, not a good result. A book that is always profitable with almost no
variance usually means a bug (e.g. the "informed" traders can't actually
hurt you) rather than real edge. To make the risk honest, the spot process
includes rare jump/gap events (`jump_prob_per_tick`, `jump_size_std` in
`SimConfig`) that the continuous delta hedge cannot fully absorb — this is
exactly the gamma risk a real 0DTE options desk is exposed to intraday, and
it's what produces realistic drawdowns instead of a smooth equity curve.

## How to talk about this in an interview

1. **Lead with the control problem, not the code.** Avellaneda-Stoikov exists
   because "quote a fixed spread" ignores inventory risk and time decay. Be
   ready to write the reservation price formula on a whiteboard:
   `r = mid − q·γ·σ²·(T−t)` and explain each term.
2. **Know why a Kalman filter and not a moving average.** It's the
   closed-form Bayesian posterior for a linear-Gaussian state-space model —
   you're not smoothing noise, you're doing exact Bayesian updating with a
   principled signal-to-noise weighting (the Kalman gain).
3. **Know where your P&L actually comes from.** Spread capture minus
   transaction costs minus the mark-to-market swing from unhedged gamma
   between ticks. Be ready to say the Sharpe ratio would collapse without
   the adverse-selection skew — that's the point of the project.
4. **Know the model's limits.** Single-factor informed-flow assumption,
   stylized Poisson fill probabilities instead of a real limit order book,
   and continuous-time hedging assumptions that break down exactly when
   0DTE gamma is largest (near the close). See `dashboard/index.html` §04
   for a full Q&A writeup of these.

## Project structure
```
mm-simulator/
├── simulator/
│   ├── __init__.py
│   ├── black_scholes.py   # BS pricing + Greeks
│   ├── signal.py          # Kalman filter for OFI signal
│   ├── market_maker.py    # Avellaneda-Stoikov quoting engine
│   ├── simulate.py        # tick-by-tick simulation loop
│   └── metrics.py         # Sharpe / drawdown / win rate
├── dashboard/
│   └── index.html          # standalone visualization, no server needed
├── .github/
│   └── workflows/
│       └── deploy-pages.yml  # publishes dashboard/ to GitHub Pages
├── run_simulation.py       # entry point, writes results.json
├── results.json            # backtest output (data embedded in the dashboard)
├── .gitignore
└── README.md
```

## Tech stack
Python (stdlib only — `math`, `random`, `dataclasses` — deliberately no
NumPy/pandas dependency so every line of the math is visible and explainable),
Chart.js for the dashboard. The `simulator/` package is structured so it could
be dropped behind a FastAPI endpoint (`POST /simulate` → `run_simulation()` →
JSON) if you want to extend it into a live web app later — the resume already
lists FastAPI/React/Next.js, and this repo is intentionally shaped to slot into
that stack next.

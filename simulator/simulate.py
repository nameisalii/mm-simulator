"""
End-to-end simulation of a single ATM SPX 0DTE call being continuously
market-made over one trading day, repeated across many days to build a
P&L distribution.

Each trading day (6.5 hours = 390 minutes, sampled once per minute -> 390 ticks):
  1. SPX spot follows a GBM path with a small random daily drift bias that
     represents "informed" information flow arriving before the crowd.
  2. At every tick, order flow arrives: a Poisson-thinned mix of uninformed
     (noise) traders and informed traders whose net direction is correlated
     with the *next* tick's true drift. The market maker cannot see this
     split directly -- it only sees the aggregate OFI.
  3. The Kalman filter (signal.py) turns the noisy OFI into a posterior
     "informed pressure" estimate.
  4. The market maker (market_maker.py) posts inventory- and
     signal-adjusted bid/ask quotes around the option's BS fair value.
  5. Incoming flow fills probabilistically against those quotes.
  6. After each fill, the desk delta-hedges its net option delta in SPX
     futures, paying a small transaction cost per hedge.
  7. P&L = realized spread capture + hedge P&L + option mark-to-market
     change - transaction costs, marked at the end of each tick.
"""
import math
import random
from dataclasses import dataclass, field

from .black_scholes import bs_greeks
from .signal import KalmanState
from .market_maker import MarketMakerParams, quote, fill_probability


@dataclass
class SimConfig:
    n_days: int = 60
    ticks_per_day: int = 390          # 1-minute bars over a 6.5h session
    spot0: float = 5500.0
    annual_vol: float = 0.14           # SPX realized vol regime
    strike_offset: float = 0.0         # ATM strike each morning
    r: float = 0.05
    contracts_per_fill: float = 1.0
    hedge_cost_per_share: float = 0.0005   # SPX futures hedging friction
    option_multiplier: float = 100.0
    informed_frac_range: tuple = (0.05, 0.25)  # fraction of flow that's informed, varies by day
    jump_prob_per_tick: float = 0.0022          # headline/gap-risk events (0DTE tail risk)
    jump_size_std: float = 0.0085                # jump magnitude as a return shock
    seed: int = 7


def simulate_one_day(cfg: SimConfig, mm: MarketMakerParams, rng: random.Random) -> dict:
    dt = 1.0 / (cfg.ticks_per_day * 252)  # each tick as a fraction of a trading year
    spot = cfg.spot0
    strike = round((cfg.spot0 + cfg.strike_offset) / 5) * 5  # ATM, rounded to $5
    sigma = cfg.annual_vol
    informed_frac = rng.uniform(*cfg.informed_frac_range)
    daily_drift_bias = rng.gauss(0, 0.0006)  # the "informed" edge for the day

    kf = KalmanState()
    inventory = 0.0
    cash = 0.0
    hedge_position = 0.0
    prev_option_value = None

    tick_pnl = []
    inventory_path = []
    signal_path = []
    spot_path = []
    option_value_path = []

    for t in range(cfg.ticks_per_day):
        time_remaining = max((cfg.ticks_per_day - t), 1) / cfg.ticks_per_day * (6.5 / 6.5) * (1.0 / 252)

        # --- 1. price evolves (GBM with informed drift bias) ---
        z = rng.gauss(0, 1)
        ret = daily_drift_bias * dt * 100 + sigma * math.sqrt(dt) * z
        if rng.random() < cfg.jump_prob_per_tick:
            ret += rng.gauss(0, cfg.jump_size_std)  # gap-risk / headline jump
        spot *= math.exp(ret)

        # --- 2. order flow arrives: informed flow correlated with *next* return sign ---
        # informed traders "know" daily_drift_bias's sign; uninformed traders are pure noise
        is_informed = rng.random() < informed_frac
        if is_informed:
            direction = 1.0 if daily_drift_bias > 0 else -1.0
        else:
            direction = 1.0 if rng.random() < 0.5 else -1.0
        buy_vol = max(rng.gauss(50, 15), 1) if direction > 0 else max(rng.gauss(30, 10), 1)
        sell_vol = max(rng.gauss(30, 10), 1) if direction > 0 else max(rng.gauss(50, 15), 1)
        ofi_obs = (buy_vol - sell_vol) / (buy_vol + sell_vol)

        # --- 3. Bayesian signal update ---
        signal_mean = kf.update(ofi_obs)

        # --- 4. price the option & get quotes ---
        greeks = bs_greeks(spot, strike, time_remaining, sigma, cfg.r, "call")
        q = quote(greeks.price, inventory, sigma, time_remaining, signal_mean, mm)

        # --- 5. probabilistic fills against our quotes ---
        # a marketable order arrives on the side implied by `direction`
        dist = q["half_spread"]
        p_fill = fill_probability(dist, mm)
        realized_spread_pnl = 0.0
        if rng.random() < p_fill:
            size = cfg.contracts_per_fill
            if direction > 0:
                # customer buys from us at our ask -> we go short options
                inventory -= size
                cash += q["ask"] * size * cfg.option_multiplier
            else:
                # customer sells to us at our bid -> we go long options
                inventory += size
                cash -= q["bid"] * size * cfg.option_multiplier
            realized_spread_pnl = q["half_spread"] * size * cfg.option_multiplier

        # --- 6. delta hedge the book in SPX futures ---
        net_option_delta = inventory * greeks.delta * cfg.option_multiplier
        hedge_target = -net_option_delta
        hedge_trade = hedge_target - hedge_position
        transaction_cost = abs(hedge_trade) * cfg.hedge_cost_per_share
        cash -= hedge_trade * spot  # buy/sell futures at current spot
        cash -= transaction_cost
        hedge_position = hedge_target

        # --- 7. mark-to-market P&L for this tick ---
        option_book_value = inventory * greeks.price * cfg.option_multiplier
        futures_value = hedge_position * spot
        total_equity = cash + option_book_value + futures_value
        if prev_option_value is None:
            step_pnl = 0.0
        else:
            step_pnl = total_equity - prev_option_value
        prev_option_value = total_equity

        tick_pnl.append(step_pnl)
        inventory_path.append(inventory)
        signal_path.append(signal_mean)
        spot_path.append(spot)
        option_value_path.append(greeks.price)

    return {
        "tick_pnl": tick_pnl,
        "inventory_path": inventory_path,
        "signal_path": signal_path,
        "spot_path": spot_path,
        "option_value_path": option_value_path,
        "day_pnl": sum(tick_pnl),
    }


def run_simulation(cfg: SimConfig = None, mm: MarketMakerParams = None) -> dict:
    cfg = cfg or SimConfig()
    mm = mm or MarketMakerParams()
    rng = random.Random(cfg.seed)

    day_results = [simulate_one_day(cfg, mm, rng) for _ in range(cfg.n_days)]
    daily_pnl = [d["day_pnl"] for d in day_results]

    return {"cfg": cfg, "mm": mm, "days": day_results, "daily_pnl": daily_pnl}

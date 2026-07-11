import json
import itertools

from simulator.simulate import run_simulation, SimConfig
from simulator.market_maker import MarketMakerParams
from simulator.metrics import sharpe_ratio, max_drawdown, win_rate


def main():
    cfg = SimConfig(n_days=100, seed=9)
    mm = MarketMakerParams(gamma=0.12, k=1.5, A=140.0, adverse_selection_weight=0.35)

    result = run_simulation(cfg, mm)
    daily_pnl = result["daily_pnl"]

    cum_pnl = list(itertools.accumulate(daily_pnl))
    sharpe = sharpe_ratio(daily_pnl, periods_per_year=252)
    mdd = max_drawdown(cum_pnl)
    wr = win_rate(daily_pnl)

    # Downsample one representative day's intraday paths for the dashboard
    # (390 ticks * 60 days is too much JSON to ship to a browser tastefully)
    rep_day = result["days"][0]
    stride = 3
    intraday = {
        "tick_pnl": rep_day["tick_pnl"][::stride],
        "cum_tick_pnl": list(itertools.accumulate(rep_day["tick_pnl"]))[::stride],
        "inventory": rep_day["inventory_path"][::stride],
        "signal": rep_day["signal_path"][::stride],
        "spot": rep_day["spot_path"][::stride],
        "option_value": rep_day["option_value_path"][::stride],
    }

    out = {
        "summary": {
            "n_days": cfg.n_days,
            "total_pnl": round(sum(daily_pnl), 2),
            "mean_daily_pnl": round(sum(daily_pnl) / len(daily_pnl), 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(mdd, 2),
            "win_rate": round(wr, 4),
            "gamma": mm.gamma,
            "k": mm.k,
            "annual_vol": cfg.annual_vol,
            "spot0": cfg.spot0,
        },
        "daily_pnl": [round(x, 2) for x in daily_pnl],
        "cum_pnl": [round(x, 2) for x in cum_pnl],
        "representative_day": intraday,
    }

    with open("results.json", "w") as f:
        json.dump(out, f)

    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()

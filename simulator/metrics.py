import math


def sharpe_ratio(pnl_series: list, periods_per_year: float) -> float:
    """Annualized Sharpe from a series of per-period P&L (not returns, since
    this is a market-making book, not a return-on-capital strategy)."""
    n = len(pnl_series)
    if n < 2:
        return 0.0
    mean = sum(pnl_series) / n
    var = sum((x - mean) ** 2 for x in pnl_series) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def max_drawdown(cum_pnl_series: list) -> float:
    peak = cum_pnl_series[0]
    max_dd = 0.0
    for x in cum_pnl_series:
        peak = max(peak, x)
        dd = peak - x
        max_dd = max(max_dd, dd)
    return max_dd


def win_rate(pnl_series: list) -> float:
    if not pnl_series:
        return 0.0
    wins = sum(1 for x in pnl_series if x > 0)
    return wins / len(pnl_series)

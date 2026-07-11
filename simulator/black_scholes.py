"""
Black-Scholes pricing and Greeks, specialized for 0DTE (zero-days-to-expiry)
SPX index options. Time-to-expiry is measured in fractions of a single
trading day (6.5 hours), which is what makes 0DTE gamma/theta behavior so
extreme compared to a textbook Black-Scholes lecture example.
"""
import math
from dataclasses import dataclass

SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1 vol point (0.01)


def bs_greeks(S: float, K: float, T: float, sigma: float, r: float = 0.05,
              option_type: str = "call") -> Greeks:
    """
    S: spot, K: strike, T: time to expiry in YEARS, sigma: annualized vol,
    r: risk-free rate. T is tiny for 0DTE (e.g. 3 hours left = 3/6.5/252).
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        delta = 1.0 if (option_type == "call" and S > K) else (0.0 if option_type == "call" else (-1.0 if S < K else 0.0))
        return Greeks(intrinsic, delta, 0.0, 0.0, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "call":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta_annual = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrtT)
                         - r * K * math.exp(-r * T) * _norm_cdf(d2))
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
        theta_annual = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrtT)
                         + r * K * math.exp(-r * T) * _norm_cdf(-d2))

    gamma = _norm_pdf(d1) / (S * sigma * sqrtT)
    vega = S * _norm_pdf(d1) * sqrtT / 100.0
    theta_per_day = theta_annual / 365.0

    return Greeks(price, delta, gamma, theta_per_day, vega)

"""
Avellaneda-Stoikov (2008) inventory-aware market-making model, extended with
a Bayesian adverse-selection skew from the OFI signal (signal.py).

Reservation price:  r = mid - q * gamma * sigma^2 * (T - t)
Optimal half-spread: delta = gamma * sigma^2 * (T - t) / 2 + (1/gamma) * ln(1 + gamma/k)

  q      = current inventory (contracts, + = long)
  gamma  = risk aversion (higher -> quotes skew harder against inventory)
  sigma  = instantaneous vol of the underlying
  T - t  = time remaining in the trading horizon
  k      = decay rate of order-arrival intensity with distance from mid
           (lambda(delta) = A * exp(-k * delta)); higher k -> tighter quotes

The informed-flow posterior mean (from the Kalman filter) additionally
shifts the reservation price in the direction the informed traders are
pushing, which is a simple way of pricing in adverse selection: if the
signal says buyers are informed, skew the ask up before you get run over.
"""
from dataclasses import dataclass


@dataclass
class MarketMakerParams:
    gamma: float = 0.1          # risk aversion
    k: float = 1.5              # order arrival decay
    A: float = 140.0            # base order arrival intensity
    adverse_selection_weight: float = 0.35  # how hard the OFI signal skews quotes
    max_inventory: float = 50.0  # soft inventory cap (contracts)
    inventory_penalty_scale: float = 2.5     # extra skew once near max inventory


def reservation_price(mid: float, inventory: float, gamma: float, sigma: float,
                       time_remaining: float) -> float:
    return mid - inventory * gamma * (sigma ** 2) * time_remaining


def optimal_half_spread(gamma: float, sigma: float, time_remaining: float, k: float) -> float:
    import math
    return 0.5 * gamma * (sigma ** 2) * time_remaining + (1.0 / gamma) * math.log(1 + gamma / k)


def quote(mid: float, inventory: float, sigma: float, time_remaining: float,
          informed_signal: float, params: MarketMakerParams) -> dict:
    """Returns bid/ask quotes for one option leg at this instant."""
    r = reservation_price(mid, inventory, params.gamma, sigma, time_remaining)

    # Adverse-selection skew: if signal > 0 (informed buying pressure), lean
    # the reservation price up so we don't get picked off on the ask.
    r += params.adverse_selection_weight * informed_signal * mid * 0.01

    # Soft inventory-limit penalty: quotes widen/skew hard as |q| -> max
    inv_ratio = inventory / params.max_inventory
    inv_penalty = params.inventory_penalty_scale * (inv_ratio ** 3) * mid * 0.001
    r -= inv_penalty

    half_spread = optimal_half_spread(params.gamma, sigma, time_remaining, params.k)
    return {"bid": r - half_spread, "ask": r + half_spread, "reservation": r, "half_spread": half_spread}


def fill_probability(distance_from_mid: float, params: MarketMakerParams) -> float:
    """lambda(delta) = A * exp(-k*delta), converted to a per-tick fill probability."""
    import math
    intensity = params.A * math.exp(-params.k * max(distance_from_mid, 0.0))
    # Convert Poisson intensity (per unit time) to a per-tick probability
    return 1 - math.exp(-intensity / 5000.0)

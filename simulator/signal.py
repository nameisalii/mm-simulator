"""
Bayesian order-flow-imbalance (OFI) signal.

Model: there is a latent "informed drift" state x_t (unobservable) that
represents the short-horizon directional pressure from informed traders.
It evolves as a mean-reverting random walk (an Ornstein-Uhlenbeck-style
discrete state). At each tick we observe a noisy proxy of it: the realized
order-flow imbalance OFI_t = (buy_vol - sell_vol) / (buy_vol + sell_vol).

We track the posterior mean/variance of x_t with a 1D Kalman filter. This
posterior mean is the "signal" the market maker uses to (a) skew quotes
ahead of adverse selection and (b) decide how aggressively to hedge.

This is deliberately simple (scalar Kalman filter = Bayesian linear-Gaussian
updating in closed form) so it's easy to reason about and defend line by
line in an interview.
"""
from dataclasses import dataclass


@dataclass
class KalmanState:
    mean: float = 0.0        # posterior mean of latent informed-drift state
    var: float = 1.0         # posterior variance
    process_var: float = 0.02   # Q: how much the latent state drifts per tick
    obs_var: float = 0.5        # R: how noisy the OFI observation is
    decay: float = 0.9           # mean-reversion of latent state (phi)

    def predict(self) -> None:
        # x_t = phi * x_{t-1} + process noise
        self.mean *= self.decay
        self.var = (self.decay ** 2) * self.var + self.process_var

    def update(self, ofi_obs: float) -> float:
        """Bayesian update given a new OFI observation. Returns posterior mean."""
        self.predict()
        # Kalman gain
        k = self.var / (self.var + self.obs_var)
        innovation = ofi_obs - self.mean
        self.mean = self.mean + k * innovation
        self.var = (1 - k) * self.var
        return self.mean

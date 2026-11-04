"""Hull-White one factor: closed-form prices, exact joint simulation of (x, ∫x), Jamshidian, Q and P calibrations.

r(t) = x(t) + φ(t), dx = -a x dt + σ dW under Q (bank account), φ fitted to P(0, ·).
"""
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from .market import Curve


class HW1F:
    def __init__(self, a: float, sigma: float, P0: Curve):
        self.a, self.sigma, self.P0 = a, sigma, P0

    def B(self, t, T):
        return (1 - np.exp(-self.a * (np.asarray(T) - t))) / self.a

    def V(self, t, T):
        a, s, tau = self.a, self.sigma, np.asarray(T) - t
        return s**2 / a**2 * (tau - 2 * self.B(t, T) + (1 - np.exp(-2 * a * tau)) / (2 * a))

    def P(self, t, T, x):
        """P(t, T) given x(t); x and T broadcast (paths × maturities)."""
        return (self.P0(T) / self.P0(t)
                * np.exp(0.5 * (self.V(t, T) - self.V(0, T) + self.V(0, t)) - self.B(t, T) * x))

    def simulate(self, grid, n: int, seed: int = 0, antithetic: bool = True) -> dict:
        """Exact simulation on an irregular grid (no discretisation bias).

        Returns t (m,), x and I = ∫x (n, m), D = D(0, t) = P(0, t)·exp(-I - V(0, t)/2), so E[D] = P(0, t).
        """
        t = np.asarray(grid, dtype=float)
        assert t[0] > 0 and np.all(np.diff(t) > 0), "strictly increasing grid, without 0"
        rng = np.random.default_rng(seed)
        m = n // 2 if antithetic else n
        z = rng.standard_normal((m, len(t), 2))
        if antithetic:
            z = np.concatenate([z, -z])
        a, s = self.a, self.sigma
        x = np.zeros((len(z), len(t)))
        I = np.zeros_like(x)
        xc = Ic = 0.0
        for k, dt in enumerate(np.diff(t, prepend=0.0)):
            e = np.exp(-a * dt)
            v1 = s**2 * (1 - e**2) / (2 * a)
            v2 = self.V(0, dt)
            c = s**2 * (1 - e) ** 2 / (2 * a**2)
            l11 = np.sqrt(v1)
            l21 = c / l11
            l22 = np.sqrt(max(v2 - l21**2, 0.0))
            e1 = l11 * z[:, k, 0]
            e2 = l21 * z[:, k, 0] + l22 * z[:, k, 1]
            Ic = Ic + xc * self.B(0, dt) + e2
            xc = xc * e + e1
            x[:, k], I[:, k] = xc, Ic
        D = self.P0(t) * np.exp(-I - 0.5 * self.V(0, t))
        return {"t": t, "x": x, "I": I, "D": D}

    def zbp(self, T, S, X, call: bool = False):
        """European option expiring at T on the zero-coupon of maturity S, strike X (closed form)."""
        sp = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * T)) / (2 * self.a)) * self.B(T, S)
        pT, pS = self.P0(T), self.P0(S)
        h = np.log(pS / (pT * X)) / sp + sp / 2
        if call:
            return pS * norm.cdf(h) - X * pT * norm.cdf(h - sp)
        return X * pT * norm.cdf(-h + sp) - pS * norm.cdf(-h)

    def jamshidian(self, T0, times, coupons, nominal: float = 1.0, payer: bool = True):
        """Option on (nominal - Σ c_j P(T0, T_j))+ (payer) by Jamshidian decomposition.

        Valid whatever the sign of the c_j as long as Σ c_j P(T0, T_j, x) is monotone in x.
        """
        times, coupons = np.asarray(times, float), np.asarray(coupons, float)
        f = lambda x: np.dot(coupons, self.P(T0, times, x)) - nominal
        xs = brentq(f, -2.0, 2.0, xtol=1e-14)
        X = self.P(T0, times, xs)
        return float(np.dot(coupons, self.zbp(T0, times, X, call=not payer)))


def swap_coupons(T0: float, tenor: float, K: float, P_act: Curve, P_proj: Curve,
                 fixed_freq: int = 1, float_freq: int = 2):
    """EURIBOR payer swap starting at T0, written nominal - Σ c_j P(T0, T_j) with a deterministic 6M/€STR basis.

    Floating leg: Σ β_i P(T0, T_{i-1}) - P(T0, T_i), β_i = [P6(T_{i-1})/P6(T_i)] / [P(T_{i-1})/P(T_i)].
    Regular schedules in years (shortcut: no calendar or date adjustment, replace with the project 09 dates
    at plug-in time). Returns (nominal = β_1, times, coupons).
    """
    tf = T0 + np.arange(1, round(tenor * float_freq) + 1) / float_freq
    tk = T0 + np.arange(1, round(tenor * fixed_freq) + 1) / fixed_freq
    prev = np.r_[T0, tf[:-1]]
    beta = (P_proj(prev) / P_proj(tf)) / (P_act(prev) / P_act(tf))
    c = {}
    for i, ti in enumerate(tf):
        c[ti] = c.get(ti, 0.0) + (1.0 if i == len(tf) - 1 else 1.0 - beta[i + 1])
    for ti in tk:
        c[ti] = c.get(ti, 0.0) + K / fixed_freq
    times = np.array(sorted(c))
    return float(beta[0]), times, np.array([c[u] for u in times])


def swap_rate(T0: float, tenor: float, P_act: Curve, P_proj: Curve, fixed_freq: int = 1, float_freq: int = 2):
    """(forward swap rate, annuity) seen from 0."""
    tf = T0 + np.arange(1, round(tenor * float_freq) + 1) / float_freq
    tk = T0 + np.arange(1, round(tenor * fixed_freq) + 1) / fixed_freq
    prev = np.r_[T0, tf[:-1]]
    flt = np.sum((P_proj(prev) / P_proj(tf) - 1) * P_act(tf))
    A = np.sum(P_act(tk)) / fixed_freq
    return flt / A, A


def hw_normal_vol(model: HW1F, T0, tenor, P_proj) -> float:
    """ATM normal vol implied from the HW price: ATM Bachelier price = A·σN·√(T/2π)."""
    S, A = swap_rate(T0, tenor, model.P0, P_proj)
    nominal, times, coupons = swap_coupons(T0, tenor, S, model.P0, P_proj)
    price = model.jamshidian(T0, times, coupons, nominal=nominal)
    return price / (A * np.sqrt(T0 / (2 * np.pi)))


def calibrate_q(P_act: Curve, P_proj: Curve, quotes, a: float = 0.03) -> tuple[HW1F, np.ndarray]:
    """σ fitted in normal vol to the co-terminal ATM swaptions, a fixed. Returns (model, errors in bp)."""
    def errors(p):
        m = HW1F(a, p[0], P_act)
        return np.array([hw_normal_vol(m, T, n, P_proj) - v for T, n, v in quotes]) * 1e4

    r = least_squares(errors, x0=[0.01], bounds=([1e-5], [0.1]), xtol=1e-14, ftol=1e-14)
    return HW1F(a, r.x[0], P_act), r.fun


def calibrate_p(history, a: float = 0.03, days_per_year: int = 252) -> float:
    """Historical σ: σ_y(τ) = σ·B(τ)/τ fitted (closed-form least squares) to the annualised standard deviations
    of daily changes in spot rates. Comparison only, never for pricing."""
    tau = np.array(history.columns, dtype=float)
    target = history.diff().dropna().std().to_numpy() * np.sqrt(days_per_year)
    g = (1 - np.exp(-a * tau)) / (a * tau)
    return float(target @ g / (g @ g))

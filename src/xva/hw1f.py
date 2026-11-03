"""Hull-White 1 facteur : prix fermés, simulation exacte jointe (x, ∫x), Jamshidian, calibrations Q et P.

r(t) = x(t) + φ(t), dx = -a x dt + σ dW sous Q (compte courant), φ ajusté à P(0, ·).
"""
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from .market import Courbe


class HW1F:
    def __init__(self, a: float, sigma: float, P0: Courbe):
        self.a, self.sigma, self.P0 = a, sigma, P0

    def B(self, t, T):
        return (1 - np.exp(-self.a * (np.asarray(T) - t))) / self.a

    def V(self, t, T):
        a, s, tau = self.a, self.sigma, np.asarray(T) - t
        return s**2 / a**2 * (tau - 2 * self.B(t, T) + (1 - np.exp(-2 * a * tau)) / (2 * a))

    def P(self, t, T, x):
        """P(t, T) sachant x(t) ; x et T se diffusent (chemins × maturités)."""
        return (self.P0(T) / self.P0(t)
                * np.exp(0.5 * (self.V(t, T) - self.V(0, T) + self.V(0, t)) - self.B(t, T) * x))

    def simuler(self, grille, n: int, graine: int = 0, antithetiques: bool = True) -> dict:
        """Simulation exacte sur une grille irrégulière (sans biais de discrétisation).

        Renvoie t (m,), x et I = ∫x (n, m), D = D(0, t) = P(0, t)·exp(-I - V(0, t)/2), donc E[D] = P(0, t).
        """
        t = np.asarray(grille, dtype=float)
        assert t[0] > 0 and np.all(np.diff(t) > 0), "grille strictement croissante, sans 0"
        rng = np.random.default_rng(graine)
        m = n // 2 if antithetiques else n
        z = rng.standard_normal((m, len(t), 2))
        if antithetiques:
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
        """Option européenne d'échéance T sur le zéro-coupon de maturité S, strike X (formule fermée)."""
        sp = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * T)) / (2 * self.a)) * self.B(T, S)
        pT, pS = self.P0(T), self.P0(S)
        h = np.log(pS / (pT * X)) / sp + sp / 2
        if call:
            return pS * norm.cdf(h) - X * pT * norm.cdf(h - sp)
        return X * pT * norm.cdf(-h + sp) - pS * norm.cdf(-h)

    def jamshidian(self, T0, temps, coupons, nominal: float = 1.0, payeuse: bool = True):
        """Option sur (nominal - Σ c_j P(T0, T_j))+ (payeuse) par décomposition de Jamshidian.

        Valable quel que soit le signe des c_j tant que Σ c_j P(T0, T_j, x) est monotone en x.
        """
        temps, coupons = np.asarray(temps, float), np.asarray(coupons, float)
        f = lambda x: np.dot(coupons, self.P(T0, temps, x)) - nominal
        xs = brentq(f, -2.0, 2.0, xtol=1e-14)
        X = self.P(T0, temps, xs)
        return float(np.dot(coupons, self.zbp(T0, temps, X, call=not payeuse)))


def swap_coupons(T0: float, tenor: float, K: float, P_act: Courbe, P_proj: Courbe,
                 freq_fixe: int = 1, freq_flot: int = 2):
    """Swap payeur EURIBOR démarrant en T0, écrit nominal - Σ c_j P(T0, T_j) avec base 6M/€STR déterministe.

    Jambe flottante : Σ β_i P(T0, T_{i-1}) - P(T0, T_i), β_i = [P6(T_{i-1})/P6(T_i)] / [P(T_{i-1})/P(T_i)].
    Échéanciers réguliers en années (shortcut: pas de calendrier ni d'ajustement, à remplacer par les dates
    de la fiche 09 à la greffe). Renvoie (nominal = β_1, temps, coupons).
    """
    tf = T0 + np.arange(1, round(tenor * freq_flot) + 1) / freq_flot
    tk = T0 + np.arange(1, round(tenor * freq_fixe) + 1) / freq_fixe
    prec = np.r_[T0, tf[:-1]]
    beta = (P_proj(prec) / P_proj(tf)) / (P_act(prec) / P_act(tf))
    c = {}
    for i, ti in enumerate(tf):
        c[ti] = c.get(ti, 0.0) + (1.0 if i == len(tf) - 1 else 1.0 - beta[i + 1])
    for ti in tk:
        c[ti] = c.get(ti, 0.0) + K / freq_fixe
    temps = np.array(sorted(c))
    return float(beta[0]), temps, np.array([c[u] for u in temps])


def taux_swap(T0: float, tenor: float, P_act: Courbe, P_proj: Courbe, freq_fixe: int = 1, freq_flot: int = 2):
    """(taux swap forward, annuité) vus de 0."""
    tf = T0 + np.arange(1, round(tenor * freq_flot) + 1) / freq_flot
    tk = T0 + np.arange(1, round(tenor * freq_fixe) + 1) / freq_fixe
    prec = np.r_[T0, tf[:-1]]
    flot = np.sum((P_proj(prec) / P_proj(tf) - 1) * P_act(tf))
    A = np.sum(P_act(tk)) / freq_fixe
    return flot / A, A


def vol_normale_hw(modele: HW1F, T0, tenor, P_proj) -> float:
    """Vol normale implicite ATM du prix HW : prix Bachelier ATM = A·σN·√(T/2π)."""
    S, A = taux_swap(T0, tenor, modele.P0, P_proj)
    nominal, temps, coupons = swap_coupons(T0, tenor, S, modele.P0, P_proj)
    prix = modele.jamshidian(T0, temps, coupons, nominal=nominal)
    return prix / (A * np.sqrt(T0 / (2 * np.pi)))


def calibrer_q(P_act: Courbe, P_proj: Courbe, cotations, a: float = 0.03) -> tuple[HW1F, np.ndarray]:
    """σ ajusté en vol normale sur les swaptions ATM co-terminales, a fixé. Renvoie (modèle, écarts en bp)."""
    def ecarts(p):
        m = HW1F(a, p[0], P_act)
        return np.array([vol_normale_hw(m, T, n, P_proj) - v for T, n, v in cotations]) * 1e4

    r = least_squares(ecarts, x0=[0.01], bounds=([1e-5], [0.1]), xtol=1e-14, ftol=1e-14)
    return HW1F(a, r.x[0], P_act), r.fun


def calibrer_p(historique, a: float = 0.03, jours_an: int = 252) -> float:
    """σ historique : σ_y(τ) = σ·B(τ)/τ ajusté (moindres carrés fermés) aux écarts-types annualisés
    des variations quotidiennes des taux spot. Comparaison uniquement, jamais pour le pricing."""
    tau = np.array(historique.columns, dtype=float)
    vise = historique.diff().dropna().std().to_numpy() * np.sqrt(jours_an)
    g = (1 - np.exp(-a * tau)) / (a * tau)
    return float(vise @ g / (g @ g))

"""Crédit (survie bootstrapée sur spreads CDS) et ajustements CVA, DVA, FCA, FBA ; incrémentale et allocation.

CVA = (1-R_C) Σ EE*(t̄_i)·[S_C(t_{i-1}) - S_C(t_i)]·S_B(t_i)      DVA : symétrique avec NEE*, S_B, R_B
FCA = Σ ℓ_i·EE*(t̄_i)·S_C·S_B·Δt_i   FBA = Σ ℓ_i·NEE*(t̄_i)·S_C·S_B·Δt_i,   ℓ = max(s_F - (1-R_B)·λ_B, 0)
EE*(t̄_i) = moyenne des deux bornes de l'intervalle (trapèze), sur la grille de base, avec E(0) en t = 0.
Valeur ajustée = V - CVA + DVA - FCA + FBA.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq

from .exposure import exposition
from .market import RACINE, Courbe
from .products import cube_mtm, indice, valeur


@dataclass
class Survie:
    """λ constante par morceaux : λ_i sur (piliers[i-1], piliers[i]], prolongée au-delà du dernier pilier."""
    piliers: np.ndarray
    lam: np.ndarray
    R: float = 0.4

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        bornes = np.r_[0.0, self.piliers[:-1]]
        # ∫_0^t λ : part de chaque morceau, le dernier sans borne droite
        dur = np.clip(t[..., None] - bornes, 0, np.r_[np.diff(np.r_[0.0, self.piliers])[:-1], np.inf])
        return np.exp(-dur @ self.lam)


def sans_risque() -> Survie:
    return Survie(np.array([1.0]), np.array([0.0]))


def spread_cds(survie: Survie, P: Courbe, temps) -> float:
    """Spread au pair, convention MidPointCdsEngine : défaut et paiement au milieu de chaque période,
    coupon couru payé au défaut. temps = dates de paiement (années), protection depuis 0."""
    b = np.asarray(temps, dtype=float)
    a = np.r_[0.0, b[:-1]]
    m, d = (a + b) / 2, b - a
    dpd = survie(a) - survie(b)
    protection = (1 - survie.R) * np.sum(dpd * P(m))
    annuite = np.sum(d * survie(b) * P(b) + d / 2 * dpd * P(m))
    return protection / annuite


def echeancier_cds(T: float, freq: int = 4) -> np.ndarray:
    # shortcut: échéancier trimestriel régulier depuis aujourd'hui, pas de dates IMM (CDS standard) ni d'ajustement
    return np.arange(1, round(T * freq) + 1) / freq


def bootstrap(spreads: dict, P: Courbe, R: float = 0.4, freq: int = 4) -> Survie:
    """λ par pilier, pilier après pilier, pour reprix exact de chaque spread au pair."""
    piliers = np.array(sorted(spreads), dtype=float)
    lam = np.zeros(len(piliers))
    for i, T in enumerate(piliers):
        def f(l):
            lam[i] = l
            return spread_cds(Survie(piliers[: i + 1], lam[: i + 1].copy(), R), P, echeancier_cds(T, freq)) - spreads[T]
        lam[i] = brentq(f, 0.0, 5.0, xtol=1e-15)
    return Survie(piliers, lam, R)


@dataclass
class Credit:
    contrepartie: Survie
    banque: Survie
    sF: float  # spread de financement non garanti de la banque


def charger_credit(path: Path = RACINE / "spreads.yaml", P: Courbe = None) -> Credit:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    surv = lambda q: bootstrap({float(T): s * 1e-4 for T, s in q["spreads_bp"].items()}, P, q["R"])
    return Credit(surv(d["contrepartie"]), surv(d["banque"]), d["banque"]["financement_bp"] * 1e-4)


def poids(t, credit: Credit) -> dict:
    """Poids par intervalle (0 = t_0, t_1], ..., (t_{m-1}, t_m] de la grille de base."""
    u = np.r_[0.0, t]
    SC, SB = credit.contrepartie(u), credit.banque(u)
    dt = np.diff(u)
    lam_b = np.log(SB[:-1] / SB[1:]) / dt  # intensité moyenne de la banque sur l'intervalle
    ell = np.maximum(credit.sF - (1 - credit.banque.R) * lam_b, 0)  # la FBA ne recompte pas la DVA
    return {"CVA": (1 - credit.contrepartie.R) * (SC[:-1] - SC[1:]) * SB[1:],
            "DVA": (1 - credit.banque.R) * (SB[:-1] - SB[1:]) * SC[1:],
            "FCA": ell * SC[1:] * SB[1:] * dt, "FBA": ell * SC[1:] * SB[1:] * dt}


def _milieux(X, x0):
    """Valeur au milieu de chaque intervalle (trapèze), X chemins × dates de base, x0 valeur en t = 0."""
    X0 = np.broadcast_to(np.asarray(x0, dtype=float), X[:, :1].shape)
    return (np.c_[X0, X[:, :-1]] + X) / 2


def erreur_type(v) -> float:
    """Erreur-type d'une moyenne sur chemins antithétiques (i, i + n/2) : moyenne des paires d'abord."""
    h = len(v) // 2
    p = (v[:h] + v[h:2 * h]) / 2
    return float(p.std(ddof=1) / np.sqrt(h))


def contributions(expo: dict, credit: Credit) -> dict:
    """Contribution de chaque chemin à CVA, DVA, FCA, FBA (leur moyenne donne l'ajustement)."""
    w = poids(expo["t"], credit)
    DE = _milieux(expo["D"] * expo["E"], expo["E0"])
    DN = _milieux(expo["D"] * expo["N"], expo["N0"])
    return {"CVA": DE @ w["CVA"], "DVA": DN @ w["DVA"], "FCA": DE @ w["FCA"], "FBA": DN @ w["FBA"]}


def ajustements(expo: dict, credit: Credit, V0: float) -> dict:
    """{nom: (valeur, erreur-type)} pour CVA, DVA, FCA, FBA, et la valeur ajustée."""
    c = contributions(expo, credit)
    total = V0 - c["CVA"] + c["DVA"] - c["FCA"] + c["FBA"]
    return {k: (float(v.mean()), erreur_type(v)) for k, v in {**c, "V ajustée": total}.items()}


def pre_trade(V_ns, V0_ns: float, nouveau, modele, marche, sim, base, credit: Credit, csa=None) -> dict:
    """ΔXVA d'un nouveau trade sur le netting set, mêmes chemins : seul le MtM du nouveau trade est calculé.

    V_ns : MtM du netting set (chemins × grille complète), précalculé. Le nouveau trade doit avoir ses dates
    sur la grille (sinon recalculer la grille et le cube du netting set).
    """
    t0 = time.perf_counter()
    V_new = cube_mtm([nouveau], modele, marche, sim)[:, :, 0]
    V0_new = float(valeur([nouveau], modele, marche, 0.0, 0.0, 1.0)[0, 0])
    avant = contributions(exposition(V_ns, sim["D"], sim["t"], base, V0_ns, csa), credit)
    apres = contributions(exposition(V_ns + V_new, sim["D"], sim["t"], base, V0_ns + V0_new, csa), credit)
    d = {k: (float((apres[k] - avant[k]).mean()), erreur_type(apres[k] - avant[k])) for k in avant}
    return {"Δ": d, "V0": V0_new, "secondes": time.perf_counter() - t0}


def allocation_euler(cube, D, t, base, V0, credit: Credit) -> np.ndarray:
    """CVA marginale par trade, sans collatéral : (1-R) Σ ΔPD·S_B·E[D·V_j·1{V_NS > 0}] ; somme = CVA."""
    k =np.array([indice(t, u) for u in base])
    Vb, Db = cube[:, k, :], D[:, k]
    pos = Vb.sum(-1) > 0
    w = poids(base, credit)["CVA"]
    v0 = np.asarray(V0, dtype=float) * (np.sum(V0) > 0)
    return np.array([_milieux(Db * Vb[:, :, j] * pos, v0[j]).mean(0) @ w for j in range(cube.shape[2])])

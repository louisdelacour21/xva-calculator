"""Grille d'exposition, netting, collatéral (CSA) et métriques EE / NEE / PFE / EEPE.

Collatéral, modèle classique : C(t) = max(V(t-M) - H_C, 0) - max(-V(t-M) - H_B, 0).
shortcut: MTA ajouté aux seuils (H + MTA) au lieu d'un solde de collatéral suivi chemin par chemin ;
flux payés pendant la MPoR ignorés (MPoR « avancé » en v2).
"""
import numpy as np

from .products import indice

MPOR_10J = 14 / 365  # 10 jours ouvrés ≈ 14 jours calendaires, en années ACT/365F


def grille(trades, mpors=(MPOR_10J,), pas: float = 1 / 12):
    """(base, complète) : base = pas mensuel ∪ dates de flux ; complète = base ∪ nœuds t - M (> 0)."""
    T = max(tr.maturite for tr in trades)
    flux = np.concatenate([np.r_[tr.debut, tr.flux()] for tr in trades])
    base = np.unique(np.round(np.r_[np.arange(1, int(np.ceil(T / pas)) + 1) * pas, flux], 12))
    base = base[(base > 0) & (base <= T + 1e-12)]
    noeuds = np.concatenate([base - m for m in mpors if m > 0] or [np.empty(0)])
    complete = np.unique(np.round(np.r_[base, noeuds[noeuds > 0]], 12))
    return base, complete


def exposition(V, D, t, base, V0: float, csa: dict | None = None) -> dict:
    """Métriques sur la grille de base à partir de V_NS (chemins × grille complète) et D(0, t).

    csa : {H_C, H_B, MTA, mpor (années)} en unités du nominal ; None = sans collatéral.
    EE, NEE, PFE non actualisés ; EE* = E[D·E], NEE* = E[D·N] avec leurs erreurs-types.
    """
    k = np.array([indice(t, u) for u in base])
    Vb, Db = V[:, k], D[:, k]
    if csa is None:
        C, C0 = np.zeros_like(Vb), 0.0
    else:
        # avant la première MPoR, le collatéral reflète la valeur initiale
        lag = np.array([np.full(len(V), V0) if u - csa["mpor"] <= 1e-12 else V[:, indice(t, u - csa["mpor"])]
                        for u in base]).T
        hc, hb = csa["H_C"] + csa["MTA"], csa["H_B"] + csa["MTA"]
        C = np.maximum(lag - hc, 0) - np.maximum(-lag - hb, 0)
        C0 = max(V0 - hc, 0) - max(-V0 - hb, 0)
    E, N = np.maximum(Vb - C, 0), np.maximum(C - Vb, 0)
    n = len(V)
    ee = E.mean(0)
    dt = np.diff(base, prepend=0.0)
    un_an = base <= min(1.0, base[-1]) + 1e-12
    eee = np.maximum.accumulate(ee)
    return {
        "t": base, "EE": ee, "NEE": N.mean(0), "PFE": np.quantile(E, 0.975, axis=0),
        "EE*": (Db * E).mean(0), "ET_EE*": (Db * E).std(0) / np.sqrt(n),
        "NEE*": (Db * N).mean(0), "ET_NEE*": (Db * N).std(0) / np.sqrt(n),
        "EEPE": float(np.sum(eee[un_an] * dt[un_an]) / np.sum(dt[un_an])),
        "E": E, "N": N, "D": Db,  # conservés pour la CVA incrémentale et l'allocation (étape 4)
        "E0": max(V0 - C0, 0.0), "N0": max(C0 - V0, 0.0),
    }

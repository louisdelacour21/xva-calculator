"""IRS EURIBOR 6M et OIS €STR à prix fermé par chemin, vectorisés chemins × trades.

Valeur d'un trade = combinaison linéaire de P(t, T_k) : à chaque date, une matrice de ZC (chemins × échéances)
multipliée par une matrice de poids (échéances × trades), plus le coupon flottant en cours, fixé sur le chemin.
Convention : à la date t, seuls les flux payés strictement après t comptent (ex-coupon).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .hw1f import HW1F
from .market import Marche


@dataclass
class Trade:
    nom: str
    type: str            # "irs" (EURIBOR 6M) ou "ois" (€STR composé)
    sens: int            # +1 payeur du fixe, -1 receveur
    nominal: float
    K: float
    debut: float         # >= 0 ; shortcut: pas de trade déjà vivant (fixings passés), à ajouter avec les fixings de la 09
    fixe: np.ndarray     # dates de paiement fixe (années ACT/365F)
    flot: np.ndarray     # dates de paiement flottant

    @property
    def maturite(self) -> float:
        return float(max(self.fixe[-1], self.flot[-1]))

    def periodes_flot(self):
        return np.r_[self.debut, self.flot[:-1]], self.flot

    def flux(self) -> np.ndarray:
        return np.unique(np.r_[self.fixe, self.flot])


def swap(nom, type, sens, nominal, K, debut, tenor) -> Trade:
    """Échéanciers réguliers : fixe annuel ; flottant semestriel (IRS) ou annuel (OIS)."""
    ff = 2 if type == "irs" else 1
    return Trade(nom, type, sens, nominal, K, debut,
                 debut + np.arange(1, round(tenor) + 1, dtype=float),
                 debut + np.arange(1, round(tenor * ff) + 1) / ff)


def charger_portefeuille(path: Path):
    """(trades, csa) depuis un YAML ; csa en euros et en jours ouvrés de MPoR."""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sens = {"payeur": 1, "receveur": -1}
    trades = [swap(q["nom"], q["type"], sens[q["sens"]], float(q["nominal"]), float(q["K"]),
                   float(q["debut"]), float(q["tenor"])) for q in d["trades"]]
    return trades, d["csa"]


def _beta(trade: Trade, marche: Marche) -> np.ndarray:
    s, e = trade.periodes_flot()
    if trade.type == "ois":
        return np.ones(len(e))
    return (marche.euribor6m(s) / marche.euribor6m(e)) / (marche.estr(s) / marche.estr(e))


def valeur(trades, modele: HW1F, marche: Marche, t: float, x_t, D_t, etat=None) -> np.ndarray:
    """Interface value(t, state) : MtM (chemins × trades) à la date t.

    x_t, D_t : x(t) et D(0, t) par chemin ; etat(s) -> (x(s), D(0, s)) aux dates de fixing passées
    (nécessaire seulement pour un coupon en cours, donc jamais à t = 0).
    """
    x_t, D_t = np.atleast_1d(x_t), np.atleast_1d(D_t)
    ech = np.unique(np.concatenate([np.r_[tr.debut, tr.flux()] for tr in trades]))
    ech = ech[ech >= t]
    W = np.zeros((len(ech), len(trades)))
    col = {u: i for i, u in enumerate(ech)}
    courants = []
    for j, tr in enumerate(trades):
        w = tr.sens * tr.nominal
        dk = np.diff(np.r_[tr.debut, tr.fixe])
        for T, d in zip(tr.fixe, dk):
            if T > t:
                W[col[T], j] -= w * tr.K * d
        beta = _beta(tr, marche)
        for s, e, b in zip(*tr.periodes_flot(), beta):
            if s >= t:
                W[col[s], j] += w * b
                W[col[e], j] -= w
            elif e > t:
                courants.append((j, s, e, b, w))
    PU = modele.P(t, ech[None, :], x_t[:, None]) if len(ech) else np.zeros((len(x_t), 0))
    V = PU @ W
    for j, s, e, b, w in courants:
        x_s, D_s = etat(s)
        if trades[j].type == "irs":
            # δ·L payé en e, L fixé en s sur le chemin : 1 + δ·L = β / P(s, e)
            V[:, j] += w * (b / modele.P(s, e, x_s) - 1) * PU[:, col[e]]
        else:
            # €STR composé payé en e : vaut en t l'accumulé exp(∫_s^t r) = D(0,s)/D(0,t), sans actualisation
            # (la fiche écrit e^{∫r}·P(t,T), ce qui est faux : détecté par test_martingale_mtm)
            V[:, j] += w * (D_s / D_t - PU[:, col[e]])
    return V


def cube_mtm(trades, modele: HW1F, marche: Marche, sim: dict) -> np.ndarray:
    """Cube MtM (chemins × dates × trades) sur la grille de la simulation, qui doit contenir les dates de fixing."""
    t = sim["t"]

    def etat(s):
        if s == 0:
            n = sim["x"].shape[0]
            return np.zeros(n), np.ones(n)
        k = indice(t, s)
        return sim["x"][:, k], sim["D"][:, k]

    return np.stack([valeur(trades, modele, marche, u, sim["x"][:, k], sim["D"][:, k], etat)
                     for k, u in enumerate(t)], axis=1)


def indice(grille, u) -> int:
    k = int(np.searchsorted(grille, u - 1e-9))
    assert k < len(grille) and abs(grille[k] - u) < 1e-9, f"date {u} absente de la grille"
    return k

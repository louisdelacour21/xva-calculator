"""Données de marché : courbes, vols de swaptions, historique BCE.

Interface de courbe minimale : un appelable t -> P(0, t), t en années ACT/365F (scalaire ou tableau numpy).
Le moteur ne reconstruit aucune courbe : en production elles viennent de la fiche 09 via depuis_courbe_09(),
en tests d'une CourbePlate.
"""
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

RACINE = Path(__file__).resolve().parents[2] / "market"
Courbe = Callable[[np.ndarray], np.ndarray]


class CourbePlate:
    """Taux zéro continu constant : courbe de secours pour les tests unitaires."""

    def __init__(self, taux: float):
        self.taux = taux

    def __call__(self, t):
        return np.exp(-self.taux * np.asarray(t, dtype=float))


def depuis_courbe_09(courbe, jour=None) -> Courbe:
    """Point de greffe : enveloppe une Courbe / CourbeHW du paquet courbes-taux (méthode P(date), ACT/365F)."""
    jour = jour or courbe.jour
    # shortcut: t arrondi au jour (erreur < 0,5 jour), suffisant tant que la grille de simulation est en jours
    une = np.vectorize(lambda u: courbe.P(jour + timedelta(days=round(u * 365))), otypes=[float])
    return lambda t: une(np.asarray(t, dtype=float))


@dataclass
class Marche:
    estr: Courbe        # actualisation (CSA cash EUR)
    euribor6m: Courbe   # projection des fixings 6M


def marche_test(taux: float = 0.025, base: float = 0.002) -> Marche:
    """Deux courbes plates séparées d'une base : exerce le multi-courbes sans la fiche 09."""
    return Marche(CourbePlate(taux), CourbePlate(taux + base))


def vols_swaptions(path: Path = RACINE / "swaption_vols.yaml") -> list[tuple[float, float, float]]:
    """(expiry, ténor, vol normale en décimal) des swaptions ATM co-terminales."""
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [(float(q["expiry"]), float(q["tenor"]), q["vol_bp"] * 1e-4) for q in d["swaptions"]]


def historique_bce(maturites=(1, 5, 10)) -> pd.DataFrame:
    """Taux spot BCE (courbe AAA Svensson) en décimal, une colonne par maturité en années. Mesure P uniquement."""
    cols = {}
    for m in maturites:
        f = RACINE / "bce" / f"YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{m}Y.csv"
        s = pd.read_csv(f, index_col="TIME_PERIOD")["OBS_VALUE"]
        cols[m] = s / 100
    return pd.DataFrame(cols).dropna()

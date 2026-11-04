"""Validation : convergence Monte Carlo, étude P contre Q, backtest point-in-time (PIT) des expositions sous P.

Backtest : à chaque date de départ d0 de l'historique BCE, σ(P) calibré sur l'année qui précède (pas de look-ahead),
loi de V(d0 + h) simulée sous HW1F à partir de la courbe du jour, puis PIT = F_modèle(V réalisée), V réalisée
étant la valeur des mêmes flux sur la courbe observée en d0 + h. Fenêtres sans chevauchement (pas = horizon).
"""
import time

import numpy as np
import pandas as pd
from scipy import stats

from .exposure import exposition, grille
from .hw1f import HW1F, calibrer_p, calibrer_q, taux_swap
from .market import RACINE, Marche
from .products import cube_mtm, swap, valeur
from .xva import Credit, bootstrap, contributions, erreur_type

MATURITES_BCE = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
HORIZONS = {"1 month": 21, "3 months": 63, "1 year": 252}  # en jours ouvrés de l'historique


def chemins_cva(trades, modele, marche, credit, n, graine=0, csa=None, antithetiques=True):
    """Contribution de chaque chemin à la CVA (leur moyenne est la CVA)."""
    base, t = grille(trades, mpors=(csa["mpor"],) if csa else ())
    sim = modele.simuler(t, n, graine=graine, antithetiques=antithetiques)
    V0 = valeur(trades, modele, marche, 0.0, 0.0, 1.0)[0].sum()
    V = cube_mtm(trades, modele, marche, sim).sum(-1)
    return contributions(exposition(V, sim["D"], t, base, V0, csa), credit)["CVA"]


def convergence(trades, modele, marche, credit, tailles=(1_000, 2_000, 5_000, 10_000, 20_000, 50_000)):
    """CVA, erreur-type avec et sans antithétiques, temps ; gain = rapport des variances (chemins économisés)."""
    lignes = []
    for n in tailles:
        t0 = time.perf_counter()
        c = chemins_cva(trades, modele, marche, credit, n)
        sec = time.perf_counter() - t0
        b = chemins_cva(trades, modele, marche, credit, n, graine=1, antithetiques=False)
        et, et_brut = erreur_type(c), float(b.std(ddof=1) / np.sqrt(n))
        lignes.append({"N": n, "CVA": float(c.mean()), "ET": et, "ET sans antithétiques": et_brut,
                       "gain antithétique": (et_brut / et) ** 2, "secondes": sec})
    df = pd.DataFrame(lignes)
    pente = np.polyfit(np.log(df["N"]), np.log(df["ET"]), 1)[0]
    return df, float(pente)


def etude_pq(trades, marche, cotations, credit, sigma_p, n=10_000, a=0.03):
    """Même CVA (mêmes tirages) sous σ implicite, σ historique, vols ±20 %, a = 0,01 / 0,05, spreads ±25 %."""
    q = lambda c=cotations, aa=a: calibrer_q(marche.estr, marche.euribor6m, c, aa)[0]
    vol = lambda s: [(T, m, v * s) for T, m, v in cotations]
    s = credit.contrepartie
    spread = lambda f: Credit(bootstrap({T: v * f for T, v in s.spreads.items()}, marche.estr, s.R), credit.banque,
                              credit.sF)
    ref = q()
    cas = {"Q, implied σ (reference)": (ref, credit), "P, historical σ": (HW1F(a, sigma_p, marche.estr), credit),
           "vols +20%": (q(vol(1.2)), credit), "vols −20%": (q(vol(0.8)), credit),
           "a = 0.01": (q(aa=0.01), credit), "a = 0.05": (q(aa=0.05), credit),
           "spreads +25%": (ref, spread(1.25)), "spreads −25%": (ref, spread(0.75))}
    r = {}
    for nom, (m, cr) in cas.items():
        c = chemins_cva(trades, m, marche, cr, n)
        r[nom] = {"σ": m.sigma, "CVA": float(c.mean()), "ET": erreur_type(c)}
    for v in r.values():
        v["écart"] = v["CVA"] / r["Q, implied σ (reference)"]["CVA"] - 1
    return pd.DataFrame(r).T


# --- backtest PIT ----------------------------------------------------------------------------------------------

def historique_complet() -> pd.DataFrame:
    """Taux spot BCE (AAA, Svensson, composition continue) en décimal depuis 2004, colonnes = maturités en années."""
    cols = {}
    for m in MATURITES_BCE:
        s = pd.read_csv(RACINE / "bce_historique" / f"SR_{m}.csv", index_col="TIME_PERIOD", parse_dates=True)
        cols[int(m[:-1]) / (12 if m[-1] == "M" else 1)] = s["OBS_VALUE"] / 100
    return pd.DataFrame(cols).dropna()


def courbe_bce(ligne: pd.Series):
    """t -> P(0, t) : taux zéro interpolés linéairement entre maturités, plats aux extrémités."""
    tau, r = np.array(ligne.index, dtype=float), ligne.to_numpy()
    return lambda t: np.exp(-np.interp(t, tau, r) * np.asarray(t, dtype=float))


def portefeuilles_test(h: float, P) -> dict:
    """Trois netting sets démarrant à l'horizon, strikes au pair forward en d0.

    shortcut: mono-courbe (AAA BCE pour l'actualisation et la projection), seul historique gratuit et long ;
    démarrage à l'horizon pour qu'aucun fixing ne tombe dans la fenêtre.
    """
    atm = lambda n: taux_swap(h, n, P, P)[0]
    return {"payer 5Y": [swap("A", "irs", 1, 10e6, atm(5), h, 5)],
            "receiver 10Y": [swap("B", "irs", -1, 10e6, atm(10), h, 10)],
            "steepener 2Y/10Y": [swap("C1", "irs", 1, 10e6, atm(10), h, 10),
                                     swap("C2", "irs", -1, 45e6, atm(2), h, 2)]}


def pit(P0, P1, h: float, sigma: float, a: float = 0.03, n: int = 2_000, graine: int = 0) -> dict:
    """PIT de chaque portefeuille test : part des V simulées (HW1F, courbe P0, horizon h) sous la V réalisée (P1)."""
    m = HW1F(a, sigma, P0)
    x = m.simuler([h], n, graine=graine)["x"][:, 0]
    out = {}
    for nom, tr in portefeuilles_test(h, P0).items():
        v_sim = valeur(tr, m, Marche(P0, P0), h, x, np.ones(n)).sum(1)
        vus = [swap(q.nom, q.type, q.sens, q.nominal, q.K, 0.0, q.maturite - q.debut) for q in tr]
        v_real = valeur(vus, HW1F(a, sigma, P1), Marche(P1, P1), 0.0, 0.0, 1.0)[0].sum()
        out[nom] = float((v_sim < v_real).mean() + 0.5 * (v_sim == v_real).mean())
    return out


def puissance_ks(n_obs: int, facteur: float = 1.25, niveau: float = 0.05, essais: int = 2_000, graine: int = 0):
    """Probabilité que le KS rejette quand la vraie dispersion vaut facteur × celle du modèle (loi gaussienne)."""
    rng = np.random.default_rng(graine)
    u = stats.norm.cdf(facteur * rng.standard_normal((essais, n_obs)))
    return float(np.mean([stats.kstest(x, "uniform").pvalue < niveau for x in u]))


def backtest_pit(hist: pd.DataFrame, horizons=HORIZONS, fenetre: int = 252, a: float = 0.03, n: int = 2_000):
    """PIT par horizon et portefeuille (fenêtres sans chevauchement) et tableau KS avec puissance."""
    pits, lignes = {}, []
    cal = hist[[1.0, 5.0, 10.0]]
    for nom_h, k in horizons.items():
        res = []
        for i in range(fenetre, len(hist) - k, k):
            d0, d1 = hist.index[i], hist.index[i + k]
            sigma = calibrer_p(cal.iloc[i - fenetre:i + 1], a)
            res.append(pit(courbe_bce(hist.iloc[i]), courbe_bce(hist.iloc[i + k]), (d1 - d0).days / 365,
                           sigma, a, n, graine=i) | {"date": d0})
        df = pd.DataFrame(res).set_index("date")
        pits[nom_h] = df
        puiss = puissance_ks(len(df))
        for p in df.columns:
            lignes.append({"horizon": nom_h, "portefeuille": p, "fenêtres": len(df),
                           "KS": stats.kstest(df[p], "uniform").statistic,
                           "p-valeur": stats.kstest(df[p], "uniform").pvalue,
                           "puissance (vol ×1,25)": puiss, "indicatif": puiss < 0.5})
    return pits, pd.DataFrame(lignes)

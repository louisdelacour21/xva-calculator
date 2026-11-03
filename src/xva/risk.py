"""Sensibilités des XVA (bump & reprice, nombres aléatoires communs) et capital : SA-CCR taux, BA-CVA réduit.

Deltas : chocs triangulaires de 1 bp sur les taux zéro aux piliers 1, 2, 5, 10, 30 ans (somme = choc parallèle),
centrés, mêmes tirages, σ recalibré ; vega : vols de swaptions ±1 % (relatif) ; CS01 : spreads de la contrepartie
±1 bp par pilier, mêmes expositions (le crédit n'entre pas dans la simulation).
"""
import numpy as np

from .exposure import exposition, grille
from .hw1f import calibrer_q
from .market import Marche
from .products import cube_mtm, valeur
from .xva import Credit, bootstrap, contributions, erreur_type

PILIERS = np.array([1.0, 2.0, 5.0, 10.0, 30.0])


def triangle(k: int, piliers=PILIERS):
    """Poids du pilier k : 1 au pilier, linéaire vers les voisins, plat au-delà des extrêmes."""
    return lambda t: np.interp(t, piliers, np.eye(len(piliers))[k])


def choc_taux(marche: Marche, forme, h: float) -> Marche:
    """Les deux courbes choquées de h·forme(t) en taux zéro continu (choc de taux, base inchangée)."""
    def f(P):
        return lambda t: P(t) * np.exp(-h * forme(np.asarray(t, dtype=float)) * np.asarray(t, dtype=float))
    return Marche(f(marche.estr), f(marche.euribor6m))


def _rebootstrap(credit: Credit, P) -> Credit:
    """Spreads inchangés, survies recalculées sur la courbe d'actualisation choquée."""
    rb = lambda s: s if s.spreads is None else bootstrap(s.spreads, P, s.R)
    return Credit(rb(credit.contrepartie), rb(credit.banque), credit.sF)


def reevaluer(marche, cotations, trades, base, t, n, graine, credit, csa) -> dict:
    """Chaîne complète : calibration Q, simulation (graine fixe), cube, exposition, contributions par chemin."""
    modele, _ = calibrer_q(marche.estr, marche.euribor6m, cotations)
    sim = modele.simuler(t, n, graine=graine)
    cube = cube_mtm(trades, modele, marche, sim)
    V0 = valeur(trades, modele, marche, 0.0, 0.0, 1.0)[0].sum()
    return contributions(exposition(cube.sum(-1), sim["D"], t, base, V0, csa), _rebootstrap(credit, marche.estr))


def _centre(haut: dict, bas: dict) -> dict:
    """(haut - bas)/2 chemin par chemin : moyenne et erreur-type (les tirages communs annulent l'essentiel du bruit)."""
    return {k: (float(((haut[k] - bas[k]) / 2).mean()), erreur_type((haut[k] - bas[k]) / 2)) for k in haut}


def sensibilites(trades, marche, cotations, credit, csa=None, n=10_000, graine=0, h=1e-4, dvol=0.01) -> dict:
    """{choc: {ajustement: (sensibilité, ET)}} pour +1 bp par pilier et en parallèle, et +1 % des vols."""
    base, t = grille(trades, mpors=(csa["mpor"],) if csa else ())
    rev = lambda m, c: reevaluer(m, c, trades, base, t, n, graine, credit, csa)
    chocs = {f"delta {p:g}Y": triangle(k) for k, p in enumerate(PILIERS)}
    chocs["delta parallèle"] = np.ones_like
    out = {nom: _centre(rev(choc_taux(marche, f, h), cotations), rev(choc_taux(marche, f, -h), cotations))
           for nom, f in chocs.items()}
    vol = lambda s: [(T, m, v * (1 + s)) for T, m, v in cotations]
    out["vega +1 %"] = _centre(rev(marche, vol(dvol)), rev(marche, vol(-dvol)))
    return out


def cs01(expo: dict, credit: Credit, P, h: float = 1e-4) -> dict:
    """CS01 par pilier de spread de la contrepartie et en parallèle, sans resimulation."""
    s = credit.contrepartie
    rev = lambda d: contributions(expo, Credit(bootstrap(d, P, s.R), credit.banque, credit.sF))
    choc = lambda piliers, x: {T: v + x * (T in piliers) for T, v in s.spreads.items()}
    out = {f"CS01 {T:g}Y": _centre(rev(choc([T], h)), rev(choc([T], -h))) for T in s.spreads}
    out["CS01 parallèle"] = _centre(rev(choc(list(s.spreads), h)), rev(choc(list(s.spreads), -h)))
    return out


# --- SA-CCR, classe taux (CRE52) -------------------------------------------------------------------------------

ALPHA, SF_TAUX, PLANCHER = 1.4, 0.005, 0.05


def facteur_duree(S: float, E: float) -> float:
    """SD = (e^{-0,05 S} - e^{-0,05 E}) / 0,05 (CRE52.34), S et E en années."""
    return (np.exp(-0.05 * S) - np.exp(-0.05 * E)) / 0.05


def sa_ccr(trades, V0: float, csa: dict | None = None) -> dict:
    """EAD SA-CCR d'un netting set de swaps EUR (un seul hedging set). csa : {H_C, H_B, MTA, mpor_jours}.

    δ = +1 pour un payeur du fixe (gagne quand les taux montent) ; buckets par maturité < 1, 1-5, > 5 ans.
    """
    def addon(mf):
        D = np.zeros(3)
        for tr in trades:
            b = 0 if tr.maturite < 1 else (1 if tr.maturite <= 5 else 2)
            D[b] += tr.sens * tr.nominal * facteur_duree(tr.debut, tr.maturite) * mf(tr)
        en = np.sqrt(D @ D + 1.4 * D[0] * D[1] + 1.4 * D[1] * D[2] + 0.6 * D[0] * D[2])
        return D, SF_TAUX * en

    def ead(rc, C, a):
        mult = min(1.0, PLANCHER + (1 - PLANCHER) * np.exp((V0 - C) / (2 * (1 - PLANCHER) * a)))
        return ALPHA * (rc + mult * a), mult

    D, a = addon(lambda tr: np.sqrt(min(max(tr.maturite, 10 / 250), 1.0)))  # MF non margé (CRE52.48)
    e_nm, mult = ead(max(V0, 0.0), 0.0, a)
    r = {"D": D, "add-on": a, "RC": max(V0, 0.0), "multiplicateur": mult, "EAD": e_nm, "EAD non margée": e_nm}
    if csa is None:
        return r
    hc, hb = csa["H_C"] + csa["MTA"], csa["H_B"] + csa["MTA"]
    C = max(V0 - hc, 0) - max(-V0 - hb, 0)  # marge de variation détenue aujourd'hui, pas de marge initiale
    D, a = addon(lambda tr: 1.5 * np.sqrt(csa["mpor_jours"] / 250))  # MF margé (CRE52.52)
    rc = max(V0 - C, csa["H_C"] + csa["MTA"], 0.0)  # CRE52.18, NICA = 0
    e_m, mult = ead(rc, C, a)
    return r | {"D": D, "add-on": a, "RC": rc, "multiplicateur": mult, "EAD": min(e_m, e_nm)}  # plafond CRE52.2


# --- BA-CVA réduit (MAR50) --------------------------------------------------------------------------------------

# MAR50.16, tableau 1 : (secteur, IG ou HY/NR) -> RW. shortcut: recopié pour les secteurs utilisés, à revérifier
# contre le texte avant publication.
RW_MAR50 = {("souverain", "IG"): 0.005, ("souverain", "HY"): 0.02, ("financier", "IG"): 0.05,
            ("financier", "HY"): 0.12, ("industrie", "IG"): 0.03, ("industrie", "HY"): 0.07,
            ("autre", "IG"): 0.05, ("autre", "HY"): 0.12}


def maturite_effective(trades) -> float:
    # shortcut: maturité résiduelle pondérée par les nominaux, plancher 1 an, au lieu de la duration des flux (CRE32.47)
    return max(1.0, sum(tr.nominal * tr.maturite for tr in trades) / sum(tr.nominal for tr in trades))


def scva(rw: float, netting_sets) -> float:
    """SCVA_c = RW_c · Σ M·EAD·DF / α, DF = (1 - e^{-0,05 M}) / (0,05 M) ; netting_sets = [(M, EAD)]."""
    return rw * sum(M * e * (1 - np.exp(-0.05 * M)) / (0.05 * M) for M, e in netting_sets) / ALPHA


def ba_cva_reduit(scvas, rho: float = 0.5, ds: float = 0.65) -> float:
    """K = DS · √[(ρ Σ SCVA)² + (1 - ρ²) Σ SCVA²] (MAR50.14) ; la CVA réglementaire exclut la DVA."""
    s = np.asarray(scvas, dtype=float)
    return float(ds * np.sqrt((rho * s.sum()) ** 2 + (1 - rho**2) * np.sum(s**2)))

"""Tableau XVA du portefeuille démo, avec et sans CSA, allocation d'Euler et pré-trade : python -m xva.demo"""
import time
from pathlib import Path

from .exposure import exposition, grille
from .hw1f import calibrer_q
from .market import marche_test, vols_swaptions
from .products import charger_portefeuille, cube_mtm, swap, valeur
from .xva import ajustements, allocation_euler, charger_credit, pre_trade

PORTEFEUILLE = Path(__file__).resolve().parents[2] / "portfolios" / "demo.yaml"


def main(n: int = 10_000, graine: int = 0):
    marche = marche_test()
    modele, _ = calibrer_q(marche.estr, marche.euribor6m, vols_swaptions())
    credit = charger_credit(P=marche.estr)
    trades, c = charger_portefeuille(PORTEFEUILLE)
    csa = {"H_C": c["H_C"], "H_B": c["H_B"], "MTA": c["MTA"], "mpor": c["mpor_jours"] * 14 / 10 / 365}
    base, t = grille(trades, mpors=(csa["mpor"],))
    t0 = time.perf_counter()
    sim = modele.simuler(t, n, graine=graine)
    cube = cube_mtm(trades, modele, marche, sim)
    print(f"σ(Q) = {modele.sigma:.4%}, {n} chemins, {len(t)} dates, cube en {time.perf_counter() - t0:.1f} s")
    V0s = valeur(trades, modele, marche, 0.0, 0.0, 1.0)[0]
    V, V0 = cube.sum(-1), V0s.sum()
    print(f"V0 = {V0 / 1e6:,.2f} M€")
    for nom, k in (("sans CSA", None), ("CSA", csa)):
        r = ajustements(exposition(V, sim["D"], t, base, V0, k), credit, V0)
        print(f"\n{nom}\n" + "\n".join(f"  {q:10s} {v / 1e3:10,.1f} k€  (ET {e / 1e3:,.1f})" for q, (v, e) in r.items()))
    alloc = allocation_euler(cube, sim["D"], t, base, V0s, credit)
    print("\nAllocation d'Euler de la CVA (sans CSA) :")
    for tr, a in sorted(zip(trades, alloc), key=lambda z: -z[1]):
        print(f"  {tr.nom:7s} {a / 1e3:10,.1f} k€")
    nouveau = swap("NOUVEAU", "irs", 1, 50e6, 0.026, 0.0, 10)
    for nom, k in (("sans CSA", None), ("CSA", csa)):
        r = pre_trade(V, V0, nouveau, modele, marche, sim, base, credit, k)
        d = "  ".join(f"Δ{q} {v / 1e3:,.1f} k€ (ET {e / 1e3:,.1f})" for q, (v, e) in r["Δ"].items())
        print(f"\nPré-trade {nouveau.nom} payeur 50 M€ 10 ans 2,60 %, {nom} : {d}  en {r['secondes']:.2f} s")


if __name__ == "__main__":
    main()

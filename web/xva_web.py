"""Même calcul que app.py, pour la page statique (Pyodide dans un Web Worker) : demarrer() une fois, calculer(p) à chaque réglage."""
from xva.demo import PORTEFEUILLE
from xva.exposure import exposition, grille
from xva.hw1f import calibrer_q, taux_swap
from xva.market import marche_test, vols_swaptions
from xva.products import charger_portefeuille, cube_mtm, swap, valeur
from xva.xva import ajustements, allocation_euler, charger_credit, pre_trade

MPORS = {5: 7 / 365, 10: 14 / 365, 20: 28 / 365}
N = 10_000
_s = {}


def demarrer(n=N):
    marche = marche_test()
    modele, _ = calibrer_q(marche.estr, marche.euribor6m, vols_swaptions())
    trades, c = charger_portefeuille(PORTEFEUILLE)
    base, t = grille(trades, mpors=tuple(MPORS.values()))
    sim = modele.simuler(t, n, graine=0)
    cube = cube_mtm(trades, modele, marche, sim)
    V0s = valeur(trades, modele, marche, 0.0, 0.0, 1.0)[0]
    credit = charger_credit(P=marche.estr)
    alloc = allocation_euler(cube, sim["D"], t, base, V0s, credit)
    V, V0 = cube.sum(-1), float(V0s.sum())
    _s.update(marche=marche, modele=modele, credit=credit, base=base, sim=sim, V=V, V0=V0)
    e0 = exposition(V, sim["D"], sim["t"], base, V0, None)
    return {"N": n, "sigma": float(modele.sigma), "V0": V0, "base": [float(b) for b in base],
            "atm": {T: float(taux_swap(0.0, T, marche.estr, marche.euribor6m)[0]) for T in range(1, 11)},
            "csa": {"seuil": c["H_C"] / 1e6, "mta": c["MTA"] / 1e6, "mpor": c["mpor_jours"]},
            "EE0": e0["EE"].tolist(), "PFE0": e0["PFE"].tolist(),
            "alloc": [[tr.nom, float(a)] for tr, a in zip(trades, alloc)]}


def calculer(p):
    """p : avec_csa, seuil, mta (M€), mpor (jours), sens, nominal (M€), tenor (ans), K (décimal)."""
    s = _s
    csa = ({"H_C": p["seuil"] * 1e6, "H_B": p["seuil"] * 1e6, "MTA": p["mta"] * 1e6, "mpor": MPORS[p["mpor"]]}
           if p["avec_csa"] else None)
    expo = exposition(s["V"], s["sim"]["D"], s["sim"]["t"], s["base"], s["V0"], csa)
    nouveau = swap("NOUVEAU", "irs", 1 if p["sens"] == "payeur" else -1, p["nominal"] * 1e6, p["K"], 0.0, p["tenor"])
    pt = pre_trade(s["V"], s["V0"], nouveau, s["modele"], s["marche"], s["sim"], s["base"], s["credit"], csa)
    return {"xva": ajustements(expo, s["credit"], s["V0"]), "delta": pt["Δ"], "V0_new": pt["V0"],
            "secondes": pt["secondes"], "EE": expo["EE"].tolist(), "PFE": expo["PFE"].tolist(),
            "EEPE": float(expo["EEPE"])}


if __name__ == "__main__":
    import json
    d = demarrer(2_000)
    r = calculer({"avec_csa": True, **d["csa"], "sens": "payeur", "nominal": 50, "tenor": 10, "K": d["atm"][10]})
    json.dumps([d, r])  # tout doit passer en JSON vers la page
    assert len(r["EE"]) == len(d["base"]) and r["xva"]["CVA"][0] > 0
    print("ok", round(r["xva"]["CVA"][0] / 1e3, 1), "k€")

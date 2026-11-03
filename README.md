# Calculateur XVA : CVA, DVA, FVA d'un netting set de swaps EUR

Moteur Monte Carlo Hull-White 1F vectorisé : profils d'exposition collatéralisés, CVA/DVA/FVA,
CVA incrémentale pré-trade et sensibilités, validé contre QuantLib et la formule fermée des swaptions.

Fiche projet : `projets-cv/fiches/pdf/05-calculateur-xva.pdf` (dossier Application Linkedin).

## Avancement

| # | Étape | État |
|---|-------|------|
| 1 | Dépôt, interface de courbe, HW1F, simulation exacte | fait (courbes plates en attendant la 09) |
| 2 | Jamshidian, calibration Q, comparaison P | fait |
| — | Greffe des courbes de la fiche 09 | en attente de la 09 |
| 3 | IRS 6M / OIS vectorisés ; grille, netting, CSA, métriques | à faire |
| 4 | Crédit, CVA/DVA/FVA, incrémentale | à faire |
| 5 | Sensibilités CRN, SA-CCR, BA-CVA | à faire |
| 6 | Backtest PIT, convergence, étude P/Q | à faire |
| 7 | Streamlit, GIF ; rapport, README, LinkedIn | à faire |

## Courbes et greffe de la fiche 09 (`src/xva/market.py`)

Le moteur ne connaît qu'une interface : un appelable `t -> P(0, t)`, `t` en années ACT/365F.
Un objet `Marche` porte deux courbes, `estr` (actualisation) et `euribor6m` (projection).

- En attendant la 09 : `marche_test()` donne deux courbes plates séparées d'une base de 20 bp,
  ce qui exerce déjà le multi-courbes (β ≠ 1).
- À la greffe : `depuis_courbe_09(courbe)` enveloppe une `Courbe` / `CourbeHW` du paquet `courbes-taux`
  (méthode `P(date)`). Il suffira de construire
  `Marche(depuis_courbe_09(estr), depuis_courbe_09(euribor6m))` à partir de `courbes.curves.courbe_estr`
  et `courbe_euribor6m`, puis d'ajouter `courbes-taux` aux dépendances.
- Reste à faire à ce moment-là : remplacer les échéanciers réguliers en années de `swap_coupons`
  par les dates TARGET2 de la 09 (marqué `shortcut:` dans le code).

## Hull-White 1F (`src/xva/hw1f.py`)

- Prix ZC fermé `P(t, T | x)`, simulation exacte jointe de `(x, ∫x)` sur grille irrégulière,
  facteur d'actualisation `D(0, t) = P(0, t)·exp(-I - V/2)` donc `E[D] = P(0, t)` sans biais,
  variables antithétiques.
- Jamshidian valable pour des coupons de signe quelconque (la base 6M/€STR crée de petits coupons négatifs).
- Calibration Q : `a = 0,03` fixé, `σ` ajusté en vol normale sur les swaptions ATM co-terminales 10 ans
  (`market/swaption_vols.yaml`, **synthétique** faute de source gratuite).
- Calibration P (comparaison uniquement) : `σ·B(τ)/τ` ajusté aux écarts-types annualisés des variations
  quotidiennes des taux spot BCE 1, 5 et 10 ans (un an d'historique, `market/bce/`).

Premiers chiffres (courbes plates, vols synthétiques) : σ(Q) = 0,84 %, écart de vol max 3,6 bp ;
σ(P) = 0,58 %, soit **-31 %** : une CVA calibrée sur l'historique serait sous-estimée d'environ autant.

## Tests

```bash
pip install -e .[test]
pytest
```

11 tests : martingale `E[D] = P(0, t)`, moments de `x` et `∫x`, cohérence ZC futur, option sur ZC et
swaption Jamshidian contre QuantLib 1.44, Jamshidian contre Monte Carlo en multi-courbes, parité
payeuse/receveuse, calibrations Q et P retrouvant un σ connu, adaptateur de la fiche 09.

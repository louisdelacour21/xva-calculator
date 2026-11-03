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
| 3 | IRS 6M / OIS vectorisés ; grille, netting, CSA, métriques | fait |
| 4 | Crédit, CVA/DVA/FVA, incrémentale | fait |
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

## Produits (`src/xva/products.py`)

- IRS EURIBOR 6M (base 6M/€STR déterministe) et OIS €STR, payeur ou receveur, démarrage immédiat ou à terme.
- `valeur(trades, modèle, marché, t, x_t, D_t, état)` : l'interface `value(t, state)` où se branchera
  le pricer neuronal. Chaque date = une matrice de ZC (chemins × échéances) × une matrice de poids
  (échéances × trades) ; seul le coupon en cours, fixé sur le chemin, est ajouté trade par trade.
- `cube_mtm` : cube MtM chemins × dates × trades, conservé pour la CVA incrémentale.
- Correction de la fiche : la jambe €STR en cours vaut `exp(∫_s^t r) − P(t, e)`, pas `exp(∫r)·P(t, e) − P(t, e)`
  (erreur détectée par le test de martingale).
- Pas encore de trade déjà vivant (fixings passés) : à ajouter avec les fixings de la 09.

## Exposition (`src/xva/exposure.py`)

- Grille : mensuelle ∪ dates de flux ∪ nœuds `t − MPoR`.
- Collatéral classique `C(t) = max(V(t−M) − H_C, 0) − max(−V(t−M) − H_B, 0)`, MTA ajouté aux seuils
  (approximation écrite), flux pendant la MPoR ignorés.
- EE, NEE, PFE 97,5 %, EE* et NEE* actualisés avec erreurs-types, EEPE (EE effective sur un an).

Portefeuille démo (`portfolios/demo.yaml`, 7 swaps fictifs, 435 M€ de nominal, σ calibré Q, 10 000 chemins,
240 dates, cube en 4 s sur le PC) :

| | EEPE | EE max | PFE max |
|---|---|---|---|
| Sans CSA | 6,2 M€ | 9,3 M€ | 36,8 M€ |
| CSA (H = 10 M€, MTA 0,5 M€, MPoR 10 j) | 5,3 M€ | 5,4 M€ | 13,9 M€ |

## Crédit et XVA (`src/xva/xva.py`)

- Survie : λ constante par morceaux bootstrapée sur les spreads CDS 1, 3, 5, 7, 10 ans, R = 40 %,
  convention MidPoint (défaut au milieu de la période, coupon couru payé au défaut), CDS trimestriels réguliers
  (pas de dates IMM). Spreads **synthétiques** dans `market/spreads.yaml` (contrepartie BBB, banque A).
- CVA, DVA, FCA, FBA selon les formules de la fiche, EE* au milieu de chaque intervalle par trapèze.
  Financement : `ℓ = max(s_F − (1 − R_B)·λ_B, 0)`, donc la FBA ne recompte pas la DVA.
- Erreurs-types calculées par paire antithétique.
- `pre_trade` : ΔXVA d'un nouveau trade sur les mêmes chemins, seul son MtM est calculé.
- `allocation_euler` : CVA marginale par trade (sans collatéral), somme exacte = CVA.

```bash
python -m xva.demo
```

Portefeuille démo, σ calibré Q, 10 000 chemins (k€, erreur-type entre parenthèses) :

| | CVA | DVA | FCA | FBA | V ajustée (V0 = 5 090) |
|---|---|---|---|---|---|
| Sans CSA | 659,7 (3,2) | 227,8 (2,4) | 81,2 (0,3) | 34,2 (0,4) | 4 609,1 |
| CSA | 446,3 (0,7) | 159,5 (1,1) | 57,0 (< 0,1) | 23,5 (0,2) | 4 767,7 |

Le CSA retire 32 % de la CVA. Allocation d'Euler (sans CSA) : IRS-01 394 k€, IRS-03 126 k€, IRS-05 103 k€.
Pré-trade d'un IRS payeur 50 M€ 10 ans à 2,60 % : ΔCVA = 146,5 k€ (ET 0,9) sans CSA, 28,0 k€ (ET 0,2) avec CSA,
en 1,4 à 1,6 s.

## Tests

```bash
pip install -e .[test]
pytest
```

30 tests, environ 30 s :
- HW1F : martingale `E[D] = P(0, t)`, moments de `x` et `∫x`, cohérence ZC futur, option sur ZC et
  swaption Jamshidian contre QuantLib 1.44, Jamshidian contre Monte Carlo en multi-courbes, parité,
  calibrations Q et P retrouvant un σ connu, adaptateur de la fiche 09.
- Produits : IRS multi-courbes contre QuantLib (< 1e-10 du nominal), OIS contre QuantLib (1e-8, capitalisation
  TARGET jour par jour) ; `E[D·V(t)]` = valeur des flux restants, y compris en milieu de période ;
  EE* d'un payeur seul = swaption sur le swap résiduel (± 3 ET) ; PFE au quantile fermée dans l'IC binomial.
- Exposition : netting sous-additif, swaps miroirs → 0, seuil infini = sans collatéral, collatéral parfait
  à MPoR nulle → 0, EE croissante en MPoR et loi √MPoR (± 15 % hors dates de flux), démo avec et sans CSA.
- XVA : le bootstrap reprice les spreads ; spread au pair contre QuantLib `MidPointCdsEngine` (< 0,1 bp ;
  `IsdaCdsEngine` de la fiche remplacé par MidPoint, la convention du moteur) ; CVA d'un swap seul = somme
  de swaptions fermées (< 2 % et 3 ET) ; somme des allocations d'Euler = CVA ; pré-trade = recalcul complet ;
  CVA de la banque = DVA de la contrepartie ; ℓ = 0 → pas de FVA.

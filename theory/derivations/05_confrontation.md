# T5 — Export des formules et confrontation aux données

*Partie analytique. Scripts : `checks/test_theory.py`, `checks/t5_trous_echantillonnage.py`, `checks/t5_erratum_E2_reel.py`,
`checks/t5_exposants_reels.py`. Résultats de simulation : partie numérique, B1 (`results/b1_theory_comparison.md`,
`results/b1_aggregate.csv`, lus sans modification) et A5.*

## 1. `src/aerot/theory.py` v0.2

| fonction | contenu | dérivation | statut |
|---|---|---|---|
| `solve_kappa`, `order_parameters` | κ auto-cohérent, χ, df, γ, $B_0$, $Q_0$, biais/variance par mode | `02` §3–4 | vérifié (sympy + gaussien) |
| `gen_error`, `train_error`, `learning_curve` | $E_{\rm gen}$, $E_{\rm train}$ (T = 0 ou β fini) | `02` §4, `03` §2 | vérifié gaussien + rotation (B1) |
| `optimal_ridge` | $\lambda_{\rm r}$ minimisant $E_{\rm gen}$ | `03` §4 | vérifié gaussien |
| `free_energy_rs` | $\langle\ln Z\rangle$ RS et entropie | `02` §3.3, §4.3 | vérifié (exact gaussien) |
| `memoire_ridgeless` | formules du mémoire, originales / corrigées (E1, E2) | `01` | vérifié sympy |
| `ood_error` | $E_{\rm OOD}$ = biais population + variance à levier | `04` §2 | vérifié gaussien, jouet 1D, rotation (B1) |
| `ood_error_truncated`, `pointwise_error` | pinv tronquée (mémoire) ; courbe $E(\omega)$ | `04` §2 | contrôle `ref_Eood` exact |
| `generalized_ratio_spectrum`, `runge_decomposition` | spectre de concentration $\mu_j$ ; identité de Runge | `04` §4 | identité exacte vérifiée |

Conventions : moments **non centrés**, $H=\frac12\sum_\mu(\omega_\mu-W\cdot\phi_\mu)^2+\frac{\lambda_{\rm r}}2\|W\|^2$ (le ridge se compare aux $\lambda_k$ via $\lambda_{\rm r}/P$),
erreurs par échantillon sans ½. `ood_error` avertit (RuntimeWarning) si des modes absents de `test_cov` sont appris.
Tests : `python theory/checks/test_theory.py` (~5 s) — tous passent.

## 2. Confrontation aux simulations de la partie numérique (B1, $\omega_{\max}=4\pi/T$, rendu `analytic` ss=4)

4 templates (gaussienne, quart de cercle, croix, segment) × {uniforme, créneau} × $P\in[5,7000]$ × ridges $\lambda_{\rm r}=r\,P\lambda_1$,
$r\in\{10^{-2},10^{-4},10^{-6},10^{-8}\}$ ; données de rotation **et** substituts gaussiens de mêmes seconds moments ; 5 tirages.

* **Substituts gaussiens** : $E_{\rm train}$, $E_{\rm gen}$, $E_{\rm OOD}$ théoriques à quelques % pour $P\ge80$ (tous templates, tous ridges) :
  le calcul de répliques et son implémentation sont validés sur les vrais spectres (rang effectif petit, $N=7840$ non
  pertinent : seuls $P$, $\lambda_{\rm r}$ et le spectre comptent, comme annoncé en `02` §2).
* **Données de rotation, $E_{\rm OOD}$ (gap)** : sur les 64 points $P\ge320$ (4 templates × 4 ridges), écart médian **0.8 %** (substituts :
  0.4 %) ; > 5 % seulement à $r=10^{-8}$ pour la croix et le segment (+5 à +12 %, et ×3 / +53 % pour le segment à $P=3000$ / 7000).
  Dans ces cas ${\rm df}(\kappa)\approx10^3$–$2\cdot10^3$ est comparable à $P$ : $x={\rm df}\ln P/P>1$, hors du domaine de validité dégagé au §3
  (le substitut gaussien lui-même fluctue de −43 % à $P=3000$, proche de l'interpolation, 5 tirages). À $P=80$ : écart médian 14 %
  (max 59 %) ; à $P\le40$ : 12 % (max ×2.9). (Recalcul : `checks/t5_ood_b1.py`, `ood_error` avec les matrices test complètes du cache
  numérique contre `b1_aggregate.csv`.) La formule `04` §2.3 prédit donc **quantitativement** l'erreur dans le gap des vraies vidéos, y compris la
  dépendance au ridge (ex. segment, $P=7000$ : $3.78\cdot10^{-2}$ à $r=10^{-2}$ → $1.88\cdot10^{-2}$ à $r=10^{-6}$, théorie et simulation).
* **Données de rotation, $E_{\rm gen}$ en distribution** : accord à quelques % pour $P$ grand, mais écarts importants dans la transition
  $P\sim$ nombre de modes appris (médiane réelle / théorie jusqu'à ×25 pour la gaussienne à $P=80$, $r=10^{-6}$ ; ×900 à $P=40$, $r=10^{-8}$).
  Les substituts restent à ×2 près : c'est un **écart à l'équivalence gaussienne**, pas une erreur de calcul (§3).
* Point de vigilance (signalé par la partie numérique) : avec `test_cov_in_train_basis` tronquée à 1000 modes, `ood_error` est
  biaisée à petit ridge pour les bords francs ; les comparaisons ci-dessus utilisent la matrice test complète.

## 3. Quand l'équivalence gaussienne tient-elle sur la variété de rotation ? — *critère proposé, testé sur B1*

**Mécanisme** (`03` §1.3, `04`) : sur une variété 1D, les $P$ valeurs de $\omega$ tirées i.i.d. laissent des **trous** ; le plus grand mesure
$\approx\omega_{\max}\ln P/P$. Chaque trou est un **mini-gap** : l'estimateur y extrapole, avec l'amplification de Runge de `04` §4. Le modèle
gaussien ignore cette géométrie. Les trous sont inoffensifs s'ils sont petits devant l'échelle de résolution des modes appris,
$\approx\omega_{\max}/{\rm df}(\kappa)$ (un mode propre de rang $k$ oscille $\sim k/2$ fois sur l'intervalle). D'où le paramètre sans dimension
$$x=\frac{{\rm df}(\kappa)\,\ln P}{P}\qquad(\text{plus grand trou / échelle de résolution}).$$
**Test** (`checks/t5_trous_echantillonnage.py`, 160 points de B1, loi uniforme ; figure `figures/t5_equivalence_gaussienne.png`) :

| $x$ | n | médiane (réel / théorie) | max | substitut / théorie |
|---|---|---|---|---|
| [0, 0.1) | 14 | 1.00 | 1.2 | 0.98 |
| [0.1, 0.3) | 12 | 0.99 | 1.4 | 0.99 |
| [0.3, 1) | 18 | 0.97 | 1.8 | 0.97 |
| [1, 3) | 78 | 1.30 | 156 | 0.97 |
| [3, 100) | 38 | 1.92 | 884 | 1.07 |

Les quatre templates s'effondrent sur la même courbe : **$x<1$ ⇒ équivalence gaussienne à ×1.8 près (médiane ≈ 1)** ; au-delà, écarts
croissants, énormes seulement pour la gaussienne à petit ridge. Lecture : dans ce régime la théorie prédit une erreur minuscule
(résidu $\sim10^{-12}$, `04` §6) que les mini-gaps amplifient d'un facteur $\mu\sim10^2$–$10^8$ ; pour les bords francs l'erreur théorique est
déjà dominée par un résidu plus gros et l'amplification relative est modeste (×1.3–×10).
**$E_{\rm OOD}$ est moins sensible** (≤ ×2.3 d'après B1) : il est dominé par le biais population du grand gap, que les petits trous
modifient peu. Statut : critère heuristique (ordre de grandeur), non dérivé par les répliques ; il donne la limite de validité pratique
des formules : $P\gtrsim{\rm df}(\kappa)\ln P$.

## 4. L'expérience de gap du mémoire

Partie numérique (A5, message du 06:50) : meilleure concordance avec **$\omega_{\max}=2\pi/T$** (valeur du *texte* du mémoire) et une coupure
**$\lambda_k/\lambda_1>10^{-8}$** (= `rcond` $10^{-4}$ sur les valeurs singulières de $\Phi$) : pics dans le gap $5.7\cdot10^{-6}$ / $3.4\cdot10^{-3}$ / $5.0\cdot10^{-4}$ / $1.6\cdot10^{-3}$
(gaussienne / segment / quart de cercle / croix) contre $9.3\cdot10^{-6}$ / $1.5\cdot10^{-3}$ / $1.2\cdot10^{-3}$ / $1.3\cdot10^{-2}$ dans le mémoire.
C'est exactement la condition dégagée en `04` §6.1 : la dichotomie n'existe que si (i) le seuil résout tout le spectre (fini) du template
lisse et (ii) le gap est assez court devant la largeur de bande (nombre de Slepian petit). La gaussienne et les bords francs sont
reproduits en ordre de grandeur ; la croix du mémoire reste ~8× au-dessus (géométrie de la croix non identique, probablement).

## 5. Impact de l'erratum E2 sur les vrais spectres — *négligeable* (`checks/t5_erratum_E2_reel.py`)

Les $\mathcal J$ du mémoire (covariance centrée) se reconstruisent des moments non centrés par Sherman–Morrison, avec
$a=\bar\phi^\top\Sigma^+\bar\phi$ : $\mathcal J_{\phi\phi}=a/(1-a)$, $\mathcal J_{\phi c}=(b-\bar\omega a)/(1-a)$, $\mathcal J_{cc}=(d-2\bar\omega b+\bar\omega^2a)+(b-\bar\omega a)^2/(1-a)$ ($b=\bar\phi^\top\Sigma^+C$,
$d=C^\top\Sigma^+C$). Sur les 12 spectres uniformes (6 templates × 2 $\omega_{\max}$, seuils $10^{-4}$ et $10^{-8}$), l'écart relatif entre le $X$ du
mémoire (signe erroné) et le $X$ corrigé ($=\sigma^2_{\rm res}$) est **≤ $1.4\cdot10^{-6}$**. Raison structurelle : $1-a$ est l'erreur de la meilleure
prédiction linéaire de la constante 1, or **l'image $t=0$ est identique pour tout $\omega$** (template non tourné), donc la constante est
exactement représentable ($1-a\sim10^{-7}$–$10^{-14}$, reste de troncature) : le modèle contient de fait un **biais (intercept)**, et le terme
d'erreur, en $O(1/\mathcal J_{\phi\phi})=O(1-a)$, disparaît. Les errata E1–E2 sont des erreurs de calcul réelles mais sans conséquence
numérique pour ces données (ils en auraient pour des données sans direction constante).

## 6. Exposants spectraux des vrais templates et lois d'échelle (`checks/t5_exposants_reels.py`)

| template (uniforme) | $2\pi/T$ | $4\pi/T$ |
|---|---|---|
| gaussienne | $\lambda_k\sim e^{-0.83k}$, $K(10^{-12})=34$ | $e^{-0.46k}$, $K=61$ |
| dipôle | $e^{-0.86k}$, $K=33$ | $e^{-0.50k}$, $K=56$ |
| quart de cercle | $a=2.25$, $b=2.32$ | $a=2.67$, $b=2.59$ |
| croix | $a=2.41$, $b=2.54$ | $a=2.78$, $b=2.90$ |
| segment | $a=2.36$, $b=2.25$ | $a=2.81$, $b=2.62$ |
| arc | $a=3.01$, $b=2.68$ | $a=3.39$, $b=2.93$ |

($\lambda_k\sim k^{-a}$ et $\sum_{j\ge k}t_j\sim k^{-(b-1)}$ ajustés sur $20\le k\le600$.) Conséquences (`03` §5) : pour les bords francs, $b<2a+1$, donc
sans bruit et quasi sans ridge $E_{\rm gen}\sim P^{-(b-1)}\approx P^{-1.3}$ à $P^{-1.9}$ asymptotiquement (pentes locales de la théorie entre
$P=100$ et 3000 : $-0.6$ à $-1.2$, régime pré-asymptotique) ; au ridge optimal avec un bruit effectif $\sigma^2$, $(\sigma^2/P)^{(b-1)/b}$, soit
$P^{-0.56}$ à $P^{-0.66}$. Pour les templates lisses, l'erreur en distribution sature vite au plancher de mauvaise spécification
($\sim10^{-12}$) : en distribution, les deux familles généralisent très bien ; **la différence spectaculaire est hors distribution** (`04`).

# T4 — Généralisation hors distribution : l'erreur dans le gap

*Partie analytique. Conventions de `docs/CONVENTIONS.md` ($P$ échantillons, $\Sigma$ second moment non centré, $\lambda_{\rm r}$ ridge,
erreurs par échantillon sans ½). Vérifications : `checks/t4_ood_gauss_check.py`, `checks/t4_ood_taille_finie.py`,
`checks/t4_toy_gap.py`, `checks/t4_spectres_reels.py`, `checks/t4_runge.py`, `checks/t4_ridge_P.py`.
Formules exportées : `src/aerot/theory.py` v0.2 (`ood_error`, `ood_error_truncated`, `pointwise_error`,
`generalized_ratio_spectrum`, `runge_decomposition`).*

**Statut des résultats.** §2 : dérivé (répliques) et vérifié numériquement sur données gaussiennes (≤ 3 %) et sur une
variété 1D jouet (1–3 %). §4 : identité exacte (algèbre linéaire), vérifiée à la précision machine sur tous les vrais spectres.
§5–7 : évaluations numériques des formules sur les spectres population exportés par la partie numérique (`data/spectra/`).

---

## 1. Cadre : changement de covariables dans la variable latente

Entraînement : $\omega\sim\rho_{\rm tr}$ = créneau $\mathcal U([0,\omega_{\max}/3]\cup[2\omega_{\max}/3,\omega_{\max}])$ ; test : $\omega\sim\rho_{\rm te}=\mathcal U[\omega_{\max}/3,2\omega_{\max}/3]$ (le gap).
La carte $\omega\mapsto\phi(\omega)$ et la cible $\omega$ sont les mêmes : c'est un **changement de covariables** pur (la loi de la
cible sachant l'entrée ne change pas, elle est même déterministe). Mais le modèle linéaire est **mal spécifié** ($\omega$ n'est pas
exactement une fonction linéaire de $\phi$), donc le meilleur prédicteur linéaire dépend de la loi :
$\theta_{\rm tr}=\Sigma_{\rm tr}^{-1}C_{\rm tr}\neq\theta_{\rm te}=\Sigma_{\rm te}^{-1}C_{\rm te}$. Toute l'erreur dans le gap vient de là (§4).

Pour tout $W$, l'erreur de test est une **forme quadratique** qui ne dépend que des seconds moments de test :
$$E_{\rm te}(W)=\langle(\omega-W\cdot\phi)^2\rangle_{\rm te}=\langle\omega^2\rangle_{\rm te}-2\,C_{\rm te}\cdot W+W^\top\Sigma_{\rm te}W .$$
On l'exprime dans la base propre d'**entraînement** $\Sigma_{\rm tr}=\sum_k\lambda_ku_ku_k^\top$ :
$(\Sigma_{\rm te})_{kl}=u_k^\top\Sigma_{\rm te}u_l$ (non diagonale), $c^{\rm te}_k=u_k\cdot C_{\rm te}$, $\langle\omega^2\rangle_{\rm te}$ (clés `test_cov_in_train_basis`,
`test_c_proj`, `test_omega2` des fichiers `*_creneau_*.npz`). Aucune hypothèse n'est faite sur $\rho_{\rm te}$ au-delà de ces moments.

Deux quantités sans dimension serviront :
* le **levier** du mode d'entraînement $k$ : $r_k=(\Sigma_{\rm te})_{kk}/\lambda_k=\|e_k\|^2_{\rm te}$, où $e_k=u_k\cdot\phi/\sqrt{\lambda_k}$ est la $k$-ième
  fonction propre (normée sur $\rho_{\rm tr}$) prolongée dans le gap ; en distribution $r_k=1$ ;
* le **spectre généralisé** du faisceau $(\Sigma_{\rm te},\Sigma_{\rm tr})$ : $\Sigma_{\rm te}v_j=\mu_j\Sigma_{\rm tr}v_j$, $v_j^\top\Sigma_{\rm tr}v_j=1$. Les fonctions
  $g_j=v_j\cdot\phi$ sont orthonormées sur $\rho_{\rm tr}$ et orthogonales sur $\rho_{\rm te}$ avec $\|g_j\|^2_{\rm te}=\mu_j$ : $\mu_j\gg1$ = direction
  **concentrée dans le gap** (peu d'énergie d'entraînement, beaucoup d'énergie de test). C'est l'analogue discret des fonctions
  de Slepian (concentration en temps de signaux à bande limitée).

## 2. Calcul de répliques de $E_{\rm OOD}(P,\lambda_{\rm r})$

### 2.1 L'erreur de test est une observable « à une réplique »
On reprend le calcul RS de `02` (modèle gaussien centré de mêmes seconds moments d'**entraînement**,
$\omega=\theta_\star\cdot\phi+\varepsilon$, $\theta_{\star,k}=c_k/\lambda_k$, $\varepsilon\sim\mathcal N(0,\sigma^2_{\rm res})$ indépendant, $\sigma^2_{\rm res}=\langle\omega^2\rangle_{\rm tr}-\sum_kc_k^2/\lambda_k$).
Pour une observable $O(W)$ qui **ne dépend pas des données d'entraînement** (c'est le cas de $E_{\rm te}(W)$), la moyenne
(données + Gibbs) s'écrit
$$\overline{\langle O\rangle}=\lim_{n\to0}\ \mathbb E_{\mathcal D}\!\int\prod_{a=1}^n dW^a\,O(W^1)\,e^{-\beta\sum_aH(W^a)} ,$$
et comme $O=O(1)$ alors que l'exposant est extensif, le col n'est pas déplacé : $\overline{\langle O\rangle}$ est la moyenne de $O(W^1)$ sous la
**mesure répliquée au col RS**. (Variante équivalente : ajouter $h\,O(W)$ au Hamiltonien et dériver $\langle\ln Z\rangle$ en $h=0$ ;
par stationnarité du col seul le terme explicite contribue.)

### 2.2 Mesure effective au col : modèle de suite gaussienne
Au col (`02` §3.2–3.4), la mesure sur $x_k=(w^1_k,\dots,w^n_k)$, $w^a=W^a-\theta_\star$, se factorise par mode :
$$\propto\exp\Big[-\tfrac12(\lambda_kR+\beta\lambda_{\rm r})\sum_a(x^a)^2+\tfrac12\lambda_k\hat q\Big(\sum_ax^a\Big)^2-\beta\lambda_{\rm r}\theta_k\sum_ax^a\Big].$$
Le terme $(\sum_ax^a)^2$ se découple par Hubbard–Stratonovich, $e^{\frac12\lambda_k\hat q s^2}=\mathbb E_{z_k}e^{\sqrt{\lambda_k\hat q}\,z_ks}$, $z_k\sim\mathcal N(0,1)$ : **à $z_k$
fixé, les répliques sont indépendantes** et gaussiennes, de moyenne $m_k(z_k)=\frac{\sqrt{\lambda_k\hat q}\,z_k-\beta\lambda_{\rm r}\theta_k}{\lambda_kR+\beta\lambda_{\rm r}}$ et de variance
$\frac1{\lambda_kR+\beta\lambda_{\rm r}}$. Avec $\lambda_kR+\beta\lambda_{\rm r}=\frac{\beta P}{1+\chi}(\lambda_k+\kappa)$ et $\hat q=\frac{P\beta^2(Q_0+\sigma^2)}{(1+\chi)^2}$, $E_{\rm gen}=\sigma^2+Q_0$ (à $T=0$) :
$$\boxed{W_k\ \stackrel{\rm loi}{=}\ \frac{\lambda_k\theta_{\star,k}+\sqrt{\lambda_kE_{\rm gen}/P}\;z_k}{\lambda_k+\kappa}\ +\ \sqrt{\frac{1+\chi}{\beta P(\lambda_k+\kappa)}}\;\xi^a_k,\qquad z_k,\ \xi^a_k\ \text{i.i.d. }\mathcal N(0,1)}$$
Le $z_k$ (commun aux répliques) représente les **fluctuations d'un jeu de données à l'autre**, $\xi^a_k$ les fluctuations thermiques.
C'est l'équivalence « ridge ↔ modèle de suite gaussienne à bruit effectif $E_{\rm gen}/P$ » (cf. Cheng–Montanari 2022, *Dimension free
ridge regression* ; Bach 2023). Contrôles : $\mathbb E W_k=L_k\theta_{\star,k}$ ($L_k=\lambda_k/(\lambda_k+\kappa)$) ;
$\sum_k\lambda_k\big[{\rm Var}_zW_k+(1-L_k)^2\theta_{\star,k}^2\big]=\gamma E_{\rm gen}+B_0=Q_0$ ✔ (cohérence avec l'équation du col de `02`).

**Point clé : les $z_k$ sont indépendants d'un mode à l'autre.** La covariance (sur les jeux) de $W$ est donc **diagonale dans la
base propre d'entraînement**, même si $\Sigma_{\rm te}$ ne l'est pas.

### 2.3 Résultat
En moyennant la forme quadratique $E_{\rm te}(W)$ :
$$\boxed{E_{\rm OOD}=\underbrace{E_{\rm te}(\bar W_\kappa)}_{\text{biais (population)}}+\underbrace{\frac{E_{\rm gen}}{P}\sum_k(\Sigma_{\rm te})_{kk}\frac{\lambda_k}{(\lambda_k+\kappa)^2}}_{\text{variance}}\ \Big[+\ \frac{1+\chi}{\beta P}\sum_k\frac{(\Sigma_{\rm te})_{kk}}{\lambda_k+\kappa}\Big]_{T>0},\qquad\bar W_{\kappa,k}=\frac{c_k}{\lambda_k+\kappa}}$$
avec $\kappa$, $\chi$ et $E_{\rm gen}$ **de la distribution d'entraînement** (`02`, `03`). Autrement dit :
* **biais** = erreur de test du prédicteur ridge « population » de régularisation effective $\kappa$ (toute la mauvaise spécification
  et le changement de loi sont là, **termes croisés de $\Sigma_{\rm te}$ compris**) ;
* **variance** = $\frac{E_{\rm gen}}{P}\sum_kr_kL_k^2$ : chaque mode appris coûte $E_{\rm gen}/P$ **multiplié par son levier** $r_k$. On note
  $\mathcal A(\kappa)=\sum_kr_kL_k^2/\sum_kL_k^2$ le **facteur d'amplification OOD de la variance** ($\mathcal A=1$ en distribution).

En distribution ($\Sigma_{\rm te}=\Sigma_{\rm tr}$, $C_{\rm te}=C_{\rm tr}$) : biais $=\sigma^2+B_0$, variance $=\gamma E_{\rm gen}$, somme $=E_{\rm gen}$ ✔ (`03` §2).
La formule coïncide avec les résultats connus de régression ridge/noyau sous changement de covariables (Canatar–Bordelon–Pehlevan,
*Out-of-distribution generalization in kernel regression*, NeurIPS 2021 ; Tripuraneni–Adlam–Pennington 2021 pour les features
aléatoires) ; on l'obtient ici directement comme observable du calcul RS de `02`.

### 2.4 Corrections de taille finie (non capturées par les répliques)
Pour $P\gg{\rm df}$ on peut développer exactement $W^*=(X^\top X/P+\lambda_{\rm r}/P)^{-1}X^\top\omega/P$ au second ordre (données gaussiennes,
Isserlis). On trouve, avec $\kappa'=\lambda_{\rm r}/P$ :
$\mathbb EW^*_k=\bar W_{\kappa,k}-\frac1PL_k^2(1-L_k)\theta_{\star,k}+O(P^{-2})$ et ${\rm Cov}(W^*)=\frac{E_{\rm gen}}P\,{\rm diag}\big(\frac{\lambda_k}{(\lambda_k+\kappa)^2}\big)+\frac1Pvv^\top+O(P^{-2})$,
$v_k=L_k(1-L_k)\theta_{\star,k}$. Les termes $\frac{\rm df}{P}$ sont ceux des répliques ; les termes $O(1/P)$ **non multipliés par df**
(dérive de la moyenne, covariance de rang 1) sont des corrections de taille finie. Mesurées : $\frac1Pv^\top\Sigma_{\rm te}v\lesssim2\cdot10^{-3}$ en relatif
dans le test gaussien (§3). En OOD elles sont toutefois **amplifiées** par le gradient $2(\Sigma_{\rm te}\bar W-C_{\rm te})$, grand quand le biais
OOD est grand (en distribution ce gradient est $O(\kappa)$).

## 3. Vérification sur données gaussiennes — *vérifié*

`checks/t4_ood_gauss_check.py [NTRIAL]` : $N=250$, $\lambda_k=k^{-1.3}$, $t_k=0.4k^{-1.2}$, $\sigma^2=0.03$ ; loi de test gaussienne de
covariance $\Sigma_{\rm te}=BB^\top$ **non diagonale** (modes 20–60 mélangés et amplifiés) et cible de test **mal spécifiée**
($\theta_{\rm te}\neq\theta_\star$). $E_{\rm te}(W)$ exact pour chaque $W$ simulé ; 600 jeux par point.

| $P$ | $\lambda_{\rm r}$ | $E_{\rm OOD}$ simulé | théorie (biais + variance) |
|---|---|---|---|
| 60 | 0.01 | 3.794 ± 0.052 | 3.685 (2.833 + 0.853) |
| 60 | 1 | 4.052 ± 0.029 | 4.041 (3.790 + 0.251) |
| 400 | 0.3 | 0.965 ± 0.012 | 0.975 (0.805 + 0.170) |
| 1000 | 0 | 0.2303 ± 0.0035 | 0.2308 (0.159 + 0.072) |
| 1000 | 3 | 1.707 ± 0.008 | 1.702 (1.652 + 0.051) |

Décomposition empirique (800 jeux) : la **variance** (trace de $\Sigma_{\rm te}$ contre la covariance empirique de $W$, termes croisés
compris) est reproduite à 1–2 % et la partie **non diagonale** de la covariance est négligeable (≤ 2 % de la variance) ;
les écarts résiduels portent sur le biais, via le terme linéaire $2(\Sigma_{\rm te}\bar W-C_{\rm te})\cdot(\mathbb EW-\bar W)$ dont l'incertitude
statistique domine. Le seul point à ~3 % ($P=60$, $\lambda_{\rm r}=0.01$, 2–3σ selon les tirages) est une fluctuation propre à cette
réalisation de $\Sigma_{\rm te}$ : `checks/t4_ood_taille_finie.py` (16 réalisations indépendantes, $N=125\to1000$, $P=0.24N$, même régime)
donne un écart relatif moyen $+0.7\pm1.5\,\%$, $-0.2\pm1.4\,\%$, $-0.2\pm1.3\,\%$, $+0.6\pm1.3\,\%$ : **pas de biais systématique**.

## 4. Mauvaise spécification et « identité de Runge » (P = ∞) — *exact, vérifié*

### 4.1 Énoncé
Soit $F$ un espace de prédicteurs (ici $F_K$ = combinaisons des $K$ premiers modes d'entraînement : pinv tronquée du
mémoire ; on traite $P=\infty$, donc biais pur), $f_{\rm tr}\in F$ le meilleur prédicteur sur $\rho_{\rm tr}$ et $g_j$ les fonctions propres
généralisées (§1) de $F$. Pour **toute** référence $f_{\rm ref}\in F$, avec $\varepsilon=\omega-f_{\rm ref}$ :
$$E_{\rm te}(f_{\rm tr})=E^{\min}_{\rm te}(F)+\sum_j\frac{\big(\mu_j\langle\varepsilon,g_j\rangle_{\rm tr}-\langle\varepsilon,g_j\rangle_{\rm te}\big)^2}{\mu_j},\qquad E^{\min}_{\rm te}(F)=\min_{f\in F}\|\omega-f\|^2_{\rm te}.$$
*Preuve.* $f_{\rm tr}=\sum_ja_jg_j$, $a_j=\langle\omega,g_j\rangle_{\rm tr}$ ; avec $b_j=\langle\omega,g_j\rangle_{\rm te}$ et $\langle g_i,g_j\rangle_{\rm te}=\mu_j\delta_{ij}$ :
$\|\omega-f_{\rm tr}\|^2_{\rm te}=\|\omega\|^2_{\rm te}-2\sum_ja_jb_j+\sum_j\mu_ja_j^2=\big[\|\omega\|^2_{\rm te}-\sum_jb_j^2/\mu_j\big]+\sum_j(\mu_ja_j-b_j)^2/\mu_j$, et le
premier crochet est $E^{\min}_{\rm te}$. Enfin $f_{\rm ref}=\sum_jx_jg_j$ contribue $\mu_jx_j$ à $\mu_ja_j$ et à $b_j$, donc disparaît de la différence. ∎

**Choix naturel** : $f_{\rm ref}=f_u$, meilleur prédicteur sur la loi **uniforme** $\rho_u=\frac23\rho_{\rm tr}+\frac13\rho_{\rm te}$ (celle qu'on aurait eue sans gap),
de résidu $\varepsilon_u$. L'orthogonalité $\frac23\langle\varepsilon_u,g_j\rangle_{\rm tr}+\frac13\langle\varepsilon_u,g_j\rangle_{\rm te}=0$ donne $\langle\varepsilon_u,g_j\rangle_{\rm te}=-2\langle\varepsilon_u,g_j\rangle_{\rm tr}$, d'où
$$\boxed{E_{\rm OOD}(F)=E^{\min}_{\rm te}(F)+\sum_j\frac{(\mu_j+2)^2}{\mu_j}\,e_j^2,\qquad e_j=\langle\varepsilon_u,g_j\rangle_{\rm tr},\qquad\sum_je_j^2\le\|\varepsilon_u\|^2_{\rm tr}}$$
et la **borne de type Lebesgue** $E_{\rm OOD}\le E^{\min}_{\rm te}+\frac{(\mu_{\max}+2)^2}{\mu_{\max}}\|\varepsilon_u\|^2_{\rm tr}\approx E^{\min}_{\rm te}+\mu_{\max}\,\varepsilon_u^2$.

### 4.2 Lecture
* L'erreur dans le gap = (erreur minimale atteignable dans le gap avec ces features) + **résidu de mauvaise spécification
  projeté sur les directions concentrées dans le gap, amplifié par leur facteur de concentration $\mu_j$**.
* Si $\omega$ était exactement linéaire en $\phi$ ($\varepsilon_u=0$), l'erreur OOD serait $E^{\min}_{\rm te}=0$ quel que soit le gap : **sans mauvaise
  spécification, pas d'erreur de changement de covariables** (à $P=\infty$). Le changement de loi n'agit qu'en amplifiant le résidu.
* C'est **exactement la structure du phénomène de Runge** : pour l'interpolation polynomiale, erreur $\le(1+\Lambda_n)\,E_n(f)$ avec $E_n$
  l'erreur de meilleure approximation et $\Lambda_n$ la constante de Lebesgue. Ici $\varepsilon_u^2(K)$ joue le rôle de $E_n^2$ et
  $\mu_{\max}(K)$ celui de $\Lambda_n^2$. Quand $K$ augmente (seuil plus fin), $\varepsilon_u^2$ **décroît** et $\mu_{\max}$ **croît** : l'erreur dans le gap
  converge si et seulement si la première l'emporte. Nuance importante vis-à-vis du mémoire : la cible ($\omega$, linéaire) est la
  même pour tous les templates ; ce qui change est l'**espace d'approximation** $F_K$ (les fonctions $\omega\mapsto\phi_{x,t}(\omega)$).
* Vérifié (`checks/t4_runge.py`) : l'identité tient à $10^{-9}$ près (précision de `test_cov`) sur tous les templates, les deux $\omega_{\max}$
  et ~10 valeurs de $K$ chacun. $E^{\min}_{\rm te}(F_K)$ devient négligeable (2 à 8 ordres de grandeur sous $E_{\rm OOD}$) dès $K\gtrsim5$–$20$ : **l'erreur dans le gap est entièrement le
  second terme** (décalage entre meilleurs prédicteurs d'entraînement et de test). La borne de Lebesgue est lâche (1–3 ordres) :
  le résidu n'est pas concentré sur la direction $\mu_{\max}$.

### 4.3 Avec ridge et $P$ fini
Le ridge remplace la troncature dure par le filtre $L_k$ ; le biais de §2.3 est l'erreur de test de $\bar W_\kappa$ (la décomposition
ci-dessus s'applique à la troncature ; pour le ridge on évalue directement $E_{\rm te}(\bar W_\kappa)$). La variance ajoute
$\frac{E_{\rm gen}}P\sum_kr_kL_k^2$ : les modes à fort levier $r_k$ (petite valeur propre d'entraînement, énergie dans le gap) sont ceux
qui amplifient à la fois le résidu (biais) et le bruit d'échantillonnage (variance).

## 5. Modèle jouet : largeur de bande et gap — *vérifié (théorie vs simulation sur la variété)*

`checks/t4_toy_gap.py` : $\phi_k(\omega)=\sqrt{2s_k}\cos(\pi k\omega)$, $k<160$, $\omega\in[0,1]$, créneau $[0,\frac13]\cup[\frac23,1]$, gap $[\frac13,\frac23]$, cible $y=\omega$ ;
(A) « lisse » $s_k=e^{-k^2/2B^2}$ (bande douce de largeur $B$) ; (B) « bords » $s_k=(1+k)^{-2}$, $k\le k_{\max}$. Théorie = `ood_error`
($P=300$, $\lambda_{\rm r}/P=10^{-4}\lambda_1$) ; simulation = 40 jeux i.i.d. **sur la variété 1D** (non gaussienne). ${\rm Var}(\omega)_{\rm gap}=9.3\cdot10^{-3}$.

| famille | #$\mu>1$ | #$\mu>10^3$ | $E_{\rm OOD}$ pop. tronq. $10^{-4}$ | théorie $P=300$ | simulation (moy. ± err.) |
|---|---|---|---|---|---|
| lisse $B=2$ | 5 | 2 | $3.3\cdot10^{-4}$ | $2.53\cdot10^{-4}$ | $2.23(9)\cdot10^{-4}$ |
| lisse $B=4$ | 9 | 5 | $4.3\cdot10^{-2}$ | $4.51\cdot10^{-2}$ | $4.56(4)\cdot10^{-2}$ |
| lisse $B=8$ | 12 | 8 | $1.44\cdot10^{-1}$ | $1.471\cdot10^{-1}$ | $1.482(4)\cdot10^{-1}$ |
| lisse $B=16$ | 15 | 9 | $2.02\cdot10^{-1}$ | $2.025\cdot10^{-1}$ | $2.041(4)\cdot10^{-1}$ |
| bords $k_{\max}=6$ | 3 | 0 | $1.9\cdot10^{-4}$ | $2.68\cdot10^{-4}$ | $2.70(22)\cdot10^{-4}$ |
| bords $k_{\max}=12$ | 5 | 2 | $4.4\cdot10^{-4}$ | $8.66\cdot10^{-4}$ | $8.68(9)\cdot10^{-4}$ |
| bords $k_{\max}=24$ | 9 | 5 | $7.5\cdot10^{-4}$ | $9.00\cdot10^{-4}$ | $9.13(10)\cdot10^{-4}$ |
| bords $k_{\max}=48$ | 11 | 7 | $1.06\cdot10^{-3}$ | $1.029\cdot10^{-3}$ | $1.036(6)\cdot10^{-3}$ |

Enseignements. (i) La formule des répliques décrit l'erreur OOD **sur une vraie variété 1D** à 1–3 % près (13 % pour $B=2$, régime à
très peu de modes). (ii) Ici c'est la famille « lisse » à **large bande** qui est catastrophique (erreur > ${\rm Var}(\omega)_{\rm gap}$ : pire que
prédire la moyenne) : la cible $y=\omega$ a des coefficients $\propto k^{-2}$ (bords de $[0,1]$), donc un résidu $\varepsilon_u$ qui ne décroît
qu'algébriquement, tandis que les nombreuses directions concentrées dans le gap ont un coût ridge $\propto1/s_k$ faible jusqu'à $k\sim B$ ;
pour la famille « bords », les directions oscillantes coûtent $\propto k^2$ et sont éteintes par le ridge. **La régularité seule ne décide
pas** : ce qui compte est la compétition résidu × amplification de §4, qui dépend du spectre **et** de la géométrie (bande × longueur du gap).

## 6. Vrais templates — *évaluation des formules sur les spectres exportés*

`checks/t4_spectres_reels.py`, `checks/t4_runge.py` (spectres population, rendu `analytic` ss=4, $d=28$, $T=10$). $E_{\rm OOD}$ de la pinv tronquée
($P=\infty$) au seuil $\lambda_k/\lambda_1>\tau$ ; « frac » = part de l'énergie de test ${\rm Tr}\Sigma_{\rm te}$ portée par les `K_export` modes exportés.

**$\omega_{\max}=2\pi/T$** (gap $[0.209,0.419]$, ${\rm Var}(\omega)_{\rm gap}=3.7\cdot10^{-3}$) :

| template | $K_{\rm export}$ | frac | $\tau=10^{-4}$ | $10^{-6}$ | $10^{-8}$ | $10^{-10}$ | $10^{-12}$ | #$\mu>1$ | $\mu_{\max}$ |
|---|---|---|---|---|---|---|---|---|---|
| gaussienne | 28 | 0.998 | $2.8\cdot10^{-4}$ | $7.7\cdot10^{-5}$ | $2.1\cdot10^{-6}$ | $1.3\cdot10^{-6}$ | $8.0\cdot10^{-7}$ | 8 | $4\cdot10^{8}$ |
| dipôle | 28 | 0.999 | $3.8\cdot10^{-4}$ | $1.2\cdot10^{-5}$ | $4.1\cdot10^{-6}$ | $5.7\cdot10^{-7}$ | $2.8\cdot10^{-7}$ | 9 | $3\cdot10^{8}$ |
| quart de cercle | 1000 | 0.91 | $1.5\cdot10^{-4}$ | $1.2\cdot10^{-4}$ | $1.3\cdot10^{-4}$ | = | = | 263 | $2\cdot10^{6}$ |
| croix | 1000 | 0.89 | $2.2\cdot10^{-3}$ | $1.9\cdot10^{-3}$ | $4.1\cdot10^{-4}$ | = | = | 357 | $1\cdot10^{6}$ |
| segment | 1000 | 0.51 | $4.1\cdot10^{-3}$ | $2.8\cdot10^{-3}$ | $1.7\cdot10^{-3}$ | = | = | 138 | $3\cdot10^{5}$ |
| arc | 430 | 0.91 | $1.8\cdot10^{-4}$ | $1.5\cdot10^{-4}$ | $4.0\cdot10^{-4}$ | = | = | 105 | $5\cdot10^{5}$ |

**$\omega_{\max}=4\pi/T$** (gap $[0.419,0.838]$, ${\rm Var}(\omega)_{\rm gap}=1.5\cdot10^{-2}$) :

| template | $K_{\rm export}$ | frac | $\tau=10^{-4}$ | $10^{-6}$ | $10^{-8}$ | $10^{-10}$ | $10^{-12}$ | #$\mu>1$ | $\mu_{\max}$ |
|---|---|---|---|---|---|---|---|---|---|
| gaussienne | 47 | 0.85 | $1.6\cdot10^{-2}$ | $9.0\cdot10^{-3}$ | $4.4\cdot10^{-3}$ | $2.6\cdot10^{-3}$ | $1.3\cdot10^{-3}$ | 16 | $1\cdot10^{10}$ |
| dipôle | 46 | 0.90 | $5.4\cdot10^{-3}$ | $3.5\cdot10^{-3}$ | $2.4\cdot10^{-3}$ | $3.9\cdot10^{-4}$ | $2.0\cdot10^{-4}$ | 17 | $8\cdot10^{9}$ |
| quart de cercle | 1000 | 0.73 | $1.4\cdot10^{-2}$ | $1.2\cdot10^{-2}$ | $1.0\cdot10^{-2}$ | = | = | 256 | $1\cdot10^{6}$ |
| croix | 1000 | 0.88 | $1.1\cdot10^{-2}$ | $8.6\cdot10^{-3}$ | $2.3\cdot10^{-3}$ | = | = | 389 | $3\cdot10^{5}$ |
| segment | 1000 | 0.37 | $3.2\cdot10^{-2}$ | $2.2\cdot10^{-2}$ | $2.1\cdot10^{-2}$ | = | = | 136 | $6\cdot10^{4}$ |
| arc | 626 | 0.81 | $1.4\cdot10^{-2}$ | $1.2\cdot10^{-2}$ | $8.4\cdot10^{-3}$ | = | = | 204 | $1\cdot10^{6}$ |

(« = » : seuil au-delà des `K_export` = 1000 modes exportés — valeur figée par la troncature de l'export ; les valeurs du calcul
numérique à $\tau=10^{-12}$ avec tous les modes sont du même ordre : segment $1.3\cdot10^{-3}$, croix $9.7\cdot10^{-4}$, quart de cercle
$2.3\cdot10^{-4}$ à $2\pi/T$, `results/a5_gap_summary_*`.) À $\tau=10^{-4}$ nos valeurs reproduisent `ref_Eood_pop_trunc1e-4` et la
colonne « moy. gap » de la partie numérique.

**Matrices de test complètes à $4\pi/T$** (cache de la partie numérique `results/b1_theory_cache/`, tous les modes
$>10^{-14}\lambda_1$ : 53 pour la gaussienne, 1436–1765 pour les bords francs ; Fig. `figures/t4_gap_runge.png` (b)) : à résolution
complète, gaussienne $6.1\cdot10^{-4}$, dipôle $2\cdot10^{-4}$ (fichier exporté, déjà complet), segment $1.8\cdot10^{-3}$, quart de cercle $7.9\cdot10^{-3}$,
croix $1.1\cdot10^{-2}$ (après un minimum $3.9\cdot10^{-3}$ à $K=706$ : **remontée** quand on ajoute les derniers modes). Les bords francs finissent
donc par être « résolus » à la coupure pixel ($\lambda\sim10^{-9}$–$10^{-14}\lambda_1$, > 1400 modes), mais leur convergence est erratique ;
l'écart lisse / bords francs n'est plus que d'un facteur 3–20 (contre 300–2000 à $2\pi/T$).

**Courbes de Runge** (`checks/t4_runge.py`, extraits) :

| | $K$ | $\varepsilon_u^2(K)$ | $\mu_{\max}(K)$ | $E_{\rm OOD}(K)$ |
|---|---|---|---|---|
| gaussienne $2\pi/T$ | 12 / 16 / 21 / 28 | $2.9\cdot10^{-6}$ / $7.5\cdot10^{-8}$ / $2.7\cdot10^{-10}$ / $1.7\cdot10^{-13}$ | $4\cdot10^2$ / $1\cdot10^4$ / $2\cdot10^6$ / $4\cdot10^8$ | $2.8\cdot10^{-4}$ / $7.7\cdot10^{-5}$ / $2.1\cdot10^{-6}$ / $8.0\cdot10^{-7}$ |
| quart de cercle $2\pi/T$ | 10 / 55 / 191 / 1000 | $2.3\cdot10^{-5}$ / $2.6\cdot10^{-6}$ / $5.8\cdot10^{-8}$ / $7.8\cdot10^{-9}$ | $16$ / $1.5\cdot10^3$ / $6.6\cdot10^3$ / $1.7\cdot10^6$ | $2.2\cdot10^{-4}$ / $1.0\cdot10^{-4}$ / $1.1\cdot10^{-4}$ / $1.3\cdot10^{-4}$ |
| croix $2\pi/T$ | 10 / 55 / 191 / 1000 | $8.1\cdot10^{-4}$ / $9.4\cdot10^{-5}$ / $8.8\cdot10^{-7}$ / $4.3\cdot10^{-8}$ | $2$ / $2\cdot10^2$ / $5\cdot10^3$ / $1.4\cdot10^6$ | $2.4\cdot10^{-3}$ / $1.9\cdot10^{-3}$ / $1.9\cdot10^{-3}$ / $4.1\cdot10^{-4}$ |
| gaussienne $4\pi/T$ | 16 / 23 / 33 / 47 | $1.8\cdot10^{-3}$ / $2.4\cdot10^{-4}$ / $7.6\cdot10^{-6}$ / $4.5\cdot10^{-8}$ | $45$ / $1\cdot10^4$ / $4\cdot10^5$ / $1\cdot10^{10}$ | $2.3\cdot10^{-2}$ / $1.0\cdot10^{-2}$ / $6.0\cdot10^{-3}$ / $1.3\cdot10^{-3}$ |

### 6.1 Explication de la dichotomie
1. **Templates lisses (gaussienne, dipôle) à $\omega_{\max}=2\pi/T$** : spectre super-exponentiel et **effectivement fini** ($K(10^{-12})=28$).
   Quand le seuil résout tout le spectre, le résidu $\varepsilon_u^2$ décroît plus vite que $\mu_{\max}$ ne croît ($\varepsilon_u^2$ perd 7 décades de $K=12$
   à 28, $\mu_{\max}$ en gagne 6) : l'erreur dans le gap **converge** vers $\sim10^{-6}$ ($2\cdot10^{-7}$ au ridge optimal, §7). C'est la
   « continuation analytique » : une fonction à bande limitée est déterminée dans le gap par ses valeurs hors du gap, de façon
   mal conditionnée ($\mu_{\max}\sim10^8$) mais convergente parce que la cible est représentée presque exactement.
2. **Templates à bords francs** : spectre en loi de puissance jusqu'à la coupure pixel (> 1000 modes), résolu seulement à
   $\tau\sim10^{-9}$–$10^{-14}$ et au prix de > 1400 modes. Le résidu
   ne décroît qu'algébriquement (quart de cercle : $\varepsilon_u^2\approx K^{-1.7}$ environ) tandis que le nombre de directions concentrées
   dans le gap croît avec $K$ (#$\mu>1$ = 100–400) et $\mu_{\max}$ aussi : le produit **stagne** à un plateau $10^{-4}$–$2\cdot10^{-3}$ ($2\pi/T$),
   voire **remonte** quand on affine le seuil (arc : $1.5\cdot10^{-4}\to4.0\cdot10^{-4}$ ; non-monotonie typique de Runge). L'erreur d'entraînement
   reste pourtant $10^{-9}$–$10^{-7}$ : le gap amplifie le résidu d'un facteur $10^4$–$10^5$.
3. **La dichotomie dépend du seuil** : à $\tau=10^{-4}$ (le seuil annoncé par le mémoire) la gaussienne ($2.8\cdot10^{-4}$) est **du même
   ordre** que le quart de cercle ($1.5\cdot10^{-4}$) : pas de dichotomie. Elle apparaît pour $\tau\lesssim10^{-8}$ : gaussienne $2\cdot10^{-6}$ (max sur
   le gap $5.7\cdot10^{-6}$ d'après la partie numérique, cohérent avec le « ~1e-5 » du mémoire) contre $10^{-4}$–$2\cdot10^{-3}$ pour les bords
   francs. Un seuil $10^{-4}$ sur les **valeurs singulières** de $\Phi$ (`rcond` de `pinv`) équivaut à $\tau=10^{-8}$ sur les $\lambda_k$ : c'est
   l'hypothèse la plus économique pour réconcilier nos chiffres avec le mémoire (à confirmer par la partie numérique, A5).
4. **La dichotomie dépend de la géométrie** ($\omega_{\max}$) : doubler $\omega_{\max}$ double la longueur du gap à largeur de bande fixée ; le nombre
   de Slepian (#$\mu>1$) double (gaussienne 8 → 16, dipôle 9 → 17), $\mu_{\max}$ passe de $4\cdot10^8$ à $10^{10}$, et 15 % de l'énergie de test
   de la gaussienne se trouve dans des directions de valeur propre d'entraînement $<10^{-12}\lambda_1$ (frac = 0.85) : **le gap n'est plus
   reconstructible depuis les données d'entraînement** à ces seuils. Résultat : gaussienne $1.3\cdot10^{-3}$ à $\tau=10^{-12}$, $6\cdot10^{-4}$ en résolvant
   tout ($10^{-14}$) ; dichotomie réduite à un facteur 3–20. C'est ce que la partie numérique a observé à $4\pi/T$ (pic ~5e-2 au seuil
   $10^{-4}$ ; aucune convention ne donne une gaussienne < 3e-4, message du 06:50).

Figure `figures/t4_gap_runge.png` : (a) $E_{\rm OOD}(\tau)$ à $2\pi/T$ ; (b) à $4\pi/T$ (matrices complètes) ; (c) contributions
$(\mu_j+2)^2\mu_j^{-1}e_j^2$ en fonction de $\mu_j$ (gaussienne : quelques directions, contributions $\le5\cdot10^{-7}$ ; quart de cercle : des centaines
de directions à $\mu_j\in[1,10^6]$, contributions jusqu'à $10^{-4}$) ; (d) $E_{\rm OOD}(\kappa_0)$ et $E_{\rm gen}$ (créneau) en fonction du ridge.

**Confrontation aux simulations (partie numérique, B1, message 06:50)** : `ood_error` avec les matrices test complètes colle aux
simulations sur substituts gaussiens à quelques % pour $P\ge80$ et aux **vraies données de rotation** pour $P\gtrsim300$ (4 templates,
4 ridges, $4\pi/T$) ; dans la transition $P\sim$ nombre de modes, l'écart médian réel/théorie reste $\le\times2.3$ pour $E_{\rm OOD}$ (jusqu'à
$\times250$ pour $E_{\rm gen}$ de la gaussienne à petit ridge : trous d'échantillonnage, `03` §1.3). Avec la matrice test tronquée à 1000 modes,
`ood_error` est biaisée à petit ridge pour les bords francs (segment : $2\cdot10^{-2}$ au lieu de $2.5\cdot10^{-3}$) : les modes appris au-delà de
`K_export` comptent dans le gap.

### 6.2 La quantité spectrale qui contrôle l'erreur
$$\mathcal R_{\rm gap}(F)=\sum_j\frac{(\mu_j+2)^2}{\mu_j}\,\langle\varepsilon_u,g_j\rangle^2_{\rm tr}\quad(\text{à }P=\infty),\qquad\text{plus }\ \frac{E_{\rm gen}}{P}\sum_kr_kL_k^2\ \text{ à }P\text{ fini}.$$
C'est la **densité du résidu de mauvaise spécification sur le spectre de concentration** $\{\mu_j\}$ du faisceau $(\Sigma_{\rm te},\Sigma_{\rm tr})$. Ses deux
ingrédients ont chacun une origine lisible :
* $\varepsilon_u^2(K)$ : vitesse d'approximation de la cible par les $K$ premiers modes — **super-exponentielle** pour un template lisse
  (spectre angulaire $S_m\propto e^{-z_0}I_m(z_0)$, `03` §6), **algébrique** pour un template à bords francs ($S_m\sim m^{-2}$) ;
* $\{\mu_j\}$ : concentration dans le gap — fixée par le **nombre de Slepian** $N_S\approx$ (largeur de bande du noyau $\nu_{\max}\approx(T-1)m_{\max}$) ×
  (longueur du gap)$/\pi$, et par la manière dont le spectre pondère les hautes fréquences.
L'analogie de Runge du mémoire est donc **correcte dans son principe et précisable** : ce n'est pas la lenteur de la décroissance
spectrale seule (le jouet §5 montre une bande « lisse » large catastrophique), mais la compétition (erreur d'approximation) ×
(constante de type Lebesgue $\mu$) ; la lenteur de la décroissance spectrale des bords francs garantit que cette compétition est perdue.

## 7. $P$ fini et ridge (formule des répliques sur les vrais spectres) — `checks/t4_ridge_P.py`

Grille $\kappa_0=\lambda_{\rm r}/P\in[10^{-14},10^{-2}]\lambda_1$ ; $E_{\rm OOD}$ minimal sur la grille et comparaison au ridge optimal **en distribution** (créneau).

| template ($2\pi/T$) | $P=\infty$ : min $E_{\rm OOD}$ ($\kappa_0^*/\lambda_1$) | $P=100$ | $P=1000$ | $E_{\rm OOD}$ au ridge optimal en distribution ($P=10^4$) | $\mathcal A(\kappa^*)$ |
|---|---|---|---|---|---|
| gaussienne | $1.8\cdot10^{-7}$ ($2\cdot10^{-11}$) | $3.1\cdot10^{-7}$ | $1.9\cdot10^{-7}$ | $7.9\cdot10^{-7}$ | $\sim4\cdot10^6$ |
| dipôle | $3.9\cdot10^{-8}$ ($6\cdot10^{-12}$) | $1.2\cdot10^{-7}$ | $4.8\cdot10^{-8}$ | $2.7\cdot10^{-7}$ | $\sim6\cdot10^6$ |
| quart de cercle | $8.9\cdot10^{-5}$ ($6\cdot10^{-8}$) | $1.3\cdot10^{-4}$ | $9.0\cdot10^{-5}$ | $1.3\cdot10^{-4}$ | $\sim6\cdot10^2$ |
| croix | $4.1\cdot10^{-4}$ (bord de grille*) | $2.0\cdot10^{-3}$ | $7.7\cdot10^{-4}$ | $4.1\cdot10^{-4}$ | $\sim3\cdot10^3$ |
| segment | $1.7\cdot10^{-3}$ (bord*) | $4.2\cdot10^{-3}$ | $2.1\cdot10^{-3}$ | $1.7\cdot10^{-3}$ | $\sim4\cdot10^2$ |
| arc | $1.5\cdot10^{-4}$ ($10^{-6}$) | $1.7\cdot10^{-4}$ | $1.5\cdot10^{-4}$ | $3.9\cdot10^{-4}$ | $\sim2\cdot10^2$ |

(* limité par `K_export` = 1000 : les modes au-delà sont ignorés côté test, ce qui biaise $E_{\rm OOD}$ à petit ridge — cf. §6 ; à $4\pi/T$ avec les matrices
complètes, le ridge optimal du gap est intérieur pour tous les templates, p. ex. croix $1.4\cdot10^{-3}$ en $\kappa_0=10^{-8}$ contre $9\cdot10^{-3}$ en $10^{-10}$.)

* **Le ridge optimal pour le gap n'est pas celui de la distribution d'entraînement** : il est plus grand (il coupe les directions à fort
  levier), et l'erreur dans le gap au ridge optimal « en distribution » peut être 2.5–7 fois plus grande (gaussienne, dipôle, arc). Le
  minimum de $E_{\rm OOD}(\kappa)$ est intérieur : courbe **en U en fonction du seuil**, version ridge de la non-monotonie de Runge.
* **Variance** : le facteur d'amplification $\mathcal A$ est énorme pour les templates lisses ($10^6$–$10^7$) mais multiplié par $E_{\rm gen}\sim10^{-16}$ :
  négligeable, l'erreur dans le gap est un **biais** et ne diminue pas avec $P$ au-delà de $P\sim10^2$. Pour les bords francs,
  $E_{\rm gen}\sim10^{-7}$–$10^{-6}$ et $\mathcal A\sim10^2$–$10^3$ : la variance compte encore à $P\lesssim10^3$ (croix : $2\cdot10^{-3}$ à $P=100$ contre $4\cdot10^{-4}$ à $P=\infty$).
* **Réserve** (cf. `03` §1.3) : pour les templates lisses, le régime « sans bruit, quasi sans ridge » est celui où l'équivalence gaussienne
  est optimiste sur une variété 1D (trous d'échantillonnage) ; les prédictions à $P$ fini y sont des ordres de grandeur. Le calcul
  numérique rapporte d'ailleurs $E_{\rm OOD}(P=2000,\tau=10^{-4})\approx E_{\rm OOD}(P=\infty)$ (A5), en accord avec « biais dominant ».

**Prédiction testable (fragilité du succès lisse).** Le succès de la gaussienne repose sur $\mu_{\max}\sim10^8$ appliqué à un résidu $\sim10^{-13}$.
Tout bruit effectif (bruit de pixel, float32, rendu) de variance $\sigma^2_{\rm n}$ impose un ridge effectif $\kappa\gtrsim\sigma^2_{\rm n}$ qui éteint les directions à
fort $\mu$ : on s'attend à ce que l'erreur dans le gap de la gaussienne remonte vers la valeur au seuil correspondant
(p. ex. $\sim3\cdot10^{-4}$ si $\kappa\sim10^{-4}\lambda_1$), alors que celle des bords francs, déjà au plateau, change peu.

## 8. Résumé T4
1. **Formule des répliques** (§2.3) : $E_{\rm OOD}=E_{\rm te}(\bar W_\kappa)+\frac{E_{\rm gen}}P\sum_k(\Sigma_{\rm te})_{kk}\frac{\lambda_k}{(\lambda_k+\kappa)^2}$, $\kappa$ et $E_{\rm gen}$ d'entraînement ;
   covariance des poids diagonale dans la base d'entraînement (modèle de suite gaussienne). Vérifiée (gaussien ≤ 3 %, variété 1D 1–3 %).
2. **Identité de Runge** (§4) : à $P=\infty$, $E_{\rm OOD}=E^{\min}_{\rm te}+\sum_j\frac{(\mu_j+2)^2}{\mu_j}\langle\varepsilon_u,g_j\rangle^2_{\rm tr}$ : erreur = résidu de mauvaise spécification
   projeté sur les directions concentrées dans le gap × facteur de concentration. Sans mauvaise spécification, pas d'erreur OOD.
3. **Dichotomie** (§6) : convergence (lisse, spectre fini résolu, $2\pi/T$, seuil fin) vs plateau/non-monotonie (bords francs, spectre en loi
   de puissance jamais résolu). Elle n'existe pas au seuil $10^{-4}$ sur $\lambda_k$ ni à $\omega_{\max}=4\pi/T$ ; elle apparaît à $2\pi/T$ pour $\tau\lesssim10^{-8}$.
4. **Ridge** (§7) : ridge optimal OOD > ridge optimal en distribution ; erreur dans le gap dominée par le biais (insensible à $P$).

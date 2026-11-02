# T2 — Répliques avec ridge : erreur d'entraînement (et de généralisation) pour tout $\alpha$

*Partie analytique. Conventions de `docs/CONVENTIONS.md` : $P$ échantillons, $\Sigma=\langle\phi\phi^\top\rangle$ non centré,
$\Sigma=\sum_k\lambda_ku_ku_k^\top$, $c_k=u_k\cdot C$, ridge $\lambda_{\rm r}$.*
Vérifications : `checks/t2_sympy_ridge.py` (sympy), `checks/t2_ridge_rs_check.py` (erreurs et $\langle\ln Z\rangle$ vs simulations),
`checks/t2_entropie_ridge.py` (entropie / log-det).

## 1. Modèle

**Modèle gaussien centré (choix (b) de `01` §1)** : $(\phi,\omega)\sim\mathcal N\big(0,[[\Sigma,C],[C^\top,\langle\omega^2\rangle]]\big)$.
Dans un modèle gaussien joint, la décomposition
$$\omega=\theta_\star\cdot\phi+\varepsilon,\qquad\theta_\star=\Sigma^{-1}C\ \ (\theta_{\star,k}=c_k/\lambda_k),\qquad
\varepsilon\sim\mathcal N(0,\sigma^2),\ \ \sigma^2\equiv\sigma^2_{\rm res}=\langle\omega^2\rangle-\sum_k\frac{c_k^2}{\lambda_k}$$
est **exacte** avec $\varepsilon$ **indépendant** de $\phi$ (décorrélé ⇒ indépendant pour un vecteur gaussien). On note
$t_k=\lambda_k\theta_{\star,k}^2=c_k^2/\lambda_k$ la « puissance du professeur » dans le mode $k$ ($\sum_kt_k+\sigma^2=\langle\omega^2\rangle$).
Si $\Sigma$ est singulière, $\theta_\star$ est pris dans l'image de $\Sigma$ (pour des données réelles $C\in{\rm Im}\,\Sigma$) ;
si l'on ne dispose que des $K$ premiers modes, la puissance non résolue $\langle\omega^2\rangle-\sum_{k\le K}t_k$ est comptée dans $\sigma^2$.

$$H(W)=\frac12\sum_{\mu=1}^P(\omega_\mu-W\cdot\phi_\mu)^2+\frac{\lambda_{\rm r}}2\|W\|^2,\qquad Z=\int_{\mathbb R^N}dW\,e^{-\beta H(W)} .$$

## 2. Scaling du ridge

Le minimiseur est $W^*=(X^\top X+\lambda_{\rm r}\mathbb I)^{-1}X^\top\omega$ avec $X^\top X\simeq P\,\Sigma$ : le ridge se compare donc
aux valeurs propres **via $\lambda_{\rm r}/P$**. Dans la limite thermodynamique ($N\to\infty$, $\alpha=P/N$ fixé, valeurs
propres $O(1)$ de densité limite $\rho(\lambda)$, i.e. ${\rm Tr}\,\Sigma=O(N)$), la limite non triviale est
$$\lambda_{\rm r}=P\,\tilde\lambda\qquad(\tilde\lambda=O(1)\ \text{fixé}),$$
c.-à-d. un ridge **extensif**, ce qui revient à écrire le coût sous la forme usuelle $\frac1P\sum_\mu(\cdot)^2+\tilde\lambda\|W\|^2$.
Avec $\lambda_{\rm r}=O(1)$ le ridge disparaît à la limite (ridgeless) ; avec $\lambda_{\rm r}=O(N^2)$ il écrase tout ($W^*\to0$).
Toutes les formules ci-dessous sont écrites avec $\lambda_{\rm r}$ et $P$ explicites et ne font **pas** intervenir $N$ :
elles s'appliquent telles quelles à un spectre réel de rang effectif petit (vidéos).

## 3. Calcul de répliques

### 3.1 Moyenne sur les données
Avec $w^a=W^a-\theta_\star$, le résidu de la réplique $a$ est $u_a=\omega-W^a\cdot\phi=\varepsilon-w^a\cdot\phi$ : gaussien
centré de covariance $\sigma^2+Q^{ab}$, où
$$\boxed{Q^{ab}=w^{a\top}\Sigma\,w^b}\qquad\text{(paramètre d'ordre ; } \sigma^2+Q^{aa}=\langle(\omega-W^a\cdot\phi)^2\rangle=E_{\rm gen}(W^a)\text{)} .$$
Comme dans `01` §2 (ici sans moyenne) :
$\big\langle e^{-\frac\beta2\sum_au_a^2}\big\rangle=\det\big(\mathbb I_n+\beta(Q+\sigma^2\mathbb J_n)\big)^{-1/2}$.

### 3.2 Terme entropique (dépendant du spectre)
On impose $Q^{ab}$ par $\delta(Q^{ab}-w^{a\top}\Sigma w^b)=\int d\hat Q^{ab}e^{\frac12\hat Q^{ab}(Q^{ab}-w^{a\top}\Sigma w^b)}$. Dans la base
propre, l'intégrale sur les $w$ se **factorise par mode** : pour le mode $k$, $x_k=(w^1_k,\dots,w^n_k)$,
$$\int d^nx\ e^{-\frac12x^\top(\lambda_k\hat Q+\beta\lambda_{\rm r}\mathbb I)x-\beta\lambda_{\rm r}\theta_k\mathbb 1\cdot x-\frac{n}{2}\beta\lambda_{\rm r}\theta_k^2}
=(2\pi)^{n/2}\det(M_k)^{-1/2}e^{\frac12\beta^2\lambda_{\rm r}^2\theta_k^2\,\mathbb 1^\top M_k^{-1}\mathbb 1-\frac n2\beta\lambda_{\rm r}\theta_k^2},$$
$M_k=\lambda_k\hat Q+\beta\lambda_{\rm r}\mathbb I_n$. **Sans ridge**, $\prod_k\det(\lambda_k\hat Q)^{-1/2}=(\det\Sigma)^{-n/2}(\det\hat Q)^{-N/2}$ : le spectre
se factorise en une constante (c'est l'invariance de `01` §3). **Avec ridge**, $\lambda_k$ et $\beta\lambda_{\rm r}$ se
concurrencent mode par mode : c'est l'origine de la dépendance spectrale.

### 3.3 Ansatz RS et limite $n\to0$
$Q=\delta Q\,\mathbb I_n+Q_0\mathbb J_n$, $\hat Q=R\,\mathbb I_n-\hat q\,\mathbb J_n$. Avec
$\ln\det(a\mathbb I+b\mathbb J)=n(\ln a+b/a)+O(n^2)$, $\mathbb 1^\top(a\mathbb I+b\mathbb J)^{-1}\mathbb 1=n/a+O(n^2)$ et
$\frac12{\rm Tr}\hat QQ=\frac n2[R(\delta Q+Q_0)-\hat q\,\delta Q]+O(n^2)$ (sympy, test [1]) :
$\langle Z^n\rangle\simeq e^{n\,G_{\rm RS}}$, $\langle\ln Z\rangle={\rm extr}\,G_{\rm RS}$ avec
$$\begin{aligned}G_{\rm RS}={}&\frac N2\ln2\pi+\frac12\big[R(\delta Q+Q_0)-\hat q\,\delta Q\big]-\frac P2\Big[\ln(1+\beta\delta Q)+\frac{\beta(Q_0+\sigma^2)}{1+\beta\delta Q}\Big]\\
&+\sum_k\Big[-\frac12\ln(\lambda_kR+\beta\lambda_{\rm r})+\frac{\lambda_k\hat q+\beta^2\lambda_{\rm r}^2\theta_k^2}{2(\lambda_kR+\beta\lambda_{\rm r})}-\frac{\beta\lambda_{\rm r}\theta_k^2}{2}\Big].\end{aligned}$$

### 3.4 Équations du col (sympy, test [2])
$$\partial_{Q_0}:\ R=\frac{P\beta}{1+\beta\delta Q},\qquad\partial_{\delta Q}:\ \hat q=\frac{P\beta^2(Q_0+\sigma^2)}{(1+\beta\delta Q)^2},$$
$$\partial_{\hat q}:\ \delta Q=\sum_k\frac{\lambda_k}{\lambda_kR+\beta\lambda_{\rm r}},\qquad
\partial_R:\ Q_0=\sum_k\frac{\lambda_k(\lambda_k\hat q+\beta^2\lambda_{\rm r}^2\theta_k^2)}{(\lambda_kR+\beta\lambda_{\rm r})^2}.$$
Posons $\chi=\beta\,\delta Q$ et
$$\boxed{\kappa=\frac{\lambda_{\rm r}(1+\chi)}{P}}\ .$$
Alors $\lambda_kR+\beta\lambda_{\rm r}=\frac{\beta P}{1+\chi}(\lambda_k+\kappa)$ et l'équation en $\delta Q$ devient
$\chi=(1+\chi)\,{\rm df}(\kappa)/P$, soit $1+\chi=\big(1-{\rm df}(\kappa)/P\big)^{-1}$, d'où l'**équation auto-cohérente**
$$\boxed{\kappa\Big(1-\frac{{\rm df}(\kappa)}{P}\Big)=\frac{\lambda_{\rm r}}{P},\qquad{\rm df}(\kappa)=\sum_k\frac{\lambda_k}{\lambda_k+\kappa}}$$
(solution positive unique pour $\lambda_{\rm r}>0$ : $g(\kappa)=\kappa(1-{\rm df}/P)$ est $<0$ tant que ${\rm df}>P$ et strictement
croissante ensuite). **Tous les $\beta$ se simplifient** : $\chi$ et $Q_0$ ne dépendent pas de la température (le Gibbs est
gaussien, sa moyenne est l'estimateur ridge pour tout $\beta$ ; seule sa largeur $\delta Q=\chi/\beta$ en dépend). Avec
$$\gamma=\frac1P\sum_k\frac{\lambda_k^2}{(\lambda_k+\kappa)^2},\qquad B_0=\kappa^2\sum_k\frac{t_k}{(\lambda_k+\kappa)^2}=\kappa^2\sum_k\frac{\lambda_k\theta_{\star,k}^2}{(\lambda_k+\kappa)^2},$$
l'équation en $R$ donne $Q_0=\gamma(Q_0+\sigma^2)+B_0$.

## 4. Résultats

### 4.1 Paramètres d'ordre
$$\chi=\beta\delta Q=\frac{{\rm df}(\kappa)}{P-{\rm df}(\kappa)}>0,\qquad Q_0=\frac{\gamma\sigma^2+B_0}{1-\gamma},\qquad
\langle W_k\rangle=\frac{\lambda_k}{\lambda_k+\kappa}\theta_{\star,k}\ \ (\text{« apprenabilité » }L_k=\tfrac{\lambda_k}{\lambda_k+\kappa}).$$
### 4.2 Erreurs (conventions communes : par échantillon, sans 1/2, sans terme ridge)
$$\boxed{E_{\rm gen}=\sigma^2+Q_0+\delta Q=\frac{\sigma^2+B_0}{1-\gamma}+\frac{\chi}{\beta}}\qquad
\boxed{E_{\rm train}=\frac{Q_0+\sigma^2}{(1+\chi)^2}+\frac{\chi}{\beta(1+\chi)}}$$
(la seconde par $E_{\rm train}=-\frac2P\partial_{\beta_{\rm perte}}G_{\rm RS}$, sympy test [3]). À $T=0$ :
$$E_{\rm gen}=\frac{\sigma^2+B_0}{1-\gamma},\qquad E_{\rm train}=\Big(1-\frac{{\rm df}}P\Big)^2E_{\rm gen}=\Big(\frac{\lambda_{\rm r}}{P\kappa}\Big)^2E_{\rm gen}.$$
**Vérification numérique** (`t2_ridge_rs_check.py`, $N=300$, $\lambda_k=5k^{-1.5}$, $\beta=3$, 60 tirages ; extrait) :

| $P$ | $\lambda_{\rm r}$ | $E_{\rm train}$ sim | RS | $E_{\rm gen}$ sim | RS | $\langle\ln Z\rangle/N$ sim | RS |
|---|---|---|---|---|---|---|---|
| 90 | 1 | 0.4312(43) | 0.4284 | 2.009(14) | 2.016 | −0.0621 | −0.0610 |
| 300 | 0.03 | 0.2888(3) | 0.2896 | 2.061(6) | 2.044 | 0.4823 | 0.4793 |
| 600 | 10 | 0.8269(37) | 0.8332 | 1.0574(25) | 1.0588 | −4.2092 | −4.2216 |

Accord au % près (écarts de taille finie $O(1/N)$), y compris pour l'**énergie libre complète** (donc l'entropie).

### 4.3 Entropie : levée des pathologies du mémoire
Le Gibbs étant gaussien, $S=\beta\langle H\rangle+\langle\ln Z\rangle=\frac N2\big(1+\ln\frac{2\pi}\beta\big)-\frac12\langle\ln\det(X^\top X+\lambda_{\rm r}\mathbb I)\rangle$
avec l'équivalent déterministe issu du col (vérifié, `t2_entropie_ridge.py`, écart $<10^{-3}$ relatif) :
$$\big\langle\ln\det(X^\top X+\lambda_{\rm r}\mathbb I)\big\rangle\simeq\sum_k\ln\Big(\lambda_{\rm r}+\frac{P\lambda_k}{1+\chi}\Big)+P\ln(1+\chi)-\frac{P\chi}{1+\chi}.$$
* $\delta Q=\chi/\beta>0$ **pour tout $\alpha$** : comme $\kappa(1-{\rm df}/P)=\lambda_{\rm r}/P>0$, on a toujours ${\rm df}<P$. Plus de $\delta q<0$.
* $Z<\infty$ et $S$ fini pour tout $\alpha$ dès que $\lambda_{\rm r}>0$. Pour $\alpha<1$ (rang de $\Sigma$ $>P$) et
  $\lambda_{\rm r}\to0^+$ : $1+\chi\simeq P\kappa_0/\lambda_{\rm r}\to\infty$ et $S\simeq-\frac{N-P}{2}\ln\lambda_{\rm r}\to+\infty$ :
  c'est la trace des $N-P$ **directions plates** (dégénérescence), qui donnent une entropie $+\infty$, pas $-\infty$ (cf. erratum E3).
* $S\sim-\frac N2\ln\beta\to-\infty$ à $T\to0$ reste vrai **avec** ridge : c'est la signature normale d'un fondamental
  unique à degrés de liberté continus.

### 4.4 Limites
* **Ridgeless, $P>{\rm rang}\,\Sigma=N$** : $\kappa\to0$, ${\rm df}\to N$, $\chi\to\frac{1}{\alpha-1}$ ⇒ $\delta Q=\frac1{\beta(\alpha-1)}$ (le $\delta q^*$ du
  mémoire), $\gamma\to1/\alpha$, $B_0\to0$ : $E_{\rm gen}\to\frac{\sigma^2}{1-1/\alpha}$, $E_{\rm train}\to\sigma^2(1-1/\alpha)$ ✔ (`01`).
* **Ridgeless, $P<{\rm rang}\,\Sigma$ (sur-paramétré)** : $\kappa\to\kappa_0>0$ solution de ${\rm df}(\kappa_0)=P$ ;
  $E_{\rm train}\simeq(\lambda_{\rm r}/P\kappa_0)^2E_{\rm gen}\to0$ (**interpolation**), et
  $E_{\rm gen}\to\frac{\sigma^2+B_0(\kappa_0)}{1-\gamma(\kappa_0)}$ = erreur de l'interpolateur de norme minimale.
  Isotrope ($\lambda_k\equiv\lambda$) : $\kappa_0=\lambda(1/\alpha-1)$, $\gamma=\alpha$,
  $E_{\rm gen}=\frac{\sigma^2}{1-\alpha}+(1-\alpha)\|\theta_\star\|_\Sigma^2$ (Hastie et al. 2022, Thm 1).
* **Pic d'interpolation** $P={\rm rang}\,\Sigma$ : $\gamma\to1$, $E_{\rm gen}\to\infty$ si $\lambda_{\rm r}=0$ ; tout $\lambda_{\rm r}>0$ le régularise.
* **Ridge fort** $\lambda_{\rm r}/P\gg\lambda_1$ : $\kappa\simeq\lambda_{\rm r}/P$, $W^*\to0$, $E\to\langle\omega^2\rangle$.
* **Grand $P$ devant le nombre effectif de modes** (cas des vidéos, ${\rm df}\lesssim40$) : $\kappa\simeq\lambda_{\rm r}/P$, $\gamma=O({\rm df}/P)$ :
  $E_{\rm gen}\simeq\sigma^2+\sum_k\frac{\kappa^2t_k}{(\lambda_k+\kappa)^2}+\frac{\sigma^2+B_0}{P}\sum_k\frac{\lambda_k^2}{(\lambda_k+\kappa)^2}$ :
  risque ridge de population + variance $\propto{\rm df}_2/P$. $\kappa$ agit comme un **seuil spectral** (modes $\lambda_k\gg\kappa$ appris,
  $\lambda_k\ll\kappa$ ignorés) : c'est la version « douce » de la troncature SVD du mémoire (seuil $10^{-4}\lambda_{\max}$).

### 4.5 Forme thermodynamique
Avec $\lambda_{\rm r}=P\tilde\lambda$ et une densité spectrale $\rho$ :
$\kappa\big(1-\frac1\alpha\int\rho(\lambda)\frac{\lambda}{\lambda+\kappa}d\lambda\big)=\tilde\lambda$, $\gamma=\frac1\alpha\int\rho\frac{\lambda^2}{(\lambda+\kappa)^2}$ :
$E_{\rm train}(\alpha,\tilde\lambda)=(\tilde\lambda/\kappa)^2E_{\rm gen}(\alpha,\tilde\lambda)$, pour **tout** $\alpha$.

## 5. Lien avec la littérature

* **Hastie–Montanari–Rosset–Tibshirani (2019/2022)** : ridge $\frac1n\|y-X\beta\|^2+\lambda\|\beta\|^2$ ⇒ $\lambda_{\rm HMRT}=\lambda_{\rm r}/P$ ;
  leur cas isotrope et leur Thm pour covariance générale donnent biais $=\kappa^2\sum t_k/(\lambda_k+\kappa)^2/(1-\gamma)$ et variance
  $\sigma^2\gamma/(1-\gamma)$, identiques à 4.2 (notre $E_{\rm gen}$ inclut le bruit irréductible $\sigma^2$).
* **Bordelon–Canatar–Pehlevan (2020)** (courbes d'apprentissage spectrales des noyaux) : $\kappa_{\rm BCP}=\lambda+\kappa_{\rm BCP}\sum_k\frac{\lambda_k}{P\lambda_k+\kappa_{\rm BCP}}$
  est notre équation avec $\kappa_{\rm BCP}=P\kappa$, $\lambda=\lambda_{\rm r}$ ; leur $E_g=\frac{1}{1-\gamma}\sum_k\frac{\kappa^2}{(\lambda_k+\kappa)^2}t_k$ (+ bruit, Canatar et al. 2021) = 4.2.
  Leur « apprenabilité » mode par mode $\lambda_k/(\lambda_k+\kappa)$ = notre $L_k$ (biais spectral).
* **Loureiro et al. (2021)** (modèle de covariables gaussiennes, professeur–élève, répliques) : mêmes équations du col
  (leurs $\hat V,\hat q,\hat m$) spécialisées à la perte carrée ; ils discutent aussi l'universalité gaussienne pour des
  features réalistes — pertinent pour les vidéos (voir `03`).
* **Mignacco–Mori (2025)**, arXiv:2507.07907 (cité par le mémoire ; résumé vérifié) : équations différentielles fermées
  pour les paramètres d'ordre de la SGD en grande dimension, et conception de protocoles (curriculum, dropout adaptatif,
  bruit) comme problème de **contrôle optimal** minimisant l'erreur de généralisation. C'est un cadre **dynamique** : il ne
  contient pas ce calcul statique d'équilibre, mais partage l'idée de réduire l'apprentissage à quelques paramètres
  d'ordre. Le ridge $\lambda_{\rm r}$ joue ici le rôle d'un hyper-paramètre dont on peut chercher la valeur optimale
  (`03` §5) — l'analogue statique d'un protocole optimal ; un ridge équivaut aussi à un arrêt précoce de la descente de
  gradient (filtre spectral $1-e^{-\eta t\lambda_k}$ vs $\lambda_k/(\lambda_k+\kappa)$), ce qui donne un pont avec le « paramètre
  d'entraînement » évoqué p. 13 du mémoire.

## 6. Ce qu'il faut retenir pour la suite
1. Le paramètre d'ordre du calcul avec ridge **est** l'erreur de généralisation : $E_{\rm gen}=\sigma^2+Q^{aa}$. T3 en découle directement.
2. La dépendance au template passe par $(\lambda_k,t_k=c_k^2/\lambda_k)$ et un seul scalaire auto-cohérent $\kappa$.
3. Pour $\alpha<1$ (au sens $P<$ rang effectif) le ridge est indispensable au calcul et fixe l'échelle spectrale $\kappa$.

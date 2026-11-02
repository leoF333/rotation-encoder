# T3 — Erreur de généralisation en distribution (répliques, ridge)

*Partie analytique. Conventions communes. Vérifications : `checks/t3_*.py`, `checks/toy1d.py`.*
*(Document rédigé au fil de l'eau ; statut indiqué par section.)*

## 1. Modélisation

### 1.1 Décomposition professeur + résidu
On reprend le modèle gaussien centré de `02` : $\omega=\theta_\star\cdot\phi+\varepsilon$, $\theta_\star=\Sigma^{-1}C$, $\varepsilon\perp\phi$,
$\sigma^2=\sigma^2_{\rm res}=\langle\omega^2\rangle-\sum_kc_k^2/\lambda_k$. Dans la base propre de $\Sigma$ :
$\theta_{\star,k}=c_k/\lambda_k$, puissance du professeur $t_k=c_k^2/\lambda_k$.

### 1.2 Lecture « noyau » (utile pour les données réelles)
L'encodeur linéaire est une régression ridge à **noyau linéaire** $K(\omega,\omega')=\phi(\omega)\cdot\phi(\omega')$. L'opérateur
$T_Kf(\omega)=\int K(\omega,\omega')f(\omega')\rho(d\omega')$ a les **mêmes valeurs propres non nulles** que $\Sigma$, de
fonctions propres orthonormées dans $L^2(\rho)$ : $e_k(\omega)=u_k\cdot\phi(\omega)/\sqrt{\lambda_k}$. Alors
$\langle\omega\,e_k\rangle_\rho=c_k/\sqrt{\lambda_k}$ : **$t_k$ est le carré du $k$-ième coefficient de la fonction cible $\omega\mapsto\omega$
dans la base de Mercer**, et $\sigma^2_{\rm res}$ est le carré de la norme de la partie de la cible **hors** de l'espace engendré
par les $e_k$ (mauvaise spécification). C'est exactement le cadre de Bordelon–Canatar–Pehlevan (2020) / Canatar et al.
(2021), où la partie de la cible hors de l'espace des features se comporte comme un **bruit effectif**.

### 1.3 Validité de l'équivalence gaussienne pour des données sur une variété 1D — *tests numériques*
Pour les vidéos, $\phi(\omega)$ est une fonction **déterministe** d'un scalaire : les coordonnées $z_k=u_k\cdot\phi$ sont
décorrélées mais fortement dépendantes et bornées, et le « bruit » $\varepsilon(\omega)$ est une fonction déterministe de
$\omega$ (hétéroscédastique vis-à-vis des $z_k$). Le calcul de répliques ne voit que les deux premiers moments ; la
question est de savoir si l'erreur moyenne du ridge en dépend davantage.

**Modèle jouet** (`checks/toy1d.py`) : $\omega\sim\mathcal U[0,1]$, $\phi_0=\sqrt{\lambda_0}$, $\phi_k=\sqrt{2\lambda_k}\cos(\pi k\omega)$ ($N=300$),
donc $\Sigma={\rm diag}(\lambda_k)$ **exactement**, spectre choisi ($\lambda_k=(1+k)^{-2}$ ou $e^{-0.15k}$), cibles $y=\omega$
($t_k=8/(\pi^4k^4)$, $k$ impair) ou marche $y=\mathbb 1[\omega>0.37]$ ($t_k\sim k^{-2}$). $E_{\rm gen}$ calculé exactement pour
chaque $W$. Comparaison théorie / simulation sur la variété / simulation gaussienne de mêmes $(\lambda_k,c_k)$
(`checks/t3_variete1d_equivalence.py`, `checks/t3_variete1d_queues.py`, 40–200 tirages) :

* la simulation **gaussienne** reproduit la théorie à quelques % (attendu) ;
* sur la **variété 1D**, la théorie est précise (≲ 5–20 %) dès que l'erreur est dominée par le **bruit effectif**
  ($\sigma^2_{\rm res}$) ou par le **biais du ridge** : ex. marche, $P=320$ : théorie $1.28\cdot10^{-3}$ vs $1.23(11)\cdot10^{-3}$ ;
  $y=\omega$, $\lambda_{\rm r}/P\lambda_0=10^{-3}$, $P=320$ : $4.06\cdot10^{-6}$ vs $4.8(3)\cdot10^{-6}$ ;
* dans le régime **sans bruit et quasi sans ridge** avec une cible très lisse, la distribution de $E_{\rm gen}$ sur les jeux
  d'entraînement i.i.d. a une **queue lourde** : moyenne ≫ médiane (ex. spectre exponentiel, $y=\omega$, $P=80$ : théorie
  $2.8\cdot10^{-7}$, médiane $4.5\cdot10^{-6}$, moyenne $5.5\cdot10^{-5}$), corrélée positivement (≈ 0.5) à la taille du **plus grand
  trou** entre points d'entraînement. Avec un échantillonnage **régulier** (grille), l'erreur est au contraire *inférieure*
  à la théorie ($5.7\cdot10^{-8}$).

**Conclusion.** L'équivalence gaussienne est fiable pour l'ordre de grandeur et la dépendance spectrale dans le régime
« bruit effectif / ridge », qui est celui des expériences (le mémoire utilise une troncature $10^{-4}$) ; elle **manque un effet
géométrique propre aux variétés 1D** : les trous d'échantillonnage (de taille $\sim\ln P/P$ pour un tirage i.i.d.) doivent être
franchis par extrapolation, ce qui est exactement le mécanisme du gap (T4) à petite échelle. Les prédictions de théorie
dans le régime ultra-lisse sans bruit sont à lire comme des **ordres de grandeur optimistes** (médiane) pour des $\omega$ i.i.d.

## 2. Résultat des répliques : $E_{\rm gen}$ = biais + variance, mode par mode — *dérivé (`02`), vérifié numériquement*

Le calcul RS de `02` donne directement l'erreur de généralisation, puisque le paramètre d'ordre est
$\sigma^2+Q^{aa}=\langle(\omega-W^a\cdot\phi)^2\rangle$ (c'est le « terme de test » du mémoire, mais moyenné sur la loi des
données au lieu d'un point $\omega_t$ ; aucun terme $\gamma$ n'est nécessaire). À $T=0$ :
$$\boxed{E_{\rm gen}=\sigma^2_{\rm res}+\underbrace{\sum_k\frac{\kappa^2\,t_k}{(\lambda_k+\kappa)^2}}_{\text{biais spectral}}\;+\;\underbrace{\frac{E_{\rm gen}}{P}\sum_k\Big(\frac{\lambda_k}{\lambda_k+\kappa}\Big)^2}_{\text{variance}}
\ \Longleftrightarrow\ E_{\rm gen}=\frac{\sigma^2_{\rm res}+B_0}{1-\gamma}}$$
avec $\kappa(1-{\rm df}(\kappa)/P)=\lambda_{\rm r}/P$. Mode par mode (`theory.order_parameters` : `bias_modes`, `var_modes`) :
* **biais** du mode $k$ : $(1-L_k)^2t_k$, $L_k=\frac{\lambda_k}{\lambda_k+\kappa}$ = fraction apprise du mode ($\langle W_k\rangle=L_k\theta_{\star,k}$).
  Les modes $\lambda_k\gg\kappa$ sont appris, les modes $\lambda_k\ll\kappa$ sont ignorés : **biais spectral** ;
* **variance** du mode $k$ : $L_k^2E_{\rm gen}/P$ — chaque mode appris coûte $E_{\rm gen}/P$ (bruit effectif ré-injecté
  par l'auto-cohérence : le facteur $1/(1-\gamma)$ est une « renormalisation » du bruit par les erreurs des autres modes).

Vérifications : `checks/test_theory.py` (6 couples $(P,\lambda_{\rm r})$, spectre $k^{-1.3}$, accord ≤ 3 %),
`checks/t2_ridge_rs_check.py`, figure `figures/t3_double_descente_lois_echelle.png` (a).

## 3. Double descente — *vérifié numériquement* (Fig. t3 (a))

À $\lambda_{\rm r}\to0$ : $P<{\rm rang}$ ⇒ $\kappa\to\kappa_0$ (${\rm df}(\kappa_0)=P$), interpolation ($E_{\rm train}=0$) ;
$P>{\rm rang}$ ⇒ $\kappa=0$, $E_{\rm gen}=\sigma^2_{\rm res}/(1-{\rm rang}/P)$. Au seuil $P={\rm rang}$, $\gamma\to1$ : **pic** en
$|P-{\rm rang}|^{-1}$ (le $1/(\alpha-1)$ de $q_0^*$ dans le mémoire, `01` §7.2). Tout $\lambda_{\rm r}>0$ régularise le pic ;
pour $\lambda_{\rm r}$ assez grand la courbe devient monotone. Figure (a) : $N=200$, $\lambda_k=k^{-1.2}$, $t_k=0.5k^{-1.5}$,
$\sigma^2=0.02$, théorie (traits) vs simulation gaussienne (points, 30 tirages) : accord à quelques %.
**Pour un spectre réel** de rang numérique $K\ll N$ (troncature à $10^{-4}\lambda_{\max}$ dans le mémoire), le pic est
en $P\approx K$ (≈ 7–45 selon le template, Fig. 6–7 du mémoire), **pas** en $P=N=7840$.

## 4. Ridge optimal — *dérivé, vérifié* (`checks/t3_ridge_optimal.py`)

$\lambda_{\rm r}\mapsto\kappa$ étant croissante, $\partial_\kappa E_{\rm gen}=0$ donne la **condition d'optimalité**
$$\kappa\,(1-\gamma)\sum_k\frac{t_k\lambda_k}{(\lambda_k+\kappa)^3}=\frac{\sigma^2_{\rm res}+B_0}{P}\sum_k\frac{\lambda_k^2}{(\lambda_k+\kappa)^3}$$
(vérifiée à $10^{-5}$ près à l'optimum numérique). **Professeur « isotrope »** $t_k=\tau^2\lambda_k$ : la condition se réduit à
$\kappa(1-{\rm df}/P)=\sigma^2/(P\tau^2)$, soit
$$\boxed{\lambda_{\rm r}^*=\sigma^2_{\rm res}/\tau^2}\quad\text{indépendant de }P\text{ (estimateur bayésien : prior }\theta\sim\mathcal N(0,\tau^2\mathbb I)).$$
Vérifié : $\lambda_{\rm r}^*=0.07143=\sigma^2/\tau^2$ pour $P=50,300,1000$. En général, $\lambda^*_{\rm r}$ dépend de $P$, du spectre et
de l'alignement $t_k$ ; simulation gaussienne ($P=100$) : minimum simulé en $\lambda_{\rm r}\approx1$, théorie $\lambda^*_{\rm r}=0.86$, courbes
$E_{\rm gen}(\lambda_{\rm r})$ superposées à 1–3 %. Remarque : quand la cible est très alignée sur les premiers modes, l'optimum
peut être $\lambda_{\rm r}^*\to0$ (voire négatif, Kobak et al. 2020) ; `theory.optimal_ridge` cherche $\lambda_{\rm r}>0$.

## 5. Asymptotiques et lois d'échelle en $P$ — *dérivé, vérifié* (`checks/t3_lois_echelle.py`, Fig. t3 (b))

On note $k_\kappa$ l'indice où $\lambda_{k_\kappa}=\kappa$. Estimations : ${\rm df}(\kappa)\approx k_\kappa$ (à $O(1/a)$ près pour un spectre
exponentiel), $B_0\approx\sum_{k>k_\kappa}t_k+\kappa^2\sum_{k<k_\kappa}t_k/\lambda_k^2$, et
$1-\gamma=\frac1P\sum_kL_k(1-L_k)+(1-{\rm df}/P)$ ; au point ridgeless ${\rm df}=P$ : $1-\gamma=\frac1P\sum_kL_k(1-L_k)$ = (nombre de
modes dans la « zone de transition » $\lambda_k\approx\kappa$)$/P$.

**A. Loi de puissance** $\lambda_k=k^{-a}$ ($a>1$), $t_k=k^{-b}$ ($b>1$) — typique des **templates à bords francs**.
| régime | loi | vérif. numérique (pente mesurée / attendue) |
|---|---|---|
| sans bruit, ridgeless ($\kappa_0\sim P^{-a}$) | $E_{\rm gen}\sim P^{-(b-1)}$ (si $b<2a+1$, sinon $P^{-2a}$) | $-1.008/-1$, $-1.998/-2$, $-2.996/-3$, $-3.02/-3$ (sat.) |
| bruit $\sigma^2$, ridgeless | $E_{\rm gen}\to\sigma^2/(1-\gamma_\infty)=a\,\sigma^2$ : sur-apprentissage **tempéré** | $E/\sigma^2=2.01$ ($a=2$), $1.56$ ($a=1.5$) |
| bruit, ridge optimal | $E_{\rm gen}-\sigma^2\sim(\sigma^2/P)^{(b-1)/b}$ ; saturation $P^{-2a/(2a+1)}$ si $b>2a+1$ | $-0.682/-0.667$, $-0.753/-0.750$, $-0.754/-0.75$ (sat.) |

($\gamma_\infty=1-1/a$ car $\int_0^\infty\frac{dx}{(1+x^a)^2}=(1-\frac1a)\int_0^\infty\frac{dx}{1+x^a}$.) On retrouve les taux « source/capacité »
classiques (Caponnetto–De Vito 2007 ; Cui et al. 2021) avec $b=1+2ar$.

**B. Spectre exponentiel** $\lambda_k=e^{-ak}$ — typique des **templates lisses** (§6 : même super-exponentiel).
| régime | loi | vérif. |
|---|---|---|
| sans bruit, ridgeless, $t_k=e^{-bk}$ | $E_{\rm gen}\simeq C\,aP\sum_{k>P}t_k\sim P\,e^{-\min(b,2a)P}$ | pente de $\ln E$ : $-0.385/-0.4$ ; $E/(aP\cdot\text{queue})=1.62$ constant |
| sans bruit, ridgeless, $t_k=k^{-b}$ | $E_{\rm gen}\sim aP\cdot P^{1-b}/(b-1)\sim P^{2-b}$ (**une puissance de $P$ perdue**) | $-1.20/-1$ (pré-asympt.) |
| bruit, ridgeless | $E_{\rm gen}-E^{\sigma=0}\simeq aP\,\sigma^2$ : sur-apprentissage **catastrophique** | $12,18,24,30$ vs $aP=12,18,24,30$ ✔ |
| bruit, ridge optimal, $t_k=e^{-bk}$ | $E_{\rm gen}-\sigma^2\simeq\frac{\sigma^2}{bP}\ln\frac{P}{\sigma^2}$ (quasi paramétrique) | $(E-\sigma^2)P/\sigma^2\approx36$ vs $\ln(P/\sigma^2)/b\approx33$ |
| bruit, ridge optimal, $t_k=k^{-b}$ | $E_{\rm gen}-\sigma^2\sim(\sigma^2/P)^{(b-1)/b}$ | $-0.660/-0.667$ |

L'origine du facteur $aP$ : pour un spectre exponentiel, ${\rm df}$ et ${\rm df}_2$ ne diffèrent que par les $\sim1/a$ modes de la zone de
transition, donc $1-\gamma\simeq1/(aP)$ au point d'interpolation.

**Lecture « régularité du template ↔ généralisation » (en distribution).**
1. À ridge **optimal**, le taux ne dépend que de la décroissance $b$ de la **cible** dans la base propre (et du niveau de bruit
   effectif), pas de celle du spectre $a$ : spectres lisses et à bords francs généralisent au même taux (Fig. t3 (b), traits
   pleins). Le spectre fixe la **valeur** de $\lambda^*_{\rm r}$ (exponentiellement petite pour un spectre exponentiel) et le
   pré-facteur aux petits $P$ (alignement : un spectre qui décroît vite dans l'indice où la cible est concentrée favorise les petits $P$).
2. **Sans régularisation**, un spectre lisse (exponentiel) est **plus fragile** : l'interpolation amplifie le bruit effectif
   (mauvaise spécification, erreurs d'arrondi) d'un facteur $aP$, contre un facteur $a$ pour un spectre en loi de puissance.
   C'est la version théorique du constat pratique du mémoire (p. 8) : pour les templates lisses, l'inversion directe échoue et
   une troncature spectrale (= ridge) est indispensable.
3. La troncature numérique à $K$ modes ajoute un plancher $\sigma^2_{{\rm res},K}=\langle\omega^2\rangle-\sum_{k\le K}t_k$ ; il est plus haut
   quand $K$ est petit (templates lisses) **si** la cible a du poids au-delà du mode $K$.
Conclusion : en distribution, la dichotomie lisse / bords francs est **modérée** et dépend du régime de régularisation ; la
dichotomie spectaculaire du mémoire (gap) a une autre origine, traitée en T4.

## 6. Application aux templates en rotation : structure du noyau — *dérivé, vérifié (continuum)* (`checks/t3_noyau_rotation.py`)

Avec $\phi_{x,t}(\omega)=s_0(R_{-\omega t}x)$ (image $t$ = template tourné de $\omega t$), l'invariance par rotation de la mesure
$dx$ donne un **noyau stationnaire** en $\omega$ :
$$K(\omega,\omega')=\sum_{t=0}^{T-1}\mathcal A\big(t(\omega-\omega')\big),\qquad \mathcal A(\theta)=\int s_0(x)s_0(R_\theta x)\,dx=\sum_{m\in\mathbb Z}S_me^{im\theta},$$
$S_m=2\pi\int|s_m(r)|^2r\,dr\ge0$ = **spectre de puissance angulaire** du template ($s_0=\sum_ms_m(r)e^{im\theta}$). Le terme $t=0$
est constant (image 0 identique pour tout $\omega$) : c'est la direction de la moyenne. $K$ ne contient que les fréquences
$\nu=mt$, de poids $w_\nu=\sum_{t\,|\,\nu,\,t<T}S_{\nu/t}$. Sur l'intervalle $[0,\omega_{\max}]$ (ou le créneau), les valeurs propres
$\lambda_k$ sont, en ordre de grandeur, les poids $w_\nu$ regroupés par cellules de résolution $\Delta\nu\approx\pi/\omega_{\max}$ (≈ 5 pour
$\omega_{\max}=2\pi/10$) ; la décroissance de $\lambda_k$ reflète donc celle de $S_m$ :

* **Blob gaussien** (centre à $r_0$, largeur $\sigma$) : $\mathcal A(\theta)=\pi\sigma^2e^{-z_0(1-\cos\theta)}$ (recouvrement de deux gaussiennes), d'où
  exactement $$S_m=\pi\sigma^2e^{-z_0}I_m(z_0),\qquad z_0=\frac{r_0^2}{2\sigma^2}$$ (vérifié numériquement à 5 chiffres pour $m\le25$).
  Décroissance **super-exponentielle** $S_m\sim(ez_0/2m)^m/\sqrt{m}$ au-delà de $m\approx z_0$ : c'est la version intégrée de
  l'argument de Jacobi–Anger du mémoire (Eq. 3, qui porte sur $I_k(rr_0/\sigma^2)$ à rayon fixé). Nombre de modes utiles
  $\sim$ quelques $z_0\times T\,\omega_{\max}/\pi$.
* **Bords francs** (croix, quart de disque, segment) : $\mathcal A(\theta)=\mathcal A(0)-c|\theta|+\dots$ (l'aire de la différence
  symétrique croît linéairement avec l'angle), d'où $$S_m\sim m^{-2}$$ (pentes mesurées $-1.99$ pour les trois), jusqu'à la
  coupure de pixelisation $m\sim\pi r_{\max}/\text{pixel}$ ($\approx40$ pour $d=28$). Donc $a\approx2$ dans le « bulk ».
* **Symétrie d'ordre 4** (croix) : $S_m=0$ pour $m\not\equiv0\pmod4$ (vérifié) : un mode angulaire sur quatre, d'où moins de
  modes dominants que pour le quart de cercle, comme observé dans le mémoire — sans changer l'exposant de la queue.

**Largeur de bande.** Les fréquences de $K$ sont bornées par $\nu_{\max}\approx(T-1)\,m_{\max}$ où $m_{\max}$ = ordre angulaire
au-delà duquel $S_m$ est négligeable : $m_{\max}\approx$ quelques $z_0$ (≈ 10) pour la gaussienne, $\approx40$ (pixel) pour les
bords francs. Cette **largeur de bande** $W=\nu_{\max}$ est la quantité clé du gap (T4).

## 7. Résumé T3
1. $E_{\rm gen}(\alpha,\lambda_{\rm r})$ exact dans le modèle gaussien (formule encadrée §2), biais spectral mode par mode,
   double descente (pic en $P$ = rang effectif), ridge optimal (condition + cas bayésien).
2. Lois d'échelle (tableaux §5) : à ridge optimal, taux fixé par la cible ($b$) ; sans ridge, spectre exponentiel ⇒
   amplification $aP$ du bruit effectif (catastrophique), loi de puissance ⇒ facteur $a$ (tempéré).
3. Équivalence gaussienne sur variété 1D : bonne dans le régime bruit/ridge, optimiste (queues lourdes, trous
   d'échantillonnage) dans le régime lisse sans bruit.
4. Structure du noyau des vidéos : stationnaire, spectre $\leftrightarrow$ $S_m$ (Bessel pour la gaussienne, $m^{-2}$ pour les bords).

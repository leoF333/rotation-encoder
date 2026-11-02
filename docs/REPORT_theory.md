# Encodeur linéaire de la vitesse de rotation : théorie des répliques, généralisation et extrapolation dans un gap

*Rapport de la partie analytique — version du 18/09/2026. Document de synthèse ; les dérivations complètes, les vérifications
et les chiffres détaillés sont dans `theory/derivations/01…05_*.md`, les scripts dans `theory/checks/`, les formules
exécutables dans `src/aerot/theory.py` (v0.2). Conventions : `docs/CONVENTIONS.md`.*

---

## Résumé

On reprend le calcul de répliques du mémoire (encodeur linéaire $\hat\omega=W\cdot\phi(\omega)$ appris par moindres carrés sur $P$ vidéos
d'un objet en rotation) et on l'étend dans trois directions : régularisation ridge, erreur de généralisation en distribution,
et erreur hors distribution dans un « gap » de vitesses absentes de l'entraînement.

1. **Re-dérivation.** Le calcul du mémoire est correct dans sa structure mais contient une erreur de signe ($m^*$) et un
   double comptage de la moyenne. Une fois corrigé, il redonne exactement le résultat connu des moindres carrés gaussiens,
   $E_{\rm train}=\sigma^2_{\rm res}(1-1/\alpha)$, et le paramètre d'ordre $q_0^*$ du mémoire **est** l'erreur de généralisation
   $\sigma^2_{\rm res}\,\alpha/(\alpha-1)$ (pic de double descente). Sans régularisation, la théorie ne dépend du template **que** par
   la variance résiduelle $\sigma^2_{\rm res}$ : le spectre n'y joue aucun rôle.
2. **Ridge.** Avec $\frac{\lambda_{\rm r}}2\|W\|^2$, tout se ramène à une échelle spectrale auto-cohérente $\kappa$,
   $\kappa(1-{\rm df}(\kappa)/P)=\lambda_{\rm r}/P$, valable pour tout $\alpha$. Les pathologies du mémoire ($\delta q<0$, entropie) disparaissent
   et la dépendance au spectre apparaît.
3. **Généralisation en distribution.** $E_{\rm gen}$ = bruit effectif + biais spectral mode par mode + variance, avec double
   descente au rang effectif, ridge optimal et lois d'échelle. En distribution, templates lisses et à bords francs
   généralisent tous deux très bien : la différence spectaculaire est ailleurs.
4. **Gap (hors distribution).** Formule des répliques pour $E_{\rm OOD}$ et **identité exacte « de Runge »** : l'erreur dans le gap est le
   résidu de mauvaise spécification projeté sur les directions concentrées dans le gap, amplifié par leur facteur de concentration.
   La dichotomie du mémoire (gaussienne ~1e-5, bords francs 1e-3–1e-2) s'explique par la compétition entre la vitesse
   d'approximation (super-exponentielle pour un template lisse, algébrique pour des bords francs) et l'amplification. Elle n'existe
   qu'à $\omega_{\max}=2\pi/T$ avec une coupure spectrale fine ($\lambda_k/\lambda_1>10^{-8}$).
5. **Confrontation.** Les formules collent aux simulations de la partie numérique sur les vraies vidéos (écart médian 0.8 % sur
   $E_{\rm OOD}$ pour $P\ge320$). L'équivalence gaussienne vaut tant que $x={\rm df}(\kappa)\ln P/P<1$ ; au-delà, les trous d'échantillonnage
   de la variété 1D agissent comme des mini-gaps.

---

## 1. Cadre

**Données.** Un template $s_0$ ($28\times28$) est tourné d'un angle $\omega t$ à chaque image $t=0,\dots,T-1$ ($T=10$) ; la vidéo aplatie
$\phi(\omega)\in\mathbb R^N$, $N=7840$, décrit une **variété de dimension 1**. On apprend $\omega$ par un encodeur linéaire sans biais :
$$H(W)=\frac12\sum_{\mu=1}^P(\omega_\mu-W\cdot\phi_\mu)^2+\frac{\lambda_{\rm r}}2\|W\|^2 .$$
Seconds moments **non centrés** : $\Sigma=\langle\phi\phi^\top\rangle=\sum_k\lambda_ku_ku_k^\top$, $C=\langle\omega\phi\rangle$, $c_k=u_k\cdot C$.
Erreurs par échantillon, sans facteur ½ ; $\alpha=P/N$.

**Modèle gaussien équivalent.** Le calcul de répliques traite $(\phi,\omega)$ comme gaussien de mêmes seconds moments. On écrit alors
exactement $\omega=\theta_\star\cdot\phi+\varepsilon$ avec $\theta_{\star,k}=c_k/\lambda_k$ (meilleur prédicteur linéaire) et $\varepsilon$ indépendant de $\phi$,
de variance $\sigma^2_{\rm res}=\langle\omega^2\rangle-\sum_kc_k^2/\lambda_k$. On appelle $t_k=c_k^2/\lambda_k$ la **puissance du professeur** dans le mode $k$ :
c'est le carré du $k$-ième coefficient de la cible dans la base de Mercer du noyau $K(\omega,\omega')=\phi(\omega)\cdot\phi(\omega')$. Le résidu de
mauvaise spécification (la part de $\omega$ non linéaire en $\phi$) joue le rôle d'un bruit effectif.

**Structure du noyau des vidéos** (`03` §6). Par invariance de la mesure par rotation, $K(\omega,\omega')=\sum_t\mathcal A(t(\omega-\omega'))$ est
**stationnaire**, avec $\mathcal A(\theta)=\sum_mS_me^{im\theta}$ et $S_m$ le **spectre de puissance angulaire** du template. Pour un blob gaussien
(centre $r_0$, largeur $\sigma$) : $S_m=\pi\sigma^2e^{-z_0}I_m(z_0)$, $z_0=r_0^2/2\sigma^2$, soit une décroissance super-exponentielle (version intégrée
de l'argument de Jacobi–Anger du mémoire). Pour un template à bords francs : $S_m\sim m^{-2}$ jusqu'à la coupure de pixelisation.
La croix n'a que les ordres $m\equiv0\pmod4$. Les valeurs propres $\lambda_k$ héritent de ces décroissances.

---

## 2. Le calcul du mémoire, re-dérivé et corrigé (`derivations/01_reproduction_memoire.md`)

On a refait tout le calcul de l'annexe : moyenne sur le désordre, paramètres d'ordre $(q^{ab},\tilde q^a,m^a)$, élimination des champs
conjugués, ansatz RS, limite $n\to0$, équations du col, énergie libre et entropie. Chaque étape a été vérifiée avec sympy
(`checks/t1_sympy_memoire.py`) et sur des simulations gaussiennes (`checks/t1_numerique_ols.py`).

**Résultats.**
* Au col corrigé, $\tilde q^{*2}+2m^*+\sigma^2_\omega+c^*=\sigma^2_{\rm res}$ : la combinaison qui apparaît dans l'énergie libre est la variance résiduelle
  du meilleur prédicteur linéaire. Donc
  $$E_{\rm train}=\sigma^2_{\rm res}\Big(1-\frac1\alpha\Big),\qquad E_{\rm gen}=q_0^*+\tilde q^{*2}=\sigma^2_{\rm res}\frac{\alpha}{\alpha-1}\qquad(\alpha>1,\ \lambda_{\rm r}=0),$$
  ce qui coïncide avec le résultat exact des moindres carrés gaussiens ($\sigma^2_{\rm res}(P-N)/P$ et $\sigma^2_{\rm res}(P-1)/(P-N-1)$).
  Simulations : accord à 1 % pour $\alpha=1.25$ à 4.
* Le « $q_0^*$ » laissé sans interprétation p. 28 est l'**erreur de généralisation**. Sa divergence en $1/(\alpha-1)$ est le **pic
  d'interpolation** (double descente).
* $\delta q^*=1/(\beta(\alpha-1))$ ne dépend pas du spectre, et c'est normal. Par le changement de variables $V=\mathcal M^{1/2}W$, **toute** la
  théorie sans ridge ne dépend du template que par $\sigma^2_{\rm res}$.
* Entropie : la dépendance en $\alpha$ de l'Eq. 21 est exacte (vérifié contre Marchenko–Pastur) à une constante près.
  $s\to-\infty$ à $T\to0$ est la signature normale d'un fondamental **unique** pour des degrés de liberté continus (comme l'oscillateur
  harmonique classique). La vraie pathologie de $\alpha<1$ est inverse : $N-P$ directions plates, donc $Z=+\infty$.

**Portée pour les expériences.** $\mathcal M$ a un rang effectif $K\approx30$–$45\ll N$ : $\alpha=P/N$ n'est pas le bon paramètre. Après la
troncature spectrale du mémoire, on a exactement $E_{\rm train}=\sigma^2_{{\rm res},K}(1-K/P)$. Le cadre adapté est la théorie ridge à spectre
quelconque (§3), où $N$ disparaît.

---

## 3. Ridge : une échelle spectrale auto-cohérente (`derivations/02_ridge_entrainement.md`)

Le ridge se compare aux valeurs propres via $\lambda_{\rm r}/P$. La limite thermodynamique non triviale est donc $\lambda_{\rm r}=P\tilde\lambda$ (ridge
extensif, ce qui équivaut au coût usuel $\frac1P\sum_\mu(\cdot)^2+\tilde\lambda\|W\|^2$). Le calcul RS se factorise par mode propre. Toutes les
températures se simplifient, et tout est gouverné par
$$\boxed{\kappa\Big(1-\frac{{\rm df}(\kappa)}P\Big)=\frac{\lambda_{\rm r}}P,\qquad{\rm df}(\kappa)=\sum_k\frac{\lambda_k}{\lambda_k+\kappa}}$$
($\kappa$ a la dimension d'une valeur propre : les modes $\lambda_k\gg\kappa$ sont appris, les modes $\lambda_k\ll\kappa$ ignorés). Avec $\gamma=\frac1P\sum_k\frac{\lambda_k^2}{(\lambda_k+\kappa)^2}$ et
$B_0=\kappa^2\sum_k\frac{t_k}{(\lambda_k+\kappa)^2}$ :
$$E_{\rm gen}=\frac{\sigma^2_{\rm res}+B_0}{1-\gamma}+\frac{\chi}{\beta},\qquad E_{\rm train}=\Big(1-\frac{\rm df}P\Big)^2E_{\rm gen}\ (T=0),\qquad\chi=\beta\delta Q=\frac{\rm df}{P-{\rm df}}>0 .$$
Ces formules valent pour **tout** $\alpha$. $\delta Q>0$ toujours, $Z$ et l'entropie sont finis, et à $\lambda_{\rm r}\to0^+$ avec $\alpha<1$ on a
$S\simeq-\frac{N-P}2\ln\lambda_{\rm r}\to+\infty$ (les directions plates). Limites : ridgeless $P>$ rang (on retrouve §2) ; ridgeless $P<$ rang
(interpolateur de norme minimale, ${\rm df}(\kappa_0)=P$) ; cas isotrope de Hastie–Montanari–Rosset–Tibshirani. L'équation est celle de
Bordelon–Canatar–Pehlevan (2020) avec $\kappa_{\rm BCP}=P\kappa$. Vérifications : sympy (équations du col) et simulations gaussiennes des
erreurs **et** de $\langle\ln Z\rangle$ (au % près).

Le ridge est aussi la version « douce » de la troncature SVD du mémoire : $\kappa$ agit comme un seuil spectral.

---

## 4. Généralisation en distribution (`derivations/03_generalisation.md`)

$$E_{\rm gen}=\sigma^2_{\rm res}+\underbrace{\sum_k(1-L_k)^2t_k}_{\text{biais spectral}}+\underbrace{\frac{E_{\rm gen}}P\sum_kL_k^2}_{\text{variance}},\qquad L_k=\frac{\lambda_k}{\lambda_k+\kappa}\ \ (\langle W_k\rangle=L_k\theta_{\star,k}).$$
* **Double descente** au rang **effectif** du spectre (≈ 7–45 pour les vidéos), pas en $P=N$. Tout ridge la régularise.
* **Ridge optimal** : condition de stationnarité explicite. Si $t_k=\tau^2\lambda_k$, on obtient $\lambda_{\rm r}^*=\sigma^2_{\rm res}/\tau^2$, indépendant de $P$
  (estimateur bayésien).
* **Lois d'échelle.** Pour $\lambda_k\sim k^{-a}$, $t_k\sim k^{-b}$ (bords francs) : $E_{\rm gen}\sim P^{-(b-1)}$ sans bruit ; sur-apprentissage *tempéré*
  ($E\to a\sigma^2$) sans ridge avec bruit ; $(\sigma^2/P)^{(b-1)/b}$ au ridge optimal. Pour un spectre exponentiel (templates lisses), le
  sur-apprentissage sans ridge est *catastrophique* ($E-E^{\sigma=0}\simeq aP\sigma^2$). Sur les vrais spectres (`05` §6) : $a\approx2.3$–$3.4$ et
  $b\approx2.3$–$2.9$ pour les bords francs ; décroissance $e^{-0.5k}$ à $e^{-0.85k}$ et 30–60 modes non négligeables pour les templates lisses.
* **Lecture.** Au ridge optimal, le taux est fixé par la cible ($b$), pas par le spectre. Sans régularisation, un spectre lisse est
  **plus fragile** au bruit effectif, ce qui explique pourquoi le mémoire devait tronquer. En distribution, la dichotomie
  lisse / bords francs est donc modérée.
* **Équivalence gaussienne sur une variété 1D** (modèle jouet, `03` §1.3) : bonne dans le régime « bruit effectif / ridge ». Elle est
  optimiste dans le régime lisse sans bruit, où l'erreur a une queue lourde gouvernée par le plus grand trou entre points
  d'entraînement. C'est quantifié sur les vraies vidéos au §6.

---

## 5. L'erreur dans le gap (`derivations/04_generalisation_gap.md`)

**Cadre.** Entraînement sur le créneau $[0,\frac{\omega_{\max}}3]\cup[\frac{2\omega_{\max}}3,\omega_{\max}]$, test sur le gap. C'est un changement de covariables
pur (même cible, même carte $\phi$). Mais le modèle linéaire est mal spécifié, donc les meilleurs prédicteurs d'entraînement et de
test diffèrent. Pour tout $W$, $E_{\rm te}(W)=\langle\omega^2\rangle_{\rm te}-2C_{\rm te}\cdot W+W^\top\Sigma_{\rm te}W$ ne dépend que des moments de test, exprimés dans la
base propre d'entraînement.

### 5.1 Formule des répliques
L'erreur de test est une observable « à une réplique » : on la moyenne sous la mesure répliquée au col RS de §3. Un découplage de
Hubbard–Stratonovich montre qu'au col, chaque poids se comporte comme dans un **modèle de suite gaussienne** :
$$W_k\stackrel{\rm loi}{=}\frac{\lambda_k\theta_{\star,k}+\sqrt{\lambda_kE_{\rm gen}/P}\,z_k}{\lambda_k+\kappa},\qquad z_k\ \text{i.i.d.}\ \mathcal N(0,1),$$
et la covariance des poids est **diagonale dans la base d'entraînement**. D'où
$$\boxed{E_{\rm OOD}=E_{\rm te}(\bar W_\kappa)+\frac{E_{\rm gen}}P\sum_k(\Sigma_{\rm te})_{kk}\frac{\lambda_k}{(\lambda_k+\kappa)^2},\qquad\bar W_{\kappa,k}=\frac{c_k}{\lambda_k+\kappa}}$$
(avec $\kappa$ et $E_{\rm gen}$ de la distribution d'entraînement). Le premier terme est un **biais** : l'erreur de test du prédicteur ridge
« population », qui contient toute la mauvaise spécification. Le second est une **variance** où chaque mode appris est pondéré par son
**levier** $r_k=(\Sigma_{\rm te})_{kk}/\lambda_k$ (énergie de test / énergie d'entraînement de la $k$-ième fonction propre). En distribution, on retrouve
§4. La formule rejoint Canatar–Bordelon–Pehlevan (2021) pour les noyaux.
Vérifications :
* données gaussiennes avec $\Sigma_{\rm te}$ non diagonale et cible de test mal spécifiée : ≤ 3 %, sans biais systématique pour
  $N=125\to1000$ ;
* variété 1D jouet : 1–3 % ;
* vraies vidéos (§6).

### 5.2 Identité de Runge
On se place à $P=\infty$ avec la pinv tronquée aux $K$ premiers modes. Soit $\Sigma_{\rm te}v_j=\mu_j\Sigma_{\rm tr}v_j$ le **spectre de concentration**
($g_j=v_j\cdot\phi$ a une énergie 1 sur l'entraînement et $\mu_j$ dans le gap ; $\mu_j\gg1$ correspond à une direction concentrée dans le gap,
analogue discret des fonctions de Slepian). Soit $\varepsilon_u$ le résidu du meilleur prédicteur sur la loi uniforme (sans gap). Alors,
exactement :
$$\boxed{E_{\rm OOD}(K)=E^{\min}_{\rm te}(K)+\sum_j\frac{(\mu_j+2)^2}{\mu_j}\,\langle\varepsilon_u,g_j\rangle^2_{\rm tr}}\ \le\ E^{\min}_{\rm te}+\frac{(\mu_{\max}+2)^2}{\mu_{\max}}\,\|\varepsilon_u\|^2_{\rm tr}.$$
* **Sans mauvaise spécification, il n'y a pas d'erreur dans le gap.** Le changement de loi n'agit qu'en amplifiant le résidu.
* Sur les vrais spectres, $E^{\min}_{\rm te}$ est négligeable : l'erreur dans le gap est **entièrement** le second terme.
* C'est la structure du phénomène de Runge : erreur ≤ (1 + constante de Lebesgue) × meilleure approximation. Ici $\mu_{\max}(K)$ joue le
  rôle de la constante de Lebesgue au carré et $\|\varepsilon_u\|^2(K)$ celui de l'erreur de meilleure approximation. Quand la coupure
  s'affine, la seconde décroît et la première croît : l'erreur converge si et seulement si l'approximation gagne.

### 5.3 La dichotomie du mémoire, expliquée
| | gaussienne | dipôle | quart de cercle | croix | segment | arc |
|---|---|---|---|---|---|---|
| $2\pi/T$, $\tau=10^{-4}$ | $2.8\cdot10^{-4}$ | $3.8\cdot10^{-4}$ | $1.5\cdot10^{-4}$ | $2.2\cdot10^{-3}$ | $4.1\cdot10^{-3}$ | $1.8\cdot10^{-4}$ |
| $2\pi/T$, $\tau=10^{-8}$ | $2.1\cdot10^{-6}$ | $4.1\cdot10^{-6}$ | $1.3\cdot10^{-4}$ | $4\cdot10^{-4}$ | $1.7\cdot10^{-3}$ | $4.0\cdot10^{-4}$ |
| $4\pi/T$, résolution complète | $6\cdot10^{-4}$ | $2\cdot10^{-4}$ | $7.9\cdot10^{-3}$ | $1.1\cdot10^{-2}$ | $1.8\cdot10^{-3}$ | — |

($E_{\rm OOD}$ moyen dans le gap, $P=\infty$, pinv tronquée à $\lambda_k/\lambda_1>\tau$ ; figure `figures/t4_gap_runge.png`.)
1. **Template lisse, spectre résolu.** Le spectre est super-exponentiel et effectivement fini ($K(10^{-12})=28$ à $2\pi/T$). Le résidu perd
   7 décades quand $\mu_{\max}$ en gagne 6 : l'erreur converge vers $10^{-6}$–$10^{-7}$. C'est la continuation analytique d'une fonction à bande
   limitée, mal conditionnée ($\mu_{\max}\sim10^8$) mais convergente.
2. **Bords francs.** Le spectre suit une loi de puissance jusqu'à la coupure pixel (> 1400 modes). Le résidu ne décroît qu'algébriquement,
   et des centaines de directions ont $\mu_j\in[1,10^6]$. L'erreur stagne à $10^{-4}$–$2\cdot10^{-3}$, voire **remonte** quand on affine la coupure
   (non-monotonie de type Runge). L'erreur d'entraînement reste pourtant $10^{-9}$–$10^{-7}$.
3. **La dichotomie dépend de la coupure.** À $\tau=10^{-4}$ sur les $\lambda_k$, elle n'existe pas. Elle apparaît pour $\tau\lesssim10^{-8}$, soit
   `rcond` $=10^{-4}$ sur les valeurs singulières de $\Phi$, ce qui est l'hypothèse de la partie numérique. Pics dans le gap :
   $5.7\cdot10^{-6}$ (gaussienne) contre $5\cdot10^{-4}$–$3.4\cdot10^{-3}$ (bords francs), à comparer au mémoire ($9.3\cdot10^{-6}$ contre $1.2\cdot10^{-3}$–$1.3\cdot10^{-2}$).
4. **La dichotomie dépend de la géométrie.** À $\omega_{\max}=4\pi/T$, le gap est deux fois plus long à largeur de bande fixée. Le nombre de
   directions concentrées double, $\mu_{\max}$ passe à $10^{10}$, et 15 % de l'énergie de test de la gaussienne tombe hors des modes
   d'entraînement résolus : l'écart lisse / bords francs se réduit à ×3–20.

**Quantité spectrale qui contrôle l'erreur.** C'est la distribution du résidu de mauvaise spécification sur le spectre de
concentration $\{\mu_j\}$ du faisceau $(\Sigma_{\rm te},\Sigma_{\rm tr})$ (plus, à $P$ fini, la variance $\frac{E_{\rm gen}}P\sum_kr_kL_k^2$). Ses deux ingrédients :
* la vitesse d'approximation de $\omega$ par les modes, fixée par la décroissance de $S_m$ (régularité du template) ;
* le nombre de Slepian, soit (largeur de bande $\approx(T-1)m_{\max}$) × (longueur du gap)$/\pi$.

L'analogie de Runge du mémoire est donc juste dans son principe, mais la décroissance lente du spectre n'est pas la cause unique :
un modèle jouet à large bande lisse est catastrophique (`04` §5). C'est la **compétition** approximation / amplification qui décide.

### 5.4 Ridge et $P$ fini
* Le ridge optimal pour le gap est plus grand que celui qui optimise l'erreur en distribution, car il coupe les directions à fort
  levier. Au ridge optimal « en distribution », l'erreur dans le gap est 2.5–7 fois plus grande (gaussienne, dipôle, arc).
* Pour les templates lisses, l'erreur dans le gap est un biais : la variance a un levier $\mathcal A\sim10^6$ mais multiplie
  $E_{\rm gen}\sim10^{-16}$, donc $E_{\rm OOD}$ ne dépend pas de $P$ au-delà de $\sim10^2$ (confirmé par la partie numérique).
* **Prédiction testable.** Le succès lisse repose sur $\mu_{\max}\sim10^8$ multiplié par un résidu $\sim10^{-13}$. Un petit bruit de pixel ou le
  passage en float32 devrait faire remonter l'erreur de la gaussienne dans le gap, beaucoup moins celle des bords francs.

---

## 6. Confrontation aux simulations (`derivations/05_confrontation.md`)

Les simulations de la partie numérique (B1) portent sur $4\pi/T$, 4 templates, $P\in[5,7000]$ et 4 ridges, avec les vraies vidéos et des
substituts gaussiens.
* **Substituts gaussiens** : $E_{\rm train}$, $E_{\rm gen}$, $E_{\rm OOD}$ théoriques à quelques % dès $P\ge80$. Le calcul et son implémentation sont
  validés sur des spectres réels, de rang effectif petit.
* **Vraies vidéos, gap** : écart médian **0.8 %** pour $P\ge320$ (64 points), ce qui confirme quantitativement la formule de §5.1. Il faut
  pour cela une matrice de test sur **tous** les modes non nuls : tronquée à 1000 modes, elle biaise le résultat à petit ridge
  (`ood_error` émet un avertissement).
* **Validité de l'équivalence gaussienne** (critère proposé et testé, `figures/t5_equivalence_gaussienne.png`). Sur la variété, le plus
  grand trou d'échantillonnage vaut $\approx\omega_{\max}\ln P/P$. Chaque trou est un **mini-gap** où joue l'amplification de §5.2. Avec
  $x={\rm df}(\kappa)\ln P/P$ :
  * pour $x<1$, le rapport réel / théorie a une médiane de 1 et un maximum de ×1.8, sur les 4 templates ;
  * pour $x>1$, les écarts grandissent (jusqu'à ×900 pour la gaussienne à petit ridge, dont l'erreur théorique est minuscule).

  En pratique, les formules valent pour $P\gtrsim{\rm df}(\kappa)\ln P$.
* **Erratum E2 sur données réelles** : impact $\le1.4\cdot10^{-6}$ relatif. L'image $t=0$ ne dépend pas de $\omega$, donc la constante est
  exactement représentable et le modèle possède de fait un intercept.

---

## 7. Errata du mémoire

| | page / équation | nature | correction / commentaire |
|---|---|---|---|
| **E1** | p. 7, 14, Eq. 8, annexe p. 20 | gaussienne de moyenne $\bar\phi$ **et** de covariance $\mathcal M=\langle\phi\phi^\top\rangle$ non centrée : double comptage de la moyenne | covariance centrée, ou modèle centré à moments non centrés (`01` §1) |
| **E2** | Eq. 18 p. 16, p. 28–29 | signe de $\bar\omega$ dans $m^*$ : correct $m^*=\frac{\mathcal J_{\phi c}(\mathcal J_{\phi c}-\bar\omega)}{1+\mathcal J_{\phi\phi}}-\mathcal J_{cc}$ | $E_{\rm train}$ final faux de $\frac{4\mathcal J_{\phi c}^2\mathcal J_{\phi\phi}\bar\omega^2}{(1+\mathcal J_{\phi\phi})^2\det A}$ ; négligeable sur les vraies vidéos (`05` §5) |
| **E3** | p. 17, Eq. 21, conclusion | $s\to-\infty$ interprété comme pathologique | normal pour des variables continues (fondamental unique) ; la dégénérescence ($\alpha<1$) donne $Z=+\infty$ |
| **E4** | p. 17 | « régime sous-paramétré » pour $\alpha<1$ | sur-paramétré |
| **E5** | p. 17, 29 | $\langle\epsilon\rangle$ présenté comme erreur quadratique | c'est $\langle H\rangle/N$ (avec ½, par dimension) ; $E_{\rm train}=(2/\alpha)\langle\epsilon\rangle$ |
| **E6** | intro p. 4, conclusion | « la dynamique d'erreur est gouvernée par le spectre » | non démontré par le calcul sans ridge, qui ne dépend que de $\sigma^2_{\rm res}$ ; vrai avec ridge / troncature (§3–5) |
| remarque | p. 7–8, 15 | $\mathcal M$ quasi singulière (rang effectif 30–45) | $\alpha=P/N$ non pertinent ; paramètre effectif $P/K$ ou ${\rm df}(\kappa)$ |
| remarque | p. 11–12 (Fig. 6–7) | dichotomie du gap attribuée à la seule décroissance spectrale | elle dépend aussi du seuil (≲ 1e-8) et de $\omega_{\max}$ ($2\pi/T$) ; mécanisme précis : §5.2–5.3 |

---

## 8. Conclusions, limites et suites

**Ce qui est établi.**
1. Le calcul du mémoire, une fois corrigé, est exact dans le modèle gaussien et coïncide avec les moindres carrés.
2. Le ridge fait apparaître la dépendance spectrale par une unique équation auto-cohérente.
3. L'erreur de généralisation se décompose mode par mode.
4. L'erreur dans le gap est une amplification, par la concentration spectrale, du résidu de mauvaise spécification. C'est une version
   quantitative du phénomène de Runge, qui explique la dichotomie observée et ses conditions d'existence.
5. Toutes les formules sont exportées dans `src/aerot/theory.py` et validées sur les vraies vidéos.

**Limites.**
* Le calcul est replica-symétrique dans un modèle gaussien équivalent. Il est exact pour des données gaussiennes, mais seulement
  asymptotique ($O(1/P)$ non multiplié par df négligé, `04` §2.4). Sur la variété 1D, il ne vaut que pour $x={\rm df}\ln P/P<1$.
* Le critère des mini-gaps est heuristique : une théorie quantitative (statistique des trous × amplification de Runge locale)
  reste à écrire.
* À $2\pi/T$, les matrices de test exportées sont tronquées à 1000 modes. Le balayage en ridge des bords francs y reste donc
  indicatif.
* La géométrie exacte des templates du mémoire (croix en particulier) n'est pas connue. Les valeurs sont reproduites en ordre de
  grandeur, pas au facteur près.

**Suites naturelles.**
* Théorie des mini-gaps pour $E_{\rm gen}$ sur une variété, en combinant la statistique des espacements et l'identité de Runge.
* Test de la prédiction de fragilité : bruit de pixel ou float32 et erreur dans le gap.
* Ridge optimal pour le gap comme problème de contrôle : lien avec l'arrêt précoce, dont le filtre $1-e^{-\eta t\lambda_k}$ remplace
  $\lambda_k/(\lambda_k+\kappa)$, et avec Mignacco–Mori (2025).
* Encodeurs non linéaires (autoencodeurs du mémoire) : le spectre de concentration $\mu_j$ reste défini pour les features apprises.

---

## Annexe : fichiers

| fichier | contenu |
|---|---|
| `derivations/01_reproduction_memoire.md` | T1 : re-dérivation ligne à ligne, errata E1–E6 |
| `derivations/02_ridge_entrainement.md` | T2 : répliques avec ridge, équation de κ, entropie |
| `derivations/03_generalisation.md` | T3 : $E_{\rm gen}$, double descente, ridge optimal, lois d'échelle, noyau des rotations |
| `derivations/04_generalisation_gap.md` | T4 : $E_{\rm OOD}$, identité de Runge, dichotomie, ridge |
| `derivations/05_confrontation.md` | T5 : theory.py v0.2, confrontation B1, critère des trous, E2 réel, exposants |
| `checks/test_theory.py` | tests de `src/aerot/theory.py` (tous passent) |
| `checks/t1_* … t5_*`, `fig_t3.py`, `fig_t4.py` | vérifications sympy / numériques et figures |
| `figures/t3_double_descente_lois_echelle.png`, `t4_gap_runge.png`, `t5_equivalence_gaussienne.png` | figures |

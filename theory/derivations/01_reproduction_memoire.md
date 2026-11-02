# T1 — Re-dérivation du calcul de répliques du mémoire (ridge nul)

*Partie analytique — document de travail. Statut des résultats indiqué à chaque section.*
Vérifications : `theory/checks/t1_sympy_memoire.py` (sympy), `theory/checks/t1_numerique_ols.py` (simulations).

## 0. Notations et table de correspondance

Dans ce document on **garde les notations du mémoire** pour faciliter la comparaison ligne à ligne, avec
la correspondance suivante vers `docs/CONVENTIONS.md` :

| Mémoire | Ici (T1) | Conventions communes |
|---|---|---|
| $M$ (nombre d'échantillons) | $P$ | $P$ |
| $\mathcal M=\langle\phi\phi^\top\rangle$ | $\mathcal M$ (voir §1 : doit être la covariance **centrée** $\Sigma_c$) | $\Sigma$ (non centré), $\Sigma_c$ (centré) |
| $\Sigma = \mathcal M^{-1}$ (p. 15) | $\mathcal M^{-1}$ (on évite la lettre $\Sigma$) | — |
| $C=\langle\omega\phi\rangle$ | $C$ (doit être la covariance centrée $C_c=\langle\omega\phi\rangle-\bar\omega\bar\phi$) | $C$ (non centré) |
| $\sigma_\omega^2$ | variance de $\omega$ | $\sigma_\omega^2$ |
| $\mathcal J_{cc},\mathcal J_{\phi c},\mathcal J_{\phi\phi}$ (= $\mathcal S_{..}$ p. 28–29) | idem | — |
| $\lambda$ (ridge) | $\lambda_{\rm r}$ | $\lambda_{\rm r}$ |
| $\langle\epsilon\rangle$ | $e = \langle H\rangle/N$ (énergie par dimension, avec le 1/2) | $E_{\rm train}=\frac{2}{\alpha}e$ à $T=0$ |

$\alpha = P/N$, $n$ = nombre de répliques, $\mathbb I_n$ identité, $\mathbb J_n$ matrice pleine de 1, $\mathbb 1_n$ vecteur de 1.

## 1. Modèle, hypothèse gaussienne et problème de la moyenne

Hamiltonien (sans ridge ni terme de test, comme dans le mémoire) :
$$H(W)=\frac12\sum_{\mu=1}^P(\omega_\mu-W\cdot\phi_\mu)^2,\qquad Z=\int_{\mathbb R^N} dW\,e^{-\beta H(W)} .$$

**Hypothèse gaussienne (Eq. 8).** Le mémoire pose
$(\phi,\omega)\sim\mathcal N\big((\bar\phi,\bar\omega),\ [[\mathcal M, C],[C^\top,\sigma_\omega^2]]\big)$
avec $\mathcal M=\langle\phi\phi^\top\rangle$ et $C=\langle\omega\phi\rangle$ définis (p. 7 et 14) comme des moments
**non centrés**. C'est incohérent : une gaussienne de moyenne $\bar\phi$ et de covariance $\mathcal M$ a pour second moment
$\mathcal M+\bar\phi\bar\phi^\top\neq\mathcal M$ (**double comptage de la moyenne**). Le calcul qui suit n'utilise
$[[\mathcal M,C],[C^\top,\sigma^2_\omega]]$ que comme **covariance** (Eq. 21 du mémoire, p. 21 : l'identité
$\mathbb E e^{i\alpha\cdot x}=e^{i\alpha\cdot\mu-\frac12\alpha^\top\Sigma\alpha}$ est appliquée avec cette matrice comme covariance).
Deux corrections cohérentes, équivalentes pour tout ce qui suit (vérifié numériquement, §8) :

* **(a)** garder la moyenne et remplacer $\mathcal M\to\Sigma_c=\mathcal M-\bar\phi\bar\phi^\top$,
  $C\to C_c=C-\bar\omega\bar\phi$, $\sigma^2_\omega=$ variance. Toutes les formules du mémoire restent alors valides
  (après correction de $m^*$, §5) avec des $\mathcal J$ **centrés**.
* **(b)** modèle **centré** de même second moment : $(\phi,\omega)\sim\mathcal N(0,[[\Sigma,C],[C^\top,\langle\omega^2\rangle]])$,
  moments non centrés, $\bar\phi=\bar\omega=0$. C'est le choix retenu pour T2–T4 (et `CONVENTIONS.md`), car un
  encodeur **sans biais** ne voit que les seconds moments non centrés : $\langle(\omega-W\cdot\phi)^2\rangle=\langle\omega^2\rangle-2C\cdot W+W^\top\Sigma W$.

Si l'on injecte littéralement les moments non centrés comme covariance **et** la moyenne, on obtient
$\sigma^2_{\rm res}+e_0^2/(1+\mathcal J^{u}_{\phi\phi})-\bar\omega^2$ au lieu de $\sigma^2_{\rm res}$ (voir §8 ; numériquement
$-0.44$ au lieu de $0.05$ : valeur **négative**, absurde). Erratum **E1**.

Au premier ordre en $1/N$ les modèles (a) et (b) donnent les mêmes erreurs : la moyenne n'est qu'une direction
(de rang 1) parmi $N$ ; on montre en §7.3 que $\mathbb E\,E_{\rm train}=\sigma^2_{\rm res}(1-N/P)$ reste vrai à cet ordre
avec $\sigma^2_{\rm res}$ **non centré**.

## 2. Moyenne sur le désordre (p. 20–22) — *vérifié*

Pour $n$ répliques, les échantillons étant i.i.d.,
$\langle Z^n\rangle=\int\prod_a dW^a\,\big\langle e^{-\frac\beta2\sum_a(\omega-W^a\cdot\phi)^2}\big\rangle^P .$
Avec $u_a=\omega-W^a\cdot\phi$ et $\vec z\sim\mathcal N(0,\beta\mathbb I_n)$ : $e^{-\frac\beta2\|u\|^2}=\mathbb E_z e^{iu\cdot z}$.
Comme $u$ est gaussien (fonction affine de $(\phi,\omega)$), de moyenne $\tilde q^a=\bar\omega-\bar\phi\cdot W^a$ et de
covariance
$$q^{ab}=\mathrm{Cov}(u_a,u_b)=W^{a\top}\mathcal MW^b-C\cdot(W^a+W^b)+\sigma^2_\omega\equiv q_i^{ab}+m^a+m^b+\sigma^2_\omega ,$$
($q_i^{ab}=W^{a\top}\mathcal MW^b$, $m^a=-C\cdot W^a$), on obtient (Eq. 10), en intégrant d'abord sur $u$ puis sur $z$ :
$$\Big\langle e^{-\frac\beta2\sum_a u_a^2}\Big\rangle=\frac{\exp\!\big(-\frac\beta2\,\tilde q^\top(\mathbb I_n+\beta q)^{-1}\tilde q\big)}{\sqrt{\det(\mathbb I_n+\beta q)}} .$$
C'est exact (intégrale gaussienne ; le complété du carré p. 21 est correct : $\mu=(\mathbb I_n+\beta q)^{-1}\tilde q$,
$\nu=\beta\tilde q^\top(\mathbb I_n+\beta q)^{-1}\tilde q$, et $\det q\det(q^{-1}+\beta\mathbb I)=\det(\mathbb I+\beta q)$).
**Interprétation** (utile pour la suite) : $q^{aa}+(\tilde q^a)^2=\langle(\omega-W^a\cdot\phi)^2\rangle$ est exactement
l'**erreur de population** (de généralisation) du poids $W^a$.

## 3. Paramètres d'ordre et élimination des champs conjugués (p. 22–25) — *vérifié*

On introduit $q_i^{ab},m^a,\tilde q^a$ par des deltas de Fourier (champs $\hat q_i,\hat m,\hat{\tilde q}$). L'intégrale sur
les $W^a$ est gaussienne :
$\int\prod_a dW^a e^{-\frac12\sum_{ab}\hat q_i^{ab}W^{a\top}\mathcal MW^b+\sum_a J_a^\top W^a}
\propto(\det\hat q_i)^{-N/2}(\det\mathcal M)^{-n/2}\exp\big(\tfrac12\sum_{ab}(\hat q_i^{-1})_{ab}J_a^\top\mathcal M^{-1}J_b\big)$,
$J_a=C\hat m^a+\bar\phi\hat{\tilde q}^a$, ce qui fait apparaître $A=[[\mathcal J_{cc},\mathcal J_{\phi c}],[\mathcal J_{\phi c},\mathcal J_{\phi\phi}]]$
avec $\mathcal J_{xy}=x^\top\mathcal M^{-1}y$. Avec $X=(m\,|\,\tilde q-\bar\omega\mathbb 1_n)$, $\mathcal J=(\hat m\,|\,\hat{\tilde q})$ ($n\times2$) :
$$S_{\rm conj}=-\tfrac12{\rm Tr}(\hat q_iq_i)-{\rm Tr}(\mathcal J^\top X)-\tfrac12{\rm Tr}(\mathcal J^\top\hat q_i^{-1}\mathcal JA)+\tfrac N2\ln\det\hat q_i .$$
Col : $\mathcal J=-\hat q_iXA^{-1}$, d'où $S_{\rm conj}=-\frac12{\rm Tr}\,\hat q_i(q_i-XA^{-1}X^\top)+\frac N2\ln\det\hat q_i$,
puis $\hat q_i=N(q_i-XA^{-1}X^\top)^{-1}$ et
$$S_{\rm eff}=\frac{\beta P}{2}\tilde q^\top(\mathbb I_n+\beta q)^{-1}\tilde q+\frac P2\ln\det(\mathbb I_n+\beta q)-\frac N2\ln\det\big(q_i-XA^{-1}X^\top\big)+\text{cst}.$$
Les étapes du mémoire (p. 24–25) sont correctes. Les constantes omises
($-\frac{Nn}2+\frac{Nn}2\ln N+\frac n2\ln\det\mathcal M$ + facteurs $2\pi$) ne dépendent pas des paramètres d'ordre : elles
n'affectent ni l'énergie ni les équations du col, mais **fixent la constante de l'entropie** (§6).

**Remarque (pourquoi le spectre disparaît).** Le terme entropique ne dépend du spectre de $\mathcal M$ que par
$\det\mathcal M$ : par le changement de variables $V=\mathcal M^{1/2}W$, les contraintes deviennent
$V^a\cdot V^b=q_i^{ab}$, $(\mathcal M^{-1/2}C)\cdot V^a=-m^a$, $(\mathcal M^{-1/2}\bar\phi)\cdot V^a=\bar\omega-\tilde q^a$ : un problème
**isotrope** en $V$. Le volume correspondant vaut $\det(q_i-XA^{-1}X^\top)^{(N-n-2)/2}$ à des constantes près, ce qui redonne
le terme $-\frac N2\ln\det(\cdot)$ à l'ordre dominant. **Sans ridge, le résultat ne peut donc dépendre des données que
par les trois scalaires $\mathcal J_{cc},\mathcal J_{\phi c},\mathcal J_{\phi\phi}$ (et $\bar\omega,\sigma^2_\omega$)** : c'est la
raison profonde de l'indépendance spectrale de $\delta q^*$ (§5) — ce n'est pas une erreur. Le ridge
$\frac{\beta\lambda_{\rm r}}{2}\|W\|^2=\frac{\beta\lambda_{\rm r}}2\|\mathcal M^{-1/2}V\|^2$ brise cette invariance (T2).

## 4. Ansatz RS et limite $n\to0$ (p. 25–27) — *vérifié symboliquement* (`t1_sympy_memoire.py`, test [1])

RS : $q_i=\delta q\,\mathbb I_n+q_{i0}\mathbb J_n$ ($\delta q=Q_i-q_{i0}$), $m^a=m$, $\tilde q^a=\tilde q$ ; alors
$q=\delta q\,\mathbb I_n+q_0\mathbb J_n$ avec $q_0=q_{i0}+2m+\sigma^2_\omega$, et $XA^{-1}X^\top=c\,\mathbb J_n$ avec
$$c(m,\tilde q)=(A^{-1})_{11}m^2+2(A^{-1})_{12}m\tilde\omega+(A^{-1})_{22}\tilde\omega^2=\frac{\mathcal J_{\phi\phi}m^2-2\mathcal J_{\phi c}m\tilde\omega+\mathcal J_{cc}\tilde\omega^2}{\mathcal J_{cc}\mathcal J_{\phi\phi}-\mathcal J_{\phi c}^2},\qquad\tilde\omega=\tilde q-\bar\omega .$$
(L'expression de $c$ p. 29 est **identique** à l'Eq. 13 : pas d'erreur de signe, test [3].)
Avec $(a\mathbb I+b\mathbb J)^{-1}=\frac1a\mathbb I-\frac{b}{a(a+nb)}\mathbb J$ et $\det(a\mathbb I+b\mathbb J)=a^{n-1}(a+nb)$,
le développement à l'ordre $n$ donne l'Eq. (14) :
$$\frac{S^{RS}_{\rm eff}}{nN}=\frac{\alpha\beta}{2}\frac{\tilde q^2+q_0}{1+\beta\delta q}+\frac\alpha2\ln(1+\beta\delta q)-\frac12\ln\delta q-\frac12\frac{q_0-2m-\sigma^2_\omega-c(m,\tilde q)}{\delta q}+O(n).$$
**Ordre des limites.** Prendre $n\to0$ avant d'extrémiser revient à extrémiser le coefficient d'ordre $n$ : c'est la
procédure standard, et elle est ici sans danger : le problème est quadratique/convexe (le Gibbs est gaussien), l'ansatz
RS est exact, et le résultat coïncide avec la théorie des matrices aléatoires (§7, §8). La phrase « par confiance absolue
envers le dieu des répliques » peut être remplacée par cette justification.

## 5. Équations du col (p. 27–28) — *vérifié symboliquement* (tests [2], [4]) — **erreur sur $m^*$**

* $\partial_{q_0}$ : $\frac{\alpha\beta}{1+\beta\delta q}=\frac1{\delta q}\Rightarrow\boxed{\delta q^*=\frac1{\beta(\alpha-1)}}$ (correct).
  $S$ est **linéaire** en $q_0$ : cette équation fixe $\delta q$, et $q_0$ est fixé par l'équation en $\delta q$.
* $\partial_m$ : $(A^{-1})_{11}m+(A^{-1})_{12}\tilde\omega=-1$ ; $\partial_{\tilde q}$ : $(A^{-1})_{12}m+(1+(A^{-1})_{22})\tilde\omega=-\bar\omega$ (corrects).
* Solution (sympy) :
$$\boxed{\tilde q^*=\frac{\bar\omega-\mathcal J_{\phi c}}{1+\mathcal J_{\phi\phi}}}\ \text{(correct)},\qquad
\boxed{m^*=\frac{\mathcal J_{\phi c}(\mathcal J_{\phi c}-\bar\omega)}{1+\mathcal J_{\phi\phi}}-\mathcal J_{cc}}\ \text{(le mémoire écrit }+\bar\omega\text{)} .$$
  Vérification directe : au col, $W$ vaut (pour ces projections) l'optimum de population
  $W_\star=(\mathcal M+\bar\phi\bar\phi^\top)^{-1}(C+\bar\omega\bar\phi)$ ; Sherman–Morrison donne
  $C\cdot W_\star=\mathcal J_{cc}+\frac{\bar\omega\mathcal J_{\phi c}-\mathcal J_{\phi c}^2}{1+\mathcal J_{\phi\phi}}$, donc $m^*=-C\cdot W_\star$
  comme ci-dessus. **Erratum E2** (Eq. 18 et p. 28–29) : le signe de $\bar\omega$ dans $m^*$. Conséquence : avec le $m^*$ du
  mémoire, la quantité finale $\tilde q^{*2}+2m^*+\sigma^2_\omega+c$ est fausse de
  $\frac{4\mathcal J_{\phi c}^2\mathcal J_{\phi\phi}\bar\omega^2}{(1+\mathcal J_{\phi\phi})^2(\mathcal J_{cc}\mathcal J_{\phi\phi}-\mathcal J_{\phi c}^2)}$ (≥ 0 ; nul
  seulement si $\bar\omega=0$ ou $\mathcal J_{\phi c}=0$). Or ici $\bar\omega=\omega_{\max}/2\neq0$.
* $\partial_{\delta q}$ : $q_0^*=\frac{1}{\alpha-1}\big[\tilde q^{*2}+\alpha(2m^*+\sigma^2_\omega+c^*)\big]$ (Eq. 19, correct).

**Résultat clé (nouveau, vérifié, test [6])** : au col corrigé,
$$\boxed{\ \tilde q^{*2}+2m^*+\sigma^2_\omega+c^*=\sigma^2_{\rm res}\equiv\sigma^2_\omega-\mathcal J_{cc}+\frac{(\bar\omega-\mathcal J_{\phi c})^2}{1+\mathcal J_{\phi\phi}}=\min_W\langle(\omega-W\cdot\phi)^2\rangle\ }$$
c'est-à-dire la **variance résiduelle du meilleur prédicteur linéaire sans biais** (erreur de population minimale), qui
s'écrit aussi $\sigma^2_{\rm res}=\langle\omega^2\rangle-C_u^\top\Sigma^{-1}C_u$ avec les moments **non centrés** $\Sigma=\mathcal M+\bar\phi\bar\phi^\top$,
$C_u=C+\bar\omega\bar\phi$ (Sherman–Morrison ; §8 numérique).

## 6. Énergie libre, énergie, entropie (p. 28–29) — *vérifié* (tests [5], [7])

$$f(\alpha,\beta)=\frac{S^{RS}}{nN\beta}=\frac{\alpha-1}2\,\sigma^2_{\rm res}+\frac{\alpha}{2\beta}\ln\frac{\alpha}{\alpha-1}+\frac1{2\beta}\ln\big((\alpha-1)\beta\big)\quad(\text{Eq. 20 avec }X\to\sigma^2_{\rm res}).$$
$q_0^*$ disparaît parce que $S$ est linéaire en $q_0$ avec un coefficient nul au col (c'est l'équation $\partial_{q_0}S=0$).

* Énergie par dimension : $e=\partial_\beta(\beta f)=\frac{\alpha-1}{2}\sigma^2_{\rm res}+\frac{1}{2\beta}$ (équipartition :
  $N$ modes quadratiques confinés, $\frac1{2\beta}$ chacun).
* **Normalisation (erratum E5, imprécision)** : $e=\langle H\rangle/N$ contient le facteur $\frac12$ et est rapportée à $N$ ; ce
  n'est pas « l'erreur quadratique moyenne ». L'erreur d'entraînement par échantillon (convention commune) est
  $$\boxed{E_{\rm train}=\frac{2}{\alpha}e\big|_{T=0}=\Big(1-\frac1\alpha\Big)\sigma^2_{\rm res}=\sigma^2_{\rm res}\Big(1-\frac NP\Big)}$$
  — **exactement** le résultat classique des moindres carrés gaussiens (§7).
* Entropie : $s=\beta^2\partial_\beta f=\frac{\alpha-1}2\ln(\alpha-1)-\frac\alpha2\ln\alpha+\frac12-\frac{\ln\beta}{2}$ (Eq. 21 correcte),
  **à une constante près** fixée par les termes omis. Calcul exact (Gibbs gaussien
  $W\sim\mathcal N(W_{\rm OLS},(\beta X^\top X)^{-1})$ + Marchenko–Pastur, `t1_entropie_logdet.py`) :
  $$s_{\rm exact}=\underbrace{\tfrac{\alpha-1}2\ln(\alpha-1)-\tfrac\alpha2\ln\alpha+\tfrac12-\tfrac{\ln\beta}2}_{\text{Eq. 21}}+\tfrac12+\tfrac12\ln 2\pi-\tfrac12\ln N-\tfrac1{2N}\ln\det\mathcal M .$$
  La dépendance en $\alpha$ du RS est donc exacte.

### 6.1 Interprétation de $s\to-\infty$ à $T=0$ (erratum E3 : interprétation)
C'est **normal** pour des degrés de liberté continus : chaque direction de $W$ est confinée dans un puits harmonique de
largeur $\propto\beta^{-1/2}$, d'entropie différentielle $\frac12\ln(2\pi e/(\beta\kappa))$ qui tend vers $-\infty$ (comme pour un
oscillateur harmonique classique). Le troisième principe est un énoncé **quantique** ; il ne s'applique pas à une mesure de
Lebesgue sur $\mathbb R^N$. Ce n'est **pas** un « effondrement non physique » ni un signe de dégénérescence : au contraire,
$s\to-\infty$ signale un état fondamental **unique** (non dégénéré). La dégénérescence produit l'effet **inverse** :
pour $\alpha<1$, $X^\top X$ a $N-P$ valeurs propres nulles, $H$ a $N-P$ directions plates, et
$Z=\int dW e^{-\beta H}=+\infty$ (entropie $+\infty$). Le RS le signale par $\delta q^*<0$ et $\ln(\alpha-1)$ non défini : le col
n'existe pas. Le ridge rend $Z$ fini (T2).

### 6.2 Coquille « sous-paramétré » (erratum E4)
p. 17 : « des pathologies dans le régime **sous**-paramétré. Ainsi $\delta q^*$ devient négative pour $\alpha<1$ » :
$\alpha=P/N<1$ est le régime **sur**-paramétré (plus de paramètres que de données) ; la conclusion (p. 18) emploie le bon terme.

## 7. Confrontation au résultat exact des moindres carrés gaussiens

### 7.1 Énoncé
Modèle centré (b) : $\omega=\theta_\star\cdot\phi+\varepsilon$, $\theta_\star=\Sigma^{-1}C$, $\varepsilon\sim\mathcal N(0,\sigma^2_{\rm res})$
**indépendant** de $\phi$ (dans un modèle gaussien joint, décorrélé ⇔ indépendant). Pour $P>N$,
$W_{\rm OLS}-\theta_\star=(X^\top X)^{-1}X^\top\varepsilon$, donc exactement (à $N,P$ finis) :
$$\mathbb E\,E_{\rm train}=\frac{\sigma^2_{\rm res}}{P}\mathbb E\,{\rm Tr}(\mathbb I-H_X)=\sigma^2_{\rm res}\frac{P-N}{P},\qquad
\mathbb E\,E_{\rm gen}=\sigma^2_{\rm res}\Big(1+\frac{N}{P-N-1}\Big)=\sigma^2_{\rm res}\frac{P-1}{P-N-1}.$$
Limite thermodynamique : $E_{\rm train}\to\sigma^2_{\rm res}(1-1/\alpha)$ ✔ (répliques), $E_{\rm gen}\to\sigma^2_{\rm res}\,\frac{\alpha}{\alpha-1}$.

### 7.2 $q_0^*$ est l'erreur de généralisation (réponse au « Interprétation ? » p. 28)
D'après §2, l'erreur de population de la réplique $a$ est $q^{aa}+(\tilde q^a)^2=\delta q+q_0+\tilde q^2$. À $T=0$
($\delta q^*\to0$) :
$$\boxed{E_{\rm gen}=q_0^*+\tilde q^{*2}=\frac{\alpha}{\alpha-1}\,\sigma^2_{\rm res}}\qquad(\text{sympy, test [8]}).$$
Le mémoire contenait donc **déjà** l'erreur de généralisation en distribution (ridgeless, $\alpha>1$) sans l'avoir
identifiée : $q_0^*$ est la variance du résidu sur un **nouvel** échantillon. Sa divergence en $1/(\alpha-1)$ (notée p. 29) est
le **pic d'interpolation** (double descente) à $\alpha\to1^+$ ; $f$ reste fini car $E_{\rm train}\to0$. On retrouve le
rapport universel ridgeless $E_{\rm train}/E_{\rm gen}=(1-1/\alpha)^2$.

### 7.3 Effet de la moyenne non nulle (modèle (a)) — *argument + numérique*
Avec $\phi=\bar\phi+\xi$, $\xi\sim\mathcal N(0,\Sigma_c)$, le résidu $\varepsilon=\omega-\theta_\star\cdot\phi$ ($\theta_\star$ non centré) vérifie
$\langle\varepsilon\phi\rangle=0$ mais $\langle\varepsilon\rangle=e_0\neq0$ : il n'est **pas** indépendant de $\phi$. On écrit
$\varepsilon=\eta-e_0\bar\phi^\top\Sigma_c^{-1}\xi$ avec $\eta\perp\xi$, $\eta\sim\mathcal N(e_0,\sigma^2_{\rm res}-e_0^2(1+\mathcal J))$,
$\mathcal J=\bar\phi^\top\Sigma_c^{-1}\bar\phi$. La partie linéaire en $\phi$ est annulée par $\mathbb I-H_X$, d'où
$(\mathbb I-H_X)\varepsilon=(\mathbb I-H_X)\eta'$, $\eta'_\mu\overset{iid}{\sim}\mathcal N(e_0(1+\mathcal J),\sigma^2_{\rm res}-e_0^2(1+\mathcal J))$
indépendant de $X$, et $\mathbb E\,\mathbb 1^\top(\mathbb I-H_X)\mathbb 1\simeq (P-N)/(1+\mathcal J)$ (projection aléatoire), donc
$\mathbb E\,E_{\rm train}\simeq\sigma^2_{\rm res}(1-N/P)$ au premier ordre. Vérifié numériquement (§8).

## 8. Vérification numérique (`checks/t1_numerique_ols.py`) — *vérifié sur données gaussiennes*

$N=150$, spectre centré $\lambda_k=k^{-1.2}$, moyenne $\bar\phi\neq0$, $\bar\omega=0.7$, bruit $0.05$, 300 tirages.

| quantité | valeur |
|---|---|
| $\sigma^2_{\rm res}$ (non centré, direct) | 0.050011 |
| formule répliques corrigée ($\mathcal J$ centrés, $m^*$ corrigé) | 0.050011 |
| formule avec le $m^*$ du mémoire | 0.050032 (faux, écart faible ici car $\mathcal J_{\phi c}$ petit) |
| moments non centrés injectés + moyenne (double comptage, $\sigma^2_\omega$=variance) | **−0.440** (absurde) |

| $\alpha$ | $E_{\rm train}$ sim | $\sigma^2_{\rm res}(1-1/\alpha)$ | $E_{\rm gen}$ sim | $\sigma^2_{\rm res}\alpha/(\alpha-1)$ |
|---|---|---|---|---|
| 1.25 | 0.00979(13) | 0.00990 | 0.2565(36) | 0.2528 (fini : $\frac{P-1}{P-N-1}\sigma^2=0.2583$) |
| 1.5 | 0.01682(16) | 0.01667 | 0.1498(11) | 0.1500 |
| 2 | 0.02541(17) | 0.02501 | 0.0999(5) | 0.1000 |
| 4 | 0.03763(14) | 0.03751 | 0.06673(13) | 0.06668 |

## 9. Discussion des points suspects

1. **Moyenne / double comptage** : erreur réelle (E1), corrigée en §1.
2. **$\delta q^*$ indépendant du spectre** : correct (§3 : invariance $V=\mathcal M^{1/2}W$ ; exactement
   $\delta q=\beta^{-1}{\rm Tr}[\mathcal M(X^\top X)^{-1}]=\beta^{-1}{\rm Tr}(G^\top G)^{-1}\simeq\frac{1}{\beta(\alpha-1)}$ avec $X=G\mathcal M^{1/2}$).
   Plus généralement **toute** la théorie ridgeless ne dépend du template que par le scalaire $\sigma^2_{\rm res}$. L'affirmation
   de l'introduction (« démontrant analytiquement que la dynamique d'erreur est gouvernée par le spectre du template ») n'est
   donc **pas** démontrée par ce calcul (**E6**, sur-interprétation) : la dépendance spectrale n'apparaît qu'avec un ridge (T2–T3)
   ou en rang fini.
3. **Limite $N\to\infty$ et données sur une variété 1D** : le résultat exige $\mathcal M$ inversible (les $\mathcal J$ contiennent
   $\mathcal M^{-1}$). Pour les vidéos, $\mathcal M$ est numériquement de rang effectif $K\approx30$–$45\ll N=7840$ (mémoire p. 7–8,
   Fig. 6–7). Le bon paramètre n'est alors pas $\alpha=P/N$ mais $P/K$ : après la troncature spectrale utilisée dans le
   mémoire (cut-off $10^{-4}$), on a exactement $E_{\rm train}=\sigma^2_{\rm res,K}(1-K/P)$, $E_{\rm gen}=\sigma^2_{\rm res,K}\,P/(P-K)$
   ($\sigma^2_{\rm res,K}$ = résidu du meilleur prédicteur restreint aux $K$ modes). Le régime « $N\to\infty$, $\alpha$ fixé, spectre
   $O(1)$ » n'est pas celui des expériences ; la formulation adaptée est la théorie ridge à spectre quelconque (T2), dans
   laquelle $N$ n'intervient plus et le nombre de modes effectif est ${\rm df}(\kappa)$. La troncature du mémoire est une
   régularisation spectrale « dure », proche d'un ridge $\lambda_{\rm r}/P\approx10^{-4}\lambda_{\max}$.
4. **Signes/facteurs dans $\tilde q^*,m^*,c$** : $\tilde q^*$ et $c$ corrects ; $m^*$ faux (E2).
5. **Entropie $\to-\infty$** : normal (E3) ; pathologie réelle de $\alpha<1$ = divergence de $Z$ (directions plates).
6. **Ordre des limites** : sans danger ici (§4).
7. **Normalisation de $\langle\epsilon\rangle$** : c'est $\langle H\rangle/N$, pas l'erreur par échantillon (E5).

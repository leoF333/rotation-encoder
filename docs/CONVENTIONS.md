# Conventions communes (numérique ↔ analytique)

Ce fichier fixe les notations et normalisations partagées par les parties numérique et analytique.
Toute modification doit être signalée dans ce fichier.

## Notations (et correspondance avec le mémoire)

Le mémoire utilise la lettre M à la fois pour le nombre d'échantillons et pour la
matrice de second moment. On lève l'ambiguïté ainsi :

| Objet | Ici | Mémoire |
|---|---|---|
| taille d'image | $d$ (défaut 28) | $d$ |
| nombre d'images par vidéo | $T$ (défaut 10), $t = 0,\dots,T-1$ | $T$ |
| dimension ambiante | $N = d^2 T$ (défaut 7840) | $N$ |
| nombre d'échantillons d'entraînement | $P$ | $M$ |
| ratio d'échantillonnage | $\alpha = P/N$ | $\alpha = M/N$ |
| datapoint (vidéo aplatie) | $\phi(\omega) \in \mathbb{R}^N$ | $\phi(\omega)$ |
| second moment **non centré** | $\Sigma = \langle \phi\phi^\top\rangle_\rho$ | $M$ |
| moyenne | $\bar\phi = \langle\phi\rangle_\rho$, $\bar\omega = \langle\omega\rangle_\rho$ | idem |
| covariance centrée | $\Sigma_c = \Sigma - \bar\phi\bar\phi^\top$ | (non distinguée) |
| corrélation entrée-cible (non centrée) | $C = \langle \omega\,\phi\rangle_\rho$ | $C$ |
| second moment de la cible | $\langle\omega^2\rangle_\rho$ ; variance $\sigma_\omega^2$ | $\sigma_\omega^2$ |
| spectre | $\Sigma = \sum_k \lambda_k u_k u_k^\top$, $\lambda_1 \ge \lambda_2 \ge \dots$ | $\lambda_k$ |
| projections cible | $c_k = u_k\cdot C$ | — |
| régularisation ridge | $\lambda_{\rm r}$ (pour éviter la confusion avec les $\lambda_k$) | $\lambda$ |

## Modèle génératif (défauts, à confirmer / ajuster par la partie numérique)

- Rotation autour du centre de la grille $((d-1)/2, (d-1)/2)$ ; l'image $t$ est le template
  tourné d'un angle $\omega t$.
- Distribution d'entraînement pleine : $\omega \sim \mathcal{U}[0, \omega_{\max}]$.
- Distribution « créneau » (gap) : $\mathcal{U}([0,\omega_{\max}/3] \cup [2\omega_{\max}/3, \omega_{\max}])$ ;
  distribution de test OOD : $\mathcal{U}[\omega_{\max}/3, 2\omega_{\max}/3]$.
- $\omega_{\max}$ : le texte du mémoire dit $2\pi/T$ (Fig. 4 : axe $\omega \in [0.05, 0.63]$), mais les
  axes des Fig. 6–7 (expérience de gap) montent à $\approx 1.26 = 4\pi/T$, gap grisé
  $\approx[0.42, 0.84]$. Les deux valeurs sont à tester ; la valeur retenue par expérience
  est consignée dans le worklog numérique.
- Encodeur linéaire sans biais : $\hat\omega = W\cdot\phi$, $W\in\mathbb{R}^N$.

## Normalisation des erreurs

- Hamiltonien : $H(W) = \frac12\sum_{\mu=1}^P (\omega_\mu - W\cdot\phi_\mu)^2 + \frac{\lambda_{\rm r}}{2}\|W\|^2$.
- **Erreur d'entraînement rapportée** : $E_{\rm train} = \frac1P\sum_\mu (\omega_\mu - W^*\cdot\phi_\mu)^2$
  (moyenne par échantillon, **sans** le facteur 1/2, sans le terme ridge).
- **Erreur de généralisation (en distribution)** : $E_{\rm gen} = \langle(\omega - W^*\cdot\phi(\omega))^2\rangle_{\omega\sim\rho}$,
  moyennée ensuite sur le tirage du jeu d'entraînement.
- **Erreur OOD (gap)** : même définition avec $\omega \sim \rho_{\rm test}$ (uniforme sur le gap), l'entraînement
  se faisant sur la distribution créneau.
- **Erreur de test ponctuelle** (celle du mémoire, courbe $E(\omega)$) : $(\omega - W^*\cdot\phi(\omega))^2$ à $\omega$ fixé.
- Toute formule exportée dans `src/aerot/theory.py` doit préciser si elle renvoie ces quantités
  avec ou sans facteur 1/2, par échantillon ou par dimension, et le scaling de $\lambda_{\rm r}$.

## Formats d'échange

- `data/spectra/<template>_<distribution>_d<d>_T<T>_wmax<valeur>.npz` (écrit par la partie numérique) :
  clés `eigvals` (ordre décroissant, second moment non centré), `c_proj` ($c_k = u_k\cdot C$),
  `omega2` ($\langle\omega^2\rangle$), `omega_mean`, `phi_mean_proj` ($u_k\cdot\bar\phi$), `N`, `d`, `T`, `wmax`,
  `n_samples_used`, et si pertinent `eigvals_centered`, `c_proj_centered`, ainsi que, pour les
  expériences de gap, la matrice test exprimée dans la base propre d'entraînement
  (`test_cov_in_train_basis` tronquée aux K premiers modes, `test_c_proj`, `test_omega2`).
  Un fichier `data/spectra/README.md` décrit exactement le contenu.
- `src/aerot/theory.py` (écrit par la partie analytique) : fonctions pures numpy prenant un spectre
  (et projections) + $(P$ ou $\alpha$, $\lambda_{\rm r})$ et renvoyant les prédictions théoriques
  ($E_{\rm train}$, $E_{\rm gen}$, $E_{\rm OOD}$, paramètres d'ordre). Docstrings en français.

# Spectres exportés par la partie numérique

Script générateur : `scripts/a2_export_spectra.py` (code : `src/aerot/`).
Toutes les quantités sont des moyennes **population** sur ρ(ω), calculées par quadrature de
Gauss-Legendre composite (8 nœuds par panneau, `n_samples_used` nœuds au total, poids normalisés à 1),
en float64 sur CPU, via l'astuce de Gram. Ce ne sont **pas** des matrices empiriques à P échantillons :
pour comparer à des simulations à P fini, tirer les ω i.i.d. (cf. `src/aerot/render.py`).

## Nom des fichiers

`<template>_<distribution>_d<d>_T<T>_wmax<ωmax à 4 décimales>[<suffixe>].npz`

- templates : `gaussian`, `dipole`, `quarter_circle`, `cross`, `disk` (témoin, invariant par rotation),
  `segment`, `arc` ; géométrie décrite dans
  `src/aerot/templates.py`.
- distributions : `uniform` = U[0, ωmax] ; `creneau` = U([0, ωmax/3] ∪ [2ωmax/3, ωmax]).
- ωmax ∈ {2π/T = 0.6283, 4π/T = 1.2566} (T = 10, d = 28, N = 7840).
- **sans suffixe : rendu par défaut `analytic` ss = 4** (fonction continue du template évaluée aux coordonnées
  tournées, moyennée sur 4×4 sous-pixels = anti-crénelage « boîte »). Choix motivé par le témoin négatif :
  le disque centré (invariant par rotation) est exactement de rang 1 (λ_2/λ_1 ~ 1e-15), alors que le rendu
  `interp` lui donne λ_2/λ_1 = 2e-4 et ~18 modes > 1e-4 λ_1, et ajoute à la gaussienne une queue spectrale
  parasite en loi de puissance (K(1e-8) = 244 au lieu de 45).
- suffixe `_interp` : ancien rendu (template rastérisé au centre des pixels puis tourné par interpolation
  bilinéaire, comme `grid_sample`) — conservé comme variante de sensibilité au rendu.

## Clés (conventions de `docs/CONVENTIONS.md`)

| clé | contenu |
|---|---|
| `eigvals` | λ_k de Σ = ⟨φφᵀ⟩_ρ (non centré), ordre décroissant, longueur n (rang ≤ n ; valeurs < 1e-15 λ_1 = bruit numérique) |
| `c_proj` | c_k = u_k · C, C = ⟨ω φ⟩_ρ (signe de u_k arbitraire mais commun à `phi_mean_proj`) |
| `phi_mean_proj` | u_k · φ̄ |
| `omega2`, `omega_mean`, `omega_var` | ⟨ω²⟩_ρ, ⟨ω⟩_ρ, variance |
| `eigvals_centered`, `c_proj_centered` | spectre de Σ_c = Σ − φ̄φ̄ᵀ et projections de C_c = ⟨(ω−ω̄)(φ−φ̄)⟩ sur ses vecteurs propres |
| `N`, `d`, `T`, `wmax`, `n_samples_used` | paramètres |
| `K_1e4`, `K_1e8`, `K_1e12` (+ `_centered`) | nombre de modes avec λ_k/λ_1 > seuil |
| `trace` | Tr Σ = ⟨‖φ‖²⟩ |
| `ref_Etrain_pop_trunc1e-4` | erreur d'entraînement population ⟨(ω − W·φ)²⟩_ρ avec W = Σ_{λ_k>1e-4 λ_1} (c_k/λ_k) u_k |

Pour `creneau` uniquement (test = U[ωmax/3, 2ωmax/3], `n_test_samples_used` nœuds) :

| clé | contenu |
|---|---|
| `test_cov_in_train_basis` | matrice K_export × K_export : u_kᵀ Σ_test u_l, Σ_test = ⟨φφᵀ⟩_test (non centré), u_k base propre d'entraînement |
| `test_c_proj` | u_k · C_test |
| `test_phi_mean_proj` | u_k · φ̄_test |
| `test_omega2`, `test_omega_mean` | ⟨ω²⟩_test, ⟨ω⟩_test |
| `test_trace` | Tr Σ_test (pour mesurer la part d'énergie test hors des K_export premiers modes) |
| `K_export` | min(K_1e12, 1000) |
| `ref_Eood_pop_trunc1e-4` | ⟨(ω − W·φ)²⟩_test avec le W tronqué ci-dessus (valeur de contrôle pour les formules) |

Normalisation des erreurs : par échantillon, **sans** facteur 1/2 (cf. CONVENTIONS).
Contrôle : E_OOD(W) = test_omega2 − 2 Σ_k W_k test_c_proj_k + Σ_kl W_k W_l test_cov_kl, avec
W_k = c_proj_k/eigvals_k pour les modes retenus, redonne `ref_Eood_pop_trunc1e-4`.

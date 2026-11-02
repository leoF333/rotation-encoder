# Scripts — one command per figure

Package `src/aerot/` (installed with `pip install -e .`) :

| module | contenu |
|---|---|
| `templates.py` | templates continus s0(x, y) (géométrie mesurée sur les figures du mémoire) |
| `render.py` | génération vectorisée des vidéos φ(ω) (rendu `analytic` ss=4 par défaut, ou `interp`), lois de ω, quadratures |
| `spectra.py` | spectres de Σ = ⟨φφᵀ⟩ par l'astuce de Gram (float64), pseudo-inverse tronquée, ridge, quantités test |
| `linreg.py` | régression ridge / pinv à P échantillons (chemin ridge exact), substituts gaussiens de mêmes moments |
| `continuum.py` | modèle sans pixels : spectre de puissance angulaire E_m et « features » exactes |
| `ae.py` | autoencodeurs PyTorch (MPS), entraînement, métriques du latent |
| `theory.py` | formules des répliques (κ auto-cohérent, E_train, E_gen, E_OOD, identité de Runge) |
| `fits.py`, `plotting.py`, `paths.py` | ajustements R², style des figures, chemins |

Set `PY=python` (the environment of `environment.yml`) and run every command from the repository root.
Tous les scripts sont idempotents : les résultats déjà calculés dans `results/` sont réutilisés. Chaque figure
`figures/X.png` a ses paramètres dans `figures/X.json`.

## Une commande par figure

| figure(s) | commande | durée indicative |
|---|---|---|
| `a1_fig2_repro.png`, `a1_templates.png`, `a1_rendering.png` | `$PY scripts/a1_data_figures.py` | < 1 min |
| spectres `data/spectra/*.npz` (prérequis de A2, B1) | `$PY scripts/a2_export_spectra.py` (variante : `--method interp --ss 1 --suffix _interp`) | ~5 min |
| `a2_spectra.png`, `a2_densities.png`, `a2_rendering_spectra.png`, `a2_jacobi_anger.png` (+ `results/a2_K_table.md`, `a2_symmetry.md`) | `$PY scripts/a2_spectra_analysis.py` | ~4 min |
| `a3_pca.png` | `$PY scripts/a3_pca.py` | ~1 min |
| `a4_fig3_repro.png`, `a4_fig4_latent.png` | `$PY scripts/a4_autoencoders.py` | ~7 min (GPU) |
| `a5_gap_memoir_wmax4piT_analytic4.png`, `a5_gap_regressors_wmax4piT_analytic4.png` | `$PY scripts/a5_gap.py --wmax 4` | ~2 min |
| `a5_gap_memoir_wmax2piT_analytic4.png`, `a5_gap_regressors_wmax2piT_analytic4.png` | `$PY scripts/a5_gap.py --wmax 2` | ~2 min |
| `a5_gap_*_wmax4piT_interp1.png` | `$PY scripts/a5_gap.py --wmax 4 --method interp --ss 1` | ~2 min |
| `a5_gap_noise.png` (bruit gaussien / float32) | `$PY scripts/a5_gap_variants.py --part noise` | ~3 min |
| `a5_gap_vs_memoir.png`, `a5_gaussian_sigma_r0.png` (+ `results/a5_variants_summary.md`) | `$PY scripts/a5_gap_variants.py --part conventions && $PY scripts/a5_gap_variants.py --part sigma_r0 && $PY scripts/a5_gap_variants.py --part report` | ~8 min |
| données B1 (`results/b1/*.json`) | `for p in main finite large stratified labelnoise; do $PY scripts/b1_replicas.py --part $p; done` | ~15 + 20 + 20 + 5 + 15 min |
| matrices test complètes (cache de la théorie E_OOD, `results/b1_theory_cache/`, non commité) | `$PY scripts/export_full_test_cov.py --wmax 2` | ~5 min |
| `b1_learning_curves_uniform.png`, `b1_learning_curves_creneau.png`, `b1_gaussian_equivalence.png`, `b1_finite_size.png`, `b1_memoire_formula.png` (+ `results/b1_theory_comparison.md`, `b1_large_alpha.md`, `b1_memoire_formula.md`) | `$PY scripts/b1_analysis.py` | ~1 min (cache) |
| `b2_angular_power.png`, `b2_scaling.png`, `b2_gap_vs_spectrum.png` (+ `results/b2_fits.md`) | `$PY scripts/b2_spectral_laws.py --part all` | ~15 min |
| données B3 (`results/b3/`) | `$PY scripts/b3_architectures.py --template arc && for t in gaussian cross; do $PY scripts/b3_architectures.py --template $t --configs lin-shallow,lin-mid,lin-deep,mlp-deep,deep-deep,lin-deep-k2; done` | ~50 + 2×18 min (GPU) |
| `b3_pareto_arc.png`, `b3_latents_arc.png` (+ `results/b3_summary_arc.md`) | `$PY scripts/b3_analysis.py --template arc` | < 1 min |
| `b3_pareto_gaussian.png`, `b3_latents_gaussian.png` | `$PY scripts/b3_analysis.py --template gaussian` | < 1 min |
| `b3_pareto_cross.png`, `b3_latents_cross.png` | `$PY scripts/b3_analysis.py --template cross` | < 1 min |
| références MSE de B3 (`results/b3_baselines.md` : vidéo moyenne, PCA de rang k) | `$PY scripts/b3_baselines.py` | < 1 min |
| sonde « décodeur partagé » g(z·t) (sortie console, cf. rapport §5) | `$PY scripts/b3_shared_probe.py` | ~1.5 min (GPU) |


# A5 — quelles conventions reproduisent l'expérience de gap du mémoire ?

Cibles (numérisées sur les Fig. 6–7 du PDF, `results/memoire_digit/`) : pic de E(ω) dans le gap = gaussian 9.3e-06, segment 1.5e-03, quarter_circle 1.2e-03, cross 1.3e-02 ; E < 1e-7 hors du gap.

D = écart quadratique moyen en décades (log10) sur les 4 pics. Rendu : analytic4 = fonction continue anti-crénelée 4×4 (défaut), analytic1 = échantillonnage ponctuel, interp1 = rotation bilinéaire de l'image. gap « thirds » = [ω_max/3, 2ω_max/3] (texte du mémoire), « fig » = [0.35, 0.65]·ω_max (zone grisée des figures). Régresseur : pinvτ = pseudo-inverse tronquée à λ_k/λ_1 > τ (τ = 1e-8 ⇔ rcond = 1e-4 sur les valeurs singulières de Φ) ; bias_ = avec biais (covariance centrée). Population : quadrature n = 2000.

| rang | rendu | ω_max | gap | unité | régresseur | gaussienne | segment | quart de cercle | croix | dipôle | D (décades) | max hors gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | analytic4 | 2π/T | thirds | rad | pinv1e-08 | 5.7e-06 | 3.4e-03 | 5.0e-04 | 1.6e-03 | 1.8e-05 | 0.53 | 1.8e-06 |
| 2 | analytic4 | 2π/T | thirds | rad | bias_pinv1e-08 | 4.1e-06 | 5.1e-03 | 2.7e-04 | 3.5e-03 | 1.5e-05 | 0.54 | 2.4e-06 |
| 3 | analytic1 | 2π/T | fig | rad | pinv1e-08 | 3.3e-06 | 1.0e-02 | 3.9e-04 | 2.4e-03 | 3.0e-06 | 0.64 | 4.0e-05 |
| 4 | interp1 | 2π/T | thirds | rad | pinv1e-08 | 6.2e-05 | 5.6e-03 | 2.7e-04 | 4.0e-03 | 2.6e-05 | 0.65 | 9.8e-09 |
| 5 | analytic1 | 2π/T | thirds | rad | pinv1e-08 | 6.0e-06 | 9.4e-03 | 6.4e-04 | 1.2e-03 | 1.8e-05 | 0.67 | 4.5e-05 |
| 6 | analytic4 | 4π/T | thirds | deg | bias_pinv1e-04 | 1.7e-06 | 2.0e-03 | 1.4e-03 | 9.4e-04 | 1.7e-06 | 0.68 | 1.9e-03 |
| 7 | interp1 | 2π/T | fig | rad | pinv1e-08 | 3.0e-05 | 4.6e-03 | 1.8e-04 | 1.8e-03 | 3.5e-06 | 0.69 | 1.3e-08 |
| 8 | analytic4 | 4π/T | thirds | deg | pinv1e-04 | 7.4e-06 | 1.6e-03 | 2.1e-04 | 9.0e-04 | 7.4e-06 | 0.70 | 3.0e-03 |
| 9 | analytic4 | 2π/T | thirds | rad | pinv1e-12 | 3.5e-06 | 1.8e-02 | 6.4e-04 | 2.0e-03 | 1.2e-06 | 0.72 | 4.7e-06 |
| 10 | analytic4 | 2π/T | fig | rad | pinv1e-08 | 3.3e-06 | 5.5e-03 | 2.0e-04 | 9.9e-04 | 3.0e-06 | 0.77 | 3.4e-06 |
| 11 | analytic1 | 2π/T | fig | rad | bias_pinv1e-08 | 1.8e-06 | 6.9e-03 | 2.1e-04 | 1.5e-03 | 1.6e-05 | 0.78 | 1.2e-04 |
| 12 | analytic1 | 2π/T | thirds | rad | bias_pinv1e-08 | 4.2e-06 | 7.7e-03 | 1.6e-04 | 1.2e-03 | 1.5e-05 | 0.78 | 1.4e-04 |
| 13 | analytic1 | 2π/T | thirds | rad | bias_pinv1e-04 | 7.5e-05 | 2.2e-03 | 7.6e-05 | 4.1e-03 | 6.6e-04 | 0.80 | 2.4e-05 |
| 14 | analytic4 | 2π/T | fig | rad | pinv1e-12 | 4.0e-07 | 8.0e-03 | 1.0e-03 | 4.1e-03 | 1.7e-07 | 0.82 | 2.8e-06 |
| 15 | analytic1 | 2π/T | thirds | rad | pinv1e-12 | 1.1e-06 | 9.4e-03 | 6.4e-04 | 1.2e-03 | 1.3e-06 | 0.82 | 4.5e-05 |
| 16 | analytic1 | 2π/T | fig | rad | bias_pinv1e-04 | 5.5e-05 | 1.8e-03 | 5.7e-05 | 3.1e-03 | 3.9e-06 | 0.83 | 2.4e-05 |
| 17 | analytic1 | 2π/T | fig | rad | pinv1e-12 | 4.1e-07 | 1.0e-02 | 3.9e-04 | 2.4e-03 | 1.8e-07 | 0.91 | 4.0e-05 |
| 18 | interp1 | 2π/T | thirds | rad | pinv1e-12 | 2.0e-05 | 6.7e-04 | 3.8e-04 | 2.3e-04 | 1.0e-04 | 0.94 | 2.9e-11 |
| 19 | analytic4 | 2π/T | fig | rad | bias_pinv1e-08 | 1.7e-06 | 5.4e-03 | 6.8e-04 | 3.0e-04 | 1.6e-05 | 0.94 | 2.9e-06 |
| 20 | analytic1 | 2π/T | fig | rad | pinv1e-04 | 4.8e-04 | 9.8e-03 | 4.4e-04 | 5.8e-03 | 1.0e-03 | 0.99 | 4.4e-05 |
| 21 | interp1 | 2π/T | fig | rad | pinv1e-04 | 4.5e-04 | 6.1e-03 | 2.0e-04 | 4.7e-03 | 9.0e-04 | 1.00 | 5.2e-05 |
| 22 | analytic4 | 2π/T | fig | rad | pinv1e-04 | 4.6e-04 | 6.7e-03 | 2.1e-04 | 5.0e-03 | 1.0e-03 | 1.01 | 5.4e-05 |
| 23 | interp1 | 2π/T | fig | rad | pinv1e-12 | 1.7e-05 | 2.0e-04 | 3.9e-04 | 2.2e-04 | 1.2e-05 | 1.03 | 5.9e-11 |
| 24 | interp1 | 2π/T | thirds | rad | pinv1e-04 | 7.4e-04 | 7.6e-03 | 3.6e-04 | 6.5e-03 | 1.2e-03 | 1.06 | 6.6e-05 |
| 25 | analytic4 | 2π/T | thirds | rad | pinv1e-04 | 7.5e-04 | 7.8e-03 | 3.3e-04 | 6.5e-03 | 1.2e-03 | 1.07 | 6.8e-05 |
| 26 | analytic1 | 2π/T | thirds | rad | pinv1e-04 | 7.7e-04 | 1.2e-02 | 6.8e-04 | 7.2e-03 | 1.3e-03 | 1.08 | 4.5e-05 |
| 27 | analytic4 | 4π/T | fig | rad | bias_pinv1e-08 | 2.7e-04 | 1.2e-01 | 5.9e-03 | 6.1e-03 | 9.5e-04 | 1.26 | 8.5e-05 |
| 28 | analytic4 | 2π/T | thirds | rad | bias_pinv1e-04 | 7.4e-05 | 3.0e-04 | 7.4e-06 | 2.9e-03 | 6.4e-04 | 1.28 | 9.4e-06 |
| 29 | analytic4 | 4π/T | thirds | rad | bias_pinv1e-08 | 1.9e-03 | 1.2e-02 | 5.5e-03 | 1.9e-02 | 3.9e-03 | 1.29 | 4.6e-05 |
| 30 | interp1 | 4π/T | fig | rad | bias_pinv1e-08 | 2.0e-03 | 7.3e-03 | 4.5e-03 | 2.7e-03 | 3.8e-03 | 1.29 | 3.3e-09 |
| 31 | analytic1 | 4π/T | fig | rad | bias_pinv1e-08 | 3.0e-04 | 1.1e-01 | 8.1e-03 | 4.4e-02 | 1.0e-03 | 1.29 | 4.6e-04 |
| 32 | interp1 | 2π/T | thirds | rad | bias_pinv1e-04 | 7.5e-05 | 3.2e-04 | 6.5e-06 | 2.6e-03 | 5.9e-04 | 1.31 | 8.0e-06 |
| 33 | interp1 | 4π/T | fig | rad | bias_pinv1e-04 | 3.5e-03 | 8.2e-03 | 4.4e-03 | 2.1e-02 | 5.8e-03 | 1.37 | 9.3e-05 |
| 34 | analytic4 | 4π/T | fig | rad | bias_pinv1e-04 | 3.6e-03 | 1.1e-02 | 4.1e-03 | 2.1e-02 | 5.9e-03 | 1.39 | 8.2e-05 |
| 35 | analytic4 | 2π/T | fig | rad | bias_pinv1e-04 | 5.4e-05 | 2.2e-04 | 4.6e-06 | 2.0e-03 | 3.4e-06 | 1.39 | 1.4e-05 |
| 36 | interp1 | 4π/T | thirds | rad | bias_pinv1e-08 | 3.5e-03 | 1.1e-02 | 6.0e-03 | 1.4e-02 | 6.4e-03 | 1.40 | 3.5e-09 |
| 37 | interp1 | 2π/T | thirds | rad | bias_pinv1e-08 | 6.5e-07 | 2.6e-04 | 4.2e-06 | 1.5e-03 | 5.0e-05 | 1.49 | 1.5e-09 |
| 38 | interp1 | 4π/T | thirds | rad | bias_pinv1e-04 | 4.8e-03 | 1.4e-02 | 5.8e-03 | 4.2e-02 | 8.3e-03 | 1.50 | 2.0e-05 |
| 39 | analytic1 | 4π/T | thirds | rad | bias_pinv1e-08 | 2.3e-03 | 7.8e-02 | 5.7e-03 | 2.6e-02 | 2.9e-03 | 1.51 | 2.9e-04 |
| 40 | analytic4 | 4π/T | thirds | rad | bias_pinv1e-04 | 4.8e-03 | 2.1e-02 | 5.8e-03 | 3.8e-02 | 7.6e-03 | 1.53 | 3.5e-05 |
| 41 | analytic1 | 4π/T | fig | rad | bias_pinv1e-04 | 3.6e-03 | 4.3e-02 | 6.4e-03 | 2.6e-02 | 6.0e-03 | 1.54 | 1.5e-04 |
| 42 | interp1 | 2π/T | fig | rad | bias_pinv1e-08 | 3.9e-07 | 1.8e-04 | 5.4e-06 | 7.8e-04 | 4.1e-06 | 1.56 | 7.5e-10 |
| 43 | interp1 | 2π/T | fig | rad | bias_pinv1e-04 | 5.4e-05 | 2.1e-04 | 1.9e-06 | 2.3e-03 | 1.0e-04 | 1.56 | 1.2e-05 |
| 44 | interp1 | 4π/T | fig | rad | pinv1e-12 | 7.7e-03 | 4.4e-03 | 1.2e-02 | 7.1e-03 | 1.0e-03 | 1.57 | 9.6e-10 |
| 45 | analytic1 | 4π/T | thirds | rad | bias_pinv1e-04 | 4.8e-03 | 5.0e-02 | 7.7e-03 | 5.0e-02 | 7.8e-03 | 1.63 | 1.3e-04 |
| 46 | interp1 | 4π/T | thirds | rad | pinv1e-12 | 1.1e-02 | 1.8e-02 | 1.4e-02 | 5.0e-03 | 3.0e-03 | 1.72 | 3.5e-10 |
| 47 | analytic1 | 4π/T | fig | rad | pinv1e-12 | 2.1e-03 | 1.4e-01 | 3.9e-02 | 4.7e-02 | 4.5e-04 | 1.73 | 5.3e-04 |
| 48 | analytic1 | 4π/T | thirds | rad | pinv1e-12 | 4.7e-03 | 1.1e-01 | 4.1e-02 | 2.9e-02 | 1.2e-03 | 1.82 | 2.6e-04 |
| 49 | analytic4 | 4π/T | fig | rad | pinv1e-08 | 9.0e-03 | 6.7e-02 | 2.6e-02 | 5.1e-03 | 4.3e-03 | 1.84 | 3.3e-05 |
| 50 | analytic4 | 4π/T | thirds | rad | pinv1e-08 | 1.7e-02 | 1.4e-02 | 3.5e-02 | 1.6e-02 | 1.0e-02 | 1.85 | 3.2e-05 |
| 51 | interp1 | 4π/T | fig | rad | pinv1e-08 | 1.5e-02 | 5.3e-02 | 1.8e-02 | 5.1e-03 | 1.9e-03 | 1.89 | 1.3e-08 |
| 52 | analytic1 | 4π/T | fig | rad | pinv1e-08 | 9.3e-03 | 1.4e-01 | 3.9e-02 | 4.7e-02 | 4.6e-03 | 1.96 | 5.3e-04 |
| 53 | interp1 | 4π/T | thirds | rad | pinv1e-08 | 2.4e-02 | 6.4e-02 | 2.9e-02 | 2.2e-02 | 3.4e-03 | 2.02 | 2.1e-08 |
| 54 | analytic1 | 4π/T | thirds | rad | pinv1e-08 | 1.7e-02 | 1.1e-01 | 4.1e-02 | 2.9e-02 | 1.1e-02 | 2.04 | 2.6e-04 |
| 55 | interp1 | 4π/T | fig | rad | pinv1e-04 | 3.5e-02 | 6.2e-02 | 2.6e-02 | 2.3e-02 | 1.7e-02 | 2.08 | 2.2e-04 |
| 56 | analytic4 | 4π/T | thirds | rad | pinv1e-12 | 4.4e-03 | 8.4e-01 | 3.5e-02 | 6.4e-02 | 1.3e-03 | 2.08 | 1.9e-04 |
| 57 | analytic4 | 4π/T | fig | rad | pinv1e-04 | 3.6e-02 | 6.4e-02 | 2.8e-02 | 2.4e-02 | 1.8e-02 | 2.09 | 1.9e-04 |
| 58 | analytic4 | 4π/T | fig | rad | pinv1e-12 | 2.0e-03 | 1.8e+00 | 5.5e-02 | 6.2e-02 | 4.2e-04 | 2.13 | 5.4e-04 |
| 59 | analytic1 | 4π/T | fig | rad | pinv1e-04 | 3.6e-02 | 9.4e-02 | 3.8e-02 | 4.2e-02 | 2.1e-02 | 2.16 | 3.8e-04 |
| 60 | interp1 | 4π/T | thirds | rad | pinv1e-04 | 4.8e-02 | 7.3e-02 | 4.0e-02 | 3.4e-02 | 2.0e-02 | 2.19 | 1.8e-04 |
| 61 | analytic4 | 4π/T | thirds | rad | pinv1e-04 | 4.9e-02 | 7.6e-02 | 4.1e-02 | 3.8e-02 | 2.2e-02 | 2.20 | 2.1e-04 |
| 62 | analytic1 | 4π/T | thirds | rad | pinv1e-04 | 5.0e-02 | 1.3e-01 | 5.0e-02 | 6.1e-02 | 2.2e-02 | 2.27 | 4.2e-04 |
| 63 | analytic4 | 4π/T | thirds | deg | pinv1e-08 | 7.5e-11 | 2.6e-03 | 3.9e-03 | 3.1e-03 | 7.4e-11 | 2.58 | 1.7e-03 |
| 64 | analytic4 | 4π/T | thirds | deg | pinv1e-12 | 1.6e-11 | 2.6e-03 | 4.0e-03 | 3.7e-03 | 1.6e-11 | 2.91 | 1.7e-03 |
| 65 | analytic4 | 4π/T | thirds | deg | bias_pinv1e-08 | 1.6e-11 | 2.8e-03 | 4.0e-03 | 3.6e-03 | 1.6e-11 | 2.91 | 1.7e-03 |
| — | *mémoire* | *4π/T (axes)* | *fig* | | *« 1e-4 »* | *9.3e-06* | *1.5e-03* | *1.2e-03* | *1.3e-02* | — | 0 | *< 1e-7* |

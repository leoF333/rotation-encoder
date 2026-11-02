# B3 — balayage d'architectures (template gaussian)

Entraînement sur le créneau de [0.05, 2π/T] (gap = tiers central), P vidéos, 6400 pas d'Adam, 3 seeds. Valeurs : médiane sur les seeds (± écart-type) ; « latent » = R² de la meilleure fonction monotone ω = f(z) sur tout l'intervalle (angle polaire si k = 2), « pire seed » = minimum ; « effondrés » = latent constant. Pareto : non dominé sur (MSE distribution, MSE gap, 1 − latent au pire seed).

| config | k | P | paramètres | MSE distrib. | MSE gap | latent (méd.) | latent (pire seed) | R² 5-NN | R² monotone dans le gap | R² linéaire z(ω) | temps (s) | effondrés | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lin-shallow | 1 | 4000 | 2.35e+04 | 9.53e-03 ± 8.0e-05 | 2.06e-02 ± 1.3e-04 | 0.98149 | 0.9811 | 0.9772 | 1.0000 | 0.925 | 5 | 0/3 |  |
| lin-mid | 1 | 4000 | 2.02e+06 | 1.89e-03 ± 2.5e-04 | 1.09e-02 ± 6.1e-05 | 1.00000 | 0.9992 | 1.0000 | 1.0000 | 0.833 | 14 | 0/3 |  |
| lin-deep | 1 | 4000 | 8.32e+06 | 6.90e-06 ± 9.5e-06 | 2.01e-02 ± 6.5e-03 | 0.82794 | 0.7982 | 0.8456 | 0.9667 | 0.679 | 48 | 0/3 |  |
| mlp-deep | 1 | 4000 | 1.03e+07 | 1.14e-05 ± 5.0e-06 | 6.25e-03 ± 4.2e-04 | 1.00000 | 0.9996 | 1.0000 | 1.0000 | 0.878 | 55 | 0/3 | **oui** |
| deep-deep | 1 | 4000 | 1.04e+07 | 2.04e-06 ± 7.1e-07 | 8.47e-03 ± 1.2e-03 | 0.99220 | 0.9857 | 0.9913 | 0.7928 | 0.882 | 56 | 0/3 | **oui** |
| lin-deep-k2 | 2 | 4000 | 8.33e+06 | 7.78e-07 ± 2.2e-07 | 1.52e-02 ± 1.5e-03 | 0.90992 | 0.8331 | 1.0000 | 0.9440 | nan | 48 | 0/3 | **oui** |

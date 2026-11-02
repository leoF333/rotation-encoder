# B3 — balayage d'architectures (template cross)

Entraînement sur le créneau de [0.05, 2π/T] (gap = tiers central), P vidéos, 6400 pas d'Adam, 3 seeds. Valeurs : médiane sur les seeds (± écart-type) ; « latent » = R² de la meilleure fonction monotone ω = f(z) sur tout l'intervalle (angle polaire si k = 2), « pire seed » = minimum ; « effondrés » = latent constant. Pareto : non dominé sur (MSE distribution, MSE gap, 1 − latent au pire seed).

| config | k | P | paramètres | MSE distrib. | MSE gap | latent (méd.) | latent (pire seed) | R² 5-NN | R² monotone dans le gap | R² linéaire z(ω) | temps (s) | effondrés | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lin-shallow | 1 | 4000 | 2.35e+04 | 7.95e-02 ± 3.6e-04 | 1.48e-01 ± 6.2e-04 | 0.87419 | 0.8111 | 0.8961 | 0.4230 | 0.528 | 5 | 0/3 | **oui** |
| lin-mid | 1 | 4000 | 2.02e+06 | 1.57e-02 ± 1.7e-03 | 1.96e-01 ± 6.6e-03 | 0.64712 | 0.3989 | 0.8284 | 0.7185 | 0.246 | 14 | 0/3 |  |
| lin-deep | 1 | 4000 | 8.32e+06 | 5.03e-03 ± 1.4e-03 | 1.89e-01 ± 1.9e-02 | 0.74794 | 0.4374 | 0.7911 | 0.9944 | 0.270 | 48 | 0/3 |  |
| mlp-deep | 1 | 4000 | 1.03e+07 | 5.12e-03 ± 8.3e-05 | 1.97e-01 ± 9.0e-03 | 0.78002 | 0.6077 | 0.8691 | 0.9841 | 0.307 | 56 | 0/3 |  |
| deep-deep | 1 | 4000 | 1.04e+07 | 4.73e-03 ± 1.4e-04 | 1.95e-01 ± 7.3e-03 | 0.80454 | 0.7820 | 0.8597 | 0.9974 | 0.283 | 56 | 0/3 | **oui** |
| lin-deep-k2 | 2 | 4000 | 8.33e+06 | 1.33e-03 ± 1.1e-04 | 1.55e-01 ± 6.2e-03 | 0.79258 | 0.7260 | 0.9974 | 0.8865 | nan | 48 | 0/3 | **oui** |

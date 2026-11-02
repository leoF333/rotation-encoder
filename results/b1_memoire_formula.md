# B1 — test de la formule du mémoire E_train = (1 − 1/α)·σ²_res

Bruit d'étiquette σ_ε = 0.001 (σ²_res ≈ σ_ε² = 1.0e-06), ρ uniforme, ω_max = 4π/T, moyenne sur 3 tirages. « rang » = nombre de modes conservés par la pinv 1e-10 (≈ rang numérique du jeu de P vidéos).

| template | type | P | α = P/N | E_train (pinv 1e-10) | (1 − 1/α)σ² (mémoire) | σ²(1 − r/P), r = min(P, rang pop.) | E_train (ridge 1e-8) | répliques (ridge 1e-8) |
|---|---|---|---|---|---|---|---|---|
| gaussian | real | 30 | 0.00 | 3.812e-08 | nan | 0.000e+00 | 1.337e-07 | 1.679e-11 |
| gaussian | surrogate | 30 | 0.00 | 4.088e-31 | nan | 0.000e+00 | 9.009e-12 | 1.679e-11 |
| gaussian | real | 100 | 0.01 | 4.463e-07 | nan | 4.700e-07 | 4.923e-07 | 5.476e-07 |
| gaussian | surrogate | 100 | 0.01 | 4.078e-07 | nan | 4.700e-07 | 4.647e-07 | 5.476e-07 |
| gaussian | real | 300 | 0.04 | 8.106e-07 | nan | 8.233e-07 | 8.340e-07 | 8.464e-07 |
| gaussian | surrogate | 300 | 0.04 | 7.789e-07 | nan | 8.233e-07 | 7.981e-07 | 8.464e-07 |
| gaussian | real | 1000 | 0.13 | 9.408e-07 | nan | 9.470e-07 | 9.473e-07 | 9.537e-07 |
| gaussian | surrogate | 1000 | 0.13 | 9.403e-07 | nan | 9.470e-07 | 9.485e-07 | 9.537e-07 |
| gaussian | real | 3000 | 0.38 | 9.724e-07 | nan | 9.823e-07 | 9.743e-07 | 9.846e-07 |
| gaussian | surrogate | 3000 | 0.38 | 9.792e-07 | nan | 9.823e-07 | 9.808e-07 | 9.846e-07 |
| gaussian | real | 7000 | 0.89 | 9.866e-07 | nan | 9.924e-07 | 9.876e-07 | 9.935e-07 |
| gaussian | surrogate | 7000 | 0.89 | 9.852e-07 | nan | 9.924e-07 | 9.866e-07 | 9.935e-07 |
| gaussian | real | 12000 | 1.53 | 9.911e-07 | 3.467e-07 | 9.956e-07 | 9.920e-07 | 9.962e-07 |
| gaussian | surrogate | 12000 | 1.53 | 9.794e-07 | 3.467e-07 | 9.956e-07 | 9.799e-07 | 9.962e-07 |
| cross | real | 30 | 0.00 | 2.641e-31 | nan | 0.000e+00 | 3.365e-14 | 2.912e-14 |
| cross | surrogate | 30 | 0.00 | 3.828e-31 | nan | 0.000e+00 | 2.800e-14 | 2.912e-14 |
| cross | real | 100 | 0.01 | 6.586e-31 | nan | 0.000e+00 | 4.348e-12 | 1.170e-12 |
| cross | surrogate | 100 | 0.01 | 7.999e-31 | nan | 0.000e+00 | 1.133e-12 | 1.170e-12 |
| cross | real | 300 | 0.04 | 8.805e-10 | nan | 0.000e+00 | 1.808e-09 | 4.760e-11 |
| cross | surrogate | 300 | 0.04 | 1.179e-30 | nan | 0.000e+00 | 4.687e-11 | 4.760e-11 |
| cross | real | 1000 | 0.13 | 4.675e-09 | nan | 0.000e+00 | 3.851e-08 | 2.468e-08 |
| cross | surrogate | 1000 | 0.13 | 2.287e-30 | nan | 0.000e+00 | 2.587e-08 | 2.468e-08 |
| cross | real | 3000 | 0.38 | 6.547e-07 | nan | 5.137e-07 | 6.765e-07 | 6.637e-07 |
| cross | surrogate | 3000 | 0.38 | 6.397e-07 | nan | 5.137e-07 | 6.633e-07 | 6.637e-07 |
| cross | real | 7000 | 0.89 | 1.018e-06 | nan | 7.916e-07 | 1.026e-06 | 9.915e-07 |
| cross | surrogate | 7000 | 0.89 | 9.803e-07 | nan | 7.916e-07 | 9.906e-07 | 9.915e-07 |
| cross | real | 12000 | 1.53 | 1.131e-06 | 3.467e-07 | 8.784e-07 | 1.136e-06 | 1.096e-06 |
| cross | surrogate | 12000 | 1.53 | 1.073e-06 | 3.467e-07 | 8.784e-07 | 1.081e-06 | 1.096e-06 |

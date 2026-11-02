# Autoencodeurs — template arc, ω ~ U[0.05, 2π/T], latent k = 1, 3 seeds (moyenne ± écart-type)

| architecture | mse_test | spearman_abs | R2_monotone | R2_knn | R2_lin_z | R2_sin_z | n_params | train_time_s |
|---|---|---|---|---|---|---|---|---|
| sym3 | 0.003767 ± 8.3e-05 | 0.9784 ± 0.001 | 0.9789 ± 0.0016 | 0.9739 ± 0.0024 | 0.9117 ± 0.0024 | 0.9845 ± 0.0014 | 23521 | 10.01 ± 0.12 |
| sym5 | 0.002614 ± 0.0027 | nan ± nan | 0.9138 ± 0.12 | 0.6608 ± 0.48 | -inf ± nan | nan ± nan | 2015393 | 24.63 ± 2.6 |
| asym | 5.837e-05 ± 9.9e-06 | 1 ± 6.8e-08 | 1 ± 4.1e-10 | 1 ± 2.3e-07 | 0.9815 ± 0.0079 | 0.9933 ± 0.004 | 8323777 | 94.64 ± 2.3 |

mse = moyenne par pixel sur le test ; R2_monotone = régression isotone de ω sur z ; R2_knn = 5-NN leave-one-out ; R2_lin_z / R2_sin_z = ajustements de z(ω) comme la Fig. 4 (mémoire : 0.6409 / 0.9061).

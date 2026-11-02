# B1 — régime α > 1 (P > N), ρ uniforme

Mémoire : E_train = (1 − 1/α)·X (X = variance résiduelle, formule corrigée : σ²_res). Théorie (T2, λ_r → 0) : E_train = (1 − df/P)² E_gen avec df = nombre de modes appris.

| template | type | P | α | régresseur | E_train (moy.) | E_gen (moy.) | σ²_res (pop., modes > 1e-14 λ₁) | (1−1/α)σ²_res | théorie répliques |
|---|---|---|---|---|---|---|---|---|---|
| cross | real | 12000 | 1.53 | pinv1e-04 | 3.923e-05 | 4.025e-05 | 2.388e-07 | 8.280e-08 | nan |
| cross | real | 12000 | 1.53 | pinv1e-10 | 2.611e-07 | 3.483e-07 | 2.388e-07 | 8.280e-08 | nan |
| cross | real | 12000 | 1.53 | ridge1e-02 | 6.538e-03 | 6.564e-03 | 2.388e-07 | 8.280e-08 | 6.563e-03 |
| cross | real | 12000 | 1.53 | ridge1e-04 | 1.748e-05 | 1.812e-05 | 2.388e-07 | 8.280e-08 | 1.745e-05 |
| cross | real | 12000 | 1.53 | ridge1e-06 | 6.088e-07 | 6.764e-07 | 2.388e-07 | 8.280e-08 | 5.860e-07 |
| cross | real | 12000 | 1.53 | ridge1e-08 | 2.652e-07 | 3.483e-07 | 2.388e-07 | 8.280e-08 | 2.159e-07 |
| cross | real | 20000 | 2.55 | pinv1e-04 | 3.955e-05 | 3.998e-05 | 2.388e-07 | 1.452e-07 | nan |
| cross | real | 20000 | 2.55 | pinv1e-10 | 2.779e-07 | 3.304e-07 | 2.388e-07 | 1.452e-07 | nan |
| cross | real | 20000 | 2.55 | ridge1e-02 | 6.562e-03 | 6.598e-03 | 2.388e-07 | 1.452e-07 | 6.562e-03 |
| cross | real | 20000 | 2.55 | ridge1e-04 | 1.752e-05 | 1.784e-05 | 2.388e-07 | 1.452e-07 | 1.747e-05 |
| cross | real | 20000 | 2.55 | ridge1e-06 | 6.233e-07 | 6.649e-07 | 2.388e-07 | 1.452e-07 | 5.958e-07 |
| cross | real | 20000 | 2.55 | ridge1e-08 | 2.817e-07 | 3.318e-07 | 2.388e-07 | 1.452e-07 | 2.272e-07 |
| cross | surrogate | 12000 | 1.53 | pinv1e-04 | 3.941e-05 | 4.005e-05 | 2.388e-07 | 8.280e-08 | nan |
| cross | surrogate | 12000 | 1.53 | pinv1e-10 | 2.100e-07 | 4.830e-07 | 2.388e-07 | 8.280e-08 | nan |
| cross | surrogate | 12000 | 1.53 | ridge1e-02 | 6.571e-03 | 6.592e-03 | 2.388e-07 | 8.280e-08 | 6.563e-03 |
| cross | surrogate | 12000 | 1.53 | ridge1e-04 | 1.745e-05 | 1.781e-05 | 2.388e-07 | 8.280e-08 | 1.745e-05 |
| cross | surrogate | 12000 | 1.53 | ridge1e-06 | 5.880e-07 | 6.926e-07 | 2.388e-07 | 8.280e-08 | 5.860e-07 |
| cross | surrogate | 12000 | 1.53 | ridge1e-08 | 2.163e-07 | 4.481e-07 | 2.388e-07 | 8.280e-08 | 2.159e-07 |
| cross | surrogate | 20000 | 2.55 | pinv1e-04 | 3.956e-05 | 3.991e-05 | 2.388e-07 | 1.452e-07 | nan |
| cross | surrogate | 20000 | 2.55 | pinv1e-10 | 2.207e-07 | 4.651e-07 | 2.388e-07 | 1.452e-07 | nan |
| cross | surrogate | 20000 | 2.55 | ridge1e-02 | 6.574e-03 | 6.612e-03 | 2.388e-07 | 1.452e-07 | 6.562e-03 |
| cross | surrogate | 20000 | 2.55 | ridge1e-04 | 1.748e-05 | 1.769e-05 | 2.388e-07 | 1.452e-07 | 1.747e-05 |
| cross | surrogate | 20000 | 2.55 | ridge1e-06 | 5.940e-07 | 6.845e-07 | 2.388e-07 | 1.452e-07 | 5.958e-07 |
| cross | surrogate | 20000 | 2.55 | ridge1e-08 | 2.264e-07 | 4.358e-07 | 2.388e-07 | 1.452e-07 | 2.272e-07 |
| gaussian | real | 12000 | 1.53 | pinv1e-04 | 9.307e-07 | 9.329e-07 | 1.110e-16 | 3.849e-17 | nan |
| gaussian | real | 12000 | 1.53 | pinv1e-10 | 1.072e-12 | 1.109e-12 | 1.110e-16 | 3.849e-17 | nan |
| gaussian | real | 12000 | 1.53 | ridge1e-02 | 4.327e-04 | 4.325e-04 | 1.110e-16 | 3.849e-17 | 4.350e-04 |
| gaussian | real | 12000 | 1.53 | ridge1e-04 | 1.593e-06 | 1.621e-06 | 1.110e-16 | 3.849e-17 | 1.596e-06 |
| gaussian | real | 12000 | 1.53 | ridge1e-06 | 1.601e-08 | 1.670e-08 | 1.110e-16 | 3.849e-17 | 1.614e-08 |
| gaussian | real | 12000 | 1.53 | ridge1e-08 | 1.081e-10 | 1.150e-10 | 1.110e-16 | 3.849e-17 | 1.071e-10 |
| gaussian | real | 20000 | 2.55 | pinv1e-04 | 9.464e-07 | 9.507e-07 | 1.110e-16 | 6.750e-17 | nan |
| gaussian | real | 20000 | 2.55 | pinv1e-10 | 1.098e-12 | 1.108e-12 | 1.110e-16 | 6.750e-17 | nan |
| gaussian | real | 20000 | 2.55 | ridge1e-02 | 4.364e-04 | 4.427e-04 | 1.110e-16 | 6.750e-17 | 4.349e-04 |
| gaussian | real | 20000 | 2.55 | ridge1e-04 | 1.603e-06 | 1.621e-06 | 1.110e-16 | 6.750e-17 | 1.596e-06 |
| gaussian | real | 20000 | 2.55 | ridge1e-06 | 1.613e-08 | 1.624e-08 | 1.110e-16 | 6.750e-17 | 1.614e-08 |
| gaussian | real | 20000 | 2.55 | ridge1e-08 | 1.068e-10 | 1.082e-10 | 1.110e-16 | 6.750e-17 | 1.071e-10 |
| gaussian | surrogate | 12000 | 1.53 | pinv1e-04 | 9.460e-07 | 9.541e-07 | 1.110e-16 | 3.849e-17 | nan |
| gaussian | surrogate | 12000 | 1.53 | pinv1e-10 | 1.091e-12 | 1.099e-12 | 1.110e-16 | 3.849e-17 | nan |
| gaussian | surrogate | 12000 | 1.53 | ridge1e-02 | 4.342e-04 | 4.328e-04 | 1.110e-16 | 3.849e-17 | 4.350e-04 |
| gaussian | surrogate | 12000 | 1.53 | ridge1e-04 | 1.594e-06 | 1.610e-06 | 1.110e-16 | 3.849e-17 | 1.596e-06 |
| gaussian | surrogate | 12000 | 1.53 | ridge1e-06 | 1.615e-08 | 1.612e-08 | 1.110e-16 | 3.849e-17 | 1.614e-08 |
| gaussian | surrogate | 12000 | 1.53 | ridge1e-08 | 1.072e-10 | 1.081e-10 | 1.110e-16 | 3.849e-17 | 1.071e-10 |
| gaussian | surrogate | 20000 | 2.55 | pinv1e-04 | 9.314e-07 | 9.430e-07 | 1.110e-16 | 6.750e-17 | nan |
| gaussian | surrogate | 20000 | 2.55 | pinv1e-10 | 1.094e-12 | 1.100e-12 | 1.110e-16 | 6.750e-17 | nan |
| gaussian | surrogate | 20000 | 2.55 | ridge1e-02 | 4.357e-04 | 4.395e-04 | 1.110e-16 | 6.750e-17 | 4.349e-04 |
| gaussian | surrogate | 20000 | 2.55 | ridge1e-04 | 1.593e-06 | 1.602e-06 | 1.110e-16 | 6.750e-17 | 1.596e-06 |
| gaussian | surrogate | 20000 | 2.55 | ridge1e-06 | 1.620e-08 | 1.636e-08 | 1.110e-16 | 6.750e-17 | 1.614e-08 |
| gaussian | surrogate | 20000 | 2.55 | ridge1e-08 | 1.071e-10 | 1.076e-10 | 1.110e-16 | 6.750e-17 | 1.071e-10 |

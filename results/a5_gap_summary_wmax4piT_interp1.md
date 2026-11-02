# Expérience de gap — rendu interp ss=1, ω_max = 4π/T = 1.2566, gap = [0.419, 0.838]

Valeurs : max de E(ω) dans le gap / moyenne dans le gap / max hors gap.

| template | régresseur | K | max gap | moy. gap | max hors gap |
|---|---|---|---|---|---|
| gaussian | trunc1e-04 | 19 | 4.80e-02 | 1.51e-02 | 9.68e-05 |
| gaussian | trunc1e-08 | 174 | 2.44e-02 | 6.57e-03 | 1.05e-08 |
| gaussian | trunc1e-12 | 1415 | 1.13e-02 | 2.62e-03 | 3.95e-12 |
| gaussian | ridge1e-03 |  | 5.87e-02 | 2.16e-02 | 1.92e-03 |
| gaussian | ridge1e-06 |  | 3.04e-02 | 8.47e-03 | 2.05e-06 |
| gaussian | ridge1e-09 |  | 2.11e-02 | 5.53e-03 | 1.01e-09 |
| gaussian | P2000_trunc1e-04 |  | 4.88e-02 | 1.55e-02 | 1.15e-04 |
| segment | trunc1e-04 | 88 | 7.34e-02 | 3.13e-02 | 7.17e-05 |
| segment | trunc1e-08 | 709 | 6.46e-02 | 2.35e-02 | 8.76e-09 |
| segment | trunc1e-12 | 1684 | 5.63e-03 | 1.67e-03 | 1.28e-10 |
| segment | ridge1e-03 |  | 7.46e-02 | 3.31e-02 | 2.16e-03 |
| segment | ridge1e-06 |  | 7.04e-02 | 2.80e-02 | 1.24e-06 |
| segment | ridge1e-09 |  | 4.52e-02 | 1.68e-02 | 1.36e-09 |
| segment | P2000_trunc1e-04 |  | 7.37e-02 | 3.15e-02 | 1.18e-04 |
| quarter_circle | trunc1e-04 | 59 | 4.05e-02 | 1.42e-02 | 1.05e-04 |
| quarter_circle | trunc1e-08 | 418 | 2.95e-02 | 9.81e-03 | 3.69e-09 |
| quarter_circle | trunc1e-12 | 1836 | 1.43e-02 | 5.30e-03 | 1.03e-11 |
| quarter_circle | ridge1e-03 |  | 4.45e-02 | 1.67e-02 | 2.20e-03 |
| quarter_circle | ridge1e-06 |  | 3.45e-02 | 1.17e-02 | 2.21e-06 |
| quarter_circle | ridge1e-09 |  | 2.47e-02 | 8.37e-03 | 1.02e-09 |
| quarter_circle | P2000_trunc1e-04 |  | 4.10e-02 | 1.45e-02 | 9.55e-05 |
| cross | trunc1e-04 | 59 | 3.42e-02 | 1.07e-02 | 1.84e-04 |
| cross | trunc1e-08 | 465 | 2.19e-02 | 7.75e-03 | 1.67e-08 |
| cross | trunc1e-12 | 1387 | 3.76e-03 | 7.19e-04 | 2.49e-10 |
| cross | ridge1e-03 |  | 7.03e-02 | 1.85e-02 | 6.42e-03 |
| cross | ridge1e-06 |  | 3.13e-02 | 1.01e-02 | 5.49e-06 |
| cross | ridge1e-09 |  | 1.38e-02 | 5.31e-03 | 6.95e-09 |
| cross | P2000_trunc1e-04 |  | 3.73e-02 | 1.10e-02 | 2.62e-04 |
| dipole | trunc1e-04 | 18 | 2.04e-02 | 5.00e-03 | 6.83e-05 |
| dipole | trunc1e-08 | 161 | 3.40e-03 | 8.79e-04 | 2.06e-08 |
| dipole | trunc1e-12 | 1225 | 2.27e-03 | 6.40e-04 | 3.75e-12 |
| dipole | ridge1e-03 |  | 3.29e-02 | 1.04e-02 | 3.75e-03 |
| dipole | ridge1e-06 |  | 9.36e-03 | 2.38e-03 | 3.42e-06 |
| dipole | ridge1e-09 |  | 2.06e-03 | 7.24e-04 | 5.25e-09 |
| dipole | P2000_trunc1e-04 |  | 2.42e-02 | 6.52e-03 | 1.17e-03 |

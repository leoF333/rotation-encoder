# Independent verification

Re-implementation, from the written derivations and conventions only (no import of `aerot`), of the key numbers:
rendering control (rank-one disk), PCA, gap experiment and its robustness to rendering details, the replica formula for
the gap error against ridge simulations on the real videos, and the saddle-point equations (sympy).

```
cd verification
python v1_pca.py && python v2_gap.py && python v2b_sensibilite.py
python v3_ood.py gaussian && python v3_ood.py cross && python v4_sympy_memoire.py && python v2c_figures.py
```

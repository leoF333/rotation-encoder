# Development notes

- Install in editable mode: `pip install -e ".[full,test]"`.
- Run the fast tests: `pytest` (~1 min, CPU). The theory tests compare replica formulas to direct Gaussian simulations.
- Scripts are idempotent and read/write `results/` and `figures/`; see `scripts/README.md` for one command per figure.
- Conventions (notation, normalisation of errors, scaling of the ridge) are fixed in `docs/CONVENTIONS.md`; `src/aerot/theory.py` follows them.
- Code and comments are in French (the project was written for a French institution); the README and docstring headers are in English.

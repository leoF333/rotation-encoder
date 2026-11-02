"""Project paths (relative to the repository root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # repository root
FIG = ROOT / "figures"
RES = ROOT / "results"
LOGS = ROOT / "logs"
DATA = ROOT / "data"
SPECTRA = DATA / "spectra"
THEORY_FIG = ROOT / "theory" / "figures"

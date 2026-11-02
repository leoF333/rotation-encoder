"""Style commun des figures (matplotlib). Couleur fixe par template (la couleur suit l'entité)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEMPLATE_COLOR = {
    "gaussian": PALETTE[0],
    "quarter_circle": PALETTE[1],
    "dipole": PALETTE[2],
    "segment": PALETTE[3],
    "cross": PALETTE[4],
    "arc": PALETTE[5],
    "disk": PALETTE[6],
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
GAP_SHADE = "#e34948"


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })


def save(fig, path, params: dict | None = None):
    """Sauve la figure (png) et ses paramètres (json à côté)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    if params is not None:
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(params, f, indent=1, default=str)

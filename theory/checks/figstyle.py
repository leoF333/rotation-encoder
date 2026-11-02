"""Style commun des figures analytiques (palette de référence dataviz, ordre fixe des séries)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.markersize": 5, "font.size": 10, "legend.frameon": False, "axes.titlesize": 11,
    })

"""Shared figure style: one place for fonts, sizes, colors, and saving.

Every experiment script calls `set_style()` once and saves through
`save_fig()` so all figures share the same look (200 dpi PNG, Okabe-Ito
colorblind-safe palette).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200

# Okabe-Ito colorblind-safe palette (https://jfly.uni-koeln.de/color/).
OKABE_ITO = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}
# Ordered cycle for series plots.
CYCLE = [OKABE_ITO[k] for k in
         ("blue", "vermillion", "green", "orange", "purple", "skyblue", "yellow")]


def set_style() -> None:
    """Apply the repository-wide matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.markersize": 4,
    })


def save_fig(fig: plt.Figure, path: Path | str) -> None:
    """Tight-layout, save at the shared DPI, and close."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)

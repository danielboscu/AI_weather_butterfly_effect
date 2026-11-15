# Shared matplotlib style for the figures, based on the plotting configuration by Weidong Li (University of Chicago).

small_size = 13
medium_size = 15
large_size = 17

params = {
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    "font.size": medium_size,
    "axes.titlesize": large_size,
    "axes.labelsize": medium_size,
    "xtick.labelsize": small_size,
    "ytick.labelsize": small_size,
    "legend.fontsize": medium_size,
    "figure.titlesize": large_size,

    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "patch.linewidth": 0.5,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    "axes.grid": False,
    "grid.alpha": 0.3,

    "axes.spines.top": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.spines.right": True,

    # LaTeX text rendering (needs a LaTeX install, e.g. module load texlive)
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "mathtext.fontset": "cm",
}

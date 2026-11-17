"""Shared figure style for the TEMPO-SAM manuscript.
Large fonts, thick axes, bright colours and legends without frames.
Import in every analysis script:  from plot_style import apply_style, finish_panel, save_fig
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Bright, high-contrast palette (blue, orange, green, magenta, amber, sky, yellow, black)
COLORS = ["#0057FF", "#FF5A00", "#00A651", "#E6007E",
          "#FFB000", "#00B8FF", "#FFE600", "#000000"]

# Legends never carry a frame, whatever a script passes.
from matplotlib.axes import Axes as _Axes
from matplotlib.figure import Figure as _Figure
def _noframe(fn):
    def wrapped(self, *a, **k):
        k["frameon"] = False
        return fn(self, *a, **k)
    return wrapped
if not getattr(_Axes.legend, "_noframe", False):
    _Axes.legend = _noframe(_Axes.legend); _Axes.legend._noframe = True
    _Figure.legend = _noframe(_Figure.legend); _Figure.legend._noframe = True

def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 22,
        "axes.labelsize": 25,
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "legend.fontsize": 18,
        "legend.frameon": False,
        "axes.linewidth": 2.5,
        "axes.edgecolor": "black",
        "xtick.major.width": 2.0, "ytick.major.width": 2.0,
        "xtick.major.size": 7,    "ytick.major.size": 7,
        "xtick.minor.width": 1.4, "ytick.minor.width": 1.4,
        "xtick.minor.size": 4,    "ytick.minor.size": 4,
        "xtick.direction": "in",  "ytick.direction": "in",
        "xtick.top": True,        "ytick.right": True,
        "lines.linewidth": 3.0,
        "lines.markersize": 10,
        "axes.prop_cycle": mpl.cycler(color=COLORS),
        "savefig.dpi": 600,
        "figure.dpi": 120,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,       # TrueType embedding (no Type 3 fonts)
        "ps.fonttype": 42,
    })

def finish_panel(ax, label=None):
    """Thick spines on all sides + optional bold panel label (a, b, c ...)."""
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(2.2); sp.set_color("black")
    if label:
        ax.text(-0.16, 1.04, label, transform=ax.transAxes,
                fontsize=30, fontweight="bold", va="bottom", ha="left")

def save_fig(fig, path_noext, border=False):
    """Save PNG (600 dpi) + PDF; no frame around the image (border=True adds one)."""
    if border:
        # frame drawn around the tight bounding box of all artists, so that labels
        # or legends extending beyond the nominal figure area stay inside the frame
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        p = 0.10
        W, H = fig.get_size_inches()
        fig.patches.append(mpl.patches.Rectangle(
            ((bb.x0 - p) / W, (bb.y0 - p) / H), (bb.width + 2 * p) / W, (bb.height + 2 * p) / H,
            transform=fig.transFigure, fill=False, edgecolor="black", linewidth=3.0, zorder=1000, clip_on=False))
    fig.savefig(path_noext + ".png", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(path_noext + ".pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

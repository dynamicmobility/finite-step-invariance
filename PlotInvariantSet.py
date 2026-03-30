"""
PlotInvariantSet.py  —  sample from the final invariant ellipsoid and plot
                         the Poincaré return points.

Reads the final ellipsoid from an ellipsoid_log CSV (last row = E*), samples
N points from it, applies the Poincaré map, and colour-codes return points by
whether they land inside (W, blue) or outside (V, red) the ellipsoid.

For 3D systems all three 2D projections are shown side by side, matching
the layout of PlotAlgorithmSets.py.  Colours match the hero figure
(PlotAlgorithmSteps.py).

Usage:
    python PlotInvariantSet.py <ellipsoid_log_csv>

    e.g.:
    python PlotInvariantSet.py ExpanderContractor_ellipsoid_log_20260328_210024.csv
    python PlotInvariantSet.py CompassGaitWalker_ellipsoid_log_20260328_211440.csv
"""

import sys
import importlib
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from ellipsoids import Ellipsoid, sample_from, plot_projection

# =========================================================================
# Paper style  (matches PlotAlgorithmSteps.py / PlotAlgorithmSets.py)
# =========================================================================
FONT_FS = 10
TICK_FS = 10
LEG_FS  = 10

mpl.rcParams.update({
    'font.family':     'serif',
    'font.size':       FONT_FS,
    'axes.labelsize':  FONT_FS,
    'axes.titlesize':  FONT_FS,
    'legend.fontsize': LEG_FS,
    'xtick.labelsize': TICK_FS,
    'ytick.labelsize': TICK_FS,
    'lines.linewidth': 1.5,
    'axes.linewidth':  0.8,
    'figure.dpi':      150,
})

# =========================================================================
# Colours — hero figure palette (PlotAlgorithmSteps.py)
# =========================================================================
C = {
    'Xi': '#AAAAAA',   # grey       — raw samples
    'W':  '#1155AA',   # dark blue  — return points inside E*
    'V':  '#BB2222',   # dark red   — return points outside E*
    'E':  '#000000',   # black      — ellipsoid outline
}
PT  = 18    # scatter point size
ALF = 0.75  # scatter alpha

# =========================================================================
# Hyperparameters
# =========================================================================
N = 500   # samples drawn from E*

# =========================================================================
# Arguments
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('csv', help='ellipsoid_log CSV produced by IterativeAlgorithmMain.py')
args = parser.parse_args()

# =========================================================================
# Load final ellipsoid from last row of CSV
# =========================================================================
df = pd.read_csv(args.csv, index_col='iteration')
d      = sum(1 for col in df.columns if col.startswith('c'))
q_cols = [f'q{i}{j}' for i in range(d) for j in range(d)]
c_cols = [f'c{k}'    for k in range(d)]

Q_star = df[q_cols].values[-1].reshape(d, d)
c_star = df[c_cols].values[-1]

E_star   = Ellipsoid(d)
E_star.Q = Q_star
E_star.c = c_star.reshape(d, 1)

# =========================================================================
# Sample and map
# =========================================================================
from pathlib import Path
sys_name = Path(args.csv).stem.split('_ellipsoid_log_')[0]
sys_mod  = importlib.import_module(sys_name)

np.random.seed(42)
Xi       = sample_from(E_star, N).T          # (N, d)  rows = points
Xi_prime = sys_mod.P_batch(Xi)               # (N, d)
in_E     = E_star.contains_point(Xi_prime.T) # (N,) bool

W = Xi_prime[in_E,  :]   # contained return points — blue
V = Xi_prime[~in_E, :]   # escaped  return points — red

print(f"N={N}  |W|={in_E.sum()}  |V|={(~in_E).sum()}  "
      f"escape fraction={100*(~in_E).mean():.1f}%")

# =========================================================================
# Plot
# =========================================================================
pairs   = [(i, j) for i in range(d) for j in range(i + 1, d)]
n_pairs = len(pairs)

fig, axes = plt.subplots(1, n_pairs, figsize=(3.0 * n_pairs, 3.0))
if n_pairs == 1:
    axes = [axes]

for ax, (di, dj) in zip(axes, pairs):
    ax.set_aspect('auto')

    # E* outline — solid black
    plot_projection(Q_star, c_star, 1, ax, diml=di, dimh=dj,
                    color=C['E'], linestyle='-', linewidth=1.5,
                    alpha=0.9, zorder=4)

    # Raw samples (grey, behind)
    ax.scatter(Xi[:, di], Xi[:, dj],
               s=PT, color=C['Xi'], alpha=ALF, zorder=2)

    # Return points: inside (blue) and outside (red)
    if len(W):
        ax.scatter(W[:, di], W[:, dj],
                   s=PT, color=C['W'], alpha=ALF, zorder=3)
    if len(V):
        ax.scatter(V[:, di], V[:, dj],
                   s=PT, color=C['V'], alpha=ALF, zorder=3)

    ax.set_xlabel(f'$x_{di + 1}$')
    ax.set_ylabel(f'$x_{dj + 1}$', labelpad=2)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune='both'))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune='both'))

# Axis limits from the E* outline (first line drawn on each axis)
margin = 0.3
for ax in axes:
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    xpad = margin * (x.max() - x.min())
    ypad = margin * (y.max() - y.min())
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)

legend_handles = [
    Line2D([0], [0], color=C['E'], linewidth=1.5,
           label='$\\mathcal{E}^*$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C['Xi'],
           markersize=8, label='$\\{\\mathbf{x}_j\\}$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C['W'],
           markersize=8, label='$\\mathcal{W}$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C['V'],
           markersize=8, label='$\\mathcal{V}$'),
]

axes[0].legend(handles=legend_handles, loc='upper right',
               fontsize=LEG_FS, framealpha=0.9,
               handlelength=1.2, handletextpad=0.4)

fig.tight_layout()

stem = args.csv.replace('.csv', '')
fig.savefig(f'{stem}_invariant.pdf', bbox_inches='tight')
fig.savefig(f'{stem}_invariant.png', dpi=300, bbox_inches='tight')
print(f"Saved {stem}_invariant.pdf / .png")

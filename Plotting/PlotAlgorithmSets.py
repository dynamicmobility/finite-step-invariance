"""
PlotAlgorithmSets.py  —  overlay all ellipsoid iterates from a log CSV.

Reads an ellipsoid_log CSV produced by IterativeAlgorithmMain.py and plots
every E_i on a single axes, colour-coded by iteration number.

Usage:
    python PlotAlgorithmSets.py ExpanderContractor_ellipsoid_log_20260328_143201.csv
"""

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'Systems'))
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ellipsoids import plot_projection

# =========================================================================
# Paper style  (matches PlotAlgorithmSteps.py)
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
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# =========================================================================
# Parse arguments
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('csv', help='Path to ellipsoid_log CSV file')
args = parser.parse_args()

# =========================================================================
# Load CSV and infer dimension
# =========================================================================
df = pd.read_csv(args.csv, index_col='iteration')

d      = sum(1 for col in df.columns if col.startswith('c'))
q_cols = [f'q{i}{j}' for i in range(d) for j in range(d)]
c_cols = [f'c{k}'    for k in range(d)]

Qs = df[q_cols].values.reshape(-1, d, d)   # (n_iters, d, d)
cs = df[c_cols].values                      # (n_iters, d)
n_iters = len(Qs)
indices = np.arange(n_iters)  # for colour-coding iterations

max_N_iter = 10 # maximum number of iterates to plot (including E0)
if n_iters > max_N_iter:
    indices = np.linspace(0, n_iters - 1, max_N_iter, dtype=int)
    Qs = Qs[indices]
    cs = cs[indices]
    new_n_iters = len(Qs)
    
# =========================================================================
# Plot
# =========================================================================
pairs   = [(i, j) for i in range(d) for j in range(i + 1, d)]
n_pairs = len(pairs)

cmap = mpl.colormaps['plasma']
norm = mpl.colors.Normalize(vmin=indices[1], vmax=indices[-1])

fig, axes = plt.subplots(1, n_pairs, figsize=(3.0 * n_pairs, 3.0))
if n_pairs == 1:
    axes = [axes]

for ax, (di, dj) in zip(axes, pairs):
    ax.set_aspect('auto')

    # Initial guess E0 — dashed black, drawn first (bottom layer)
    plot_projection(Qs[0], cs[0], 1, ax, diml=di, dimh=dj,
                    color='black', linestyle='--', linewidth=1.0,
                    alpha=0.9, label='$\\mathcal{E}_0$', zorder=2)

    # Subsequent iterates — plasma colormap
    for i, (Q, c) in zip(indices[1:], zip(Qs[1:], cs[1:])):
        plot_projection(Q, c, 1, ax, diml=di, dimh=dj,
                        color=cmap(norm(i)), linewidth=0.8, alpha=0.85, zorder=3)

    ax.set_xlabel(f'$x_{di + 1}$')
    ax.set_ylabel(f'$x_{dj + 1}$', labelpad=2)

# axes[0].legend(fontsize=LEG_FS, loc='upper right', framealpha=0.9)

# Scale each subplot to fit E0 (the largest ellipse, drawn first)
margin = 0.1
for ax in axes:
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    xpad = margin * (x.max() - x.min())
    ypad = margin * (y.max() - y.min())
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)

sm   = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)  # noqa: deprecated in 3.8+
# cbar = fig.colorbar(sm, ax=axes[-1], shrink=0.85, pad=0.03)
# cbar.set_label('Iteration', fontsize=FONT_FS)
# cbar.ax.tick_params(labelsize=TICK_FS)
# cbar.set_ticks(indices[1:])

fig.tight_layout()

# =========================================================================
# Save bare figures only: no ticks, labels, axes, legend, or colorbar
# =========================================================================
FIGURES_DIR = os.path.join(_ROOT, 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
stem = os.path.basename(args.csv).replace('.csv', '')

for ax in axes:
    ax.set_axis_off()

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05)
fig.canvas.draw()

if n_pairs == 1:
    fig.savefig(os.path.join(FIGURES_DIR, f'{stem}_sets_bare.pdf'),
                bbox_inches='tight', pad_inches=0)
    print(f"Saved Figures/{stem}_sets_bare.pdf")
else:
    for ax, (di, dj) in zip(axes, pairs):
        # crop to exactly this subplot's area
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        name = f'{stem}_sets_bare_x{di + 1}x{dj + 1}.pdf'
        fig.savefig(os.path.join(FIGURES_DIR, name), bbox_inches=bbox, pad_inches=0)
        print(f"Saved Figures/{name}")

print(f"({n_iters} ellipsoids, d={d}, {n_pairs} projection(s))")
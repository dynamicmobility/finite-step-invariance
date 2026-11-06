"""
PlotDogBoneComparison.py — DogBone true ROA vs ellipsoid E* vs RBF invariant set.

Styled to match PlotAlgorithmSets.py / PlotAlgorithmSteps.py:
  - 3.0 x 3.0 in square panel
  - axis window fitted to the true set with a 10% margin (same rule as E0 in PlotAlgorithmSets)
  - black solid outline for the ellipsoid, plasma colour for the other curve
  - no legend (describe the curves in the paper caption)
  - also writes a "bare" version (axes off) like PlotAlgorithmSets.py

Usage:
    python PlotDogBoneComparison.py <ellipsoid_log_csv> <rbf_log_csv>
"""

import sys
import os

import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker

from sets.ellipsoid import Ellipsoid, plot_projection
from sets.radialBasisFunction import RBF
from Systems import DogBone

# =========================================================================
# Paper style  (identical to PlotAlgorithmSets.py)
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
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

# =========================================================================
# Colours
# =========================================================================
_plasma   = mpl.colormaps['plasma']
C_ELLIPSE = _plasma(0.1)           # dark purple, solid
C_RBF     = '#FF8C00'              # bright orange, dashed
C_ROA     = '#FBDDE6'              # very light pink — true ROA fill (drawn opaque)

# =========================================================================
# Arguments
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('ellipsoid_csv', help='DogBone ellipsoid_log CSV')
parser.add_argument('rbf_csv',       help='DogBone rbf_log CSV')
args = parser.parse_args()

# =========================================================================
# Load final ellipsoid (last row of CSV)
# =========================================================================
df_ell = pd.read_csv(args.ellipsoid_csv, index_col='iteration')
d      = sum(1 for col in df_ell.columns if col.startswith('c'))
q_cols = [f'q{i}{j}' for i in range(d) for j in range(d)]
c_cols = [f'c{k}'    for k in range(d)]

Q_star   = df_ell[q_cols].values[-1].reshape(d, d)
c_star   = df_ell[c_cols].values[-1]
E_star   = Ellipsoid(d)
E_star.Q = Q_star
E_star.c = c_star.reshape(d, 1)
print(f"Loaded ellipsoid E* from {args.ellipsoid_csv}")

# =========================================================================
# Load final RBF
# =========================================================================
df_rbf = pd.read_csv(args.rbf_csv, index_col='ball')
centers = df_rbf[[f'center_{k}' for k in range(d)]].values   # (K, d)
widths  = df_rbf['width'].values                              # (K,)
thresh  = float(df_rbf['thresh'].iloc[0])

rbf = RBF(d, thresh)
rbf.num_nballs   = len(centers)
rbf.centers_opt  = centers
rbf.widths_opt   = widths
print(f"Loaded RBF ({len(centers)} balls) from {args.rbf_csv}")

# =========================================================================
# True ROA on a grid: union of two discs
# =========================================================================
GRID_N = 500
xgrid = np.linspace(-1.5, 1.5, GRID_N)
ygrid = np.linspace(-1.5, 1.5, GRID_N)
X, Y  = np.meshgrid(xgrid, ygrid)
pts_flat   = np.stack([X.ravel(), Y.ravel()], axis=1)   # (GRID_N^2, 2)
in_dogbone = DogBone._in_dogbone(pts_flat).reshape(X.shape)

# =========================================================================
# RBF level-set on the same grid (vectorised)
# =========================================================================
diffs    = pts_flat[:, None, :] - centers[None, :, :]          # (N², K, 2)
sq_dists = np.sum(diffs ** 2, axis=2)                          # (N², K)
rbf_vals = np.exp(-0.5 * sq_dists / widths[None, :] ** 2)     # (N², K)
Z_rbf    = np.log(np.maximum(rbf_vals.sum(axis=1), 1e-15)).reshape(X.shape)

# =========================================================================
# Axis window: fit the true set with a 10% margin (square, like the others)
# =========================================================================
margin = 0.1
xs, ys = X[in_dogbone], Y[in_dogbone]
xmid, ymid = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
half = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin)

# =========================================================================
# Plot
# =========================================================================
fig, ax = plt.subplots(figsize=(3.0, 3.0))

# True ROA — filled
ax.contourf(X, Y, in_dogbone.astype(float),
            levels=[0.5, 1.5], colors=[C_ROA], alpha=0.3, zorder=1)

# RBF boundary
ax.contour(X, Y, Z_rbf, levels=[thresh],
           colors=[C_RBF], linewidths=1.0, zorder=3)

# Ellipsoid boundary
plot_projection(Q_star, c_star, 1, ax, diml=0, dimh=1,
                color=C_ELLIPSE, linestyle='-', linewidth=1.0,
                alpha=0.9, zorder=4)

ax.set_xlim(xmid - half, xmid + half)
ax.set_ylim(ymid - half, ymid + half)
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$', labelpad=2)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))

fig.tight_layout()

FIGURES_DIR = os.path.join(_ROOT, 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

fig.savefig(os.path.join(FIGURES_DIR, 'DogBone_comparison.pdf'), bbox_inches='tight')
print("Saved Figures/DogBone_comparison.pdf")

# =========================================================================
# Bare version (no ticks / labels / axes), as in PlotAlgorithmSets.py
# =========================================================================
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.savefig(os.path.join(FIGURES_DIR, 'DogBone_comparison_bare.pdf'),
            bbox_inches='tight', pad_inches=0)
print("Saved Figures/DogBone_comparison_bare.pdf")
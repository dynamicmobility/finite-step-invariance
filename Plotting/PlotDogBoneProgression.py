"""
PlotDogBoneProgression.py — DogBone: progression of the ellipsoid sets E_k and of
the RBF sets at every iteration, styled like the example panel
(plasma colour = iteration, dashed black outline = initial set, colorbar on the right).

Writes:
    Figures/DogBone_ellipsoid_progression_bare.pdf
    Figures/DogBone_rbf_progression_bare.pdf   (bare: no axes, labels or colorbar)

Usage:
    python PlotDogBoneProgression.py <ellipsoid_log_csv> <rbf_log_csv> [--roa]

CSV formats
  Current logs: index column 'iteration', columns param_0, param_1, ...
  RBF rows contain flattened centers, widths, and a trailing threshold.
  Legacy RBF logs with center_{k}, width, and thresh columns are also supported.
"""

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'Systems'))

import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker

from sets.ellipsoid import Ellipsoid
from sets.radialBasisFunction import RBF
from Systems import DogBone

# =========================================================================
# Paper style  (identical to PlotDogBoneComparison.py)
# =========================================================================
FONT_FS = 10
mpl.rcParams.update({
    'font.family':     'serif',
    'font.size':       FONT_FS,
    'axes.labelsize':  FONT_FS,
    'axes.titlesize':  FONT_FS,
    'xtick.labelsize': FONT_FS,
    'ytick.labelsize': FONT_FS,
    'lines.linewidth': 1.5,
    'axes.linewidth':  0.8,
    'figure.dpi':      150,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

CMAP = mpl.colormaps['plasma']

# =========================================================================
# Arguments
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('ellipsoid_csv', help='DogBone ellipsoid_log CSV')
parser.add_argument('rbf_csv',       help='DogBone rbf_log CSV')
parser.add_argument('--roa', action='store_true',
                    help='also shade the true ROA behind the curves')
args = parser.parse_args()

FIGURES_DIR = os.path.join(_ROOT, 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================================================================
# Load ellipsoid sequence
# =========================================================================
df_ell = pd.read_csv(args.ellipsoid_csv, index_col='iteration')
d = DogBone.d
ell_iters = list(df_ell.index)
print(f"Loaded {len(ell_iters)} ellipsoids from {args.ellipsoid_csv}")

# =========================================================================
# Load RBF sequence, keeping a separate RBF object for every iteration
# =========================================================================
df_rbf = pd.read_csv(args.rbf_csv)
rbf_seq = []
rbf_iters = []
param_cols = sorted((col for col in df_rbf.columns if col.startswith('param_')),
                    key=lambda col: int(col.split('_')[1]))
if param_cols:
    if 'iteration' in df_rbf.columns:
        df_rbf = df_rbf.sort_values('iteration')
        rbf_iters = df_rbf['iteration'].tolist()
    else:
        rbf_iters = list(range(len(df_rbf)))
    for row in df_rbf[param_cols].to_numpy(dtype=float):
        rbf = RBF(d)
        rbf.load_from_row(row)
        rbf_seq.append(rbf)
else:
    # Convert legacy one-row-per-ball logs to the same params() layout.
    cen_cols = [f'center_{k}' for k in range(d)]
    groups = (df_rbf.groupby('iteration', sort=True) if 'iteration' in df_rbf.columns
              else [(1, df_rbf)])
    for it, group in groups:
        row = np.concatenate([group[cen_cols].to_numpy().ravel(),
                              group['width'].to_numpy(), [group['thresh'].iloc[0]]])
        rbf = RBF(d)
        rbf.load_from_row(row)
        rbf_seq.append(rbf)
        rbf_iters.append(it)
print(f"Loaded {len(rbf_seq)} RBF iterates from {args.rbf_csv}")

# =========================================================================
# =========================================================================
# True ROA grid (used for the optional shading and the fallback window)
# =========================================================================
GRID_N = 400
xgrid = np.linspace(-1.5, 1.5, GRID_N)
X, Y  = np.meshgrid(xgrid, xgrid)
pts_flat   = np.stack([X.ravel(), Y.ravel()], axis=1)
in_dogbone = DogBone._in_dogbone(pts_flat).reshape(X.shape)


def rbf_bbox(centers, widths, thresh):
    """Guaranteed bounding box of {log sum_k exp(-r_k^2 / 2 w_k^2) >= thresh}.
    A point in the set needs some ball with exp(-r^2/2w^2) >= e^thresh / K,
    i.e. r <= |w| * sqrt(2 * ln(K / e^thresh))."""
    K = len(widths)
    r = np.abs(widths) * np.sqrt(2.0 * np.log(K) - 2.0 * thresh)
    lo = (centers - r[:, None]).min(axis=0)
    hi = (centers + r[:, None]).max(axis=0)
    return lo, hi


def get_ellipsoid(row):
    return Ellipsoid().load_from_row(row.to_numpy(dtype=float))


def square_window(ax, margin=0.1):
    """Square window around whatever is currently plotted, with a margin."""
    ax.relim()
    ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    ax.set_xlim(xm - half, xm + half)
    ax.set_ylim(ym - half, ym + half)


def finish(fig, ax, norm, fname):
    """Bare version: no axes, ticks, labels or colorbar (as in PlotAlgorithmSets.py)."""
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(os.path.join(FIGURES_DIR, fname),
                bbox_inches='tight', pad_inches=0)
    print(f"Saved Figures/{fname}")
    plt.close(fig)


def shade_roa(ax):
    ax.contourf(X, Y, in_dogbone.astype(float), levels=[0.5, 1.5],
                colors=[CMAP(0.85)], alpha=0.25, zorder=1)


# =========================================================================
# Figure 1: ellipsoid progression
# =========================================================================
fig, ax = plt.subplots(figsize=(3.0, 3.0))

# iteration 0 (if logged) is the initial set: dashed black, as in the example
colored = [it for it in ell_iters if it != 0]
if 0 in ell_iters:
    ell = get_ellipsoid(df_ell.loc[0])
    ell.plot(ax, color='k', linestyle = '--', linewidth = 0.8, zorder = 2)
    square_window(ax)                      # window fitted to E0 + 10 % margin
    xl, yl = ax.get_xlim(), ax.get_ylim()
else:
    xl = yl = None

norm = mpl.colors.Normalize(vmin=min(colored or [0]), vmax=max(colored or [1]))
if args.roa:
    shade_roa(ax)
for it in colored:
    ell = get_ellipsoid(df_ell.loc[it])
    ell.plot(ax, color=CMAP(norm(it)), linestyle = '-', linewidth = 0.8, alpha=0.9, zorder = 3+it*1e-3)

if xl is not None:
    ax.set_xlim(xl); ax.set_ylim(yl)
else:
    square_window(ax)
finish(fig, ax, norm, 'DogBone_ellipsoid_progression_bare.pdf')

# =========================================================================
# Figure 2: RBF progression
# =========================================================================
fig, ax = plt.subplots(figsize=(3.0, 3.0))
its = [it for it in rbf_iters if it != 0] or [1]
norm = mpl.colors.Normalize(vmin=min(its), vmax=max(its) if max(its) > min(its) else min(its) + 1)

# Fit the window to the initial RBF if logged, otherwise to all iterates.
ref = [rbf for it, rbf in zip(rbf_iters, rbf_seq) if it == 0] or rbf_seq
boxes = [rbf_bbox(rbf.centers_opt, rbf.widths_opt, rbf.thresh) for rbf in ref]
lo = np.min([b[0] for b in boxes], axis=0)
hi = np.max([b[1] for b in boxes], axis=0)
mid  = 0.5 * (lo + hi)
half = 0.5 * np.max(hi - lo) * 1.2
RX0, RX1 = mid[0] - half, mid[0] + half
RY0, RY1 = mid[1] - half, mid[1] + half

if args.roa:
    shade_roa(ax)
for it, rbf in zip(rbf_iters, rbf_seq):
    if it == 0:
        rbf.plot(ax, color='k', linestyle='--', linewidth=0.8, zorder=2)
    else:
        rbf.plot(ax, color=CMAP(norm(it)), linestyle='-', linewidth=0.8, zorder=3)

ax.set_xlim(RX0, RX1)
ax.set_ylim(RY0, RY1)
finish(fig, ax, norm, 'DogBone_rbf_progression_bare.pdf')
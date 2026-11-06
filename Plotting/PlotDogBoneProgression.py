"""
PlotDogBoneProgression.py — DogBone: progression of the ellipsoid sets E_k and of
the RBF sets at every iteration, styled like the example panel
(plasma colour = iteration, dashed black outline = initial set, colorbar on the right).

Writes:
    Figures/DogBone_ellipsoid_progression_bare.pdf
    Figures/DogBone_rbf_progression_bare.pdf   (bare: no axes, labels or colorbar)

Usage:
    python PlotDogBoneProgression.py <ellipsoid_log_csv> <rbf_log_csv> [--roa]

Assumed CSV formats
  ellipsoid_log : index column 'iteration', columns q{i}{j}, c{k}   (as in PlotDogBoneComparison.py)
  rbf_log       : columns 'iteration', 'ball', center_{k}, 'width', 'thresh'
                  (one block of rows per iteration; if there is no 'iteration'
                  column, only the final RBF can be drawn)
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

from ellipsoids import Ellipsoid, plot_projection
import DogBone

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
d      = sum(1 for col in df_ell.columns if col.startswith('c'))
q_cols = [f'q{i}{j}' for i in range(d) for j in range(d)]
c_cols = [f'c{k}'    for k in range(d)]
ell_iters = list(df_ell.index)
print(f"Loaded {len(ell_iters)} ellipsoids from {args.ellipsoid_csv}")

# =========================================================================
# Load RBF sequence -> list of (iteration, centers, widths, thresh)
# =========================================================================
df_rbf = pd.read_csv(args.rbf_csv)
cen_cols = [f'center_{k}' for k in range(d)]
rbf_seq = []
if 'iteration' in df_rbf.columns:
    for it, g in df_rbf.groupby('iteration', sort=True):
        rbf_seq.append((it, g[cen_cols].values, g['width'].values,
                        float(g['thresh'].iloc[0])))
else:
    print("WARNING: rbf CSV has no 'iteration' column; drawing the final RBF only.")
    rbf_seq.append((1, df_rbf[cen_cols].values, df_rbf['width'].values,
                    float(df_rbf['thresh'].iloc[0])))
print(f"Loaded {len(rbf_seq)} RBF iterates from {args.rbf_csv}")

# =========================================================================
# True ROA grid (used for the optional shading and the fallback window)
# =========================================================================
GRID_N = 400
xgrid = np.linspace(-1.5, 1.5, GRID_N)
X, Y  = np.meshgrid(xgrid, xgrid)
pts_flat   = np.stack([X.ravel(), Y.ravel()], axis=1)
in_dogbone = DogBone._in_dogbone(pts_flat).reshape(X.shape)


def rbf_field(centers, widths, pts, shape):
    diffs    = pts[:, None, :] - centers[None, :, :]
    sq       = np.sum(diffs ** 2, axis=2)
    vals     = np.exp(-0.5 * sq / widths[None, :] ** 2)
    return np.log(np.maximum(vals.sum(axis=1), 1e-15)).reshape(shape)


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
    Q = row[q_cols].values.astype(float).reshape(d, d)
    c = row[c_cols].values.astype(float)
    return Q, c


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
    Q0, c0 = get_ellipsoid(df_ell.loc[0])
    plot_projection(Q0, c0, 1, ax, diml=0, dimh=1, color='k',
                    linestyle='--', linewidth=0.8, alpha=1.0, zorder=2)
    square_window(ax)                      # window fitted to E0 + 10 % margin
    xl, yl = ax.get_xlim(), ax.get_ylim()
else:
    xl = yl = None

norm = mpl.colors.Normalize(vmin=min(colored), vmax=max(colored))
if args.roa:
    shade_roa(ax)
for it in colored:
    Q, c = get_ellipsoid(df_ell.loc[it])
    plot_projection(Q, c, 1, ax, diml=0, dimh=1, color=CMAP(norm(it)),
                    linestyle='-', linewidth=0.8, alpha=0.9, zorder=3 + it * 1e-3)

if xl is not None:
    ax.set_xlim(xl); ax.set_ylim(yl)
else:
    square_window(ax)
finish(fig, ax, norm, 'DogBone_ellipsoid_progression_bare.pdf')

# =========================================================================
# Figure 2: RBF progression
# =========================================================================
fig, ax = plt.subplots(figsize=(3.0, 3.0))
its  = [r[0] for r in rbf_seq if r[0] != 0] or [1]
norm = mpl.colors.Normalize(vmin=min(its), vmax=max(its) if max(its) > min(its) else min(its) + 1)

# Window: fitted to the initial RBF (iteration 0) if logged, else to all iterates,
# with a 10 % margin, square. The evaluation grid covers exactly this window,
# so no level set is clipped by the grid (only by the window itself).
ref = [r for r in rbf_seq if r[0] == 0] or rbf_seq
boxes = [rbf_bbox(c, w, t) for (_, c, w, t) in ref]
lo = np.min([b[0] for b in boxes], axis=0)
hi = np.max([b[1] for b in boxes], axis=0)
mid  = 0.5 * (lo + hi)
half = 0.5 * np.max(hi - lo) * 1.2
RX0, RX1 = mid[0] - half, mid[0] + half
RY0, RY1 = mid[1] - half, mid[1] + half

RN = 600
gx = np.linspace(RX0, RX1, RN)
gy = np.linspace(RY0, RY1, RN)
RXg, RYg = np.meshgrid(gx, gy)
rpts = np.stack([RXg.ravel(), RYg.ravel()], axis=1)

if args.roa:
    shade_roa(ax)
for it, centers, widths, thresh in rbf_seq:
    Z = rbf_field(centers, widths, rpts, RXg.shape)
    # linestyles set explicitly: a negative level (thresh = log 0.25 < 0) is
    # otherwise drawn dashed by matplotlib's contour defaults.
    if it == 0:      # initial RBF: dashed black, like the ellipsoid figure
        ax.contour(RXg, RYg, Z, levels=[thresh], colors=['k'],
                   linewidths=0.8, linestyles='--', zorder=2)
    else:
        ax.contour(RXg, RYg, Z, levels=[thresh], colors=[CMAP(norm(it))],
                   linewidths=0.8, linestyles='-', zorder=3)

ax.set_xlim(RX0, RX1)
ax.set_ylim(RY0, RY1)
finish(fig, ax, norm, 'DogBone_rbf_progression_bare.pdf')
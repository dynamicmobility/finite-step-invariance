"""
PlotExpanderContractorProgression.py — Expander/contractor toy system: progression of
the RBF sets at every iteration, styled like the DogBone panels
(plasma colour = iteration, dashed black outline = initial set).

Writes:
    Figures/ExpanderContractor_rbf_progression_bare.pdf   (bare: no axes, labels or colorbar)

Usage:
    python PlotExpanderContractorProgression.py <rbf_log_csv> [--roa] [--ellipsoid_csv <ellipsoid_log_csv>]

If --ellipsoid_csv is given, the last row (final iteration) is used as the final
ellipsoid and the volume difference V(RBF) - V(ellipsoid) is printed.

CSV formats
  Current logs: columns param_0, param_1, ... (plus optional 'iteration').
  RBF rows contain flattened centers, widths, and a trailing threshold.
  Legacy RBF logs with center_{k}, width, and thresh columns are also supported.

The system module is assumed to live in Systems/ExpanderContractor.py and to expose
d, Q, v (as in the snippet). Change SYSTEM_MODULE below if it is named differently.
"""

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'Systems'))

import argparse
import importlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sets.radialBasisFunction import RBF

SYSTEM_MODULE = 'ExpanderContractor'
SYS = importlib.import_module(f'Systems.{SYSTEM_MODULE}')
NAME = 'ExpanderContractor'

# =========================================================================
# Paper style  (identical to PlotDogBoneProgression.py)
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
parser.add_argument('rbf_csv', help=f'{NAME} rbf_log CSV')
parser.add_argument('--roa', action='store_true',
                    help='also shade the true ROA (the Q-norm ball) behind the curves')
parser.add_argument('--ellipsoid_csv', default=None,
                    help='optional ellipsoid_log CSV; its final iteration is compared '
                         'to the final RBF by volume')
args = parser.parse_args()

FIGURES_DIR = os.path.join(_ROOT, 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

d = SYS.d

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
# True ROA grid: {x : (x - v)^T Q (x - v) <= 1}, centred on v.
# =========================================================================
Q = np.asarray(SYS.Q, dtype=float)
v = np.asarray(SYS.c, dtype=float).flatten()
GRID_N = 400
R = 1.5 / np.sqrt(np.linalg.eigvalsh(Q).min())
xgrid = np.linspace(v[0] - R, v[0] + R, GRID_N)
ygrid = np.linspace(v[1] - R, v[1] + R, GRID_N)
X, Y  = np.meshgrid(xgrid, ygrid)
pts_flat = np.stack([X.ravel(), Y.ravel()], axis=1) - v
in_roa = (np.einsum('ni,ij,nj->n', pts_flat, Q, pts_flat) <= 1.0).reshape(X.shape)


def rbf_bbox(centers, widths, thresh):
    """Guaranteed bounding box of {log sum_k exp(-r_k^2 / 2 w_k^2) >= thresh}.
    A point in the set needs some ball with exp(-r^2/2w^2) >= e^thresh / K,
    i.e. r <= |w| * sqrt(2 * ln(K / e^thresh))."""
    K = len(widths)
    r = np.abs(widths) * np.sqrt(2.0 * np.log(K) - 2.0 * thresh)
    lo = (centers - r[:, None]).min(axis=0)
    hi = (centers + r[:, None]).max(axis=0)
    return lo, hi


def finish(fig, ax, fname):
    """Bare version: no axes, ticks, labels or colorbar."""
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(os.path.join(FIGURES_DIR, fname),
                bbox_inches='tight', pad_inches=0)
    print(f"Saved Figures/{fname}")
    plt.close(fig)


def shade_roa(ax):
    ax.contourf(X, Y, in_roa.astype(float), levels=[0.5, 1.5],
                colors=[CMAP(0.85)], alpha=0.25, zorder=1)


# =========================================================================
# RBF progression
# =========================================================================
fig, ax = plt.subplots(figsize=(3.0, 3.0))
its = [it for it in rbf_iters if it != 0] or [1]
norm = mpl.colors.Normalize(vmin=min(its), vmax=max(its) if max(its) > min(its) else min(its) + 1)

# Fit the window to all RBF iterates.
boxes = [rbf_bbox(rbf.centers_opt, rbf.widths_opt, rbf.thresh) for rbf in rbf_seq]
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
print(f"Axis limits: x in [{RX0:.4f}, {RX1:.4f}], y in [{RY0:.4f}, {RY1:.4f}]")
print(f"  pgfplots: xmin={RX0:.4f}, xmax={RX1:.4f}, ymin={RY0:.4f}, ymax={RY1:.4f}")
finish(fig, ax, f'{NAME}_rbf_progression_bare.pdf')

# =========================================================================
# Volume comparison: final RBF set vs. optional final ellipsoid
# =========================================================================
from math import gamma, pi

# How the ellipsoid row's d x d block is parametrised.  CHECK THIS against sets/ellipsoid.py:
#   'Q': {x : (x-c)^T Q (x-c) <= 1}      volume = V_d / sqrt(det Q)
#   'P': {x : (x-c)^T P^-1 (x-c) <= 1}   volume = V_d * sqrt(det P)
#   'L': {x : x = c + L u, |u| <= 1}     volume = V_d * |det L|
ELL_FORM = 'Q'
UNIT_BALL_VOL = pi ** (d / 2) / gamma(d / 2 + 1)


def rbf_volume(rbf, n_samples=2_000_000, batch=100_000, seed=0):
    """Monte Carlo volume of {x : log sum_k exp(-|x-c_k|^2 / 2 w_k^2) >= thresh},
    sampled uniformly in its guaranteed bounding box. Returns (volume, std. error)."""
    c = np.asarray(rbf.centers_opt, dtype=float)
    w = np.asarray(rbf.widths_opt, dtype=float)
    lo, hi = rbf_bbox(c, w, rbf.thresh)
    box_vol = float(np.prod(hi - lo))
    rng = np.random.default_rng(seed)
    n_batches = max(n_samples // batch, 1)
    inside = 0
    for _ in range(n_batches):
        x = rng.uniform(lo, hi, size=(batch, d))
        z = -((x[:, None, :] - c[None, :, :]) ** 2).sum(-1) / (2.0 * w[None, :] ** 2)
        m = z.max(axis=1)
        lse = m + np.log(np.exp(z - m[:, None]).sum(axis=1))
        inside += int(np.count_nonzero(lse >= rbf.thresh))
    n = n_batches * batch
    p = inside / n
    return box_vol * p, box_vol * np.sqrt(p * (1.0 - p) / n)


def ellipsoid_volume(row):
    M = np.asarray(row[:d * d], dtype=float).reshape(d, d)
    if ELL_FORM == 'Q':
        return UNIT_BALL_VOL / np.sqrt(np.linalg.det(0.5 * (M + M.T)))
    if ELL_FORM == 'P':
        return UNIT_BALL_VOL * np.sqrt(np.linalg.det(0.5 * (M + M.T)))
    return UNIT_BALL_VOL * abs(np.linalg.det(M))


if args.ellipsoid_csv is not None:
    df_ell = pd.read_csv(args.ellipsoid_csv, index_col='iteration').sort_index()
    final_it = df_ell.index[-1]
    V_ell = ellipsoid_volume(df_ell.iloc[-1].to_numpy(dtype=float))
    V_rbf, V_rbf_se = rbf_volume(rbf_seq[-1])
    print(f"\nVolumes (final RBF = iteration {rbf_iters[-1]}, "
          f"final ellipsoid = iteration {final_it}):")
    print(f"  RBF        : {V_rbf:.6f}  (MC std. error {V_rbf_se:.1e})")
    print(f"  Ellipsoid  : {V_ell:.6f}")
    print(f"  Difference (RBF - ellipsoid): {V_rbf - V_ell:+.6f}  "
          f"({100 * (V_rbf - V_ell) / V_ell:+.1f}% of ellipsoid volume)")
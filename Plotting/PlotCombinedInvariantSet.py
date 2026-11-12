"""
PlotInvariantCombined.py  —  ONE compact figure showing both the ellipsoid
                              iterate sequence AND all four verification
                              sample sets (U, Z, W, V).

Per coordinate projection (x_i, x_j) a single panel shows:
  - E_0 (dashed black) if present in the log
  - Ellipsoid iterates E_1 ... E_k coloured by iteration (plasma)
  - The final set E* (solid black)
  - Verification samples from E*, all four sets:
        U  preimages whose return lands inside  E*   (filled circle, dark purple)
        Z  preimages whose return lands outside E*   (filled circle, orange-red)
        W  return points inside  E*                  (cross, dark purple)
        V  return points outside E*                  (cross, bright orange)

Layout:
  - one shared horizontal colour bar on top
  - panels share y-axis labels/ticks when projections share the same y variable
  - tick labels are rescaled when the range is tiny/huge

Usage:
    python PlotInvariantCombined.py <ellipsoid_log_csv> [--n 100] [--stride 1] [--width 6.0]

    e.g.  python PlotInvariantCombined.py CompassGaitWalker_ellipsoid_log_20260328_211440.csv

Writes (in ./Figures/):
    <stem>_invariant_combined.pdf
    invariant_combined_legend.pdf
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'Systems'))
import importlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from sets.ellipsoid import Ellipsoid, plot_projection

# =========================================================================
# Paper style
# =========================================================================
FONT_FS = 8
TICK_FS = 7
mpl.rcParams.update({
    'font.family':        'serif',
    'font.size':          FONT_FS,
    'axes.labelsize':     FONT_FS,
    'axes.titlesize':     FONT_FS,
    'legend.fontsize':    FONT_FS,
    'xtick.labelsize':    TICK_FS,
    'ytick.labelsize':    TICK_FS,
    'lines.linewidth':    1.0,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
    'xtick.major.size':   2.5,
    'ytick.major.size':   2.5,
    'figure.dpi':         150,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# =========================================================================
# Colour palette — combines the plasma hero palette with iterate colouring
# =========================================================================
_plasma = mpl.colormaps['plasma']
CMAP    = _plasma                          # iterate colour map

# Point-set colours (from PlotInvariantSet.py hero palette)
C = {
    'U':  _plasma(0.4),    # magenta-pink  — preimages of W  (return inside E*)
    'Z':  _plasma(0.65),   # orange-red    — preimages of V  (return outside E*)
    'W':  _plasma(0.1),    # dark purple   — contained return points
    'V':  _plasma(0.85),   # bright orange — escaped return points
    'E':  '#000000',       # black         — final ellipsoid E*
    'E0': '#000000',       # black dashed  — initial ellipsoid E_0
}

# Drawing parameters
LW_IT    = 0.5       # iterate outline width
LW_E     = 0.5       # E* outline width
LW_E0    = 0.5       # E_0 outline width
PT       = 4.        # scatter point size
ALF      = 0.75      # scatter alpha
MARGIN   = 0.12      # window padding fraction beyond E_0 outline

# =========================================================================
# Arguments
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('csv',       help='ellipsoid_log CSV produced by the iterative algorithm')
parser.add_argument('--n',       type=int,   default=100,  help='verification samples drawn from E*')
parser.add_argument('--stride',  type=int,   default=1,    help='draw every k-th iterate')
parser.add_argument('--width',   type=float, default=6.0,  help='figure width in inches')
args = parser.parse_args()

# =========================================================================
# Load ellipsoid sequence  (columns: iteration, param_0 .. param_{k-1})
# =========================================================================
df = pd.read_csv(args.csv, index_col='iteration')
p_cols = [c for c in df.columns if c.startswith('param_')]
n_bad = len(df) - len(df.dropna(subset=p_cols))
if n_bad:
    print(f"Dropping {n_bad} incomplete row(s) from the log")
df = df.dropna(subset=p_cols)

d      = Ellipsoid.get_d_from_row_length(len(p_cols))
iters  = list(df.index)


def get_qc(it):
    row = df.loc[it, p_cols].values.astype(float)
    Q = row[:d * d].reshape(d, d)
    c = row[d * d:]
    return Q, c


Q_star, c_star = get_qc(iters[-1])
Q_star = (Q_star + Q_star.T) / 2
E_star = Ellipsoid(d).load_from_row(np.concatenate([Q_star.ravel(), c_star]))
E_star.Q = Q_star
E_star.c = c_star.reshape(d, 1)
print(f"Volume of E*: {E_star.volume():.6g}")

# Iterates to draw
has_E0  = 0 in iters
colored = [it for it in iters if it != 0]
drawn   = colored[::max(1, args.stride)]
if colored and drawn[-1] != colored[-1]:
    drawn.append(colored[-1])
norm = mpl.colors.Normalize(vmin=min(colored), vmax=max(colored))
print(f"{len(iters)} ellipsoids in log; drawing {len(drawn)} coloured iterates"
      f"{' + dashed E_0' if has_E0 else ''}")
print(f"colour bar: iteration {norm.vmin:g} .. {norm.vmax:g}")

# =========================================================================
# Verification samples from E*
# =========================================================================
sys_name = Path(args.csv).stem.split('_ellipsoid_log_')[0]
sys_mod  = importlib.import_module(sys_name)

rng      = np.random.default_rng(42)
Xi       = np.real(E_star.sample_from(args.n, rng))   # (N, d)
Xi_prime = sys_mod.P_batch(Xi)                        # (N, d)
in_E     = E_star.contains_points(Xi_prime)           # (N,) bool

U = Xi[in_E,  :]
Z = Xi[~in_E, :]
W = Xi_prime[in_E,  :]
V = Xi_prime[~in_E, :]
# =========================================================================
# Figure geometry
# =========================================================================
pairs   = [(i, j) for i in range(d) for j in range(i + 1, d)]
n_pairs = len(pairs)

W_fig    = args.width
ML, MR   = 0.58, 0.07
G_FULL   = 0.60
G_SHARE  = 0.16
MB       = 0.46
PH       = 1.30
CB_GAP, CB_H, CB_TOP = 0.10, 0.10, 0.58

shares = [pairs[k][1] == pairs[k - 1][1] for k in range(1, n_pairs)]
gaps   = [G_SHARE if s else G_FULL for s in shares]
pw     = (W_fig - ML - MR - sum(gaps)) / n_pairs
H_fig  = MB + PH + CB_TOP


def rect(x, y, w, h):
    return [x / W_fig, y / H_fig, w / W_fig, h / H_fig]


fig  = plt.figure(figsize=(W_fig, H_fig))
axes = []
x    = ML
for k in range(n_pairs):
    axes.append(fig.add_axes(rect(x, MB, pw, PH)))
    x += pw + (gaps[k] if k < n_pairs - 1 else 0)

# =========================================================================
# Draw on each projection panel
# =========================================================================
for ax, (di, dj) in zip(axes, pairs):
    ax.set_aspect('auto')

    # --- E_0 (dashed, drawn first so its outline defines the window) ---
    if has_E0:
        Q0, c0 = get_qc(0)
        plot_projection(Q0, c0, 1, ax, diml=di, dimh=dj, color=C['E0'],
                        linestyle='--', linewidth=LW_E0, alpha=1.0, zorder=2)

    # --- Coloured iterates ---
    for it in drawn:
        Q, c_it = get_qc(it)
        plot_projection(Q, c_it, 1, ax, diml=di, dimh=dj,
                        color=CMAP(norm(it)),
                        linestyle='-', linewidth=LW_IT, alpha=0.95,
                        zorder=3 + 1e-3 * it)

    # --- Final set E* (solid black, on top of iterates) ---
    plot_projection(Q_star, c_star, 1, ax, diml=di, dimh=dj, color=C['E'],
                    linestyle='-', linewidth=LW_E, alpha=1.0, zorder=20)

    # --- Preimage samples coloured by fate ---
    # U: preimages whose return lands inside E*  (filled circle)
    if len(U):
        ax.scatter(U[:, di], U[:, dj],
                   s=PT, color=C['U'], alpha=ALF, zorder=25,
                   edgecolors='none', label='$\\mathcal{U}$')
    # Z: preimages whose return lands outside E*  (filled circle)
    if len(Z):
        ax.scatter(Z[:, di], Z[:, dj],
                   s=PT, color=C['Z'], alpha=ALF, zorder=26,
                   edgecolors='none', label='$\\mathcal{Z}$')

    # --- Return points coloured by fate ---
    # W: return points inside E*  (cross marker)
    if len(W):
        ax.scatter(W[:, di], W[:, dj],
                   s=PT, color=C['W'], alpha=ALF,
                   edgecolors='none', zorder=27, label='$\\mathcal{W}$')
    # V: return points outside E*  (cross marker)
    if len(V):
        ax.scatter(V[:, di], V[:, dj],
                   s=PT, color=C['V'], alpha=ALF,
                   edgecolors='none', zorder=28, label='$\\mathcal{V}$')

# =========================================================================
# Window: first line drawn (E_0 or first iterate) + margin
# =========================================================================
for ax in axes:
    xl, yl = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    xp = MARGIN * (xl.max() - xl.min())
    yp = MARGIN * (yl.max() - yl.min())
    ax.set_xlim(xl.min() - xp, xl.max() + xp)
    ax.set_ylim(yl.min() - yp, yl.max() + yp)

# Panels sharing y variable also share y limits; drop redundant y labels
for k in range(1, n_pairs):
    if shares[k - 1]:
        lo = min(axes[k].get_ylim()[0], axes[k - 1].get_ylim()[0])
        hi = max(axes[k].get_ylim()[1], axes[k - 1].get_ylim()[1])
        axes[k].set_ylim(lo, hi)
        axes[k - 1].set_ylim(lo, hi)


# =========================================================================
# Tick styling helper
# =========================================================================
def style_axis(ax, which, dim):
    axis = ax.xaxis if which == 'x' else ax.yaxis
    lim  = ax.get_xlim() if which == 'x' else ax.get_ylim()
    axis.set_major_locator(mticker.MaxNLocator(nbins=3))
    maxval = max(abs(lim[0]), abs(lim[1]))
    if maxval == 0:
        return f'$x_{dim + 1}$'
    e = int(np.floor(np.log10(maxval)))
    label = f'$x_{dim + 1}$'
    if e <= -2 or e >= 3:
        axis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _, e=e: f'{v / 10.0 ** e:g}'.replace('-', '\u2212')))
        label = f'$x_{dim + 1}$ ($\\times 10^{{{e}}}$)'
    return label


for k, (ax, (di, dj)) in enumerate(zip(axes, pairs)):
    xlabel = style_axis(ax, 'x', di)
    ylabel = style_axis(ax, 'y', dj)
    ax.set_xlabel(xlabel, labelpad=1)
    if k > 0 and shares[k - 1]:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel(ylabel, labelpad=1)

# =========================================================================
# Shared horizontal colour bar on top
# =========================================================================
cax = fig.add_axes(rect(ML, MB + PH + CB_GAP, W_fig - ML - MR, CB_H))
sm  = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
cb.set_label('Iteration', labelpad=2)
cb.outline.set_linewidth(0.5)

# =========================================================================
# Save main figure
# =========================================================================
FIGURES_DIR = os.path.join(_ROOT, 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
stem = os.path.basename(args.csv).replace('.csv', '')
out  = os.path.join(FIGURES_DIR, f'{stem}_invariant_combined.pdf')
fig.savefig(out)
print(f"Saved Figures/{stem}_invariant_combined.pdf   ({W_fig:.2f} x {H_fig:.2f} in)")

# =========================================================================
# Save bare version: no ticks, labels, axes, frame
# =========================================================================
for ax in axes:
    ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05)
bare_path = os.path.join(FIGURES_DIR, f'{stem}_invariant_combined_bare.pdf')
fig.savefig(bare_path, bbox_inches='tight', pad_inches=0)
print(f"Saved Figures/{stem}_invariant_combined_bare.pdf")

# Individual bare PDFs per projection (only if more than one)
if n_pairs > 1:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, (di, dj) in zip(axes, pairs):
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fname = f'{stem}_invariant_combined_bare_x{di + 1}x{dj + 1}.pdf'
        fig.savefig(os.path.join(FIGURES_DIR, fname),
                    bbox_inches=bbox, pad_inches=0)
        print(f"Saved Figures/{fname}")

# Print axis limits for reference
for ax, (di, dj) in zip(axes, pairs):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    print(f"x{di+1}-x{dj+1}:  xmin={x0:.4g}, xmax={x1:.4g}, ymin={y0:.4g}, ymax={y1:.4g}")

# =========================================================================
# Standalone horizontal legend
# =========================================================================
legend_handles = []
if has_E0:
    legend_handles.append(
        Line2D([0], [0], color=C['E0'], linestyle='--', linewidth=LW_E0,
               label='$\\mathcal{E}_0$'))
legend_handles.append(
    Line2D([0], [0], color=C['E'], linewidth=LW_E, label='$\\mathcal{E}^*$'))
legend_handles.extend([
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C['U'],
           markersize=5, label='$\\mathcal{U}$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C['Z'],
           markersize=5, label='$\\mathcal{Z}$'),
    Line2D([0], [0], marker='x', color='w', markeredgecolor=C['W'],
           markersize=5, markeredgewidth=0.9, label='$\\mathcal{W}$'),
    Line2D([0], [0], marker='x', color='w', markeredgecolor=C['V'],
           markersize=5, markeredgewidth=0.9, label='$\\mathcal{V}$'),
])

fig_leg = plt.figure()
leg = fig_leg.legend(handles=legend_handles, loc='center',
                     ncol=len(legend_handles), framealpha=0.9,
                     handlelength=1.6, handletextpad=0.4, columnspacing=1.2)
leg.get_frame().set_linewidth(0.6)
leg.get_frame().set_edgecolor('grey')
fig_leg.canvas.draw()
bbox = leg.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
fig_leg.savefig(os.path.join(FIGURES_DIR, 'invariant_combined_legend.pdf'),
                bbox_inches=bbox)
plt.close(fig_leg)
print("Saved Figures/invariant_combined_legend.pdf")
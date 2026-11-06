"""
Configuration:
    PT (int): Scatter point size in points (matplotlib units), default 18
    ALF (float): Scatter alpha (opacity) for most point sets, default 0.75 (0-1 scale)
    XI_ALF (float): Reduced translucency for raw samples in panel (b), default 0.25
    N_ARROW (int): Number of mapping arrows to display in panel (b), default 25
    LEG_FS (int): Legend font size in points, default 6
    margin (float): Uniform padding fraction relative to axis range, default 0.18
    bottom_extra (float): Additional fraction of axis range added to bottom for legend, default 0.35

Figure Dimensions:
    TARGET_W (float): Width of individual panel figures in inches, 2.125 (= 8.5/4)
    figsize: Combined figure is 10.0 × 2.8 inches; individual panels are 2.125 × 2.125 inches
    dpi: 150 for display, 300 for PNG export
PlotAlgorithmSteps.py  —  step-by-step illustration matching Algorithm 1.

Four panels (left to right):
  (a) Step 1: Sample points from current ellipsoid
  (b) Step 2: Apply Poincaré return map
  (c) Step 3: Partition — Z and U (preimage sets)
  (d) Step 4: Fit MVEE to U

Outputs:
  fig_steps.pdf             -combined 4-panel figure (with titles)
"""

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'Systems'))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sets.ellipsoid import Ellipsoid, sample_from

# =========================================================================
# System
# =========================================================================
import ExpanderContractor as sys

# =========================================================================
# Paper style
# =========================================================================
FONT_FS   = 7      # base font / axis label size (combined figure)
TICK_FS   = 7       # tick label size (combined figure)
LEG_FS    = 8       # legend font size (combined figure)

mpl.rcParams.update({
    'font.family':      'serif',
    'font.size':        FONT_FS,
    'axes.labelsize':   FONT_FS,
    'axes.titlesize':   FONT_FS,
    'legend.fontsize':  LEG_FS,
    'xtick.labelsize':  TICK_FS,
    'ytick.labelsize':  TICK_FS,
    'lines.linewidth':  1.5,
    'axes.linewidth':   0.8,
    'figure.dpi':       150,
    # 'text.usetex':    True,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# =========================================================================
# Colors
# =========================================================================
_plasma = mpl.colormaps['plasma']
C = {
    'Xi':         '#AAAAAA',        # neutral grey  — raw samples
    'W':          _plasma(0.1),     # dark purple   — contained return points
    'V':          _plasma(0.85),    # bright orange — escaped return points
    'U':          _plasma(0.4),     # magenta-pink  — preimages of W
    'Z':          _plasma(0.65),    # orange-red    — preimages of V
    'arrow':      '#000000',        # black         — mapping arrows
    'Ei_outline': '#000000',        # black         — ellipsoid outline
    'Ei_prev':    '#888888',        # mid grey      — dotted previous ellipsoid
}
PT        = 18      # scatter point size
ALF       = 0.75    # scatter alpha
XI_ALF    = 0.25    # translucency for raw samples in panel (b)
N_ARROW   = 25      # number of mapping arrows in panel (b)


# =========================================================================
# Helper: white-filled ellipse with solid outline
# outline_style=':' and fill_alpha=0 for the dotted-only previous ellipsoid
# =========================================================================
def plot_filled_ellipse(Q, c, ax, fill_alpha=1.0,
                        outline_color='#000000', outline_alpha=1.0,
                        outline_style='-', outline_zorder=4):
    eigvals, eigvecs = np.linalg.eigh(Q)
    theta  = np.linspace(0, 2*np.pi, 300)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    pts    = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ circle).T + c
    if fill_alpha > 0:
        ax.fill(pts[:, 0], pts[:, 1], color='white', alpha=fill_alpha, zorder=1)
    ax.plot(pts[:, 0], pts[:, 1], color=outline_color, alpha=outline_alpha,
            linestyle=outline_style, linewidth=1.5, zorder=outline_zorder)


def _autolim(ax, margin=0.6, bottom_extra=0.4):
    """margin: uniform padding; bottom_extra: additional fraction added to the
    bottom only, pushing the ellipse upward to leave room for the legend."""
    ax.autoscale(enable=True, tight=True)
    xl, xh = ax.get_xlim()
    yl, yh = ax.get_ylim()
    xpad = margin * (xh - xl)
    ypad = margin * (yh - yl)
    ax.set_xlim(xl - xpad, xh + xpad)
    ax.set_ylim(yl - ypad - bottom_extra * (yh - yl), yh + ypad)


# =========================================================================
# Warm-up iterations then illustration iteration with small N
# =========================================================================
np.random.seed(3)
N = 50

E0 = sys.initial_guess()
Ei = E0

for _ in range(1):
    Xi_tmp       = sample_from(Ei, 600).T
    Xi_prime_tmp = sys.P_batch(Xi_tmp)
    in_E_tmp     = Ei.contains_point(Xi_prime_tmp.T)
    U_tmp        = Xi_tmp[in_E_tmp, :]
    V_tmp        = Xi_prime_tmp[~in_E_tmp, :]
    if len(V_tmp) == 0:
        break
    try:
        Ei_new = Ellipsoid(sys.d)
        Ei_new.fit_p_ball(U_tmp, 2)
        Ei = Ei_new
    except np.linalg.LinAlgError:
        pass

Xi       = sample_from(Ei, N).T
Xi_prime = sys.P_batch(Xi)
in_E     = Ei.contains_point(Xi_prime.T)

V = Xi_prime[~in_E, :]
W = Xi_prime[in_E,  :]
U = Xi[in_E,        :]
Z = Xi[~in_E,       :]

Ei_new = Ellipsoid(sys.d)
Ei_new.fit_p_ball(U, 2)

print(f"|V|={len(V)}, |W|={len(W)}, |U|={len(U)}, |Z|={len(Z)}")

rng       = np.random.default_rng(7)
arrow_idx = rng.choice(N, size=min(N_ARROW, N), replace=False)


PANEL_TITLES = [
    '(a) Step 1: Sample points from $\\mathcal{E}_i$',
    '(b) Step 2: Apply Poincaré map $\\mathcal{P}$',
    '(c) Step 3: Partition Preimages',
    '(d) Step 4: Fit MVEE to $\\mathcal{U}$',
]

# =========================================================================
# Draw a single panel (idx 0-3) onto ax
# =========================================================================
def draw_panel(ax, idx, show_title=True):
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$', labelpad=2)
    if show_title:
        ax.set_title(PANEL_TITLES[idx], pad=4)

    if idx == 0:
        plot_filled_ellipse(Ei.Q, Ei.c.flatten(), ax)
        ax.scatter(Xi[:, 0], Xi[:, 1], s=PT, c=C['Xi'], alpha=ALF, zorder=3)
        _autolim(ax)
        ax.legend(handles=[
            Patch(facecolor='white', edgecolor=C['Ei_outline'],
                  label='$\\mathcal{E}_i$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C['Xi'],
                   markersize=6, label='$\\{x_i\\}$'),
        ], fontsize=LEG_FS, loc='lower center', framealpha=0.9, ncol=2,
           handlelength=1.0, handletextpad=0.4, columnspacing=0.8)

    elif idx == 1:
        plot_filled_ellipse(Ei.Q, Ei.c.flatten(), ax)
        ax.scatter(Xi[:, 0], Xi[:, 1], s=PT, c=C['Xi'], alpha=XI_ALF, zorder=2)
        for i in range(N):
            ax.annotate('',
                        xy=Xi_prime[i], xytext=Xi[i],
                        arrowprops=dict(arrowstyle='->', color=C['arrow'],
                                        lw=0.7, shrinkA=0, shrinkB=0),
                        zorder=3)
        ax.scatter(Xi_prime[in_E,  0], Xi_prime[in_E,  1],
                   s=PT, c=C['W'], alpha=ALF, zorder=4)
        ax.scatter(Xi_prime[~in_E, 0], Xi_prime[~in_E, 1],
                   s=PT, c=C['V'], alpha=ALF, zorder=4)
        _autolim(ax)
        ax.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C['W'],
                   markersize=6, label='$\\mathcal{W}:= \{\\mathcal{P}(x_i) \\mid \mathcal{P}(x_i) \\in \\mathcal{E}_i\}$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C['V'],
                   markersize=6, label='$\\mathcal{V}:= \{\\mathcal{P}(x_i) \\mid \mathcal{P}(x_i) \\notin \\mathcal{E}_i\}$'),
        ], fontsize=LEG_FS, loc='lower center', framealpha=0.9, ncol=1,
           handlelength=1.0, handletextpad=0.4, columnspacing=0.8)

    elif idx == 2:
        plot_filled_ellipse(Ei.Q, Ei.c.flatten(), ax)
        ax.scatter(Z[:, 0], Z[:, 1], s=PT, c=C['Z'], alpha=ALF, zorder=3)
        ax.scatter(U[:, 0], U[:, 1], s=PT, c=C['U'], alpha=ALF, zorder=3)
        _autolim(ax)
        ax.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C['U'],
                   markersize=6, label='$\\mathcal{U}:= \{x_i \\mid \\mathcal{P}(x_i) \\in \\mathcal{W}\}$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C['Z'],
                   markersize=6, label='$\\mathcal{Z}:= \{x_i \\mid \\mathcal{P}(x_i) \\in \\mathcal{V}\}$'),
        ], fontsize=LEG_FS, loc='lower center', framealpha=0.9, ncol=1,
           handlelength=1.0, handletextpad=0.4, columnspacing=0.8)

    elif idx == 3:
        plot_filled_ellipse(Ei.Q, Ei.c.flatten(), ax,
                            fill_alpha=0,
                            outline_color=C['Ei_prev'], outline_alpha=0.6,
                            outline_style=':', outline_zorder=2)
        plot_filled_ellipse(Ei_new.Q, Ei_new.c.flatten(), ax)
        ax.scatter(U[:, 0], U[:, 1], s=PT, c=C['U'], alpha=ALF, zorder=3)
        _autolim(ax)
        ax.legend(handles=[
            Line2D([0], [0], color=C['Ei_prev'], linestyle=':', alpha=0.6,
                   label='$\\mathcal{E}_i$'),
            Patch(facecolor='white', edgecolor=C['Ei_outline'],
                  label='$\\mathcal{E}_{i+1}$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C['U'],
                   markersize=6, label='$\\mathcal{U}$'),
        ], fontsize=LEG_FS, loc='lower center', framealpha=0.9, ncol=3,
           handlelength=1.0, handletextpad=0.4, columnspacing=0.8)


def draw_panels(axes, show_titles=True):
    for i, ax in enumerate(axes.flatten()):
        draw_panel(ax, i, show_title=show_titles)


# =========================================================================
# Combined figure (with titles) — 2×2 grid
# =========================================================================
FIGURES_DIR = os.path.join(_ROOT, 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(4.25, 4.0))
draw_panels(axes, show_titles=True)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig_steps.pdf'), bbox_inches='tight')
print("Saved fig_steps.pdf")

# =========================================================================
# Individual panels: no axes, ticks, labels, or titles
# =========================================================================
PANEL_NAMES = ['fig_step1_sample', 'fig_step2_map',
               'fig_step3_partition', 'fig_step4_mvee']

def save_individual_panels(size=2.125):
    # shared limits: draw all panels once, take the union of their limits
    lims = []
    for idx in range(4):
        f, a = plt.subplots(figsize=(size, size))
        draw_panel(a, idx, show_title=False)
        lims.append((a.get_xlim(), a.get_ylim()))
        plt.close(f)
    xlim = (min(l[0][0] for l in lims), max(l[0][1] for l in lims))
    ylim = (min(l[1][0] for l in lims), max(l[1][1] for l in lims))

    for idx, name in enumerate(PANEL_NAMES):
        fig_i, ax_i = plt.subplots(figsize=(size, size))
        draw_panel(ax_i, idx, show_title=False)
        if ax_i.get_legend() is not None:
            ax_i.get_legend().remove()
        ax_i.set_xlim(xlim); ax_i.set_ylim(ylim)
        ax_i.axis('off')
        fig_i.subplots_adjust(0, 0, 1, 1)           # axes fill the whole canvas
        fig_i.savefig(os.path.join(FIGURES_DIR, f'{name}.pdf'),
                      transparent=True)              # no bbox_inches='tight'
        plt.close(fig_i)
        
save_individual_panels()
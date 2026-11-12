"""Project saved reachable normotopes sliced onto the guard surface.

Styled like PlotInvariantCombined.py: paper rcParams, plasma colormap
(colour = time), one panel per coordinate projection, a shared horizontal
colour bar on top, and bare (axis-free) PDF exports.

Usage:
    python PlotImmraxGuardProjection.py [data.npz] [--ellipsoid-log log.csv]
                                        [--output Figures/immrax_guard_projection.pdf]
                                        [--width 6.0]

Writes (next to --output, default ./Figures/):
    <stem>.pdf
    <stem>_bare.pdf
    <stem>_bare_z{i}z{j}.pdf       (one per projection)
    immrax_guard_projection_legend.pdf
"""

import argparse
from pathlib import Path

import immrax as irx
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
from sets.ellipsoid import Ellipsoid

jax.config.update("jax_enable_x64", True)
NT = irx.L2Normotope
ROOT = Path(__file__).resolve().parent
GUARD_CHART = jnp.array([[1., -1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]).T

# =========================================================================
# Paper style  (identical to PlotInvariantCombined.py)
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

CMAP = mpl.colormaps['plasma']
C_E = '#000000'      # final iterative ellipsoid E*

LW_IT = 0.5          # normotope outline width
LW_E = 0.5           # E* outline width
MARGIN = 0.12        # window padding fraction

LABELS = [r"$z_1$", r"$z_2$", r"$z_3$"]
UNITS = ["rad", "rad/s", "rad/s"]


def guard_height_bounds(nt, gamma):
    """Same h = 2*gamma - tsw - tst bounds as in ImmraxAlgorithm."""
    normal = jnp.array([-1., -1., 0., 0.])
    direction = jnp.linalg.solve(nt.alpha.T, normal)
    center_height = normal @ nt.ox + 2*gamma
    radius = nt.y*jnp.linalg.norm(direction)
    return center_height - radius, center_height + radius


def nt_slice(nt, chart, origin):
    """Slice a 4-D normotope into guard coordinates x = chart @ z + origin."""
    Q, R = jnp.linalg.qr(nt.alpha @ chart)
    displacement = nt.alpha @ (nt.ox - origin)
    projected = Q.T @ displacement
    oz = jax.scipy.linalg.solve_triangular(R, projected)
    r = jnp.sum(displacement**2) - jnp.sum(projected**2)
    if bool(nt.y**2 < r):
        return None
    return NT(oz, R, jnp.sqrt(nt.y**2 - r))


def load_guard_intersections(path):
    """Load the entire trajectory and retain nonempty slices after T_underbar."""
    with np.load(path, allow_pickle=False) as data:
        nt_traj = jnp.asarray(data["nt_traj"])
        times = np.asarray(data["ts"])
        nominal_index = int(data["N"])
        gamma = float(data["guard_gamma"]) if "guard_gamma" in data else 0.0525
        chart = jnp.asarray(data["guard_chart"]) if "guard_chart" in data else GUARD_CHART

    if nt_traj.ndim != 2 or nt_traj.shape[1] != 21 or len(times) != len(nt_traj):
        raise ValueError("Expected nt_traj with shape (samples, 21) and matching ts.")
    if not np.isfinite(nt_traj).all() or np.any(np.asarray(nt_traj[:, -1]) < 0.):
        raise ValueError("Trajectory must contain finite normotopes with nonnegative radii.")

    normotopes = [NT.unvec(state) for state in nt_traj]
    bounds = np.asarray([guard_height_bounds(nt, gamma) for nt in normotopes])
    above = np.flatnonzero(bounds[:-1, 0] > 0.)
    if not len(above):
        raise ValueError("No T_underbar: no normotope lies entirely above the guard.")
    start = int(above[0])
    # As in alpha0_to_max_mjac_length, the last sample is T_overbar.
    if not bounds[-1, 1] < 0.:
        raise ValueError("Invalid T_overbar: the last normotope must lie entirely in h < 0.")

    # Use the same nominal pre-impact origin as ImmraxAlgorithm's slice_nt.
    origin = normotopes[nominal_index].ox
    sliced_normotopes, indices = [], []
    for index in range(start, len(normotopes)):
        sliced = nt_slice(normotopes[index], chart, origin)
        if sliced is not None:
            sliced_normotopes.append(sliced)
            indices.append(index)
    if not sliced_normotopes:
        raise ValueError("No guard intersections found after T_underbar.")
    return sliced_normotopes, times[indices], times[start], times[-1], origin, chart


def load_iterative_ellipsoid(path, origin, chart):
    """Load the last logged ellipsoid and express it in the plotting chart."""
    from Systems.CompassGaitWalker import x_star, _guard_chart

    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    ellipsoid = Ellipsoid(3).load_from_row(rows[-1, 1:])
    if ellipsoid.dimension != 3:
        raise ValueError("Expected a three-dimensional guard-coordinate ellipsoid.")
    # Iterative coordinates: x = x_star + _guard_chart @ z_iterative.
    # Plot coordinates: x = origin + chart @ z_plot.
    chart = np.asarray(chart)
    transform = np.linalg.lstsq(chart, np.asarray(_guard_chart), rcond=None)[0]
    offset = np.linalg.lstsq(chart, np.asarray(x_star) - np.asarray(origin), rcond=None)[0]
    inverse = np.linalg.inv(transform)
    Q = inverse.T @ ellipsoid.Q @ inverse
    center = transform @ ellipsoid.c.flatten() + offset
    return Ellipsoid(3).load_from_row(np.concatenate((Q.flatten(), center)))


def style_axis(ax, which, dim):
    """Few ticks; rescale tick labels when the range is tiny/huge."""
    axis = ax.xaxis if which == 'x' else ax.yaxis
    lim = ax.get_xlim() if which == 'x' else ax.get_ylim()
    axis.set_major_locator(mticker.MaxNLocator(nbins=3))
    maxval = max(abs(lim[0]), abs(lim[1]))
    base, unit = LABELS[dim], UNITS[dim]
    if maxval == 0:
        return f'{base} ({unit})'
    e = int(np.floor(np.log10(maxval)))
    label = f'{base} ({unit})'
    if e <= -2 or e >= 3:
        axis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _, e=e: f'{v / 10.0 ** e:g}'.replace('-', '−')))
        label = f'{base} ($\\times 10^{{{e}}}$ {unit})'
    return label


def plot_normotope(nt, ax, xi, yi, color):
    """Plot a sliced normotope projection and restyle whatever it drew."""
    n_lines, n_patches = len(ax.lines), len(ax.patches)
    nt.plot_projection(ax, xi, yi, color=color)
    for ln in ax.lines[n_lines:]:
        ln.set_linewidth(LW_IT)
        ln.set_alpha(0.95)
        ln.set_linestyle('-')
        ln.set_zorder(3)
    for p in ax.patches[n_patches:]:
        p.set_linewidth(LW_IT)
        p.set_zorder(3)


def fit_window(ax):
    """Window = extent of everything drawn so far + margin."""
    xs = [np.asarray(ln.get_xdata(), dtype=float) for ln in ax.lines]
    ys = [np.asarray(ln.get_ydata(), dtype=float) for ln in ax.lines]
    if not xs:
        ax.relim()
        ax.autoscale_view()
        return
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    xp = MARGIN * (xs.max() - xs.min())
    yp = MARGIN * (ys.max() - ys.min())
    ax.set_xlim(xs.min() - xp, xs.max() + xp)
    ax.set_ylim(ys.min() - yp, ys.max() + yp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path,
                        default=ROOT / "Data/immrax_reachable_trajectory.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Figures/immrax_guard_projection.pdf",
                        help="main PDF; bare variants are written next to it")
    parser.add_argument("--ellipsoid-log", type=Path,
                        help="Iterative ellipsoid CSV; defaults to the latest timestamp in Data/cgw")
    parser.add_argument("--width", type=float, default=6.0, help="figure width in inches")
    args = parser.parse_args()

    sliced_normotopes, times, t_underbar, t_overbar, origin, chart = load_guard_intersections(args.data)
    ellipsoid_log = args.ellipsoid_log
    if ellipsoid_log is None:
        logs = sorted((ROOT / "Data/cgw").glob("*CompassGaitWalker_ellipsoid_log_*.csv"))
        if not logs:
            raise FileNotFoundError("No Compass Gait Walker ellipsoid logs found in Data/cgw.")
        ellipsoid_log = logs[-1]
    iterative_ellipsoid = load_iterative_ellipsoid(ellipsoid_log, origin, chart)

    pairs = [(0, 1), (0, 2), (1, 2)]
    n_pairs = len(pairs)
    norm = mpl.colors.Normalize(times.min(),
                                times.max() if times.max() > times.min() else times.min() + 1e-6)

    # ---- figure geometry (same as PlotInvariantCombined.py) ----
    W_fig = args.width
    ML, MR = 0.58, 0.07
    G_FULL, G_SHARE = 0.60, 0.16
    MB, PH = 0.46, 1.30
    CB_GAP, CB_H, CB_TOP = 0.10, 0.10, 0.58
    shares = [pairs[k][1] == pairs[k - 1][1] for k in range(1, n_pairs)]
    gaps = [G_SHARE if s else G_FULL for s in shares]
    pw = (W_fig - ML - MR - sum(gaps)) / n_pairs
    H_fig = MB + PH + CB_TOP

    def rect(x, y, w, h):
        return [x / W_fig, y / H_fig, w / W_fig, h / H_fig]

    fig = plt.figure(figsize=(W_fig, H_fig))
    axes, x = [], ML
    for k in range(n_pairs):
        axes.append(fig.add_axes(rect(x, MB, pw, PH)))
        x += pw + (gaps[k] if k < n_pairs - 1 else 0)

    # ---- draw ----
    for ax, (xi, yi) in zip(axes, pairs):
        ax.set_aspect('auto')
        for nt, time in zip(sliced_normotopes, times):
            plot_normotope(nt, ax, xi, yi, CMAP(norm(time)))
        fit_window(ax)                      # window from the normotope slices
        iterative_ellipsoid.plot(ax, diml=xi, dimh=yi, color='#EBEBEB', alpha=1.,
                                 linestyle='-', linewidth=LW_E, zorder=20)
        ln = ax.lines[-1]                      # the E* outline just drawn
        ax.fill(ln.get_xdata(), ln.get_ydata(), facecolor='#EBEBEB', edgecolor='none', zorder=1)

    # panels sharing a y variable also share y limits
    for k in range(1, n_pairs):
        if shares[k - 1]:
            lo = min(axes[k].get_ylim()[0], axes[k - 1].get_ylim()[0])
            hi = max(axes[k].get_ylim()[1], axes[k - 1].get_ylim()[1])
            axes[k].set_ylim(lo, hi)
            axes[k - 1].set_ylim(lo, hi)

    for k, (ax, (xi, yi)) in enumerate(zip(axes, pairs)):
        xlabel = style_axis(ax, 'x', xi)
        ylabel = style_axis(ax, 'y', yi)
        ax.set_xlabel(xlabel, labelpad=1)
        if k > 0 and shares[k - 1]:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(ylabel, labelpad=1)

    # ---- shared horizontal colour bar on top ----
    cax = fig.add_axes(rect(ML, MB + PH + CB_GAP, W_fig - ML - MR, CB_H))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    cb.set_label('Time (s)', labelpad=2)
    cb.outline.set_linewidth(0.5)

    # ---- save main figure ----
    out = args.output.with_suffix('.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}   ({W_fig:.2f} x {H_fig:.2f} in)")

    # ---- bare versions: no ticks, labels, axes, frame ----
    for ax in axes:
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05)
    bare = out.with_name(f'{out.stem}_bare.pdf')
    fig.savefig(bare, bbox_inches='tight', pad_inches=0)
    print(f"Saved {bare}")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, (xi, yi) in zip(axes, pairs):
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fname = out.with_name(f'{out.stem}_bare_z{xi + 1}z{yi + 1}.pdf')
        fig.savefig(fname, bbox_inches=bbox, pad_inches=0)
        print(f"Saved {fname}")

    for ax, (xi, yi) in zip(axes, pairs):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        print(f"z{xi+1}-z{yi+1}:  xmin={x0:.4g}, xmax={x1:.4g}, ymin={y0:.4g}, ymax={y1:.4g}")

    # ---- standalone legend ----
    handles = [Line2D([0], [0], color=C_E, linewidth=LW_E, label='$\\mathcal{E}^*$')]
    fig_leg = plt.figure()
    leg = fig_leg.legend(handles=handles, loc='center', ncol=len(handles), framealpha=0.9,
                         handlelength=1.6, handletextpad=0.4, columnspacing=1.2)
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor('grey')
    fig_leg.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    legend_path = out.with_name('immrax_guard_projection_legend.pdf')
    fig_leg.savefig(legend_path, bbox_inches=bbox)
    plt.close(fig_leg)
    plt.close(fig)
    print(f"Saved {legend_path}")

    print(f"T_underbar = {t_underbar:.6f} s; T_overbar = {t_overbar:.6f} s")
    print(f"Plotted {len(sliced_normotopes)} three-dimensional guard normotopes")
    print(f"Iterative ellipsoid: final row of {ellipsoid_log}")


if __name__ == "__main__":
    main()
"""Project saved reachable normotopes sliced onto the guard surface."""

import argparse
from pathlib import Path

import immrax as irx
import jax
from equinox import filter_jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sets.ellipsoid import Ellipsoid
import Systems.CompassGaitWalker as cgw
from diffrax import Euler, ODETerm, Event, SaveAt, diffeqsolve
import optimistix as optx

jax.config.update("jax_enable_x64", True)
NT = irx.L2Normotope
ROOT = Path(__file__).resolve().parent
GUARD_CHART = jnp.array([[1., -1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]).T


def guard_height_bounds(nt, gamma):
    """Same h = 2*gamma - tsw - tst bounds as in ImmraxAlgorithm."""
    normal = jnp.array([-1., -1., 0., 0.])
    direction = jnp.linalg.solve(nt.alpha.T, normal)
    center_height = normal @ nt.ox + 2*gamma
    radius = nt.y*jnp.linalg.norm(direction)
    return center_height - radius, center_height + radius

term   = ODETerm(cgw.sys.f)
solver = Euler()
u      = 0.0
event  = Event(cgw.sys.guard, optx.Newton(5e-7, 5e-7, optx.rms_norm), direction=True)
dt     = 0.001005217149786066

# Immrax needs to roll out trajectories using Euler integration due to slight differences with Dopri8

@filter_jit
def P_single_euler(x_guard, xf):
    """Euler return map in guard coordinates centered at xf."""
    x_full = cgw._guard_chart @ x_guard + xf
    xp     = cgw.sys.reset(0, x_full)
    traj   = diffeqsolve(term, solver, t0=0, t1=2, dt0=dt,
                         y0=xp, args=u, event=event, throw=False)
    return (cgw._guard_chart_inv @ (traj.ys.flatten() - xf))[:3]

P_batch_euler = jax.vmap(P_single_euler, in_axes=(0, None))  # Shared xf; (N,3) -> (N,3)

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
    return sliced_normotopes, times[indices], origin, chart


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path,
                        default=ROOT / "Data/immrax_reachable_trajectory.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "Figures/immrax_guard_projection.png")
    parser.add_argument("--ellipsoid-log", type=Path,
                        help="Iterative ellipsoid CSV; defaults to the latest timestamp in Data/cgw")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--n-verify", type=int, default=10000, help="Number of samples to verify with")
    args = parser.parse_args()
    N_VERIFY = args.n_verify


    sliced_normotopes, times, origin, chart = load_guard_intersections(args.data)

    # Generate Monte Carlo samples on the sliced normotope
    mc_x0s_guard = []
    for nt in sliced_normotopes:
        # TODO: Make a sample_interior method in nt
        mc_x0s_i = nt.sample_boundary(jax.random.PRNGKey(0), int(N_VERIFY/len(sliced_normotopes)))
        mc_x0s_guard.append(mc_x0s_i)

    mc_x0s_guard = np.vstack(mc_x0s_guard)
    returns = P_batch_euler(mc_x0s_guard, origin)

    # Check if these samples end up within one of the sliced normotopes
    containments = []
    for nt in sliced_normotopes:
        contains = jax.vmap(nt.contains)(returns)
        containments.append(contains)

    containments = np.any(np.vstack(np.atleast_2d(containments)), axis=0)
    print(f"IMMRAX Ellipsoid:")
    print(f"{len(containments)-np.count_nonzero(containments)} escaped from {len(containments)}")

    # Load the final ellipsoid from the iterative algorithm    
    ellipsoid_log = args.ellipsoid_log
    if ellipsoid_log is None:
        logs = sorted((ROOT / "Data/cgw").glob("*CompassGaitWalker_ellipsoid_log_*.csv"))
        if not logs:
            raise FileNotFoundError("No Compass Gait Walker ellipsoid logs found in Data/cgw.")
        ellipsoid_log = logs[-1]
    rows = np.loadtxt(ellipsoid_log, delimiter=",", skiprows=1, ndmin=2)
    iterative_ellipsoid = Ellipsoid(3).load_from_row(rows[-1, 1:])

    # Sample from this ellipsoid and apply the return map    
    mc_x0s_scenario = iterative_ellipsoid.sample_from(N_VERIFY, np.random.default_rng(21))
    returns_scenario = cgw.P_batch(mc_x0s_scenario)
    containments = iterative_ellipsoid.contains_points(returns_scenario)

    print(f"Scenario Opt Ellipsoid:")
    print(f"{len(containments)-np.count_nonzero(containments)} escaped from {len(containments)}")


    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [r"$z_1$ (rad)", r"$z_2$ (rad/s)", r"$z_3$ (rad/s)"]
    norm = Normalize(times.min(), times.max() if times.max() > times.min() else times.min()+1e-6)
    cmap = plt.get_cmap("viridis")
    for ax, (xi, yi) in zip(axes, pairs):
        for nt, time in zip(sliced_normotopes, times):
            nt.plot_projection(ax, xi, yi, color=cmap(norm(time)))
        iterative_ellipsoid.plot(ax, diml=xi, dimh=yi, color="black", alpha=1.,
                                 linestyle="--", linewidth=2., label="Iterative algorithm (final)")
        ax.scatter(mc_x0s_guard[:, xi], mc_x0s_guard[:, yi], s=8, marker="o",
                   color="tab:blue", alpha=.5, label="IMMRAX samples", zorder=3)
        ax.scatter(returns[:, xi], returns[:, yi], s=12, marker="x",
                   color="tab:cyan", alpha=.5, label="IMMRAX returns", zorder=3)
        ax.scatter(mc_x0s_scenario[:, xi], mc_x0s_scenario[:, yi], s=10, marker="^",
                   color="tab:orange", alpha=.5, label="Scenario samples", zorder=3)
        ax.scatter(returns_scenario[:, xi], returns_scenario[:, yi], s=14, marker="+",
                   color="tab:red", alpha=.5, label="Scenario returns", zorder=3)
        ax.set_xlabel(labels[xi])
        ax.set_ylabel(labels[yi])
        ax.set_title(f"Guard coordinates ({xi}, {yi})")
        ax.grid(True, alpha=.3)
        ax.autoscale_view()
        ax.legend()
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="Time (s)")
    fig.suptitle(f"Projections of {len(sliced_normotopes)} sliced guard normotopes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Plotted {len(sliced_normotopes)} three-dimensional guard normotopes")
    print(f"Iterative ellipsoid: final row of {ellipsoid_log}")
    print(f"Saved plot to {args.output}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()

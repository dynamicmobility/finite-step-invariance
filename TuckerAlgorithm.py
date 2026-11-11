import argparse
import importlib
import numpy as np
from matplotlib import pyplot as plt
import Systems.CompassGaitWalker
import Systems.ExpanderContractor
import jax
from datetime import datetime
from probabilistic_bounds import binomial_tail
import pandas as pd
import warnings
from scipy.linalg import solve_discrete_lyapunov
from sets.ball import Ball
from sets.ellipsoid import Ellipsoid
from pathlib import Path
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--ellipsoid-log", type=Path,
                    help="Iterative ellipsoid CSV; defaults to the latest log in Data/cgw")
args = parser.parse_args()

# =========================================================================
# Hyperparameters
# =========================================================================
sys = Systems.CompassGaitWalker
# sys = Systems.ExpanderContractor
alpha = 0.0001
N = 200

# =========================================================================
# Initial Values
# =========================================================================

# Batched H
def H(x, r):
    return r-np.linalg.vector_norm(x-sys.c.flatten(), axis=1)

Brmax = Ball(sys.d)
Brmax.c = sys.c.flatten()
Brmax.r = 0.002

# Check to make sure Br is actually centered in the right place
# print(sys.P_single(Brmax.c))
# assert np.allclose(sys.P_single(Brmax.c), Brmax.c, atol=1e-6)

samples = Brmax.sample_from(N, np.random.default_rng(0))

distances = np.sort(np.linalg.vector_norm(samples-sys.c.flatten(), axis=1))

print(np.linalg.svd(sys.J))


forwards = sys.P_batch(samples)
assert H(samples, Brmax.r).shape == (N,)
assert H(forwards, Brmax.r).shape == (N,)

assert H(np.atleast_2d(sys.c.flatten()), 1) == 1.

# # Compare each sample with its image under the return map before the main loop.
# sample_points = np.asarray(samples)
# forward_points = np.asarray(forwards)
# fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
# for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
#     ax.quiver(
#         sample_points[:, i], sample_points[:, j],
#         forward_points[:, i] - sample_points[:, i],
#         forward_points[:, j] - sample_points[:, j],
#         angles="xy", scale_units="xy", scale=1,
#         color="gray", alpha=0.5, width=0.002, zorder=1,
#     )
#     ax.scatter(sample_points[:, i], sample_points[:, j],
#                color="red", s=12, label="Samples", zorder=2)
#     ax.scatter(forward_points[:, i], forward_points[:, j],
#                color="blue", s=12, label="Forwards", zorder=3)
#     ax.set_xlabel(f"Guard coordinate {i}")
#     ax.set_ylabel(f"Guard coordinate {j}")
#     ax.set_title(f"Coordinates ({i}, {j})")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
# fig.suptitle("Samples and their return-map images")
# plt.show()


# Main Algorithm

sample_idx = N-1

while sample_idx > 0:
    r = distances[sample_idx]
    Br = Ball(sys.d)
    Br.c = sys.c.flatten()
    Br.r = r
    samples = Br.sample_from(N, np.random.default_rng(0))
    forwards = sys.P_batch(samples)
    inside_idx = np.flatnonzero(np.linalg.vector_norm(samples-sys.c.flatten(), axis=1) < r+1e-5)
    cbf = H(forwards[inside_idx], r) - H(samples[inside_idx], r) >= -alpha*H(samples[inside_idx], r)
    print("r = ", r)
    print("Points Inside: ", len(inside_idx))
    print("Satisfies CBF: ", sum(cbf))

    if(cbf.all()):
        print(f"Converged with r={r}")
        break
    else:
        sample_idx -= 1 

Bfinal = Ball(sys.d)
Bfinal.c = sys.c.flatten()
Bfinal.r = r

fig, ax = plt.subplots()

sys.plot_true(ax, 0, 1, label="True")
Bfinal.plot(ax, 0, 1, color="blue", label="Bfinal")
if sys is Systems.CompassGaitWalker:
    log_dir = Path(__file__).resolve().parent / "Data" / "cgw"
    ellipsoid_log = args.ellipsoid_log
    if ellipsoid_log is None:
        ellipsoid_logs = sorted(log_dir.glob("*CompassGaitWalker_ellipsoid_log_*.csv"))
        if not ellipsoid_logs:
            raise FileNotFoundError(f"No Compass Gait Walker ellipsoid logs found in {log_dir}")
        ellipsoid_log = ellipsoid_logs[-1]
    rows = pd.read_csv(ellipsoid_log, index_col=0)
    iterative_ellipsoid = Ellipsoid(3).load_from_row(rows.iloc[-1].to_numpy(dtype=float))
    iterative_ellipsoid.plot(ax, 0, 1, color="black", linestyle="--", linewidth=2,
                             label="Iterative algorithm (final)")
    print(f"Iterative ellipsoid: final row of {ellipsoid_log}")
sample_points = np.asarray(samples)
forward_points = np.asarray(forwards)
ax.quiver(
    sample_points[:, 0], sample_points[:, 1],
    forward_points[:, 0] - sample_points[:, 0],
    forward_points[:, 1] - sample_points[:, 1],
    angles="xy", scale_units="xy", scale=1,
    color="gray", alpha=0.5, width=0.002, zorder=1,
)
ax.scatter(sample_points[:, 0], sample_points[:, 1],
           color="red", s=12, label="Samples", zorder=2)
ax.scatter(forward_points[:, 0], forward_points[:, 1],
           color="blue", s=12, label="Forwards", zorder=3)
ax.set_xlabel("Coordinate 0")
ax.set_ylabel("Coordinate 1")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

plt.show()

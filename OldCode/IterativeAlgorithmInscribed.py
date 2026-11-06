import numpy as np
from scipy.spatial import ConvexHull
import cvxpy as cp
from matplotlib import pyplot as plt
from sets.ellipsoid import Ellipsoid, plot_projection, sample_from
import CompassGaitWalker
import ExpanderContractor
import DogBone
import jax
from datetime import datetime
import jax.numpy as jnp
from probabilistic_bounds import binomial_tail

print("Using backend: ", jax.default_backend())


# =========================================================================
# Hyperparameters
# =========================================================================
N             = 2000
MAX_ITER      = 200
PLOT          = False
EPS_THRESHOLD = 0.03

# =========================================================================
# Model Setup
# =========================================================================
sys = CompassGaitWalker
# sys = ExpanderContractor
# sys = DogBone


# =========================================================================
# Maximum-volume ellipsoid inscribed in conv(U)
#
# Parameterisation:  E = { Bv + c : ||v||₂ ≤ 1 },  B symmetric PD
#   => standard form  (x-c)ᵀ Q (x-c) ≤ 1  with  Q = B⁻²
#
# Polytope from convex hull:  aᵢᵀ x ≤ bᵢ  for each face i
# Half-space constraint:       ||B aᵢ||₂ + aᵢᵀ c ≤ bᵢ   (SOCP)
# Objective:                   maximise  log det(B)
# =========================================================================
def fit_inscribed_ellipsoid(U: np.ndarray) -> Ellipsoid:
    """
    Fit the maximum-volume ellipsoid inscribed in the convex hull of U.

    Parameters
    ----------
    U : (N, d) array of points

    Returns
    -------
    Ellipsoid with Q, c set to the MVIE solution.
    """
    d = U.shape[1]

    hull = ConvexHull(U, qhull_options='QJ')   # QJ joggle for near-degenerate data
    # hull.equations rows: [n₁ … nᵈ  offset]  meaning  nᵀx + offset ≤ 0
    A_hs = hull.equations[:, :d]   # (F, d) face normals
    b_hs = -hull.equations[:, d]   # (F,)   RHS  nᵀx ≤ b

    B = cp.Variable((d, d), symmetric=True)
    c = cp.Variable((d,))

    constraints = [B >> 1e-6 * np.eye(d)]
    for a_i, b_i in zip(A_hs, b_hs):
        constraints.append(cp.norm(B @ a_i, 2) + a_i @ c <= b_i)

    prob = cp.Problem(cp.Maximize(cp.log_det(B)), constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise np.linalg.LinAlgError(f"MVIE SDP failed: {prob.status}")

    B_val = B.value
    c_val = c.value

    E    = Ellipsoid(d)
    Binv = np.linalg.inv(B_val)
    E.Q  = Binv @ Binv          # Q = B⁻² (B symmetric)
    E.c  = c_val.reshape(d, 1)
    E.L  = np.linalg.cholesky((E.Q + E.Q.T) / 2)
    return E


# =========================================================================
# Main Algorithm
# =========================================================================
E0 = sys.initial_guess()
print(f"Initial ellipsoid volume: {E0.volume():.6e}")

Ei = E0

for iteration in range(MAX_ITER):

    Xi       = sample_from(Ei, N).T           # (N, d)
    Xi_prime = sys.P_batch(Xi)                # (N, d)
    in_E     = Ei.contains_point(Xi_prime.T)  # (N,) bool

    U = Xi[in_E,        :]
    V = Xi_prime[~in_E, :]
    W = Xi_prime[in_E,  :]
    Z = Xi[~in_E,       :]

    n_tot = len(W) + len(V)
    n_esc = int(np.sum(~in_E))
    print(f"Iter {iteration:3d}: |V|={len(V):4d}/{n_tot}  "
          f"({100*len(V)/n_tot:.1f}% escaped)  vol={Ei.volume():.4e}")

    epsilon = binomial_tail(n_esc, N)
    print(f"Iter {iteration:3d}: epsilon={epsilon:.4f}")

    if epsilon < EPS_THRESHOLD:
        print(f"\nConverged after {iteration} iterations, epsilon={epsilon:.4f}")
        break

    if len(U) <= sys.d + 1:
        print("  Too few U points for convex hull, resampling")
        continue

    try:
        Ei = fit_inscribed_ellipsoid(U)
    except (np.linalg.LinAlgError, Exception) as e:
        print(f"  MVIE failed ({e}), resampling")
        continue

    if PLOT:
        fig, ax = plt.subplots()
        plot_projection(Ei.Q, Ei.c.flatten(), 1, ax, color='blue',   label='$\\mathcal{E}_i$')
        plot_projection(E0.Q, E0.c.flatten(), 1, ax, color='orange', label='$\\mathcal{E}_0$')
        ax.scatter(U[:, 0], U[:, 1], c='blue',   s=8, label='$\\mathcal{U}$')
        ax.scatter(W[:, 0], W[:, 1], c='green',  s=8, marker='x', label='$\\mathcal{W}$')
        ax.scatter(Z[:, 0], Z[:, 1], c='red',    s=8, label='$\\mathcal{Z}$')
        ax.scatter(V[:, 0], V[:, 1], c='yellow', s=8, marker='x', label='$\\mathcal{V}$')
        ax.set_title(f"Iteration {iteration}")
        ax.legend(fontsize=8)

        if sys.d > 2:
            fig3d = plt.figure()
            ax3d  = fig3d.add_subplot(111, projection='3d')
            ax3d.scatter(U[:, 0], U[:, 1], U[:, 2], c='blue',   s=8, label='$\\mathcal{U}$')
            ax3d.scatter(W[:, 0], W[:, 1], W[:, 2], c='green',  s=8, marker='x', label='$\\mathcal{W}$')
            ax3d.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c='red',    s=8, label='$\\mathcal{Z}$')
            ax3d.scatter(V[:, 0], V[:, 1], V[:, 2], c='yellow', s=8, marker='x', label='$\\mathcal{V}$')
            ax3d.set_title(f"Iteration {iteration} (3D)")
            ax3d.set_xlabel('$x_1$')
            ax3d.set_ylabel('$x_2$')
            ax3d.set_zlabel('$x_3$')
            ax3d.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

else:
    print(f"\nDid not converge within {MAX_ITER} iterations.")


# =========================================================================
# Result
# =========================================================================
E_star = Ei
print(f"\nFinal ellipsoid  Q =\n{E_star.Q}")
print(f"Final ellipsoid  c =\n{E_star.c.T}")
print(f"Volume: {E_star.volume():.6e}")


# =========================================================================
# Verification: sample from E* and check P(x) ∈ E*
# =========================================================================
N_verify = 5000
pts_v    = sample_from(E_star, N_verify).T
Ppts_v   = sys.P_batch(pts_v)
n_esc    = int(np.sum(~E_star.contains_point(Ppts_v.T)))
print(f"\nVerification ({N_verify} samples): {n_esc} escaped "
      f"({100*n_esc/N_verify:.2f}%)")
print("PASS" if n_esc == 0 else "FAIL — ellipsoid not verified invariant")


# =========================================================================
# Plot final ellipsoid projections
# =========================================================================
labels = ['$x_1$', '$x_2$', '$x_3$']
pairs  = [(0, 1), (0, 2), (1, 2)] if sys.d == 3 else [(0, 1)]

fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4))
if len(pairs) == 1:
    axes = [axes]

for ax, (i, j) in zip(axes, pairs):
    plot_projection(E_star.Q, E_star.c.flatten(), 1, ax,
                    diml=i, dimh=j, color='blue', label='$\\mathcal{E}^*$')
    plot_projection(E0.Q, E0.c.flatten(), 1, ax,
                    diml=i, dimh=j, color='orange', linestyle='--', label='$\\mathcal{E}_0$')
    ax.scatter(pts_v[:, i],  pts_v[:, j],  s=2, c='blue',  alpha=0.2, label='samples')
    ax.scatter(Ppts_v[:, i], Ppts_v[:, j], s=2, c='green', alpha=0.2, label='returns')
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])
    ax.legend(fontsize=7)

fig.suptitle('Final invariant ellipsoid $\\mathcal{E}^*$ (blue) vs initial $\\mathcal{E}_0$ (orange)')
plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"Ellipsoid_Inscribed_{timestamp}.txt", 'w') as f:
    f.write(f"Q =\n{E_star.Q}\n\nc =\n{E_star.c.T}\n")
plt.savefig(f"Ellipsoid_Inscribed_{timestamp}.png")
plt.show()

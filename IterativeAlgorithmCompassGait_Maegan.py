import numpy as np
from scipy.linalg import solve_discrete_lyapunov, sqrtm
from matplotlib import pyplot as plt
from ellipsoids import Ellipsoid, plot_projection, sample_from
import CompassGaitWalker
import ExpanderContractor


# =========================================================================
# Hyperparameters
# =========================================================================
N        = 5000   # samples per iteration
MAX_ITER = 200
PLOT     = False  # Shows the plot every iteration
RESCALE  = False # Rescale=True doesn't really work because it shrinks the set too fast

import jax
import jax.numpy as jnp
print("Using backend: ", jax.default_backend())

sys = CompassGaitWalker
# sys = ExpanderContractor

# =========================================================================
# Main Algorithm – Identify Invariant Ellipsoid
# =========================================================================

# Initial ellipsoid  E0 = { x : ||A0 x - b0|| <= 1 }
#   stored as (Q,c)
E0      = sys.initial_guess()

print(f"Initial ellipsoid volume: {E0.volume():.6e}")

# --- Line 3: initialise A, b, E -----------------------------------------
Ei = E0

# =========================================================================
# REPEAT ... UNTIL  V = ∅
# =========================================================================
for iteration in range(MAX_ITER):

    # --- Line 6-7: sample N points from E --------------------------------
    Xi = sample_from(Ei, N).T          # (N, d)

    # --- Line 9: apply Poincaré return map --------------------------------
    Xi_prime = sys.P_batch(Xi)             # (N, d)

    # --- Lines 11-13: partition ------------------------------------------
    in_E   = Ei.contains_point(Xi_prime.T)   # (N,) bool
    U      = Xi[in_E,        :]   # their preimages
    V      = Xi_prime[~in_E, :]   # escaped return points
    W      = Xi_prime[in_E,  :]   # surviving return points
    Z      = Xi[~in_E, :]         # preimage of escaped return points

    n_tot = len(W) + len(V)
    print(f"Iter {iteration:3d}: |V|={len(V):4d}/{n_tot}  "
          f"({100*len(V)/n_tot:.1f}% escaped)  vol={Ei.volume():.4e}")

    # --- Lines 14-19: update ellipsoid if any points escaped --------------
    if len(V) == 0:
        print(f"\nConverged — V = ∅  after {iteration} iterations.")
        break

    # fit MVEE to W (NOT UNIONED WITH U)
    WU   = np.vstack((W,U))
    VZ   = np.vstack((V,Z))
    Ei   = Ellipsoid(sys.d)
    try:
        Ei.fit_p_ball(U, 2) # Fits to preimage of returning points
    except np.linalg.LinAlgError:
        print("W not well-conditioned, resampling")
        continue
    # Ei.fit_p_ball(WU, 2)
    if PLOT:
        fig, ax = plt.subplots()
        plot_projection(Ei.Q, Ei.c.flatten(), 1, ax, color='blue',   label='$\\mathcal{E}_i$')

    # Rescale MVEE
    if RESCALE:
        distance_to_reject = Ei.mahalanobis_distance(Z.T)
        Ei.scale(min(distance_to_reject))
        if PLOT:
            plot_projection(Ei.Q, Ei.c.flatten(), 1, ax, color='cyan',   label='$\\mathcal{E}_i$ Rescale')

    if PLOT:
        # fig, ax = plt.subplots()
        plot_projection(E0.Q, E0.c.flatten(), 1, ax, color='orange', label='$\\mathcal{E}_0$')
        ax.scatter(U[:, 0], U[:, 1], c='blue',  s=8, label='$\\mathcal{U}$')
        ax.scatter(W[:, 0], W[:, 1], c='green', s=8, label='$\\mathcal{W}$')
        ax.scatter(V[:, 0], V[:, 1], c='red',   s=8, marker='x', label='$\\mathcal{V}$')

        ax.set_title(f"Iteration {iteration}")
        ax.legend(fontsize=8)

        if(sys.d > 2):
            fig3d = plt.figure()
            ax3d = fig3d.add_subplot(111, projection='3d')
            ax3d.scatter(U[:, 0], U[:, 1], U[:, 2], c='blue', s=8, label='$\\mathcal{U}$')
            ax3d.scatter(W[:, 0], W[:, 1], W[:, 2], c='green', s=8, label='$\\mathcal{W}$')
            ax3d.scatter(V[:, 0], V[:, 1], V[:, 2], c='red', s=8, marker='x', label='$\\mathcal{V}$')
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
# RETURN  E* = { x : ||A x - b|| <= 1 }
# =========================================================================
E_star = Ei
print(f"\nFinal ellipsoid  Q =\n{E_star.Q}")
print(f"Final ellipsoid  c =\n{E_star.c.T}")
print(f"Volume: {E_star.volume():.6e}")

# =========================================================================
# Verification: confirm P(x) ∈ E*  for  N_verify  samples from  E*
# =========================================================================
N_verify = 5000
pts_v    = sample_from(E_star, N_verify).T
Ppts_v   = sys.P_batch(pts_v)
n_esc    = int(np.sum(~E_star.contains_point(Ppts_v.T)))
print(f"\nVerification ({N_verify} samples): {n_esc} escaped "
      f"({100*n_esc/N_verify:.2f}%)")
print("PASS" if n_esc == 0 else "FAIL — ellipsoid not verified invariant")

# =========================================================================
# Plot final ellipsoid projections (all three 2-D coordinate pairs)
# =========================================================================
# TODO: Update this to work with either 2D or 3D
labels = ['$x_1$', '$x_2$', '$x_3$']
pairs  = [(0, 1), (0, 2), (1, 2)]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (i, j) in zip(axes, pairs):
    plot_projection(E_star.Q, E_star.c.flatten(), 1, ax,
                    diml=i, dimh=j, color='blue', label='$\\mathcal{E}^*$')
    plot_projection(E0.Q,    E0.c.flatten(),    1, ax,
                    diml=i, dimh=j, color='orange', linestyle='--', label='$\\mathcal{E}_0$')
    ax.scatter(pts_v[:, i], pts_v[:, j], s=2, c='blue',  alpha=0.2, label='samples')
    ax.scatter(Ppts_v[:, i], Ppts_v[:, j], s=2, c='green', alpha=0.2, label='returns')
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])
    ax.legend(fontsize=7)

fig.suptitle('Final invariant ellipsoid $\\mathcal{E}^*$ (blue) vs initial $\\mathcal{E}_0$ (orange)')
plt.tight_layout()
plt.savefig("Ellipsoid_Invariant.png")
plt.show()

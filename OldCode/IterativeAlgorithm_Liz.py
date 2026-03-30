import numpy as np
from scipy.linalg import solve_discrete_lyapunov, sqrtm
from matplotlib import pyplot as plt
from ellipsoids import Ellipsoid, plot_projection, sample_from
from radialBasisFunctions import sample_uniform_levelset, RBF
from probabilistic_bounds import binomial_tail
import ExpanderContractor
import DogBone
import jax
import jax.numpy as jnp
print("Using backend: ", jax.default_backend())


# =========================================================================
# Hyperparameters
# =========================================================================
N        = 2000   # samples per iteration
MAX_ITER = 200
PLOT     = False  # Shows the plot every iteration
RESCALE  = False # Rescale=True doesn't really work because it shrinks the set too fast

# =========================================================================
# Model Setup
# =========================================================================

sys = ExpanderContractor

# =========================================================================
# Main Algorithm – Identify Invariant Ellipsoid
# =========================================================================

# Initial ellipsoid  E0 = { x : ||A0 x - b0|| <= 1 }
# Initial guess is system-dependent
E0 = sys.initial_guess_rbf()

# print(f"Initial ellipsoid volume: {E0.volume():.6e}")

# --- Line 3: initialise A, b, E -----------------------------------------
Ei = E0
# =========================================================================
# REPEAT ... UNTIL  V = ∅
# =========================================================================
for iteration in range(MAX_ITER):

    # --- Line 6-7: sample N points from E --------------------------------
    Xi = sample_uniform_levelset(Ei, N)        # (N, d)

    # --- Line 9: apply Poincaré return map --------------------------------
    Xi_prime = sys.P_batch(Xi)             # (N, d)

    # --- Lines 11-13: partition ------------------------------------------
    in_E   = Ei.contains_point(Xi_prime)   # (N,) bool
    U      = Xi[in_E,        :]   # their preimages
    V      = Xi_prime[~in_E, :]   # escaped return points
    W      = Xi_prime[in_E,  :]   # surviving return points
    Z      = Xi[~in_E, :]         # preimage of escaped return points
    n_tot = len(W) + len(V)
    curr_misses = len(V)
    n_esc    = int(np.sum(~Ei.contains_point(Xi_prime)))
    print(f"Iter {iteration:3d}: |V|={len(V):4d}/{n_tot}  "
          f"({100*len(V)/n_tot:.1f}% escaped) ")

    # --- Lines 14-19: update ellipsoid if any points escaped --------------
    epsilon = binomial_tail(n_esc, N)
    print(f"Iter {iteration:3d}: epsilon={epsilon}")
    # do active verification so we can use as a termination criteria
    if epsilon < 0.05:
        print(f"\nConverged — V = ∅  after {iteration} iterations,  with epsilon = {epsilon}")
        break

    # fit MVEE to W (NOT UNIONED WITH U)
    WU   = np.vstack((W,U))
    VZ   = np.vstack((V,Z))
    Ei   = RBF(sys.d, np.log(0.25))
    try:
        Ei.fit_rbf(U, 3)  
    except np.linalg.LinAlgError:
        print("W not well-conditioned, resampling")
        continue
    # Ei.fit_p_ball(WU, 2)
    if PLOT:
        fig, ax = plt.subplots()
        ax.scatter(U[:,0],U[:,1],c='blue')
        Ei.plot_2D_rbf(-8,8,-8,8,Ei.centers_opt,Ei.widths_opt, colors=['cyan'])
        # plot_projection(Ei.Q, Ei.c.flatten(), 1, ax, color='blue',   label='$\\mathcal{E}_i$')

    if PLOT:
        # fig, ax = plt.subplots()
        E0.plot_2D_rbf(-8,8,-8,8,E0.centers_opt,E0.widths_opt, colors=['orange'])
        ax.scatter(U[:, 0], U[:, 1], c='blue',  s=8, label='$\\mathcal{U}$')
        ax.scatter(W[:, 0], W[:, 1], c='green', s=8, marker='x', label='$\\mathcal{W}$')
        ax.scatter(Z[:, 0], Z[:, 1], c='red',   s=8, label='$\\mathcal{Z}$')
        ax.scatter(V[:, 0], V[:, 1], c='yellow',   s=8, marker='x', label='$\\mathcal{V}$')

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
print(f"\nFinal rbf  centers =\n{E_star.centers_opt}")
print(f"Final rbf  widths =\n{E_star.widths_opt}")

# =========================================================================
# Verification: confirm P(x) ∈ E*  for  N_verify  samples from  E*
# =========================================================================
# N_verify = 5000
# pts_v    = sample_uniform_levelset(E_star, N_verify)
# Ppts_v   = sys.P_batch(pts_v)
# n_esc    = int(np.sum(~E_star.contains_point(Ppts_v.T)))
# print(f"\nVerification ({N_verify} samples): {n_esc} escaped "
#       f"({100*n_esc/N_verify:.2f}%)")
# print("PASS" if n_esc == 0 else "FAIL — ellipsoid not verified invariant")

# =========================================================================
# Plot final ellipsoid projections (all three 2-D coordinate pairs)
# =========================================================================
# fig, axes = plt.subplots(1, 1, figsize=(12, 4))
# E_star.plot_2D_rbf(-8,8,-8,8,E_star.centers_opt,E_star.widths_opt, colors=['blue'])
# E0.plot_2D_rbf(-8,8,-8,8,E0.centers_opt,E0.widths_opt, colors=['orange'])
# ax.scatter(pts_v[:, 0], pts_v[:, 1], s=2, c='blue',  alpha=0.2, label='samples')
# ax.scatter(Ppts_v[:, 0], Ppts_v[:, 1], s=2, c='green', alpha=0.2, label='returns')
# ax.legend(fontsize=7)

# fig.suptitle('Final invariant ellipsoid $\\mathcal{E}^*$ (blue) vs initial $\\mathcal{E}_0$ (orange)')
# plt.tight_layout()
# plt.savefig("Ellipsoid_Invariant.png")
# plt.show()

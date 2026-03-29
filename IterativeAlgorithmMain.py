import numpy as np
from scipy.linalg import solve_discrete_lyapunov, sqrtm
from matplotlib import pyplot as plt
from ellipsoids import Ellipsoid, plot_projection, sample_from
import CompassGaitWalker
import ExpanderContractor
import DogBone
import jax
from datetime import datetime
import jax.numpy as jnp
from probabilistic_bounds import binomial_tail
import pandas as pd
print("Using backend: ", jax.default_backend())


# =========================================================================
# Hyperparameters
# =========================================================================
N        = 2000   # samples per iteration
MAX_ITER = 200
PLOT     = False  # Shows the plot every iteration
RESCALE  = False # Rescale=True doesn't really work because it shrinks the set too fast
EPS_THRESHOLD = 0.03 # It doesn't really converge any lower than this

# =========================================================================
# Model Setup
# =========================================================================

sys = CompassGaitWalker
# sys = ExpanderContractor
# sys = DogBone

# =========================================================================
# Main Algorithm – Identify Invariant Ellipsoid
# =========================================================================

# Initial ellipsoid  E0 = { x : ||A0 x - b0|| <= 1 }
# Initial guess is system-dependent
E0      = sys.initial_guess()

print(f"Initial ellipsoid volume: {E0.volume():.6e}")

# --- Line 3: initialise A, b, E -----------------------------------------
Ei = E0

log_eps  = []
log_vol  = []
log_Q    = []   # flattened Q matrix at each iteration
log_c    = []   # flattened c vector at each iteration

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
    n_esc    = int(np.sum(~in_E))
    print(f"Iter {iteration:3d}: |V|={len(V):4d}/{n_tot}  "
          f"({100*len(V)/n_tot:.1f}% escaped)  vol={Ei.volume():.4e}")

    # --- Lines 14-19: update ellipsoid if any points escaped --------------
    epsilon = binomial_tail(n_esc, N)
    print(f"Iter {iteration:3d}: epsilon={epsilon}")
    # do active verification so we can use as a termination criteria

    log_eps.append(epsilon)
    log_vol.append(Ei.volume())
    log_Q.append(Ei.Q.flatten())
    log_c.append(Ei.c.flatten())

    if epsilon < EPS_THRESHOLD:
        print(f"\nConverged after {iteration} iterations,  with epsilon = {epsilon}")
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

    distance_to_reject = Ei.mahalanobis_distance(VZ.T)
    # print(min(distance_to_reject)) # If this is > 1 then there are points in V union Z that are inside the new ellipsoid
    # Rescale MVEE
    if RESCALE:
        Ei.scale(min(distance_to_reject)-1e-5)
        distance_to_reject = Ei.mahalanobis_distance(VZ.T)
        # print(distance_to_reject)
        assert min(distance_to_reject) > 1
        assert np.all(Ei.contains_point(VZ.T) == False)
        if PLOT:
            plot_projection(Ei.Q, Ei.c.flatten(), 1, ax, color='cyan',   label='$\\mathcal{E}_i$ Rescale')

    if PLOT:
        # fig, ax = plt.subplots()
        # Blue goes to Green, Red goes to Yellow
        plot_projection(E0.Q, E0.c.flatten(), 1, ax, color='orange', label='$\\mathcal{E}_0$')
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
            ax3d.scatter(W[:, 0], W[:, 1], W[:, 2], c='green', s=8, marker='x', label='$\\mathcal{W}$')
            ax3d.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c='red', s=8, label='$\\mathcal{Z}$')
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

## Save data to CSVs
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
sysname    = sys.__name__
df = pd.DataFrame({'epsilon': log_eps, 'volume': log_vol})
df.to_csv(f"{sysname}_convergence_log_{timestamp}.csv", index=False)

d      = sys.d
q_cols = [f'q{i}{j}' for i in range(d) for j in range(d)]
c_cols = [f'c{k}'    for k in range(d)]
df_ellipsoids = pd.DataFrame(
    np.hstack([np.array(log_Q), np.array(log_c)]),
    columns=q_cols + c_cols,
)
df_ellipsoids.index.name = 'iteration'
df_ellipsoids.to_csv(f"{sysname}_ellipsoid_log_{timestamp}.csv")

# =========================================================================
# Plot final ellipsoid projections (all 2-D coordinate pairs)
# =========================================================================
labels = [f'$x_{k+1}$' for k in range(sys.d)]
pairs  = [(i, j) for i in range(sys.d) for j in range(i + 1, sys.d)]

fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4))
if len(pairs) == 1:
    axes = [axes]
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
with open(f"Ellipsoid_Invariant_{timestamp}.txt", 'w') as f:
    f.write(f"Q =\n{E_star.Q}\n\nc =\n{E_star.c.T}\n")
plt.savefig(f"Ellipsoid_Invariant_{timestamp}.png")
plt.show()
import sys, os
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, sqrtm
from matplotlib import pyplot as plt
from sets.radialBasisFunction import RBF
from sets.ellipsoid import Ellipsoid, plot_projection
import Systems.CompassGaitWalker as CompassGaitWalker
import Systems.ExpanderContractor as ExpanderContractor
import Systems.DogBone as DogBone
import jax
from datetime import datetime
import jax.numpy as jnp
from probabilistic_bounds import binomial_tail
import pandas as pd
import warnings
print("Using backend: ", jax.default_backend())


# =========================================================================
# Hyperparameters
# =========================================================================
N        = 2000   # samples per iteration
MAX_ITER = 200
PLOT     = False  # Shows the plot every iteration
RESCALE  = False # Rescale=True doesn't really work because it shrinks the set too fast
EPS_THRESHOLD = 0.00 # User-defined accuracy threshold for convergence
SAVE_DATA = False # Save convergence data to CSVs

# =========================================================================
# Model Setup
# =========================================================================

# sys = CompassGaitWalker
sys = ExpanderContractor
# sys = DogBone

# =========================================================================
# Main Algorithm – Identify Invariant Ellipsoid
# =========================================================================

def main():
    # TODO: Instantiate sys and E0 based on kwargs
    E0 = sys.initial_guess()
    Ei = E0

    threshold = EPS_THRESHOLD

    log_eps  = []
    log_vol  = []
    log_theta    = []   # collects all parameters
    epsilon_min = binomial_tail(0, N)
    if(threshold < epsilon_min):
        warnings.warn(f"Threshold must be greater than {epsilon_min}, setting to epsilon_min")
        threshold = epsilon_min

    for iteration in range(MAX_ITER):

        # --- Line 6-7: sample N points from E --------------------------------
        Xi = Ei.sample_from(N)          # (N, d)

        # --- Line 9: apply Poincaré return map --------------------------------
        Xi_prime = sys.P_batch(Xi)             # (N, d)

        # --- Lines 11-13: partition ------------------------------------------
        in_E   = Ei.contains_points(Xi_prime)   # (N,) bool
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
        log_theta.append(np.asarray(Ei.params()).flatten().copy())

        if epsilon <= threshold:
            print(f"\nConverged after {iteration} iterations,  with epsilon = {epsilon}")
            break

        # fit MVEE to W (NOT UNIONED WITH U)
        WU   = np.vstack((W,U))
        VZ   = np.vstack((V,Z))
        Ei   = Ellipsoid(sys.d)
        try:
            Ei.fit(U, 2) # Fits to preimage of returning points
        except np.linalg.LinAlgError:
            print("W not well-conditioned, resampling")
            continue
        # Ei.fit(WU, 2)
        if PLOT:
            fig, ax = plt.subplots()
            Ei.plot(ax, color='blue',   label='$\\mathcal{E}_i$')

            # fig, ax = plt.subplots()
            # Blue goes to Green, Red goes to Yellow
            E0.plot(ax, color='orange', label='$\\mathcal{E}_0$')
            ax.scatter(U[:, 0], U[:, 1], c='blue',  s=8, label='$\\mathcal{U}$')
            ax.scatter(W[:, 0], W[:, 1], c='green', s=8, marker='x', label='$\\mathcal{W}$')
            ax.scatter(Z[:, 0], Z[:, 1], c='red',   s=8, label='$\\mathcal{Z}$')
            ax.scatter(V[:, 0], V[:, 1], c='yellow',   s=8, marker='x', label='$\\mathcal{V}$')

            ax.set_title(f"Iteration {iteration}")
            ax.legend(fontsize=8)

            if(sys.d == 3):
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
    pts_v    = E_star.sample_from(N_verify)
    Ppts_v   = sys.P_batch(pts_v)
    n_esc    = int(np.sum(~E_star.contains_points(Ppts_v)))
    print(f"\nVerification ({N_verify} samples): {n_esc} escaped "
        f"({100*n_esc/N_verify:.2f}%)")
    print("PASS" if n_esc == 0 else "FAIL — ellipsoid not verified invariant")

    ## Save data to CSVs
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    sysname    = sys.__name__

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
        sys.plot_true(ax, diml=i, dimh=j, label="True")
        ax.scatter(pts_v[:, i], pts_v[:, j], s=2, c='blue',  alpha=0.2, label='samples')
        ax.scatter(Ppts_v[:, i], Ppts_v[:, j], s=2, c='green', alpha=0.2, label='returns')
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.legend(fontsize=7)
    fig.suptitle('Final invariant ellipsoid $\\mathcal{E}^*$ (blue) vs initial $\\mathcal{E}_0$ (orange)')
    plt.tight_layout()
    # Save data to CSVs and other logs
    if(SAVE_DATA):
        df = pd.DataFrame({'epsilon': log_eps, 'volume': log_vol})

        set_type = type(E_star).__name__.lower()
        df_params = pd.DataFrame(log_theta)
        df_params.columns = [f'param_{k}' for k in range(df_params.shape[1])]
        df_params.index.name = 'iteration'
        df.to_csv(f"{sysname}_{set_type}_convergence_log_{timestamp}.csv", index=False)
        df_params.to_csv(f"{sysname}_{set_type}_log_{timestamp}.csv")
        with open(f"{sysname}_{set_type}_invariant_{timestamp}.txt", 'w') as f:
            f.write(f"params =\n{E_star.params()}\n")
        plt.savefig(f"{sysname}_{set_type}_invariant_{timestamp}.png")
    plt.show()


if __name__ == '__main__':
    main()
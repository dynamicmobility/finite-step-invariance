import argparse
import importlib
import numpy as np
from matplotlib import pyplot as plt
from sets.radialBasisFunction import RBF
from sets.ellipsoid import Ellipsoid
import jax
from datetime import datetime
from probabilistic_bounds import binomial_tail
import pandas as pd
import warnings



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

SYSTEMS = {
    'cec': 'Systems.ExpanderContractor',
    'nec': 'Systems.DogBone',
    'cgw': 'Systems.CompassGaitWalker',
}




def plot_set(set_, ax, diml=0, dimh=1, color='blue', label=None, **kwargs):
    return set_.plot(ax, diml=diml, dimh=dimh, color=color, label=label, **kwargs)


def main(args):
    sys = importlib.import_module(SYSTEMS[args.system])
    print("Using backend: ", jax.default_backend())
    N = args.N
    threshold = args.eps_thresh
    E0 = sys.initial_guess(args.set_type)
    Ei = E0

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
            f"({100*len(V)/n_tot:.1f}% escaped)  vol={Ei.volume()}")

        # --- Lines 14-19: update ellipsoid if any points escaped --------------
        epsilon = binomial_tail(n_esc, N)

        print(f"Iter {iteration:3d}: epsilon={epsilon}")
        # do active verification so we can use as a termination criteria

        log_eps.append(epsilon)
        log_vol.append(Ei.volume() if isinstance(Ei, Ellipsoid) else np.nan)
        log_theta.append(np.asarray(Ei.params()).flatten().copy())

        if epsilon <= threshold:
            print(f"\nConverged after {iteration} iterations,  with epsilon = {epsilon}")
            break

        # fit MVEE to W (NOT UNIONED WITH U)
        WU   = np.vstack((W,U))
        VZ   = np.vstack((V,Z))
        candidate = (Ellipsoid(sys.d) if args.set_type == 'ell'
                     else RBF(sys.d))
        try:
            candidate.fit(U) # Fits to preimage of returning points
        except np.linalg.LinAlgError:
            print("W not well-conditioned, resampling")
            continue
        Ei = candidate
        # Ei.fit(WU, 2)
        if PLOT:
            fig, ax = plt.subplots()
            plot_set(Ei, ax, color='blue',   label='$\\mathcal{E}_i$')

            # fig, ax = plt.subplots()
            # Blue goes to Green, Red goes to Yellow
            plot_set(E0, ax, color='orange', label='$\\mathcal{E}_0$')
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
    print(f"\nFinal {args.set_type} params =\n{E_star.params()}")
    if isinstance(E_star, Ellipsoid):
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
    print("PASS" if n_esc == 0 else "FAIL — set not verified invariant")

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
        plot_set(E_star, ax,
                        diml=i, dimh=j, color='blue', label='$\\mathcal{E}^*$')
        plot_set(E0, ax,
                        diml=i, dimh=j, color='orange', linestyle='--', label='$\\mathcal{E}_0$')
        if hasattr(sys, "plot_true"):
            sys.plot_true(ax, diml=i, dimh=j, label="True")
        ax.scatter(pts_v[:, i], pts_v[:, j], s=2, c='blue',  alpha=0.2, label='samples')
        ax.scatter(Ppts_v[:, i], Ppts_v[:, j], s=2, c='green', alpha=0.2, label='returns')
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.legend(fontsize=7)
    fig.suptitle(f'Final invariant {args.set_type} (blue) vs initial set (orange)')
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
    parser = argparse.ArgumentParser(description="Identify a probabilistically invariant set.")
    parser.add_argument('--N', '--n', type=int, default=N,
                        help='Samples per iteration (default: %(default)s)')
    parser.add_argument('--eps-thresh', '--eps_thresh', type=float, default=EPS_THRESHOLD,
                        help='Convergence threshold in [0, 1] (default: %(default)s)')
    parser.add_argument('--set-type', '--set_type', type=str.lower,
                        choices=['ell', 'rbf'], default='ell')
    parser.add_argument('--system', type=str.lower, choices=SYSTEMS,
                        default='cec')
    args = parser.parse_args()
    if args.N <= 0:
        parser.error('N must be positive')
    if not 0 <= args.eps_thresh <= 1:
        parser.error('eps_thresh must be in [0, 1]')
    main(args)

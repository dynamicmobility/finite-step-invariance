import numpy as np
from scipy.linalg import solve_discrete_lyapunov, sqrtm
from matplotlib import pyplot as plt
from ellipsoids import Ellipsoid, plot_projection, sample_from
from CompassGaitWalker import CompassGaitWalker
from diffrax import diffeqsolve, ODETerm, Dopri8, Event
from equinox import filter_jit
import optimistix as optx


# =========================================================================
# Hyperparameters
# =========================================================================
N        = 5000   # samples per iteration
MAX_ITER = 30
PLOT     = True  # Shows the plot every iteration
INIT_GUESS = "Contraction" # Either choose from uniform distribution or guess Q from contraction metric "Contraction" or "Uniform"
RESCALE  = False # Rescale=True doesn't really work because it shrinks the set too fast

import jax
import jax.numpy as jnp
print("Using backend: ", jax.default_backend())

# =========================================================================
# System and ODE solver
# =========================================================================
sys    = CompassGaitWalker(m=5., mh=10., a=0.5, b=0.5, gamma=0.0525)
x_star = np.array([-0.21876908, 0.32376908, 1.8083619, 1.49571536])  # fixed point
assert jnp.abs(sys.guard(0, x_star)) < 1e-5, "x_star is not on the guard"

term   = ODETerm(sys.f)
solver = Dopri8()
u      = 0.0
event  = Event(sys.guard, optx.Newton(5e-7, 5e-7, optx.rms_norm), direction=True)

# =========================================================================
# Guard surface chart: 3-D parameterisation of the 4-D guard surface
#   full_state = guard_surface_chart @ guard_coords + x_star
# =========================================================================
_c                 = jnp.array([[1, 1, 0, 0]]).T
_guard_chart       = jnp.array([[1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T
_guard_chart_inv   = jnp.linalg.inv(jnp.hstack((_guard_chart, _c)))

def guard_to_full(x_guard):
    """(3,) or (N,3) guard coords -> (4,) or (N,4) full state"""
    return (_guard_chart @ x_guard.T).T + x_star

def full_to_guard(x_full):
    """(4,) or (N,4) full state -> (3,) or (N,3) guard coords"""
    return ((_guard_chart_inv @ (x_full - x_star).T).T)[..., :3]

# =========================================================================
# Poincaré return map  P : S -> S  (guard coords -> guard coords)
# =========================================================================
@filter_jit
def P_single(x_guard):
    x_full = guard_to_full(x_guard)
    xp     = sys.reset(0, x_full)
    traj   = diffeqsolve(term, solver, t0=0, t1=2, dt0=0.001,
                         y0=xp, args=u, event=event, throw=False)
    return full_to_guard(traj.ys.flatten())

P_batch = jax.vmap(P_single, in_axes=0)   # (N,3) -> (N,3)

# sanity-check: fixed point maps to itself
assert jnp.allclose(P_single(jnp.zeros(3)), jnp.zeros(3), atol=5e-4), \
    "Fixed point does not map to itself — check system / solver"

# =========================================================================
# INPUTS  (x* = 0 in guard coords by construction)
# =========================================================================
x_star_guard = np.zeros(3)

J = np.array(jax.jacrev(P_single)(jnp.zeros(3)))   # (3,3)

eigs = np.abs(np.linalg.eigvals(J))
print("Jacobian eigenvalues:", eigs)
assert np.all(eigs < 1), "Limit cycle is not stable — Lyapunov equation has no PD solution"

# =========================================================================
# Main Algorithm – Identify Invariant Ellipsoid
# =========================================================================

# --- Line 1: solve  J^T P J - P = -Q  for  P > 0 -----------------------
#   scipy.solve_discrete_lyapunov(A, Q) solves  A X A^T - X + Q = 0
#   => set A = J^T  =>  J^T P J - P + Q = 0  =>  J^T P J - P = -Q  ✓
# P_lyap = solve_discrete_lyapunov(J.T, Q_lyap)
# assert np.all(np.linalg.eigvalsh(P_lyap) > 0), "P is not positive definite"

# Contraction-based procedure to compute initial guess set



# Initial ellipsoid  E0 = { x : ||A0 x - b0|| <= 1 }
#   stored as (Q,c) with Q = A0^T A0 = P_lyap, c = 0
E0      = Ellipsoid(3)

if INIT_GUESS == "Uniform":
    pts_0 = np.random.uniform([-0.1, -1, -0.3], [0.1, 1, 0.3], (N,3))
    E0.fit_p_ball(pts_0, 2)

elif INIT_GUESS == "Contraction":
    import cvxpy as cp

    # Contraction rate is the spectral radius of A
    contr_rate = jnp.max(eigs) + 1e-4
    print(contr_rate)
    P_cp = cp.Variable(J.shape, symmetric=True)
    cons = []
    cons.append(J.T @ P_cp @ J << contr_rate**2*P_cp)
    cons.append(P_cp >> jnp.eye(J.shape[0]))
    prob = cp.Problem(cp.Minimize(cp.trace(P_cp)), cons)
    prob.solve()

    Q0 = P_cp.value
    E0.Q    = Q0.copy() # Make E0 bigger to see if we get the same result
    E0.c    = np.zeros((3, 1))

print(f"Initial ellipsoid volume: {E0.volume():.6e}")

# --- Line 3: initialise A, b, E -----------------------------------------
Ei = E0

# =========================================================================
# REPEAT ... UNTIL  V = ∅
# =========================================================================
for iteration in range(MAX_ITER):

    # --- Line 6-7: sample N points from E --------------------------------
    Xi = sample_from(Ei, N).T          # (N, 3)

    # --- Line 9: apply Poincaré return map --------------------------------
    Xi_prime = P_batch(Xi)             # (N, 3)

    # --- Lines 11-13: partition ------------------------------------------
    in_E   = Ei.contains_point(Xi_prime.T)   # (N,) bool
    W      = Xi_prime[in_E,  :]   # surviving return points
    U      = Xi[in_E,        :]   # their preimages
    V      = Xi_prime[~in_E, :]   # escaped return points

    n_tot = len(W) + len(V)
    print(f"Iter {iteration:3d}: |V|={len(V):4d}/{n_tot}  "
          f"({100*len(V)/n_tot:.1f}% escaped)  vol={Ei.volume():.4e}")

    # --- Lines 14-19: update ellipsoid if any points escaped --------------
    if len(V) == 0:
        print(f"\nConverged — V = ∅  after {iteration} iterations.")
        break

    # fit MVEE to W (NOT UNIONED WITH U)
    WU   = np.vstack((W,U))
    Ei   = Ellipsoid(3)
    try:
        Ei.fit_p_ball(W, 2)
    except np.linalg.LinAlgError:
        print("W not well-conditioned, resampling")
        continue
    # Ei.fit_p_ball(WU, 2)
    if PLOT:
        fig, ax = plt.subplots()
        plot_projection(Ei.Q, Ei.c.flatten(), 1, ax, color='blue',   label='$\\mathcal{E}_i$')

    # Rescale MVEE
    if RESCALE:
        distance_to_reject = Ei.mahalanobis_distance(V.T)
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
Ppts_v   = P_batch(pts_v)
n_esc    = int(np.sum(~E_star.contains_point(Ppts_v.T)))
print(f"\nVerification ({N_verify} samples): {n_esc} escaped "
      f"({100*n_esc/N_verify:.2f}%)")
print("PASS" if n_esc == 0 else "FAIL — ellipsoid not verified invariant")

# =========================================================================
# Plot final ellipsoid projections (all three 2-D coordinate pairs)
# =========================================================================
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

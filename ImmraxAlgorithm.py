import numpy as np
import jax.numpy as jnp
import immrax as irx
import matplotlib.pyplot as plt
from pathlib import Path
from diffrax import SaveAt
from Systems.CompassGaitWalker import *
from Systems.CompassGaitWalker import _guard_chart, _guard_chart_inv
from diffrax import Euler

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

NT = irx.L2Normotope

# Build continuous dynamics and reset function
class ContSys(irx.System):
    def __init__ (self) :
        self.xlen = 4
        self.evolution = 'continuous'
    def f (self, t, x, *args) :
        return sys.f(t, x, *args)

def make_rollout(f, N):
    def rollout(x0, dt):
        def f_euler(xt, _):
            xtp1 = xt + f(0.0, xt, 0.0)*dt
            return (xtp1, xtp1)
        _, x = jax.lax.scan(
            f_euler, x0, length=N
        )
        return jnp.vstack((x0, x))
    return rollout

cont_sys = ContSys()
reset = sys.reset

# Build nominal trajectory (integrate dynamics from x_star)
# x0     = np.array([0.30992698, -0.21464522,  0.41259121,  1.07173842])
x0     = np.array([0.32357214, -0.21857214,  0.3706022,   1.08981166])
dt     = 0.001005217149786066
N      = 734 # Nominal trajectory hits guard at 0.7344544
EXTRA  = 50
rollout_nom = make_rollout(cont_sys.f, N+EXTRA)


nom = rollout_nom(x0, dt)
assert len(nom) == N+EXTRA+1
# WARNING: Code assumes _basis_reset(0) = x0, or reset(x[N]) = x0.

print("Reset: ", reset(0, nom[N]), " x0: ", x0)
assert np.allclose(reset(0, nom[N]), x0)
assert np.allclose(sys.guard(0, nom[N]), 0, atol=1e-7)
xf = nom[N]

# Phase plot: swing angle versus stance angle.
# fig, ax = plt.subplots()
# ax.plot(nom[:, 0], nom[:, 2])
# ax.plot(nom[:, 1], nom[:, 3])
# ax.set_xlabel(r"$\theta$ (rad)")
# ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
# ax.set_title("Nominal trajectory")
# ax.grid(True)
# fig.tight_layout()
# plt.show()



# Build sys_mjacM function

iz = irx.icentpert(jnp.zeros(4), 0.1)

sys_mjacM = irx.mjacM(sys.f)
izero = irx.interval(jnp.array([0.]))

perm = irx.Permutation((0, 3, 4, 1, 2, 5))
Mt, Mx, Mu = sys_mjacM(izero, iz + x0, izero + 1., permutations=(perm,))[0]

Mx = irx.interval(Mx); Mu = irx.interval(Mu)

gsc = irx.utils.get_sparse_corners(Mx)
gsc_Mu = irx.utils.get_sparse_corners(Mu)

print(len(gsc(Mx)), 'sparse corners in Mx')
print(len(gsc_Mu(Mu)), 'sparse corners in Mu')


# Build rollout Embsys function
At = jax.jit(jax.jacfwd(sys.f, argnums=1))

@jax.jit
def rollout_embsys (nt0) :
    init = nt0.vec()
    def emb_euler (state, _) :
        nt = NT.unvec(state)

        ix = nt.iover()

        A = At(0., nt.ox, 0.0)


        MM = sys_mjacM(0., ix, 0.0, centers=((jnp.zeros(1), nt.ox, 0.0),), 
                       permutations=(perm,))[0]

        Mx = MM[1]

        alpha_inv = nt.alpha_inv

        alpha_dot = -nt.alpha @A
        growth = jnp.max(jnp.asarray([NT.mu(alpha_dot@alpha_inv + nt.alpha@M@alpha_inv) for M in gsc(Mx)]))
        y_dot = growth * nt.y

        ox_dot = sys.f(0., nt.ox, 0.0).reshape(-1)
        nt_next = NT(nt.ox + dt*ox_dot, nt.alpha + dt*alpha_dot, nt.y + dt*y_dot)

        return (nt_next.vec(), nt_next.vec())

    _, state_traj = jax.lax.scan(emb_euler, init, length = N+EXTRA)

    return jnp.vstack((nt0.vec(), state_traj))

nt0 = NT(x0, jnp.eye(4)*500., 1.)

nt_traj = rollout_embsys(nt0)

test_basis_reset = lambda x : reset(0, _guard_chart @ x + xf)
test_mjac_basis_reset = irx.mjacM(test_basis_reset)
test_M_reset = test_mjac_basis_reset(irx.icentpert(jnp.zeros(3), 1.), permutations=(irx.Permutation((1,2,0)),))[0][0]
# print(test_M_reset)
M_reset_gsc = irx.utils.get_sparse_corners(test_M_reset)

# _nt_traj = rollout_embsys(nt0)

# # Preimpact normotope: [ox(T), alpha(T), y(T)]
# _nt_pre = NT.unvec(_nt_traj[N])
# assert np.allclose(_nt_pre.ox, xf)

# Build alpha_0_to_max_mjac_length

def normotope_guard_height_bounds(nt):
    """Exact extrema of h = 2*gamma - tsw - tst over an L2 normotope."""
    normal = jnp.array([-1., -1., 0., 0.])
    direction = jnp.linalg.solve(nt.alpha.T, normal)
    center_height = normal @ nt.ox + 2*sys.gamma
    radius = nt.y*jnp.linalg.norm(direction)
    return center_height - radius, center_height + radius


def find_t_underbar_index(nt_traj):
    """First sample entirely in h > 0, or len(nt_traj) if none exists.

    Use h = 2*gamma - tsw - tst, decreasing through zero at impact.
    This checks sampled normotopes, not the paper's transversality conditions.
    """
    def height_lower_bound(state):
        return normotope_guard_height_bounds(NT.unvec(state))[0]

    lower_bounds = jax.vmap(height_lower_bound)(nt_traj)
    # T_underbar must precede the final rollout sample.
    above_guard = (lower_bounds > 0.) & (jnp.arange(len(nt_traj)) < len(nt_traj)-1)
    return jnp.where(jnp.any(above_guard), jnp.argmax(above_guard), len(nt_traj))


def alpha0_to_mjac_lengths_slice (alpha0, nt_traj=None) :
    _nt_traj = rollout_embsys(NT(x0, alpha0, 1.)) if nt_traj is None else nt_traj
    t_underbar_index = find_t_underbar_index(_nt_traj)

    # Preimpact normotope: [ox(T), alpha(T), y(T)]
    _nt_pre = NT.unvec(_nt_traj[N])

    # \Delta^B (z) = \Delta(B(z) + ox(T))
    _basis_reset = lambda z : reset(0, _guard_chart @ z + _nt_pre.ox)

    def slice_nt (nt) :
        nt = NT.unvec(nt)

        ## Step 1: compute the sliced normotope

        # alpha B = QR 
        Q, R = jnp.linalg.qr(nt.alpha @ _guard_chart)

        # oz = R^{-1} Q^T alpha (ox - ox(T))
        oz = jax.scipy.linalg.solve_triangular(R, Q.T @ nt.alpha @ (nt.ox - _nt_pre.ox))

        # r = |alpha (ox - ox(T))|_2^2 - |Q^T alpha (ox - ox(T))|
        r = jnp.sum((nt.alpha @ (nt.ox - _nt_pre.ox))**2) - jnp.sum((Q.T @ (nt.alpha @ (nt.ox - _nt_pre.ox)))**2)

        # No intersection when y^2 < r
        empty = nt.y**2 < r
    
        def _empty () :
            # no-op the rest if the slice is empty
            return -jnp.inf

        def _nonempty () :
            # If not empty, the slice is [oz, R, sqrt(y^2 - r)]_2 (in the basis B)
            sliced = NT(oz, R, jnp.sqrt(nt.y**2 - r))

            ## Step 2: Overapproximate the reset map in the basis.
            mjac_basis_reset = irx.mjacM(_basis_reset)
            M_reset = mjac_basis_reset(sliced.iover(), centers=((sliced.ox,),), permutations=(irx.Permutation((1,2,0)),))[0][0]
            # M_reset = mjac_basis_reset(sliced.iover(), center=(sliced.ox,), permutation=irx.Permutation((1,2,0)))[0]

            def _M_to_length (M) :
                # |alpha0 M R^{-1}| sqrt{y^2 - r} + |alpha0(Delta^B(oz) - x^*)|
                # WARNING: Code assumes _basis_reset(0) = x0, or reset(x(T)) = x0.
                return NT.induced_norm(alpha0 @ M @ sliced.alpha_inv) * sliced.y + jnp.linalg.norm(alpha0 @ (_basis_reset(sliced.ox) - x0))

            # Cannot use Rohn because this is a R^{4\times 3} induced norm.
            corners = jax.vmap(_M_to_length)(jnp.asarray(M_reset_gsc(M_reset)))

            return jnp.max(corners)

        return jax.lax.cond(empty, _empty, _nonempty)

    def slice_after_t_underbar(index, state):
        return jax.lax.cond(
            index >= t_underbar_index,
            slice_nt,
            lambda _: -jnp.inf,
            state,
        )

    return jax.lax.cond(
        t_underbar_index < len(_nt_traj),
        lambda: jax.vmap(slice_after_t_underbar)(jnp.arange(len(_nt_traj)), _nt_traj),
        # No qualifying T_underbar means this candidate cannot be verified.
        lambda: jnp.full((len(_nt_traj),), jnp.inf, dtype=_nt_traj.dtype),
    )

@jax.jit
def alpha0_to_max_mjac_length (alpha0) :
    nt_traj = rollout_embsys(NT(x0, alpha0, 1.))
    # The final sample is T_overbar: the entire set must lie in h < 0.
    # Disjointness alone could also mean the set has not reached the guard.
    _, final_height_upper = normotope_guard_height_bounds(NT.unvec(nt_traj[-1]))

    def reset_bound():
        lengths = alpha0_to_mjac_lengths_slice(alpha0, nt_traj=nt_traj)
        return jnp.max(lengths)

    return jax.lax.cond(
        jnp.isfinite(final_height_upper) & (final_height_upper < 0.),
        reset_bound,
        lambda: jnp.asarray(jnp.inf, dtype=nt_traj.dtype),
    )

print("Test alpha0_to_max_mjac_length: ", alpha0_to_max_mjac_length(nt0.alpha))

# Initial Guess
import cvxpy as cp
from scipy.linalg import sqrtm
# This may need to be replace with the diffrax implementation to account for saltation matrix shenanigans

def one_step(x0):
    """Integrate from a post-impact state to the guard, then apply the reset."""
    term   = ODETerm(sys.f)
    solver = Euler()
    u      = 0.0
    event  = Event(sys.guard, optx.Newton(5e-7, 5e-7, optx.rms_norm), direction=True)
    traj = diffeqsolve(term, solver, t0=0, t1=2, dt0=dt,
                      event=event, y0=x0, args=u, saveat=SaveAt(t1=True))
    xf = traj.ys[-1]
    return sys.reset(traj.ts[-1], xf)
print("One step: ", one_step(x0), " x0: ", x0)
assert np.allclose(one_step(x0), x0)

A_cl_plus = jax.jacrev(one_step)(x0)
print("A eigs: ", jnp.linalg.eigvals(A_cl_plus))
assert (jnp.abs(jnp.linalg.eigvals(A_cl_plus)) < 1).all()
# Contraction rate is the spectral radius of A
c = jnp.max(jnp.abs(jnp.linalg.eigvals(A_cl_plus))) + 1e-4
# print(c)
P_cp = cp.Variable(A_cl_plus.shape, symmetric=True)
cons = []
cons.append(A_cl_plus.T @ P_cp @ A_cl_plus << c**2*P_cp)
cons.append(P_cp >> jnp.eye(A_cl_plus.shape[0]))
prob = cp.Problem(cp.Minimize(cp.trace(P_cp)), cons)
prob.solve()

P_plus = P_cp.value
alpha = sqrtm(P_plus)

print('P = ', P_plus)
print('alpha = ', alpha)

# Bisection Algorithm
from scipy.optimize import bisect

best_alpha = sqrtm(P_plus.copy())*100
# line search to increase size of alpha0 while keeping max post-reset length < 1
gamma = .99

def resize(alpha, min=0.1, max=5.0):
    root_fun = lambda s : alpha0_to_max_mjac_length(alpha / s) - gamma

    # precheck
    print(root_fun(min))
    print(root_fun(max))
    assert jnp.sign(root_fun(min)) is not jnp.sign(root_fun(max)), f"Sign of min {root_fun(min)} is same as sign of max {root_fun(max)}"

    scaling, r = bisect(root_fun, min, max, xtol=2e-12, full_output=True)
    alpha_new = alpha / scaling
    return alpha_new, scaling

alpha, scaling = resize(best_alpha)

original_cost = alpha0_to_max_mjac_length(best_alpha)
cost = alpha0_to_max_mjac_length(alpha)

print("Best Alpha: ", best_alpha)
print("New Alpha: ", alpha)
print(f'scaled alpha0 by {scaling:<.8f}')
print("Max Mjac Length: ", cost)
print("original Mjac Length: ", original_cost)


# Plot Trajectory

nt0_final = NT(x0, alpha, 1.)
nt_traj_final = rollout_embsys(nt0_final)

trajectory_path = Path(__file__).resolve().parent / "Data" / "immrax_reachable_trajectory.npz"
trajectory_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    trajectory_path,
    nt_traj=np.asarray(nt_traj_final),
    ts=np.arange(len(nt_traj_final)) * dt,
    dt=np.asarray(dt),
    N=np.asarray(N),
    x0=np.asarray(x0),
    alpha0=np.asarray(alpha),
    guard_gamma=np.asarray(sys.gamma),
    guard_chart=np.asarray(_guard_chart),
)
print(f"Saved reachable trajectory to {trajectory_path}")

PLOT_EVERY = 5

fig, ax = plt.subplots()

mc_x0s = nt0_final.sample_boundary(jax.random.PRNGKey(21), 50)
term   = ODETerm(sys.f)
solver = Euler()
u      = 0.0
event  = Event(sys.guard, optx.Newton(5e-7, 5e-7, optx.rms_norm), direction=True)

def solve_plot_trajectory(x_init):
    # Each initial state needs its own solve and guard-crossing time.
    return diffeqsolve(term, solver, t0=0, t1=2, dt0=dt,
                      y0=x_init, args=u, event=event,
                      saveat=SaveAt(t0=True, t1=True, steps=True))


traj = jax.vmap(solve_plot_trajectory)(mc_x0s)


for i in range(0,len(nt_traj_final),PLOT_EVERY) :
    try :
        nt = NT.unvec(nt_traj_final[i])
        # color = 'k' if i != 0 else 'b'
        nt.plot_projection(ax, 0, 2, color='b')
        nt.plot_projection(ax, 1, 3, color='orange')
    except :
        pass
ax.plot(nt_traj_final[:, 0], nt_traj_final[:, 2], color='b',
        label='Euler nominal (swing)')
ax.plot(nt_traj_final[:, 1], nt_traj_final[:, 3], color='orange',
        label='Euler nominal (stance)')
for index, (times, states) in enumerate(zip(np.asarray(traj.ts), np.asarray(traj.ys))):
    # SaveAt(steps=True) pads unused slots with infinity after termination.
    valid = np.isfinite(times) & np.all(np.isfinite(states), axis=1)
    states = states[valid]
    ax.plot(states[:, 0], states[:, 2], color='crimson', linestyle='--', alpha=.4,
            label='Dopri8 samples (swing)' if index == 0 else None)
    ax.plot(states[:, 1], states[:, 3], color='green', linestyle='--', alpha=.4,
            label='Dopri8 samples (stance)' if index == 0 else None)
ax.set_xlabel(r'$\theta$ (rad)')
ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
ax.set_title('Euler normotope tube and Dopri8 sample trajectories')
ax.grid(True, alpha=.3)
ax.legend()
fig.tight_layout()
plt.show()

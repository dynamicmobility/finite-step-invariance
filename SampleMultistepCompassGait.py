# Imports and globals
import numpy as np
import jax.numpy as jnp
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from ellipsoids import plot_ellipse, Ellipsoid, plot_projection, sample_from
from typing import Tuple
from CompassGaitWalker import CompassGaitWalker
from diffrax import diffeqsolve, ODETerm, Tsit5, Event, Dopri8, Euler, SaveAt, RESULTS
from equinox import filter_jit
import optimistix as optx
import jax
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform) # TODO: GPU this
# jax.config.update("jax_disable_jit", True)


## Setup system and ODE Solver
sys = CompassGaitWalker(m=5., mh=10., a=0.5, b=0.5, gamma=0.0525) # This has a known limit cycle
x0_nom = np.array([-0.21876908,  0.32376908,  1.8083619 ,  1.49571536]) # This point must be on the guard
assert jnp.abs(sys.guard(0, x0_nom)) < 1e-5 # x0_nom is on guard

term = ODETerm(sys.f)
solver = Dopri8()
# ctrl_func = lambda t, x: 0.0
u = 0.0

cond_fn = sys.guard
root_finder = optx.Newton(5e-7, 5e-7, optx.rms_norm)
event = Event(cond_fn, root_finder, direction=True)

def rollout_sys_to_guard(xm):
    xp = sys.reset(0, xm)
    traj = diffeqsolve(term, solver, t0=0, t1=2, dt0=0.001, y0=xp, args=u, event=event, saveat=SaveAt(steps=1), throw=False, max_steps=8192)
    ts = traj.ts
    ys = traj.ys

    return traj, ts[jnp.isfinite(ts)], ys[jnp.isfinite(ts), :]

# Define 3 vectors that are aligned with the guard surface so we can sample preimpact states
# You can do this automatically with jax.grad or by hand as shown here
c = jnp.array([[1, 1, 0, 0]]).T
guard_surface_chart = jnp.array([[1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T

# guard_pts: (N, 3)
def guard_pts_to_full(guard_pts):
    return (guard_surface_chart@guard_pts.T).T + x0_nom

# full pts: (4, N)
def full_pts_to_guard(full_pts):
    pts_guard = (jnp.linalg.inv(jnp.hstack((guard_surface_chart, c)))@(full_pts - x0_nom).T).T
    return pts_guard[..., 0:3]


E0 = Ellipsoid(3)
E0.Q = jnp.array([[848031.94076201, 58223.14845228, 283702.70316407],
 [58223.14845228, 35631.60821713, 31965.30355452],
 [283702.70316407, 31965.30355452, 129725.10278012]])

E0.scale(200)

N = 100
N_STEPS = 5

initial_pts = sample_from(E0, N).T # Preimpact states

step_pts = []

for j in range(N_STEPS):
    next_pts = np.empty((N,3))
    next_pts[:] = np.nan
    one_step_trajs = []
    print("Step: ", j)
    print("Num Remaining Trials: ", np.sum(~np.isnan(initial_pts[:, 0])))
    if(initial_pts.shape[0] == 1):
        break
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(0, initial_pts.shape[0]):
        if not np.isnan(initial_pts[i, :]).any():
            x0_pt = guard_pts_to_full(initial_pts[i, :])
            sol, ts, ys = rollout_sys_to_guard(x0_pt)
            # print(sol.result)
            if jnp.abs(sys.guard(ts[-1], ys[-1, :])) < 5e-7:
                next_pts[i, :] = full_pts_to_guard(ys[-1, :])
                one_step_trajs.append((ts, ys))

                cmap = plt.cm.viridis
                color = cmap(j / N_STEPS)
            else:
                color = 'red' 
            ax1.scatter(x0_pt[0], x0_pt[1], marker='x', color=color) # preimpact starting points
            ax1.scatter(ys[0, 0], ys[0, 1]) # post-impact starting points
            ax1.plot(ys[:, 0], ys[:, 1], color=color)
            fig.add_axes(ax1)

            ax2.scatter(x0_pt[2], x0_pt[3], marker='x', color=color) # Preimpact starting points
            ax2.scatter(ys[0, 2], ys[0, 3]) # Post-impact starting points
            ax2.plot(ys[:, 2], ys[:, 3], color=color)
            fig.add_axes(ax2)
            # print(guard_vmap(ys[-1, :][np.newaxis, :]))
            x_guard = np.linspace(np.min(ys[:, 0]), np.max(ys[:, 0]), 10)
            ax1.plot(x_guard, 2 * sys.gamma - x_guard, 'k:')
            # plt.plot(ys[:, 1], ys[:, 3])
    step_pts.append(next_pts)
    initial_pts = next_pts
    plt.show()
# plt.close()

survived_pts = step_pts[0][~np.isnan(step_pts[-1]).any(axis=1)]
assert survived_pts.shape[1] == 3
E0_return = Ellipsoid(3)
E0_return.fit_p_ball(survived_pts, 2, verbose=False)
print(E0_return.Q, E0_return.c)

initial_pts = initial_pts[~np.isnan(initial_pts).any(axis=1)]
# Compute invariant set approximation from remaining points
assert survived_pts.shape[1] == 3
Ef = Ellipsoid(3)
Ef.fit_p_ball(initial_pts, 2, verbose=True)
print(Ef.Q, Ef.c)



'''
This yields the following ellipsoid:

E0_return:
Q = [[ 1.61219434e+05 -2.58922749e+03  4.80310584e+04]
 [-2.58922749e+03  4.37112112e+01 -7.52288208e+02]
 [ 4.80310584e+04 -7.52288208e+02  1.47875945e+04]]
c =[[ 0.00514039]
 [-0.05754205]
 [-0.01663567]]

Ef:
Q = [[ 3.20426343e+08 -6.12384139e+06  8.88079594e+07]
 [-6.12384139e+06  1.17139201e+05 -1.69684421e+06]
 [ 8.88079594e+07 -1.69684421e+06  2.46190681e+07]]
c = [[-0.00127878]
 [-0.03709283]
 [ 0.00222866]]
'''
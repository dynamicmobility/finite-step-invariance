import numpy as np
import jax.numpy as jnp
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from sets.ellipsoid import plot_ellipse, Ellipsoid
from typing import Tuple
from CompassGaitWalker import CompassGaitWalker
from diffrax import diffeqsolve, ODETerm, Tsit5, Event
import optimistix as optx
import jax
import pdb

PLOT = False

# Setup system and solver of trajectory
sys = CompassGaitWalker(m=5., mh=10., a=0.5, b=0.5, gamma=0.0525) # This has a known limit cycle
# x0_nom = np.array([0.3238649,  -0.21885487,  0.37617958,  1.0930421])
# x0_nom = np.array([-0.21293414, 0.31793414, 1.92576351, 1.48154276])
# x0_nom = np.array([-0.22096612, 0.32596612, 1.64639786, 1.47784094])
x0_nom = np.array([-0.21876908,  0.32376908,  1.8083619 ,  1.49571536])
term = ODETerm(sys.f)
solver = Tsit5()
ctrl_func = lambda t, x: 0.0

cond_fn = sys.guard
root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event = Event(cond_fn, root_finder, direction=True)


# Step-to-step map of the Compass Gait Walker
# Start on pre-impact state xminus and check next intersection onto guard surface
@jax.jit
def one_step(xm):
    xp = sys.reset(0, xm)
    traj = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.001, y0=xp, args=ctrl_func, event=event)
    ys = traj.ys

    return ys.flatten()

print(x0_nom)
print(one_step(x0_nom))
print("Test one_step(x0_nom)")
assert jnp.allclose(one_step(x0_nom), x0_nom, atol=5e-4)

P = jax.vmap(one_step, 0) # Create vmap function over rows of the argument

print("Test vmapped P")

assert jnp.allclose(P(x0_nom[np.newaxis, :]), x0_nom, atol=5e-4)

# Generate the dataset
N = 1000

# Randomly sample points uniformly from some rectangular interval centered on nominal state (doesn't have to be)
I0 = np.random.uniform(-0.001, 0.001, size=(4, N)) + x0_nom[:, np.newaxis]

assert P(I0.T).shape == (N, 4)

def sample_from(e: Ellipsoid, N_pts: int) -> np.ndarray:
    # Sample Normal Distribution 
    # TODO: pull dimension of ellipsoid
    pts = np.random.standard_normal((4, N_pts))
    r = np.random.uniform(0, 1, size=(N,)) ** (1/2) # The denominator of the power is also the dimension
    pts = r*pts/np.linalg.vector_norm(pts, axis=0)
    return np.linalg.inv(sqrtm(e.Q))@pts + e.c
    

def reject_and_resample(Ei: Ellipsoid, I, Pxi, rescale=0.9) -> Tuple[np.ndarray, Ellipsoid, np.ndarray]:
    N = Pxi.shape[1]
    in_pts = I[:, Ei.contains_point(Pxi)]
    Ein = Ellipsoid(4)
    Ein.fit_p_ball(in_pts.T, 2)
    Ein.scale(rescale) # Enforce contraction rate
    return sample_from(Ein, N), Ein, in_pts


# Initialize set I0 and corresponding over-approximating ellipse

E0 = Ellipsoid(4)
# opt, init_A, init_b = p_ball(x.T, 2)
# init_b = init_b.flatten()

print(I0.shape)

E0.fit_p_ball(I0.T, 2)

Ii = I0
Ei = E0


# Test sample_from
fig, ax = plt.subplots()
sample_pts = sample_from(E0, N)
ax.scatter(sample_pts[0, :], sample_pts[1, :])
plt.show()

# Test reject_and_resample
fig, ax = plt.subplots()
Pxi = P(Ii.T).T

# P(in_pts) should lie completely within E0
# We want E_resample to tightly wrap in_pts
# P(I_resample) should also lie within E0
I_resample, E_resample, in_pts = reject_and_resample(E0, Ii, Pxi, rescale=1)

Pin_points = P(in_pts.T).T


Px_resample = P(I_resample.T).T
ax.scatter(Ii[0, :], Ii[1, :], c='black', marker='o')
ax.scatter(Pxi[0, :], Pxi[1, :], c='black', marker='x')
ax.scatter(I_resample[0, :], I_resample[1, :], c='green', marker='o')
ax.scatter(Px_resample[0, :], Px_resample[1, :], c='green', marker='x')
ax.scatter(in_pts[0, :], in_pts[1, :], c='blue', marker='o')
plot_ellipse(E0.Q, E0.c.flatten(), 1, ax, color = 'blue')
plot_ellipse(E_resample.Q, E_resample.c.flatten(), 1, ax, color = 'green')
plt.show()

# TODO: This doesn't work right now for a numerical reason?

# Loop:
# for i in range(10):
#     fig, ax = plt.subplots()
#     # Compute PXi = P(Ii)
#     Pxi = P(Ii.T).T
#     # If E(PXi) within E(Ii)
#     Epxi = Ellipsoid(4)
#     Epxi.fit_p_ball(Pxi.T, 2)
#     print(i)
#     print(Epxi.Q, Epxi.c)
#     if(PLOT):
#         ax.scatter(Ii[0, :], Ii[1, :], c='black', marker='o')
#         ax.scatter(Pxi[0, :], Pxi[1, :], c='black', marker='x')
#         plot_ellipse(Ei.Q, Ei.c.flatten(), 1, ax, color = 'green')
#         plot_ellipse(Epxi.Q, Epxi.c.flatten(), 1, ax, color = 'blue')
#     if(Ei.contains_ellipsoid(Epxi)):
#         print(f"Success! Found Ellipse after {i} iterations:")
#         print(f"Q: {Epxi.Q}, c: {Epxi.c}")
#         break
#     else:
#         Ii, Ei, _ = reject_and_resample(Ei, Ii, Pxi, rescale=1)

#     if(PLOT):
#         plt.title(f"Iteration {i}")
#         ax.scatter(Ii[0, :], Ii[1, :], c='green', marker='o')
#         plt.show()


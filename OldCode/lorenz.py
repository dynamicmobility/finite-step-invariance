import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from sets.ellipsoid import plot_ellipse, Ellipsoid
import pdb
from numpy.random import default_rng
from scipy.integrate import solve_ivp

# This is the A-norm with offset v 
def h(x, A, v):
    return np.diag((x-v).T@A@(x-v))

def lorenz(t, y, sigma=10, rho=28, beta=8/3):
    return [sigma*(y[1] - y[0]), y[0]*(rho - y[2]) - y[1], y[0]*y[1] - beta*y[2]]

def poincare_section(t, y, z0=25.0):
    return y[2] - z0
poincare_section.terminal = True
poincare_section.direction = 1  # z crossing upward (zdot > 0)


def lorenz_poincare_map(x0, z0=25.0, tmax=50.0):
    sol = solve_ivp(lorenz, [0, tmax], x0, events=lambda t, y: poincare_section(t, y, z0))
    return sol.y_events[0].T

def lorenz_samples(x0):
    Ps = []
    step = []
    for x in x0:
        p = lorenz_poincare_map(x)
        step.append(p)
    Ps.append(step)
    return Ps

# Generate the dataset
N = 1000

# Randomly sample points uniformly from some rectangular interval
I0 = np.random.uniform(-2, 2, size=(3, N))

def sample_from(e: Ellipsoid, N_pts: int):
    # Sample Normal Distribution 
    # TODO: pull dimension of ellipsoid
    pts = np.random.standard_normal((2, N_pts))
    r = np.random.uniform(0, 1, size=(N,)) ** (1/2) # The denominator of the power is also the dimension
    pts = r*pts/np.linalg.vector_norm(pts, axis=0)
    return np.linalg.inv(sqrtm(e.Q))@pts + e.c
    

def reject_and_resample(Ei: Ellipsoid, I, Pxi, rescale=0.9):
    N = Pxi.shape[1]
    in_pts = I[:, Ei.contains_point(Pxi)]
    Ein = Ellipsoid(3)
    Ein.fit_p_ball(in_pts.T, 2)
    Ein.scale(rescale) # Enforce contraction rate
    return sample_from(Ein, N), Ein, in_pts


# Initialize set I0 and corresponding over-approximating ellipse

E0 = Ellipsoid(3)
# opt, init_A, init_b = p_ball(x.T, 2)
# init_b = init_b.flatten()

# print(I0.shape)

E0.fit_p_ball(I0.T, 2)

Ii = I0
Ei = E0



# Loop:
for i in range(10):
    fig, ax = plt.subplots()
    # Compute PXi = P(Ii)
    Pxi = lorenz_samples(Ii.T)
    # If E(PXi) within E(Ii)
    Ei = Ellipsoid(3)
    Ei.fit_p_ball(Ii.T, 2)

    Epxi = Ellipsoid(3)
    Epxi.fit_p_ball(Pxi, 2)


    ax.scatter(Ii[0, :], Ii[2, :], c='black', alpha=0.5, marker='o')
    ax.scatter(Pxi[:, 0], Pxi[:, 2], c='black', marker='x')
    # plot_ellipse(Ei.Q, Ei.c.flatten(), 1, ax, color = 'green')
    # plot_ellipse(Epxi.Q, Epxi.c.flatten(), 1, ax, color = 'blue')
    # if(Ei.contains_ellipsoid(Epxi)):
        # print("Success! Found Ellipse:")
        # print(f"Q: {Epxi.Q}, c: {Epxi.c}")
        # break
    # else:
        # Ii, _, _ = reject_and_resample(Ei, Ii, Pxi, rescale=.99)
    # ax.scatter(Ii[0, :], Ii[1, :], c='green', marker='o')
    # plot_ellipse(A, v.flatten(), 1, ax, color = 'red')
    
    plt.show()


import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from ellipsoids import plot_ellipse, Ellipsoid
from typing import Tuple
import pdb

PLOT = True

# This is the A-norm with offset v 
def h(x, A, v):
    return np.diag((x-v).T@A@(x-v))

# This is a toy step-to-step map (contracts within the A-norm unit ball but expands otherwise)
def P(x, A, v):
    return (x-v)*np.sqrt(h(x, A, v)) + v

# Generate the dataset
N = 1000
A = np.array([[2, 0], [0, 1]])
v = np.array([[2], [0]])

# Randomly sample points uniformly from some rectangular interval
I0 = np.random.uniform(-4, 4, size=(2, N))



# Initialize set I0 and corresponding over-approximating ellipse

E0 = Ellipsoid(2)
# opt, init_A, init_b = p_ball(x.T, 2)
# init_b = init_b.flatten()

print(I0.shape)

# Test sample_from
# fig, ax = plt.subplots()
# sample_pts = sample_from(E0, N)
# ax.scatter(sample_pts[0, :], sample_pts[1, :])
# plt.show()

# Test reject_and_resample
# fig, ax = plt.subplots()
# Pxi = P(Ii, A, v)

# # P(in_pts) should lie completely within E0
# # We want E_resample to tightly wrap in_pts
# # P(I_resample) should also lie within E0
# I_resample, E_resample, in_pts = reject_and_resample(E0, Ii, Pxi, rescale=1)

# Pin_points = P(in_pts, A, v)


# Px_resample = P(I_resample, A, v)
# ax.scatter(Ii[0, :], Ii[1, :], c='black', marker='o')
# ax.scatter(Pxi[0, :], Pxi[1, :], c='black', marker='x')
# ax.scatter(I_resample[0, :], I_resample[1, :], c='green', marker='o')
# ax.scatter(Px_resample[0, :], Px_resample[1, :], c='green', marker='x')
# ax.scatter(in_pts[0, :], in_pts[1, :], c='blue', marker='o')
# plot_ellipse(A, v.flatten(), 1, ax, color = 'red')
# plot_ellipse(E0.A, E0.b.flatten(), 1, ax, color = 'blue')
# plot_ellipse(E_resample.A, E_resample.b.flatten(), 1, ax, color = 'green')
# plt.show()

def algorithm():
    def sample_from(e: Ellipsoid, N_pts: int) -> np.ndarray:
        # Sample Normal Distribution 
        # TODO: pull dimension of ellipsoid
        pts = np.random.standard_normal((2, N_pts))
        r = np.random.uniform(0, 1, size=(N,)) ** (1/2) # The denominator of the power is also the dimension
        pts = r*pts/np.linalg.vector_norm(pts, axis=0)
        return np.linalg.inv(sqrtm(e.Q))@pts + e.c
    

    def reject_and_resample(Ei: Ellipsoid, I, Pxi, rescale=1) -> Tuple[np.ndarray, Ellipsoid, np.ndarray]:
        N = Pxi.shape[1]
        in_pts = I[:, Ei.contains_point(Pxi)]
        in_return_pts = Pxi[:, Ei.contains_point(Pxi)]
        print(Ei.contains_point(Pxi))
        pts = np.hstack([in_return_pts, in_pts])
        Ein = Ellipsoid(2)
        Ein.fit_p_ball(pts.T, 2)
        Ein.scale(rescale) # Enforce contraction rate
        return sample_from(Ein, N), Ein, pts

    E0.fit_p_ball(I0.T, 2)

    Ii = I0
    Ei = E0

    # Loop:
    for i in range(30):
        fig, ax = plt.subplots()
        # Compute PXi = P(Ii)
        Pxi = P(Ii, A, v)
        # If E(PXi) within E(Ii)
        Epxi = Ellipsoid(2)
        Epxi.fit_p_ball(Pxi.T, 2)
        if(PLOT):
            ax.scatter(Ii[0, :], Ii[1, :], c='black', marker='o')
            ax.scatter(Pxi[0, :], Pxi[1, :], c='black', marker='x')
            plot_ellipse(Ei.Q, Ei.c.flatten(), 1, ax, color = 'green')
            plot_ellipse(Epxi.Q, Epxi.c.flatten(), 1, ax, color = 'blue')
        if(Ei.contains_ellipsoid(Epxi)):
            print(f"Success! Found Ellipse after {i} iterations:")
            print(f"Q: {Epxi.Q}, c: {Epxi.c}")
            break
        else:
            Ii, Ei, _ = reject_and_resample(Ei, Ii, Pxi, rescale=1)

        if(PLOT):
            plt.title(f"Iteration {i}")
            # ax.scatter(Ii[0, :], Ii[1, :], c='green', marker='o')
            plot_ellipse(A, v.flatten(), 1, ax, color = 'red')
            plt.show()

def scenario_approach():

    def get_samples(e: Ellipsoid, N_pts: int) -> np.ndarray:
        """
        Uniformly sample points from the ellipsoid:
            { x : (x - c)^T Q (x - c) <= 1 }
        """

        Q = e.Q
        c = e.c.reshape(-1, 1)
        n = Q.shape[0]

        # Cholesky factor: Q = L^T L
        L = np.linalg.cholesky(Q)

        # Sample from unit ball in R^n
        # Step 1: random directions
        dirs = np.random.randn(n, N_pts)
        dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)

        # Step 2: random radii with correct distribution
        radii = np.random.rand(N_pts) ** (1.0 / n)

        # Scale directions by radii
        u = dirs * radii

        # Map to ellipsoid: x = c + L^{-1} u
        x = np.linalg.solve(L, u) + c

        return x

    def resample(I, Pxi, rescale=1) -> Tuple[np.ndarray, Ellipsoid]:
        N = Pxi.shape[1]
        Ein = Ellipsoid(2)
        Ein.fit_ellipsoid(I.T, P)
        return get_samples(Ein, N), Ein



    E0.fit_ellipsoid(I0.T, P)

    Ii = I0
    prior = E0

    it = 0
    for i in range(30):
        fig, ax = plt.subplots()
        # Compute PXi = P(Ii)
        Pxi = P(Ii, A, v)


        if it > 0:
            ax.scatter(Ii[0, :], Ii[1, :], c='black', marker='o')
            ax.scatter(Pxi[0, :], Pxi[1, :], c='black', marker='x')
            plot_ellipse(prior.Q, prior.c.flatten(), 1, ax, color = 'green')
            plot_ellipse(A, v.flatten(), 1, ax, color = 'red')
            # if(Ei.contains_ellipsoid(prior)):
            #     print(f"Success! Found Ellipse after {i} iterations:")
            #     print(f"Q: {Ei.Q}, c: {Ei.c}")
            #     break
            # else:
            Ii, Ei = resample(Ii, Pxi, rescale=1)
            ax.scatter(Ii[0, :], Ii[1, :], c='cyan', marker='o')
            plot_ellipse(Ei.Q, Ei.c.flatten(), 1, ax, color = 'blue')
            plt.show()
            prior = Ei
            it +=1
        else:
            Ii, Ei = resample(Ii, Pxi, rescale=1)
            it += 1

if __name__ == "__main__":
    algorithm()
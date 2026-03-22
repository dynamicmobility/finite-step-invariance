import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from ellipsoids import plot_ellipse, Ellipsoid
from typing import Tuple
from ellipsoid_intersection import monte_carlo_intersection
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
    

    def reject_and_resample(oldellipsoid: Ellipsoid, I, Pxi, ax, rescale=1) -> Tuple[np.ndarray, Ellipsoid, np.ndarray]:
        # n = Pxi.shape[0]
        # in_pts = I[:, oldellipsoid.contains_point(Pxi)]
        # newellipsoid = Ellipsoid(2)
        # newellipsoid.fit_p_ball(in_pts.T, 2)

        in_pts_Pxi = Pxi[:, oldellipsoid.contains_point(Pxi)]
        in_pts_I   = I[:, oldellipsoid.contains_point(Pxi)]
        all_in_pts = np.hstack((in_pts_I, in_pts_Pxi))

        # out_pts_Pxi = Pxi.T[~in_idx, :]
        out_pts_I   = I[:, ~oldellipsoid.contains_point(Pxi)]
        # all_out_pts = np.vstack((out_pts_Pxi, out_pts_I))

        new_pts = np.hstack((in_pts_I, out_pts_I))
        newellipsoid = Ellipsoid(2)
        newellipsoid.fit_p_ball(in_pts_I.T, 2)
        # ax.scatter(in_pts_I[0, :], in_pts_I[1, :], c='cyan', marker='o')
        # ax.scatter(out_pts_I[0, :], out_pts_I[1, :], c='yellow', marker='o')
        # newellipsoid.rescale_ellipsoid(out_pts_I)
        plot_ellipse(newellipsoid.Q, newellipsoid.c.flatten(), newellipsoid.r[0][0], ax, color = 'orange')
        # # Ein.scale(rescale) # Enforce contraction rate
        return sample_from(newellipsoid, N), newellipsoid, all_in_pts

    E0.fit_p_ball(I0.T, 2)

    initset = I0
    prior_ellipsoid = E0

    # Loop:
    for i in range(30):
        fig, ax = plt.subplots()
        # Compute PXi = P(Ii)
        returnmap = P(initset, A, v)
        # If E(PXi) within E(Ii)
        aftermap_ellipsoid = Ellipsoid(2)
        aftermap_ellipsoid.fit_p_ball(returnmap.T, 2)
        prior_volume = prior_ellipsoid.ellipsoid_volume()
        after_volume = aftermap_ellipsoid.ellipsoid_volume()
        intersection_volume = monte_carlo_intersection(prior_ellipsoid, aftermap_ellipsoid, 3000, ax)
        print("prior volume:", prior_volume, "after volume:", after_volume, "and intersection volume:", intersection_volume)
        if(PLOT):
            ax.scatter(initset[0, :], initset[1, :], c='black', marker='o')
            ax.scatter(returnmap[0, :], returnmap[1, :], c='black', marker='x')
            plot_ellipse(prior_ellipsoid.Q, prior_ellipsoid.c.flatten(), prior_ellipsoid.r[0][0], ax, color = 'green')
            plot_ellipse(aftermap_ellipsoid.Q, aftermap_ellipsoid.c.flatten(), aftermap_ellipsoid.r[0][0], ax, color = 'blue')
            # plt.show()
        if(np.round(prior_volume, 1) == np.round(after_volume, 1) == np.round(intersection_volume, 1)):
            print(f"Success! Found Ellipse after {i} iterations:")
            print(f"Q: {aftermap_ellipsoid.Q}, c: {aftermap_ellipsoid.c}")
            break
        else:
            initset, prior_ellipsoid, _ = reject_and_resample(prior_ellipsoid, initset, returnmap, ax, rescale=1)

        if(PLOT):
            plt.title(f"Iteration {i}")
            ax.scatter(initset[0, :], initset[1, :], c='green', marker='o')
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
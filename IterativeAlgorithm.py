import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from ellipsoids import plot_ellipse, Ellipsoid
import pdb

# This is the A-norm with offset v 
def h(x, A, v):
    return np.diag((x-v).T@A@(x-v))

# This is a toy step-to-step map (contracts within the A-norm unit ball but expands otherwise)
def P(x, A, v):
    return (x-v)*np.sqrt(h(x, A, v)) + v

# Generate the dataset
N = 1000
A = np.array([[1, 0], [0, 1]])
v = np.array([[0], [0]])

# Randomly sample points uniformly from some rectangular interval
I0 = np.random.uniform(-2, 2, size=(2, N))


def sample_from(e: Ellipsoid, N_pts: int):
    # Sample Normal Distribution 
    # TODO: pull dimension of ellipsoid
    pts = np.random.standard_normal((2, N_pts))
    r = np.random.uniform(0, 1, size=(N,)) ** (1/2) # The denominator of the power is also the dimension
    pts = r*pts/np.linalg.vector_norm(pts, axis=0)
    return np.linalg.inv(sqrtm(e.A))@pts + e.b
    

def reject_and_resample(Ei: Ellipsoid, I, Pxi, rescale=0.9):
    N = Pxi.shape[1]
    in_pts = I[:, Ei.contains_point(Pxi)]
    Ein = Ellipsoid(2)
    Ein.fit_p_ball(in_pts.T, 2)
    Ein.scale(rescale) # Enforce contraction rate
    return sample_from(Ein, N), Ein, in_pts


# Initialize set I0 and corresponding over-approximating ellipse

E0 = Ellipsoid(2)
# opt, init_A, init_b = p_ball(x.T, 2)
# init_b = init_b.flatten()

print(I0.shape)

E0.fit_p_ball(I0.T, 2)

Ii = I0
Ei = E0


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


# Loop:
for i in range(10):
    fig, ax = plt.subplots()
    # Compute PXi = P(Ii)
    Pxi = P(Ii, A, v)
    # If E(PXi) within E(Ii)
    Ei = Ellipsoid(2)
    Ei.fit_p_ball(Ii.T, 2)

    Epxi = Ellipsoid(2)
    Epxi.fit_p_ball(Pxi.T, 2)
    ax.scatter(Ii[0, :], Ii[1, :], c='black', marker='o')
    ax.scatter(Pxi[0, :], Pxi[1, :], c='black', marker='x')
    plot_ellipse(Ei.A, Ei.b.flatten(), 1, ax, color = 'green')
    plot_ellipse(Epxi.A, Epxi.b.flatten(), 1, ax, color = 'blue')
    if(Ei.contains_ellipsoid(Epxi)):
        print("Success!")
        break
    else:
        Ii, _, _ = reject_and_resample(Ei, Ii, Pxi, rescale=1)
    ax.scatter(Ii[0, :], Ii[1, :], c='green', marker='o')
    plot_ellipse(A, v.flatten(), 1, ax, color = 'red')
    plt.show()


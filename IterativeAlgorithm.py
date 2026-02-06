import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from ellipsoids import p_ball, project_ellipsoid_to_2d, plot_ellipse, in_ellipsoid, Ellipsoid


# This is the A-norm with offset v 
def h(x, A, v):
    return np.diag((x-v).T@A@(x-v))

# This is a toy step-to-step map (contracts within the A-norm unit ball but expands otherwise)
def P(x, A, v):
    return (x-v)*np.sqrt(h(x, A, v)) + v

# Generate the dataset
N = 1000
A = np.array([[2, 0], [0, 1]])
v = np.array([[1], [0]])

# Randomly sample points uniformly from some rectangular interval
I0 = np.random.uniform(-2, 2, size=(2, N))


def sample_on(e: Ellipsoid):
    A_loc = e.A
    b_loc = e.b
    # TODO: Copy from immrax


# Initialize set I0 and corresponding over-approximating ellipse

E0 = Ellipsoid(2)
# opt, init_A, init_b = p_ball(x.T, 2)
# init_b = init_b.flatten()

E0.fit_p_ball(I0.T, 2)

Ii = I0
Ei = E0

# Loop:
for i in range(10):
    Pxi = P(Ii)
# Compute PXi = P(Ii)
# If E(PXi) within E(Ii)
# return
# Else
# Ii+1 = reject(I)


# Reject(I):
# Px = P(I)
# Px* = find({Px in E(I)})
# return sampleFrom(E(Px*))
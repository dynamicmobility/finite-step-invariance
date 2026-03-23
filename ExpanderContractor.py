import numpy as np
from ellipsoids import Ellipsoid


d = 2
v = np.array([[1], [1]])
A = np.array([[2, 1], [1, 1]])

assert np.all(np.linalg.eigvals(A) > 0)

# This is the A-norm with offset v 
def _h(x, A, v):
    return np.diag((x-v).T@A@(x-v))

# This is a toy step-to-step map (contracts within the A-norm unit ball but expands otherwise)
def P_batch(x):
    return ((x.T-v)*_h(x.T, A, v) + v).T

def initial_guess():
    pts = np.random.uniform(low=-2, high=2, size=(100, d))
    E0 = Ellipsoid(d)
    E0.fit_p_ball(pts, 2)
    return E0


if __name__ == '__main__':
    # Tests to make sure dimensions make sense
    x = np.array([1, 0])
    P_batch(x)

    xs = np.random.uniform(low=-2, high=2, size=(10, d))

    Xprime = P_batch(xs)
    assert Xprime.shape == xs.shape

    initial_guess()
import numpy as np
from scipy.linalg import eigvalsh
from sets.ellipsoid import Ellipsoid, plot_projection
from sets.radialBasisFunction import RBF
from matplotlib import pyplot as plt
from typing import Literal

d = 2
c = np.array([[1], [1]])
Q = np.array([[2, 1],
              [1, 1]])

assert np.all(np.linalg.eigvals(Q) > 0)

# This is the Q-norm with offset v 
def _h(x, Q, c):
    return np.diag((x-c).T@Q@(x-c))

# This is a toy step-to-step map (contracts within the Q-norm unit ball but expands otherwise)
def P_batch(x):
    return ((x.T-c)*_h(x.T, Q, c) + c).T

def initial_guess(set_type: Literal['ell', 'rbf'], rng: np.random.Generator = None, N = 100):
    if(set_type == 'ell'):
        E0  = Ellipsoid(d)
        if(rng is not None):
            c0 = rng.uniform(low=-1, high=1, size=(2,1))
            M = rng.uniform(low=-1, high=1, size=(2,2))
            Q0 = M.T@M
            # Rescale to contain E(Q, c), accounting for the center offset.
            delta = c - c0
            a = np.sqrt((delta.T @ Q0 @ delta).item())
            b = np.sqrt(eigvalsh(Q0, Q)[-1])
            wiggle = rng.uniform(low=1.0, high=2.0)
            Q0 = Q0 / (wiggle*(a + b)**2)
            E0.load_from_row(np.concat([Q0.flatten(), c0.flatten()]))
        else:
            E0.load_from_row(np.concat([0.1*np.eye(2,2).flatten(), np.array([0, 0])]))
    elif(set_type == 'rbf'):
        E0  = RBF(d, 2)
        rng = np.random.default_rng() if rng is None else rng
        pts = rng.uniform(low=-3, high=3, size=(N, d))
        E0.fit(pts)
    return E0

# def initial_guess_rbf(rng=None):
#     rng = np.random.default_rng() if rng is None else rng
#     pts = rng.uniform(low=-2, high=2, size=(100, d))
#     E0 = RBF(d, np.log(0.25))
#     E0 = E0.fit_rbf(pts, 1)
#     return E0

def plot_true(ax, diml=0, dimh=1, **kwargs):
    plot_projection(Q, c.flatten(), 1, ax, diml, dimh, alpha=0.75, color='green', **kwargs)

if __name__ == '__main__':
    # Tests to make sure dimensions make sense
    x = np.array([1, 0])
    P_batch(x)

    xs = np.random.uniform(low=-2, high=2, size=(10, d))

    Xprime = P_batch(xs)
    assert Xprime.shape == xs.shape

    initial_guess()

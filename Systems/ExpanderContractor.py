import numpy as np
from sets.ellipsoid import Ellipsoid, plot_projection
from sets.radialBasisFunction import RBF
from matplotlib import pyplot as plt
from typing import Literal

d = 2
v = np.array([[1], [1]])
Q = np.array([[2, 1],
              [1, 1]])

assert np.all(np.linalg.eigvals(Q) > 0)

# This is the Q-norm with offset v 
def _h(x, Q, v):
    return np.diag((x-v).T@Q@(x-v))

# This is a toy step-to-step map (contracts within the Q-norm unit ball but expands otherwise)
def P_batch(x):
    return ((x.T-v)*_h(x.T, Q, v) + v).T

def initial_guess(set_type: Literal['ell', 'rbf'], rng: np.random.Generator, N = 100):
    if(set_type == 'ell'):
        E0  = Ellipsoid(d)
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
    plot_projection(Q, v.flatten(), 1, ax, diml, dimh, alpha=0.75, color='green', **kwargs)

if __name__ == '__main__':
    # Tests to make sure dimensions make sense
    x = np.array([1, 0])
    P_batch(x)

    xs = np.random.uniform(low=-2, high=2, size=(10, d))

    Xprime = P_batch(xs)
    assert Xprime.shape == xs.shape

    initial_guess()
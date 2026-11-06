import numpy as np
from sets.ellipsoid import Ellipsoid, plot_projection
from sets.radialBasisFunction import RBF
from sets.parameterized_set import ParameterizedSet
from typing import Literal

d = 2

# Dogbone: two circle centers and radii
c1 = np.array([-0.6, 0.0])
c2 = np.array([ 0.6, 0.0])
R  = 0.6      # radius of each lobe
PUSH = 1.3    # expansion factor outside the t-bone

# 1.01 converges in 66 iterations
# 1.1 converges in 18 iterations
# 1.3 converges in 6 iterations

def _in_dogbone(x):
    """Boolean mask: True if x is inside either lobe. x: (N, 2)"""
    in_c1 = np.sum((x - c1)**2, axis=1) <= R**2
    in_c2 = np.sum((x - c2)**2, axis=1) <= R**2
    return in_c1 | in_c2

def _in_dynamics(x, nearest):
    # A = np.array([[0, -1], [1, 0]])
    A = np.array([[0.5, 0], [0, 0.5]])
    return (A@(x-nearest).T).T + nearest

def P_batch(x):
    """
    Poincare return map:
    - Inside t-bone: contract toward nearest center
    - Outside t-bone: push away from origin by PUSH factor
    x: (N, 2) -> (N, 2)
    """
    x      = np.atleast_2d(x)
    out    = x.copy()
    inside = _in_dogbone(x)

    # Inside: contract toward nearest lobe center
    d1         = np.sum((x - c1)**2, axis=1)
    d2         = np.sum((x - c2)**2, axis=1)
    nearest    = np.where(d1[:, None] < d2[:, None], c1, c2)   # (N, 2)
    # out[inside] = x[inside]  # Points inside do not move
    # out[inside] = x[inside] + 0.5 * (nearest[inside] - x[inside])
    out[inside] = _in_dynamics(x[inside], nearest[inside])

    # Outside: push away from origin
    out[~inside] *= PUSH

    return out

def initial_guess(set_type: Literal['ell', 'rbf'], rng: np.random.Generator, N = 100):
    """Single RBF centered at origin as initial guess."""
    if(set_type == 'ell'):
        E0  = Ellipsoid(d)
        E0.load_from_row(np.concat([np.eye(2,2).flatten(), np.array([0, 0])]))
    elif(set_type == 'rbf'):
        E0  = RBF(d)
        pts  = rng.uniform(low=-2., high=2., size=(N, d))
        E0.fit(pts, 2)
    return E0

# def initial_guess_rbf(rng=None):
#     rng = np.random.default_rng() if rng is None else rng
#     pts = rng.uniform(low=-2, high=2, size=(100, d))
#     E0 = RBF(d, np.log(0.25))
#     E0 = E0.fit_rbf(pts, 1)
#     return E0

def plot_true(ax, diml=0, dimh=1, **kwargs):
    """Plot the true ROA (union of two discs) on the given axes."""
    plot_projection(np.eye(2), c1, R, ax, diml, dimh, alpha=0.75, color='green', **kwargs)
    plot_projection(np.eye(2), c2, R, ax, diml, dimh, alpha=0.75, color='green', **kwargs)

if __name__ == '__main__':
    # Tests
    x = np.array([0.5, 0.0])
    print("Single point:", P_batch(x))

    xs     = np.random.uniform(low=-2, high=2, size=(10, d))
    Xprime = P_batch(xs)
    assert Xprime.shape == xs.shape, "Shape mismatch"
    print("Batch shape OK:", Xprime.shape)

    # Check contraction inside t-bone
    x_in = np.array([[-0.5, 0.0], [0.5, 0.0]])
    x_out = P_batch(x_in)
    d_before = np.sqrt(np.sum((x_in  - np.array([c1, c2]))**2, axis=1))
    d_after  = np.sqrt(np.sum((x_out - np.array([c1, c2]))**2, axis=1))
    assert np.all(d_after < d_before), "Should contract inside t-bone"
    print("Contraction check OK")

    # Check expansion outside t-bone
    x_out_dogbone = np.array([[1.5, 1.5], [-1.5, -1.5]])
    x_pushed    = P_batch(x_out_dogbone)
    assert np.all(np.sum(x_pushed**2, axis=1) > np.sum(x_out_dogbone**2, axis=1)), \
        "Should expand outside t-bone"
    print("Expansion check OK")

    initial_guess()
    print("Initial guess OK")
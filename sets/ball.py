import numpy as np 
import cvxpy as cp
import scipy
from scipy.special import gamma
from scipy.optimize import minimize, NonlinearConstraint
from numpy.linalg import det, inv
from scipy.linalg import null_space, sqrtm
from sets.parameterized_set import ParameterizedSet
from sets.ellipsoid import Ellipsoid

# Defines ellipsoid as ||x-xc|| <= r
class Ball(Ellipsoid):
    def __init__(self, n = 2):
        super().__init__(n)
        self.r = 0
        self.c = np.zeros((n,))

    def fit(self, data, normp=2, verbose=False):
        """Fit row-wise data and return the optimization objective."""
        center = np.mean(data, axis=0)
        row_norms = np.linalg.vector_norm(data-center, axis=1, ord=normp)
        self.r = np.max(row_norms)
        self.c = center
        return self.r, self.c

    def sample_from(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """Return uniform samples with shape (N, dimension)."""
        pts = rng.standard_normal((N, self.dimension))
        radii = rng.uniform(size=N) ** (1 / self.dimension)
        pts = radii[:, np.newaxis] * pts / np.linalg.norm(pts, axis=1)[:, np.newaxis]
        return pts*self.r + self.c

    def contains_points(self, pts: np.ndarray) -> np.ndarray:
        """Return a membership mask for row-wise points."""
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] != self.dimension:
            raise ValueError("pts must have shape (N, dimension)")
        centered = pts - self.c.reshape(-1)
        return np.linalg.vector_norm(centered, axis=1) <= self.r

    def plot(self, ax, diml = 0, dimh = 1, alpha = 1.0, color = 'green', **kwargs):
        plot_projection(np.eye(self.dimension), self.c.flatten(), self.r, ax,
                        diml=diml, dimh=dimh, alpha=alpha, color=color, **kwargs)
        return ax

    def params(self) -> np.ndarray:
        return np.concat([self.r, self.c.flatten()])

    @classmethod
    def get_d_from_row_length(cls, l):
        return int(l-1)

    def load_from_row(self, row: np.ndarray):
        """Restore params() from a 1D CSV row, excluding the iteration index."""
        row = np.asarray(row, dtype=float)
        if row.ndim != 1:
            raise ValueError(f"row must be a 1D array of parameters")
        d = self.get_d_from_row_length(row.size)
        if not np.all(np.isfinite(row)):
            raise ValueError("row must contain only finite parameters")
        r = row[0]
        c = row[1:].reshape(d, 1).copy()
        self.r = r
        self.c = c
        return self

    def contains_point(self, p):
        """Compatibility wrapper for column-wise points (dimension, N)."""
        return self.contains_points(np.asarray(p).reshape(self.dimension, -1).T)
 
    def volume(self):
        n = self.c.size()  # dimension of the ball
        volume = ((np.pi ** (n / 2)) / gamma(n / 2 + 1))*self.r**n
        return volume


    

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Plotting ellipsoids
# TODO: Inherit this from ellipsoids.py

def plot_projection(Q, c, r, ax, diml = 0, dimh = 1, alpha=0.75, color='green', **kwargs):
        P = Q / r
        n = P.shape[0]
        if n == 2:
            plot_ellipse(P, c, r, ax, alpha, color, **kwargs)
            return
        ind = [k for k in range(n) if k not in [diml, dimh]]
        Phat = P[ind, :]
        N = null_space(Phat)
        M = N[(diml, dimh), :]  # Since M is guaranteed 2x2,
        Minv = (1 / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])) * np.array(
            [[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]
        )
        P = Minv.T @ N.T @ P @ N @ Minv
        plot_ellipse(P, c[(diml, dimh),], r, ax, alpha, color, **kwargs)

def plot_ellipse(Q, c, r, ax, alpha=0.75, color='green', **kwargs):
    """
    Plot 2D ellipse defined by (x-c)^T Q (x-c) <= r
    """
    Q_2d = Q
    c_2d = c
    eigvals, eigvecs = np.linalg.eigh(Q_2d)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    # Scale the ellipse by sqrt(r)
    ellipse = eigvecs @ np.diag(np.sqrt(r) / np.sqrt(eigvals)) @ circle
    ellipse = ellipse.T + c_2d
    ax.plot(ellipse[:, 0], ellipse[:, 1], **kwargs, color=color, alpha=alpha)

    return ax


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    E0 = Ellipsoid(2)
    E1 = Ellipsoid(2)
    E2 = Ellipsoid(2)
    E3 = Ellipsoid(2)

    E0.A = np.array([[1/4, 0],[0, 1/9]])
    E0.b = np.array([[-1, 1]]).T
    E0.L = np.linalg.cholesky((E0.A + E0.A.T)/2)

    E1.A = np.array([[1/3, -1/5],[-1/5, 1/2]])
    E1.b = np.array([[-1.5, 1.5]]).T
    E1.L = np.linalg.cholesky((E1.A + E1.A.T)/2)

    E2.A = np.array([[1, 1/3],[1/3, 2]])
    E2.b = np.array([[-0.6,-0.5]]).T
    E2.L = np.linalg.cholesky((E2.A + E2.A.T)/2)

    E3.A = np.array([[1/2, 0],[0, 12]])
    E3.b = np.array([[-1, 2.4]]).T
    E3.L = np.linalg.cholesky((E3.A + E3.A.T)/2)

    fig, ax = plt.subplots()

    plot_ellipse(E0.A, E0.b.flatten(), 1, ax, color='orange')
    plot_ellipse(E1.A, E1.b.flatten(), 1, ax, color='blue')
    plot_ellipse(E2.A, E2.b.flatten(), 1, ax, color='green')
    plot_ellipse(E3.A, E3.b.flatten(), 1, ax, color='red')

    # Does the orange contain the
    print("Contains E1? ", E0.contains_ellipsoid(E1)) # blue
    print("Contains E2? ", E0.contains_ellipsoid(E2)) # green
    print("Contains E3? ", E0.contains_ellipsoid(E3)) # red

    pts = np.random.uniform(-3, 1, (2,10))

    ax.scatter(pts[0, :], pts[1, :])

    print(E0.contains_point(pts))

    print(E0.contains_point(np.array([[0, 0]]).T))

    plt.show()

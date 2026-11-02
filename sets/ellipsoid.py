import numpy as np 
import cvxpy as cp
import scipy
from scipy.special import gamma
from scipy.optimize import minimize, NonlinearConstraint
from numpy.linalg import det, inv
from scipy.linalg import null_space, sqrtm
from sets.parameterized_set import ParameterizedSet

# Defines ellipsoid as (x-c)T Q (x-c) <= 1
class Ellipsoid(ParameterizedSet):
    def __init__(self, n = 2):
        super().__init__(n)
        self.Q = np.eye(n)
        self.c = np.zeros((n, 1))
        self.L = np.linalg.cholesky((self.Q + self.Q.T)/2)
        self.A = np.eye(n)
        self.b = np.zeros((n, 1))
        self.r = 0

    def fit(self, data, normp=2, verbose=False):
        """Fit row-wise data and return the optimization objective."""
        return self._fit_p_ball(data, normp=normp, verbose=verbose)

    def sample_from(self, N: int) -> np.ndarray:
        """Return uniform samples with shape (N, dimension)."""
        pts = np.random.standard_normal((self.dimension, N))
        radii = np.random.uniform(size=N) ** (1 / self.dimension)
        pts = radii * pts / np.linalg.norm(pts, axis=0)
        return (np.linalg.inv(sqrtm(self.Q)) @ pts + self.c).T

    def contains_points(self, pts: np.ndarray) -> np.ndarray:
        """Return a membership mask for row-wise points."""
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] != self.dimension:
            raise ValueError("pts must have shape (N, dimension)")
        centered = pts - self.c.reshape(-1)
        return np.einsum("ni,ij,nj->n", centered, self.Q, centered) <= 1

    def plot(self, ax, diml = 0, dimh = 1, alpha = 1.0, color = 'green', **kwargs):
        plot_projection(self.Q, self.c, 1.0, ax, diml, dimh, color, **kwargs)

    def params(self) -> np.ndarray:
        return np.concat([self.Q.flatten(), self.c.flatten()])

    def load_from_row(self, row: np.ndarray):
        """Restore params() from a 1D CSV row, excluding the iteration index."""
        row = np.asarray(row, dtype=float)
        d = self.dimension
        if row.ndim != 1 or row.size != d * d + d:
            raise ValueError(f"row must be a 1D array of {d * d + d} parameters")
        if not np.all(np.isfinite(row)):
            raise ValueError("row must contain only finite parameters")
        Q = row[:d * d].reshape(d, d).copy()
        c = row[d * d:].reshape(d, 1).copy()
        if not np.allclose(Q, Q.T):
            raise ValueError("Q must be symmetric")
        L = np.linalg.cholesky((Q + Q.T) / 2)
        self.Q = Q
        self.c = c
        self.L = L
        self.A = L.T
        self.b = self.A @ c
        self.r = 1.0
        return self

    # Given a dataset of xs and norm degree, fit ellipse to dataset
    # Ellipse is fit in form ||Ax - b|| < 1 but saved in form Q, c
    def _fit_p_ball(self, xs, normp=2, verbose=False):
        xs = np.array(xs).T
        n = np.shape(xs)[0]
        m = np.shape(xs)[1]

        assert n == self.c.shape[0]

        A = cp.Variable((n, n), symmetric=True)
        b = cp.Variable((n,1))

        # ------------------------------------------------------------------
        # objective --------------------------------------------------------
        n = A.shape[0]                       # matrix dimension
        objective = cp.Maximize(cp.log_det(A))

        # ------------------------------------------------------------------
        # constraints ------------------------------------------------------
        residual = A @ xs - b @ np.ones((1,m))
        col_norm_constraints = [
            cp.norm(residual[:, j], normp) <= 1
            for j in range(m)
        ]

        constraints = col_norm_constraints #+ bounds 

        # ------------------------------------------------------------------
        # solve ------------------------------------------------------------

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if(verbose):
            print("Optimal value :", problem.value)
            print("A* =", A.value)
            print("b* =", b.value)
            print("volume proxy:", np.log(np.linalg.det(A.value)))
            print(problem.status)
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print("Optimization failed with status:", problem.status)
            else:
                print("Optimization succeeded.")
        Q, c, r = self._convert_A_b_to_Q_c_r(A.value, b.value) 
        self.Q = Q
        self.c = c
        self.L = np.linalg.cholesky((self.Q /r + self.Q.T/r)/2)
        self.r = r
        self.A = A.value
        self.b = b.value
        return problem.value
    
    @staticmethod 
    def _convert_A_b_to_Q_c_r(A, b):
        """
        Convert ellipsoid of form ||A x - b|| <= 1
        into standard ellipsoid form: (x - c)^T Q (x - c) <= r
        Returns (Q, c, r)
        """
        Q = A.T @ A
        Q_inv = np.linalg.inv(Q)
        c = Q_inv @ A.T @ b
        r = (b.T @ A @ Q_inv @ A.T @ b) - (b.T @ b) + 1
        return Q, c, r

    def _mahalanobis_distance(self, p):
        return np.diag((p-self.c).T@self.Q@(p-self.c))

    def contains_point(self, p):
        """Compatibility wrapper for column-wise points (dimension, N)."""
        return self.contains_points(np.asarray(p).reshape(self.dimension, -1).T)

    def _get_lb_cb(self, ell0):
        L0 = ell0.L
        L0i = np.linalg.inv(L0)
        P = L0i @ self.Q @ L0i.transpose()
        specDecomp = np.linalg.svd(P)
        lb = specDecomp[1]
        cb = specDecomp[0].T @ L0.T @ (self.c - ell0.c)
        return (lb,cb.flatten())

    def _contains_ellipsoid(self, elli1):
        """
        Python implementation of: https://arxiv.org/pdf/2211.06237
        Checks if ellipsoid 1 is in self
        Assumptions:
        - elli1.A, self.A are symmetric positive semidefinite numpy arrays
        - elli1.b, self.b are numpy vectors
        """

        # Pre-Tests
        # e_min = eigmin(elli1.P - self.P)
        e_min = np.min(np.linalg.eigvalsh(elli1.Q - self.Q))

        # elli1 must be smaller than self so elli1.A - self.A must be positive definite
        if e_min < 0:
            # print("Elli1 is larger than self")
            return False

        # If they share a center, positive definiteness is sufficient 
        elif np.allclose(elli1.c, self.c):
            # print("Elli1 shares center and is smaller than self")
            return e_min >= 0

        # If the center of elli1 is outside self, it cannot be contained
        elif not (self.contains_point(elli1.c)):
            # print("Elli1 center is outside self")
            return False

        # Main algorithm
        else:

            # print("Precheck inconclusive")
            # L = cholesky((self.P + elli2.P') / 2).L
            lb, ct = elli1.get_lb_cb(self)
            # print(f"lb: {lb}, cb: {ct}")

            # α = 1 / min(lb...)
            alpha = 1.0 / np.min(lb)

            def polPos(beta):
                return -(1.0 - beta + np.sum((beta * lb / (1.0 - beta * lb)) * (ct ** 2)))
            
            def ell_f(b, lb, cb):
                blb = b*lb
                return -b + np.sum((cb*cb)*blb/(1-blb))

            if alpha == 1.0: # TODO: Check geometric meaning of alpha
                return polPos(1.0) <= 0
            
            if alpha > 1.0 - np.linalg.norm(ct) ** 2:
                return False
            
            objective = lambda beta: -ell_f(beta, lb, ct)

            lower = alpha+1e-10
            upper =  1.0 - np.linalg.norm(ct) ** 2
            
            res = minimize(objective, x0 = (lower+upper/2), bounds=[(lower, upper)])
            f_beta = -res.x

            return f_beta[0] > -1

    # Divides Q by scale_factor
    # Which is equivalent to multiplying RHS of inequality (x-b)T Q (x-b) <= 1 by scale_factor
    def _scale(self, scale_factor):
        self.Q /= scale_factor
        self.L = np.linalg.cholesky((self.Q + self.Q.T)/2)
 
    def volume(self):
        n = self.Q.shape[1]  # dimensionality of the ellipsoid
        volume = ((np.pi ** (n / 2)) / gamma(n / 2 + 1)) / np.sqrt(det(self.Q))
        return volume


    

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Plotting ellipsoids

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

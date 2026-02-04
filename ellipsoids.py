import numpy as np 
import cvxpy as cp
from scipy.special import gamma
from scipy.optimize import minimize, NonlinearConstraint
from numpy.linalg import det, inv


# Calculate Ellipsoid
def p_ball(xs, normp):
    xs = np.array(xs).T
    n = np.shape(xs)[0]
    m = np.shape(xs)[1]

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

    print("Optimal value :", problem.value)
    print("A* =", A.value)
    print("b* =", b.value)
    print("volume proxy:", np.log(np.linalg.det(A.value)))
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Optimization failed with status:", problem.status)
    else:
        print("Optimization succeeded.")
    return problem.value, A.value, b.value

# Function to check for set containment (is a sample in the ellipsoid)
def in_ellipsoid(A, b, p):
    p = np.asarray(p,  dtype=float).ravel()
    b = np.asarray(b,  dtype=float).ravel()  
    return np.linalg.norm(A @ p - b)-1<=0

# Calculate exact volume (not proxy) of ellipsoid
def ellipsoid_volume(A: np.ndarray, b: np.ndarray) -> float:
    Q = A.T @ A
    Q_inv = inv(Q)
    c = Q_inv @ A.T @ b
    r = b.T @ A @ Q_inv @ A.T @ b - b.T @ b + 1

    n = A.shape[1]  # dimensionality of the ellipsoid

    volume = (np.pi ** (n / 2)) / gamma(n / 2 + 1) * (r ** (n / 2)) / np.sqrt(det(Q))
    return volume




# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Plotting ellipsoids

def convert_A_b_to_Q_c_r(A, b):
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

def project_ellipsoid_to_2d(A, b, diml, dimh):
    """
    Project ellipsoid ||A x - b|| <= 1 from nD to 2D.
    Returns (Q_2d, c_2d, r)
    """
    indices = [diml, dimh]
    Q, c, r = convert_A_b_to_Q_c_r(A, b)
    Q_2d = Q[np.ix_(indices, indices)]
    c_2d = c[indices]
    return Q_2d, c_2d, r

def plot_ellipse(Q, c, r, ax, alpha=0.75, color='green', **kwargs):
    """
    Plot ellipse defined by (x-c)^T Q (x-c) <= r in 2D
    """
    eigvals, eigvecs = np.linalg.eigh(Q)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    # Scale the ellipse by sqrt(r)
    ellipse = eigvecs @ np.diag(np.sqrt(r) / np.sqrt(eigvals)) @ circle
    ellipse = ellipse.T + c
    ax.plot(ellipse[:, 0], ellipse[:, 1], **kwargs, color=color, alpha=alpha)

    return ax
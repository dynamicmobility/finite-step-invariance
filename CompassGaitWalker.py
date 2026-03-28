import numpy as np
import jax.numpy as jnp
import jax
from numpy.typing import ArrayLike
from matplotlib.collections import LineCollection
from ellipsoids import Ellipsoid
from diffrax import diffeqsolve, ODETerm, Dopri8, Event
from equinox import filter_jit
import optimistix as optx

# Dynamics taken from Hybrid Limit Cycles of Walking Robots 
# Underactuated Robotics (https://underactuated.mit.edu/simple_legs.html) Also uses these dynamics but the coordinates are flipped
d = 3
g = 9.81
class CompassGaitWalker:
    '''
    Model of a compass gait walker 
    Args
    ---
        mh (float): mass at hip
        m (float): mass at each leg
        a (float): distance from foot to COM of leg
        b (float): distance from COM of leg to hip
        gamma (float): angle of ground
    '''
    def __init__(self, mh: float, m: float, a: float, b: float, gamma: float):
        super().__init__()
        self.mh = mh
        self.m = m
        self.a = a
        self.b = b
        self.l = a+b
        self.gamma = gamma

    @staticmethod
    def inv_analytic(M):
        det_M =  M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1]
        return (1/det_M)*jnp.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) # Compute inverse analytically because jax linalg inv does not support the gradient

    
    def f(self, t: float, x: jnp.ndarray, u):
        tsw, tst, tswd, tstd = x.ravel()

        M = jnp.array([[self.m*self.b**2, -self.m*self.l*self.b*jnp.cos(tst - tsw)],
                      [-self.m*self.l*self.b*jnp.cos(tst - tsw), (self.mh + self.m)*self.l**2 + self.m*self.a**2]])

        C = jnp.array([[0, self.m*self.l*self.b*jnp.sin(tst - tsw)*tstd],
                      [-self.m*self.l*self.b*jnp.sin(tst - tsw)*tswd, 0]])

        G = jnp.array([[self.m*self.b*g*jnp.sin(tsw)],
                      [-(self.mh*self.l + self.m*self.a + self.m*self.l)*g*jnp.sin(tst)]])

        B = jnp.array([[1],
                      [-1]])
        qd = jnp.array([[tswd], [tstd]])
        qdd = self.inv_analytic(M)@(B*u - C@qd - G)

        return jnp.hstack((qd.flatten(), qdd.flatten()))

    def reset(self, t:float, x: jnp.ndarray):
        tsw, tst, tswd, tstd = x.ravel()
        alpha = (tsw-tst)*0.5
        Qm = jnp.array([[-self.m*self.a*self.b, -self.m*self.a*self.b + (self.mh*self.l**2 + 2*self.m*self.a*self.l)*jnp.cos(2*alpha)],
                       [0, -self.m*self.a*self.b]])
        Qp = jnp.array([[self.m*self.b*(self.b-self.l*jnp.cos(2*alpha)), self.m*self.l*(self.l-self.b*jnp.cos(2*alpha)) + self.m*self.a**2 + self.mh*self.l**2],
                       [self.m*self.b**2, -self.m*self.b*self.l*jnp.cos(2*alpha)]])
        qdm = jnp.array([[tswd], [tstd]])
        qdp = self.inv_analytic(Qp)@Qm@qdm # Relabel velocities
        qp = jnp.array([tst, tsw])
        return jnp.hstack((qp, qdp.flatten()))

    def guard(self, t: float, y: jnp.ndarray, *args, **kwargs):
        tsw, tst, tswd, tstd = y.ravel()
        return jnp.where(tst > 0, tsw + tst - 2*self.gamma, -0.001)

    
    def draw_system(self, t, x, offset=jnp.zeros((2,)), *args, **kwargs):
        tsw, tst, tswd, tstd = x.ravel()
        ph = jnp.array([self.l*jnp.sin(tst), self.l*jnp.cos(tst)]) + offset
        pf = jnp.array([self.l*jnp.sin(tst) - self.l*(jnp.sin(tsw)), self.l*jnp.cos(tst)- self.l*jnp.cos(tsw)]) + offset
        line0 = [offset, ph] # Stance Leg 
        line1 = [ph, pf] # Swing Leg
        line2 = [offset - jnp.array([jnp.cos(-self.gamma), jnp.sin(-self.gamma)]), offset + jnp.array([jnp.cos(-self.gamma), jnp.sin(-self.gamma)])] # Floor
        return LineCollection([line0, line1, line2], colors=['r', 'g', 'k'])
    
    def sw_foot_pos(self, x):
        tsw, tst, tswd, tstd = x.ravel()
        return jnp.array([self.l*jnp.sin(tst) - self.l*(jnp.sin(tsw)), self.l*jnp.cos(tst)- self.l*jnp.cos(tsw)]) # Needed for finding the global offset that advances the robot every step
    

# =========================================================================
# System and ODE solver
# =========================================================================
sys    = CompassGaitWalker(m=5., mh=10., a=0.5, b=0.5, gamma=0.0525)
x_star = np.array([-0.21876908, 0.32376908, 1.8083619, 1.49571536])  # fixed point
assert jnp.abs(sys.guard(0, x_star)) < 1e-5, "x_star is not on the guard"

term   = ODETerm(sys.f)
solver = Dopri8()
u      = 0.0
event  = Event(sys.guard, optx.Newton(5e-7, 5e-7, optx.rms_norm), direction=True)

# =========================================================================
# Guard surface chart: 3-D parameterisation of the 4-D guard surface
#   full_state = guard_surface_chart @ guard_coords + x_star
# =========================================================================
_c                 = jnp.array([[1, 1, 0, 0]]).T
_guard_chart       = jnp.array([[1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T
_guard_chart_inv   = jnp.linalg.inv(jnp.hstack((_guard_chart, _c)))

def guard_to_full(x_guard):
    """(3,) or (N,3) guard coords -> (4,) or (N,4) full state"""
    return (_guard_chart @ x_guard.T).T + x_star

def full_to_guard(x_full):
    """(4,) or (N,4) full state -> (3,) or (N,3) guard coords"""
    return ((_guard_chart_inv @ (x_full - x_star).T).T)[..., :3]

# =========================================================================
# Poincaré return map  P : S -> S  (guard coords -> guard coords)
# =========================================================================
@filter_jit
def P_single(x_guard):
    x_full = guard_to_full(x_guard)
    xp     = sys.reset(0, x_full)
    traj   = diffeqsolve(term, solver, t0=0, t1=2, dt0=0.001,
                         y0=xp, args=u, event=event, throw=False)
    return full_to_guard(traj.ys.flatten())

P_batch = jax.vmap(P_single, in_axes=0)   # (N,3) -> (N,3)

J = jax.jacrev(P_single)(jnp.array([0., 0., 0.]))

# Either choose from uniform distribution or guess Q from contraction metric "Contraction" or "Uniform"
def initial_guess(strategy="Contraction"):
    E0 = Ellipsoid(3)
    if strategy == "Uniform":
        pts_0 = np.random.uniform([-0.1, -1, -0.3], [0.1, 1, 0.3], (2000,3))
        E0.fit_p_ball(pts_0, 2)

    elif strategy == "Contraction":
        import cvxpy as cp
        # print(J)
        eigs = np.abs(np.linalg.eigvals(J))
        # Contraction rate is the spectral radius of A
        contr_rate = jnp.max(eigs) + 1e-4
        # print(contr_rate)
        P_cp = cp.Variable(J.shape, symmetric=True)
        cons = []
        cons.append(J.T @ P_cp @ J << contr_rate**2*P_cp)
        cons.append(P_cp >> jnp.eye(J.shape[0]))
        prob = cp.Problem(cp.Minimize(cp.trace(P_cp)), cons)
        prob.solve()

        Q0 = P_cp.value
        E0.Q    = Q0.copy() # Make E0 bigger to see if we get the same result
        E0.c    = np.zeros((3, 1))

    return E0

# sanity-check: fixed point maps to itself
# assert jnp.allclose(P_single(jnp.zeros(3)), jnp.zeros(3), atol=5e-4), \
#     "Fixed point does not map to itself — check system / solver"

# =========================================================================
# INPUTS  (x* = 0 in guard coords by construction)
# =========================================================================
# x_star_guard = np.zeros(3)

# J = np.array(jax.jacrev(P_single)(jnp.zeros(3)))   # (3,3)

# eigs = np.abs(np.linalg.eigvals(J))
# print("Jacobian eigenvalues:", eigs)
# assert np.all(eigs < 1), "Limit cycle is not stable — Lyapunov equation has no PD solution"


if __name__ == "__main__":
    import numpy as np
    M = np.random.random(size=(2,2))
    print(np.linalg.inv(M))
    print(CompassGaitWalker.inv_analytic(M))
    assert np.allclose(np.linalg.inv(M), CompassGaitWalker.inv_analytic(M), atol=1e-8)
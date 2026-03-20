import jax.numpy as jnp
import jax
from numpy.typing import ArrayLike
from matplotlib.collections import LineCollection

# Dynamics taken from Hybrid Limit Cycles of Walking Robots 
# Underactuated Robotics (https://underactuated.mit.edu/simple_legs.html) Also uses these dynamics but the coordinates are flipped
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
    

if __name__ == "__main__":
    import numpy as np
    M = np.random.random(size=(2,2))
    print(np.linalg.inv(M))
    print(CompassGaitWalker.inv_analytic(M))
    assert np.allclose(np.linalg.inv(M), CompassGaitWalker.inv_analytic(M), atol=1e-8)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
import scipy
from sklearn.cluster import KMeans
from sets.parameterized_set import ParameterizedSet

class RBF(ParameterizedSet):
    def __init__(self, dims, thresh):
        super().__init__(dims)
        self.dimensions = dims
        self.num_nballs = 1
        self.centers_opt = None
        self.widths_opt = None
        self.thresh = thresh

    def fit(self, data, num_balls=None):
        """Fit row-wise data, using the current ball count by default."""
        return self.fit_rbf(data, self.num_nballs if num_balls is None else num_balls)

    def params(self) -> np.ndarray:
        """Return flattened centers followed by widths, as used by rbf_eval."""
        if self.centers_opt is None or self.widths_opt is None:
            raise ValueError("Fit the RBF before accessing parameters")
        return np.concatenate([self.centers_opt.flatten(), self.widths_opt.flatten()])

    def load_from_row(self, row: np.ndarray):
        """Restore centers and widths from a 1D parameter row without its index.

        Infer the ball count using this set's dimension. The threshold remains
        the value supplied to the constructor, since params() does not log it.
        """
        row = np.asarray(row, dtype=float)
        if row.ndim != 1 or row.size == 0 or row.size % (self.dimension + 1):
            raise ValueError("row must be a 1D array with num_balls * (dimension + 1) parameters")
        if not np.all(np.isfinite(row)):
            raise ValueError("row must contain only finite parameters")
        num_balls = row.size // (self.dimension + 1)
        split = num_balls * self.dimension
        centers = row[:split].reshape(num_balls, self.dimension).copy()
        widths = row[split:].copy()
        if np.any(widths == 0):
            raise ValueError("RBF widths must be nonzero")
        self.centers_opt = centers
        self.widths_opt = widths
        self.num_nballs = num_balls
        return self

    def sample_from(self, N: int) -> np.ndarray:
        """Sample the level set within the existing [-4, 4]^dimension box."""
        if self.centers_opt is None or self.widths_opt is None:
            raise ValueError("Fit the RBF before sampling")
        if N < 0:
            raise ValueError("N must be nonnegative")
        samples = []
        while len(samples) < N:
            point = np.random.uniform(-4, 4, size=(1, self.dimension))
            if self.contains_points(point)[0]:
                samples.append(point[0])
        return np.asarray(samples).reshape(N, self.dimension)

    def rbf_eval(self, nballs, z, dimensions, p=None, centers=None, widths=None):
        if centers is None and p is not None:
            centers = p[0:dimensions*nballs].reshape((nballs,dimensions))
        if widths is None and p is not None:
            widths = p[dimensions*nballs:(dimensions+1)*nballs]    

        c_list= []
        nfs = np.shape(centers)[0]
        for c_i in range(len(z)):
            c = 0
            for f_idx in range(nfs):
                c += (np.exp(-0.5 * np.linalg.norm(z[c_i] - centers[f_idx,:])**2 / widths[f_idx]**2))
            # help with numerical stability
            if c == 0:
                c = 1e-15
            c_list.append(c)
                
        return np.log(np.min(c_list))
    
    def rbf_stack(self, z, x, y):
        z = np.insert(z, 0, x)
        z = np.insert(z, 2, y)
        z = np.array([z])
        return z

    def plot_2D_rbf(self, xmin, xmax, ymin, ymax, centers, widths, colors=['black']):
        xgrid = np.linspace(xmin, xmax,100)
        ygrid = np.linspace(ymin, ymax,100)
        nx = np.size(xgrid)
        ny = np.size(ygrid)
        X,Y = np.meshgrid(xgrid,ygrid)
        Z = np.zeros(np.shape(X))
        for x_idx in range(nx):
            for y_idx in range(ny):
                x = X[x_idx, y_idx]
                y = Y[x_idx, y_idx]
                z = np.vstack([x,y]).T
                Z[x_idx, y_idx] = self.rbf_eval(self.num_nballs, z, self.dimensions, centers=centers, widths=widths)
        plt.contour(X, Y, Z, levels=[self.thresh], colors=colors)
        plt.plot(centers[:,0], centers[:,1], color=colors[0], marker='o', markerfacecolor='None', markersize=10, linestyle='None')


    def plot_3D_rbf(self, xmin, xmax, ymin, ymax, centers, widths, data, colors=['black']):
        xgrid = np.linspace(xmin, xmax,100)
        ygrid = np.linspace(ymin, ymax,100)
        nx = np.size(xgrid)
        ny = np.size(ygrid)
        X,Y = np.meshgrid(xgrid,ygrid)
        Z = np.zeros(np.shape(X))
        for x_idx in range(nx):
            for y_idx in range(ny):
                x = X[x_idx, y_idx]
                y = Y[x_idx, y_idx]
                [xopt, fopt, it, funclls, warnflg] =  scipy.optimize.fmin((lambda z : -(self.rbf_eval(self.num_nballs, self.rbf_stack(z, x, y), self.dimensions, centers=centers, widths=widths))), [np.mean(data[:][1])], full_output=True)
                Z[x_idx, y_idx]  = np.float64(-fopt)
        plt.contour(X, Y, Z, levels=[self.thresh])
        plt.plot(centers[:,0], centers[:,2], colors=colors, marker='o', markerfacecolor='None', markersize=10, linestyle='None')
        plt.xlabel('$x$')
        plt.ylabel('$y$')

    def rbf_solve(self, nballs, thresh, data, dimensions, centers_init=None):
        def rbf_badness(params, nballs, data, thresh):
            centers = params[0:dimensions*nballs].reshape((nballs,dimensions))
            widths = params[dimensions*nballs:(dimensions+1)*nballs]
            b = 0
            ndata = np.shape(data)[0]
            for data_idx in range(ndata):
                z = data[data_idx:data_idx + 1, :]
                lev = self.rbf_eval(self.num_nballs, z, self.dimensions, centers=centers, widths=widths)
                # heavily punish points near the boundary
                point_badness = np.exp(-(lev - thresh))
                b += point_badness
            b /= ndata
            b += 10*np.linalg.norm(widths)
            return b

        if centers_init is None:
            centers_init = 5*np.random.normal(size=dimensions*nballs)
        centers_init = centers_init.reshape((dimensions*nballs,))
        widths_init = 5*np.ones((nballs,))
        params_init = np.concatenate((centers_init, widths_init))
        
        obj = lambda p: rbf_badness(p, nballs, data, thresh)

        nlc = NonlinearConstraint((lambda p :self.rbf_eval(nballs, data, dimensions, p=p)-thresh), 0,  np.inf)

        params_opt = scipy.optimize.minimize(obj, params_init, constraints=(nlc), method="SLSQP").x
        centers_opt = params_opt[0:dimensions*nballs].reshape((nballs,dimensions))
        widths_opt = params_opt[dimensions*nballs:(dimensions+1)*nballs]
        return centers_opt, widths_opt

    def fit_rbf(self, data, num_balls):

        self.num_nballs = num_balls
        thresh = self.thresh

        kmeans = KMeans(n_clusters=self.num_nballs, random_state=0).fit(data)
        centers_init = kmeans.cluster_centers_ 

        centers_opt, widths_opt = self.rbf_solve(self.num_nballs, thresh, data, self.dimensions, centers_init=centers_init)

        self.centers_opt = centers_opt
        self.widths_opt = widths_opt

        return self

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("points must have shape (N, dimension)")
        if self.centers_opt is None or self.widths_opt is None:
            raise ValueError("Fit the RBF before testing membership")
        bool_list = []
        for p in points:
            c = 0
            for nc in range(np.shape(self.centers_opt)[0]):
                c += (np.exp(-0.5 * np.linalg.norm(p - self.centers_opt[nc, :])**2 / self.widths_opt[nc]**2))
            if c == 0:
                c = 1e-15
            bool_list.append(np.log(c) >= self.thresh)

        return np.array(bool_list, dtype=bool)

    def contains_point(self, points):
        """Compatibility wrapper for row-wise points."""
        return self.contains_points(points)


def sample_uniform_levelset(r: RBF, N: int):
    """Compatibility wrapper for RBF.sample_from."""
    return r.sample_from(N)

if __name__ == "__main__":
    from scipy.integrate import odeint
    from numpy.random import default_rng

    # make sure it works on duffing oscillator
    def chaotic_oscillator(y,t, alpha = 0.05, omega = 1.3, gamma = 0.4):
        dydt = [y[1], -alpha*y[1] + y[0] - y[0]**3+gamma*np.cos(omega*t)]
        return dydt

    def make_sample_n(sample_fn, parallel=False, pool=None):
        def sample_n(n, pool=pool):
            return np.array([sample_fn() for i in range(n)])
        return sample_n

    def sample_oscillator(x=None):
        t = np.linspace(0, 100, 1001)
        rng = default_rng()
        y0 = np.array([rng.uniform(0.95, 1.05), rng.uniform(-0.05, 0.05)])
        sol = odeint(chaotic_oscillator, y0, t)
        succ = sol[-1]
        return np.flip(succ)

    sample_oscillator_n = make_sample_n(sample_oscillator)

    ndata = 1500
    sampler = sample_oscillator_n
    data = sampler(ndata)

    r = RBF(2, np.log(0.25))
    r.fit_rbf(data, 2)
    plt.plot(data[:,0],data[:,1],'b.')
    r.plot_2D_rbf(-8,8,-8,8,r.centers_opt,r.widths_opt)
    plt.show()


    ## if no level set, just try again

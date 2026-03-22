import numpy as np
from ellipsoids import Ellipsoid, plot_ellipse 
from matplotlib import pyplot as plt

def in_ellipsoid(A, b, xs):
    return np.linalg.norm(A @ xs - b) <= 1


def ellipsoid_bounding_box(A, b):
    A_inv = np.linalg.inv(A)

    c = (A_inv @ b).flatten()
    deltas = np.linalg.norm(A_inv, axis=1)

    mins = c - deltas
    maxs = c + deltas

    return mins, maxs


def combined_bounding_box(E1, E2):
    mins1, maxs1 = ellipsoid_bounding_box(E1.A, E1.b)
    mins2, maxs2 = ellipsoid_bounding_box(E2.A, E2.b)

    mins = np.minimum(mins1, mins2)
    maxs = np.maximum(maxs1, maxs2)

    return mins, maxs

import numpy as np

    
def monte_carlo_intersection(E1, E2, n_samples, ax):
    mins, maxs = combined_bounding_box(E1, E2)

    points = np.random.uniform(mins, maxs, size=(n_samples, len(mins)))

    # ax.scatter(points[:, 0], points[:, 1], c='black', marker='o')

    def inside(A, b, pts):
        vals = (A @ pts.T - b).T
        return np.sum(vals**2, axis=1) <= 1

    in_both = inside(E1.A, E1.b, points) & inside(E2.A, E2.b, points)

    box_vol = np.prod(maxs - mins)

    return box_vol * np.mean(in_both)


if __name__ == "__main__":
    E0 = Ellipsoid(2)
    E1 = Ellipsoid(2)

    E0.A = np.array([[1/4, 0],[0, 1/9]])
    E0.b = np.array([[-1, 1]]).T
    E0.L = np.linalg.cholesky((E0.A + E0.A.T)/2)

    E1.A = np.array([[1/3, -1/5],[-1/5, 1/2]])
    E1.b = np.array([[-1.5, 1.5]]).T
    E1.L = np.linalg.cholesky((E1.A + E1.A.T)/2)

    fig, ax = plt.subplots()


    plot_ellipse(E0.A, E0.b.flatten(), 1, ax, color='orange')
    plot_ellipse(E1.A, E1.b.flatten(), 1, ax, color='blue')
    estimate = monte_carlo_intersection(E0, E1, 3000, ax)
    print("Ellipsoid 1 volume:", E0.ellipsoid_volume())
    print("Ellipsoid 2 volume:", E1.ellipsoid_volume())
    print("Estimated intersection volume:", estimate)
    plt.show()
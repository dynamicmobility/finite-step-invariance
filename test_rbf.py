"""Check and visualize the two-center RBF: python test_rbf.py."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sets.radialBasisFunction import RBF

width=0.5

def expected_membership(points, threshold):
    """Independent analytic Gaussian sum for centers at (+/-0.6, 0)."""
    x, y = points.T
    values = np.exp(-0.5 * ((x - 0.6) ** 2 + y ** 2)/width**2)
    values += np.exp(-0.5 * ((x + 0.6) ** 2 + y ** 2)/width**2)
    return np.log(values) >= threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error('--samples must be positive')
    rng = np.random.default_rng(args.seed)

    rbf = RBF(2, thresh=np.log(0.25))
    rbf.load_from_row(np.array([-0.6, 0., 0.6, 0., width, width, rbf.thresh]))

    X, Y = np.meshgrid(np.linspace(-4, 4, 41), np.linspace(-4, 4, 41))
    points = np.column_stack((X.ravel(), Y.ravel()))
    inside = rbf.contains_points(points)
    assert inside.shape == (len(points),) and inside.dtype == bool
    np.testing.assert_array_equal(inside, expected_membership(points, rbf.thresh))
    np.testing.assert_array_equal(
        rbf.contains_points(np.array([[-0.6, 0.], [0.6, 0.], [0., 0.], [4., 4.]])),
        [True, True, True, False],
    )
    assert rbf.contains_points(np.empty((0, 2))).shape == (0,)

    samples = rbf.sample_from(args.samples, rng=rng)
    assert samples.shape == (args.samples, 2)
    assert np.isfinite(samples).all()
    assert np.all(np.abs(samples) <= 4)
    assert rbf.contains_points(samples).all()
    assert expected_membership(samples, rbf.thresh).all()
    assert rbf.sample_from(0, rng=rng).shape == (0, 2)
    print(f'PASS: analytic membership checks; {len(samples)} samples inside the set.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(*points[inside].T, s=12, color='tab:blue', label='Inside')
    axes[0].scatter(*points[~inside].T, s=12, color='tab:red', label='Outside')
    axes[0].set_title('contains_points: grid classification')
    axes[1].scatter(*samples.T, s=8, alpha=0.5, color='tab:blue', label='Samples')
    axes[1].set_title(f'sample_from: {len(samples)} accepted points')
    for ax in axes:
        rbf.plot(ax, color='black', label='RBF centers')
        ax.set(xlabel='x', ylabel='y', xlim=(-4, 4), ylim=(-4, 4))
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(f'Centers at (±0.6, 0), widths = {width}, threshold = log(0.25)')
    fig.tight_layout()
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()

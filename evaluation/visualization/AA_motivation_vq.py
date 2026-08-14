"""AA -- motivation_vq.pdf: scalar vs vector quantization of a 2-D Gaussian.

Three panels at a fixed 6-bit budget: 1-D uniform, 1-D k-means, and 2-D k-means
codebooks over the same point cloud, annotated with quantization MSE. This is
the figure that motivates vector quantization.


Synthetic -- no input data. Seeded (np.random.seed(0)), but note that KMeans
itself is only reproducible up to scikit-learn's own initialization; the
centroid positions can shift slightly between versions while the MSE ordering
holds.

    python AA_motivation_vq.py [--use-simulation] [--out-dir DIR]
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import _paths
import _style

FIGURE = "motivation_vq.pdf"

N = 1000            # points
BITS_2D = 6         # codebook budget, shared across all three panels
K_1D = 2 ** (BITS_2D // 2)
K_2D = 2 ** BITS_2D
POINT_SIZE = 20


def _int_1d(k, value_range):
    """Uniform-grid centroids: the scalar-quantization baseline."""
    a, b = value_range
    step = (b - a) / k
    return a + (np.arange(k) + 0.5) * step


def _kmeans_1d(points_1d, k):
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(points_1d.reshape(-1, 1))
    return km.cluster_centers_.flatten()


def _kmeans_2d(points, k):
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(points)
    return km.cluster_centers_


def _outer_grid(centroids_1d):
    """A 1-D codebook applied per dimension spans the Cartesian product --
    which is exactly why it wastes codewords on empty regions."""
    return np.array([[cx, cy] for cx in centroids_1d for cy in centroids_1d])


def _mse(points, centroids):
    dists = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.mean(np.min(dists, axis=1))


def plot_AA_motivation_vq(seed=0):
    _style.use(_style.FONT_MOTIVATION)

    np.random.seed(seed)
    points = np.random.multivariate_normal([0, 0], [[2, 0.2], [0.2, 2]], N)

    lo, hi = np.percentile(points.flatten(), [1, 99])
    grids = [
        ("1D Uniform", _outer_grid(_int_1d(K_1D, (lo, hi)))),
        ("1D Non-uniform", _outer_grid(_kmeans_1d(points.flatten(), K_1D))),
        ("2D Non-uniform", _kmeans_2d(points, K_2D)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ticks = np.linspace(-4, 4, 5)

    for ax, (title, centroids) in zip(axes, grids):
        ax.scatter(points[:, 0], points[:, 1], s=POINT_SIZE, alpha=0.3)
        ax.scatter(centroids[:, 0], centroids[:, 1], color="red", marker="x")
        ax.set_title(title)

        mse = _mse(points, centroids)
        print(f"  MSE ({title}): {mse:.4f}")
        ax.text(0.05, 0.95, f"MSE: {mse:.2f}", transform=ax.transAxes,
                fontsize=_style.FONT_MOTIVATION, va="top")

        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(2)
        ax.set_xlim(-4.1, 4.1)
        ax.set_ylim(-4.1, 4.1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{t:.0f}" for t in ticks])
        ax.set_yticklabels([f"{t:.0f}" for t in ticks])

    plt.tight_layout()
    path = _style.save(plt, _paths.figure_path(FIGURE))
    plt.close(fig)
    return path


if __name__ == "__main__":
    _paths.parse_args(__doc__)
    plot_AA_motivation_vq()

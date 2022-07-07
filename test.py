import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

figs, axs = plt.subplots(1, figsize=(6, 6))


pts = np.array([[-16.0823566, 16.43046557],
                [38.99491754, 96.56723232],
                [47.08536459, 78.93714363],
                [9.58925423, 25.17265831],
                [11.92303703, 13.80780775],
                [-8.38087335, -6.88204839]])

hull = ConvexHull(pts)
print(in_hull(np.array([0, 20]), pts))
print(hull.simplices)
plt.plot([0], [20], 'o')
for simplex in hull.simplices:
    plt.plot(pts[simplex, 0], pts[simplex, 1], 'k-')
plt.show()
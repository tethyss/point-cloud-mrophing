from __future__ import annotations
import time
import matplotlib.pyplot as plt  # type: ignore
import tqdm  # type: ignore
import numpy as np
from scipy.spatial.distance import cdist  # type: ignore


class ThinPlateSpline:

    def __init__(self, alpha=0.0) -> None:
        self._fitted = False  # check if fitted
        self.alpha = alpha  #

        self.parameters = np.array([], dtype=np.float32)
        self.control_points = np.array([], dtype=np.float32)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ThinPlateSpline:
        """Learn f that matches Y given X

        Args:
            X (ndarray): Control point at source space (X_c)
                Shape: (n_c, d_s)
            Y (ndarray): Control point in the target space (X_t)
                Shape: (n_c, d_t)

        Returns:
            ThinPlateSpline: self
        """
        assert X.shape[0] == Y.shape[0]

        n_c, d_s = X.shape
        self.control_points = X

        phi = self._radial_distance(X)

        # Build the linear system AP = Y
        X_p = np.hstack([np.ones((n_c, 1)), X])

        A = np.vstack(
            [np.hstack([phi + self.alpha * np.identity(n_c), X_p]), np.hstack([X_p.T, np.zeros((d_s + 1, d_s + 1))])]
        )

        Y = np.vstack([Y, np.zeros((d_s + 1, Y.shape[1]))])

        self.parameters = np.linalg.solve(A, Y)
        self._fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map source space to target space

        Args:
            X (ndarray): Points in the source space
                Shape: (n, d_s)

        Returns:
            ndarray: Mapped point in the target space
                Shape: (n, d_t)
        """
        assert self._fitted

        assert X.shape[1] == self.control_points.shape[1]

        phi = self._radial_distance(X)  # n x n_c

        X = np.hstack([phi, np.ones((X.shape[0], 1)), X])  # n x (n_c + 1 + d_s)
        return X @ self.parameters

    def _radial_distance(self, X: np.ndarray) -> np.ndarray:
        """Compute the pairwise radial distances of the given points to the control points

        Input dimensions are not checked.

        Args:
            X (ndarray): N points in the source space
                Shape: (n, d_s)

        Returns:
            ndarray: The radial distance for each point to a control point (\\Phi(X))
                Shape: (n, n_c)
        """
        dist = cdist(X, self.control_points)
        dist[dist == 0] = 1  # phi(r) = r^2 log(r) ->  (phi(0) = 0)
        return dist**2 * np.log(dist)


d_s = 25
d_t = 25
n_c = 200
n = 20

X_c = np.random.normal(0, 100, (n_c, d_s))
X_t = np.random.normal(0, 2, (n_c, d_t))
X = np.random.normal(0, 100, (n, d_s))

tps = ThinPlateSpline()

tps.fit(X_c, X_t)
X1 = tps.transform(X)

plt.plot([X[:, 0], X1[:, 0]], [X[:, 1], X1[:, 1]], c=[.5, .5, 1], alpha=0.2)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X1[:, 0], X1[:, 1])
plt.show()
import numpy as np
from sklearn.neighbors import NearestNeighbors


class BorderlineSMOTE(object):
    """
    Over-sampling using Borderline SMOTE.
    This algorithm is a variant of the original SMOTE algorithm.
    Borderline samples will be detected and used to generate new synthetic samples.
    Created according to [1]_.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest neighbours to used to construct synthetic samples.
    m_neighbors : int, default=10
        Number of nearest neighbours to use to determine if a minority
        sample is in danger.

    Attributes
    ----------
    nn_k_ : estimator object
        Validated k-nearest neighbours created from the `k_neighbors` parameter.
    nn_m_ : estimator object
        Validated m-nearest neighbours created from the `m_neighbors` parameter.

    References
    ----------
    .. [1] H. Han, W. Wen-Yuan, M. Bing-Huan, "Borderline-SMOTE: a new
       over-sampling method in imbalanced data sets learning," Advances in
       intelligent computing, 878-887, 2005.
    """

    def __init__(
            self,
            k_neighbors=5,
            m_neighbors=10,
    ):
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors

    @staticmethod
    def get_neighbors_object(n_neighbors, additional_neighbor=0):
        return NearestNeighbors(n_neighbors=n_neighbors + additional_neighbor)

    def _in_danger_noise(self, nn_estimator, samples, target_class, y):
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)

        return np.bitwise_and(
            n_maj >= (nn_estimator.n_neighbors - 1) / 2,
            n_maj < nn_estimator.n_neighbors - 1,
        )

    @staticmethod
    def get_target_class_label(y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmin(counts)]

    def _get_danger_index(self, X, y, target_class):
        self.nn_m_ = self.get_neighbors_object(self.m_neighbors, 1)
        target_class_indices = np.flatnonzero(y == target_class)
        X_class = X[target_class_indices]
        self.nn_m_.fit(X)
        danger_index = self._in_danger_noise(self.nn_m_, X_class, target_class, y)
        return danger_index

    def fit_resample(self, X, y, target_class=None, s=3):
        if s > self.k_neighbors:
            raise ValueError(
                f"s cannot be greater than k ({self.k_neighbors})"
            )
        target_class = self.get_target_class_label(y) if target_class is None else target_class
        danger_index = self._get_danger_index(X, y, target_class)
        P = X[y == target_class]
        self.nn_k_ = self.get_neighbors_object(self.k_neighbors)
        self.nn_k_.fit(P)
        nns = self.nn_k_.kneighbors(P, return_distance=False)[:, :][danger_index]
        danger_points = P[danger_index]
        for p0, neighbours in zip(danger_points, nns):
            indices = np.random.choice(neighbours, s, False)
            chosen = P[indices]
            new_points = np.array([p0 + (p1 - p0) * np.random.uniform() for p1 in chosen])
            X = np.vstack((X, new_points))
            y = np.hstack((y, np.full(s, target_class)))
        return X, y
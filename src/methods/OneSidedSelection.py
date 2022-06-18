from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors


class OneSidedSelection:
    def __init__(self, n_neighbours=None, random_state=None, n_seed_S=1):
        self.n_neighbours = n_neighbours
        self.random_state = random_state
        self.n_seed_S = n_seed_S

    def _knn_estimator(self):
        if self.n_neighbours is None:
            self.knn = KNeighborsClassifier(n_neighbors=1)
        elif isinstance(self.n_neighbours, int):
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbours)
        else:
            raise ValueError()

    def fit_resample(self, X, y):
        self._knn_estimator()

        ## For reproducibility purposes
        random_state = check_random_state(self.random_state)

        ## Get the number of occurences of each class
        target_stats = Counter(y)

        ## Get the smallest class in the dataset
        minority_class = min(target_stats, key=lambda v: target_stats[v])

        idx_under = np.empty((0,), dtype=int)

        minority_class_indices = np.where(y == minority_class)[0]
        for target_class in np.unique(y):
            if target_class != minority_class:
                idx_majority = np.where(y == target_class)[0]
                select_idx_majority = random_state.randint(
                    low=0, high=target_stats[target_class], size=min(self.n_seed_S, target_stats[target_class])
                )

                idx_majority_sample = idx_majority[select_idx_majority]

                C_indices = np.append(minority_class_indices, idx_majority_sample)
                C_X = X[C_indices]
                C_y = y[C_indices]

                idx_majority_extracted = np.delete(idx_majority, select_idx_majority, axis=0)
                S_X = X[idx_majority_extracted]
                S_y = y[idx_majority_extracted]

                self.knn.fit(C_X, C_y)
                pred_S_y = self.knn.predict(S_X)

                S_missclassified_indices = np.where(pred_S_y != S_y)[0]
                idx_tmp = idx_majority_extracted[S_missclassified_indices]
                idx_under = np.concatenate((idx_under, idx_majority_sample, idx_tmp))
            else:
                idx_under = np.concatenate(
                    (idx_under, np.where(y == target_class)[0]), axis=0
                )

        X_resampled = X[idx_under]
        y_resampled = y[idx_under]

        # Tomek https://www.youtube.com/watch?v=bBMfZ9xBSnA
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_resampled)
        nns = nn.kneighbors(X_resampled, return_distance=False)[:, :]

        first = nns[:, 0]
        second = nns[:, 1]

        first_label = y_resampled[first]
        second_label = y_resampled[second]

        first_inconsistent = (first_label != minority_class) & (second_label == minority_class)
        second_inconsistent = (second_label != minority_class) & (first_label == minority_class)
        delete_it = list(set(first[np.where(first_inconsistent)]) & set(second[np.where(second_inconsistent)]))

        X_cleaned = np.delete(X_resampled, delete_it, axis=0)
        y_cleaned = np.delete(y_resampled, delete_it)

        # t = TomekLinks()
        # X_cleaned, y_cleaned = t.fit_resample(X_resampled, y_resampled)

        return X_cleaned, y_cleaned

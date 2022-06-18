from sklearn.neighbors import NearestNeighbors
import random
import numpy as np
from typing import List


class SafeLevelSmote:
    def __init__(self, k_neighbors: int = 5):
        self._k_neighbors = k_neighbors
        self._d_prim: List[List[float]] = []
        self._nearest_neighbors: NearestNeighbors = None

    def _get_neigbors(self, data: np.array, sample: np.array, k_neighbors: int = None):
        if not self._nearest_neighbors:
            self._nearest_neighbors = NearestNeighbors(n_neighbors=k_neighbors or self._k_neighbors)
            self._nearest_neighbors.fit(data)
        return self._nearest_neighbors.kneighbors([sample], return_distance=False)[0]

    def fit_resample(self, data: np.array, y: np.array) -> List[List[float]]:
        for positive_instance in data[data[:, 2] == 1]:
            k_nearest_neighbors = self._get_neigbors(data, positive_instance)
            n_neighbor_idx: int = random.choice(k_nearest_neighbors)
            positive_neigbours = [i for i in k_nearest_neighbors if data[i][2] == 1]
            slp = len(positive_neigbours)
            positive_neigbours = [i for i in self._get_neigbors(data, data[n_neighbor_idx]) if data[i][2] == 1]
            sln = len(positive_neigbours)
            if sln != 0:
                sl_ratio = slp / sln
            else:
                sl_ratio = float('inf')
            if sl_ratio == float('inf') and slp == 0:
                continue
            else:
                range_of_attrs = range(1, data.shape[1] + 1)
                s = [None for i in range_of_attrs]
                for atti in range_of_attrs:
                    if sl_ratio == float('inf') and slp != 0:
                        gap = 0
                    elif sl_ratio == 1:
                        gap = random.random()
                    elif sl_ratio > 1:
                        gap = random.random() / sl_ratio
                    elif sl_ratio < 1:
                        gap = random.uniform(1 - sl_ratio, 1)
                    dif = data[n_neighbor_idx][atti - 1] - positive_instance[atti - 1]
                    s[atti - 1] = positive_instance[atti - 1] + gap * dif
                self._d_prim.append(s)
        return self._d_prim

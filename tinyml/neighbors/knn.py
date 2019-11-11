from collections import Counter

import numpy as np

from tinyml.utils import normalize


class KNN:

    def __init__(self, n_neighbors, weights, p):
        self.n_neighbors = n_neighbors
        assert weights in ("uniform", "distance")

        self.weights = weights  
        self.p = p

        self.train_x, self.train_y = None, None

    def fit(self, x, y):
        self.train_x, self.train_y = x, y

    def predict(self, x):
        predictions = []
        for test_sample in x:
            dists = np.array([self._dist_func(test_sample, train_sample)
                              for train_sample in self.train_x])
            neighbors = np.argsort(dists)[:self.n_neighbors]

            # calculate weights
            if self.weights == "uniform":
                weights = np.ones(self.n_neighbors, dtype=float) / self.n_neighbors
            else:
                weights = normalize(1.0 / (dists[neighbors] + 1e-8))

            pred = self._agg_func(self.train_y[neighbors], weights)
            predictions.append(pred)
        return np.asarray(predictions)

    def _dist_func(self, x1, x2):
        """Minkowski-distance"""
        return np.sum((x1 - x2) ** self.p) ** (1.0 / self.p)

    def _agg_func(self, y, weights):
        """aggregation function"""
        raise NotImplementedError


class KNNClassifier(KNN):

    def __init__(self, 
                 n_neighbors=5, 
                 weights="uniform", 
                 p=2):
        super().__init__(n_neighbors, weights, p)

    def _agg_func(self, y, weights):
        score_dict = dict()
        for cls in np.unique(y):
            score_dict[cls] = weights[y == cls].sum()
        return sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)[0][0]


class KNNRegressor(KNN):

    def __init__(self,
                 n_neighbors=5,
                 weights="uniform",
                 p=2):
        super().__init__(n_neighbors, weights, p)

    def _agg_func(self, y, weights):
        return np.sum(y * weights)

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


class KDTreeNode:

    def __init__(self,
                 left=None,
                 right=None,
                 depth=None,
                 value=None):
        self.left = left
        self.right = right
        self.depth = depth
        self.value = value

        self.is_leaf = left is None and right is None
        self.is_adopted = False


class KDTree(KNN):

    def __init__(self, n_neighbors, weights, p):
        super().__init__(n_neighbors, weights, p)
        self.root = None

    def fit(self, x, y):
        self.root = self._build_tree(x, y)

    def predict(self, x):
        # TODO
        pass
        

    def _build_tree(self, x, y, curr_depth=0):
        n_samples, n_feats = x.shape
        if not n_samples:
            return None

        col = curr_depth % n_feats
        split_idx, l_idx, r_idx = self._split(x[:, col])

        left = self._build_tree(x[l_idx], y[l_idx], curr_depth + 1)
        right = self._build_tree(x[r_idx], y[r_idx], curr_depth + 1)
        return KDTreeNode(left, right, 
                          value=(x[split_idx], y[split_idx]),
                          depth=curr_depth)

    def _split(self, x):
        """return index of split point, left part and right part"""
        median = np.percentile(x, 50, interpolation="nearest")
        split_idx = list(x).index(median)
        l_idx = x < x[split_idx]
        r_idx = x > x[split_idx]
        return split_idx, l_idx, r_idx


class KDTreeClassifier(KDTree):
    
    def __init__(self):
        super().__init__()


class KDTreeRegressor(KDTree):

    def __init__(self, n_neighbors=5, weights="uniform", p=2):
        super().__init__(n_neighbors, weights, p)


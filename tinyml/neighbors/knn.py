from collections import Counter

import numpy as np

from tinyml.utils import normalize


class KDTreeNode:

    def __init__(self, left=None, right=None, depth=None, x=None, y=None):
        self.left = left
        self.right = right
        self.depth = depth
        self.x = x
        self.y = y


class KNN:

    def __init__(self, n_neighbors, weights, p, algorithm):
        self.k = n_neighbors
        assert weights in ("uniform", "distance")
        assert algorithm in ("brute", "kd_tree")

        self.algorithm = algorithm
        self.weights = weights  
        self.p = p

        self.train_x, self.train_y = None, None

    def fit(self, x, y):
        if self.algorithm == "kd_tree":
            if y.ndim == 1:
                y = y.reshape((-1, 1))
            xy = np.concatenate([x, y], axis=1)
            self.n_feats = x.shape[1]
            self.root = self._build_kd_tree(xy)
        else:  # brute-force
            self.train_x, self.train_y = x, y

    def predict(self, x):
        preds = []
        for sample in x:
            neighbors, dists = self._get_neighbors(sample)
            if self.weights == "uniform":
                weights = np.ones(self.k, dtype=float) / self.k
            else:
                weights = normalize(1.0 / (dists + 1e-8))

            pred = self._agg_func(neighbors, weights)
            preds.append(pred)
        return np.asarray(preds)

    def _dist_func(self, x1, x2):
        """Minkowski-distance"""
        return np.sum((x1 - x2) ** self.p) ** (1.0 / self.p)

    @staticmethod
    def _agg_func(y, weights):
        """aggregation function"""
        raise NotImplementedError

    def _get_neighbors(self, sample):
        if self.algorithm == "kd_tree":
            self.nodes = [None for _ in range(self.k)]
            self.dists = [-float("inf") for _ in range(self.k)]
            # get k nearest nodes and their distances
            self._search_kd_tree(self.root, sample)
            neighbors = np.ravel([n.y for n in self.nodes])
            dists = np.array(self.dists)
        else:
            dists = np.array([self._dist_func(sample, train_sample)
                              for train_sample in self.train_x])
            neighbors_idx = np.argsort(dists)[:self.k]
            dists = dists[neighbors_idx]
            neighbors = self.train_y[neighbors_idx]
        return neighbors, dists

    def _build_kd_tree(self, xy, curr_depth=0):
        n_samples = len(xy)
        if not n_samples:
            return None

        col = curr_depth % self.n_feats
        xy = xy[np.argsort(xy[:, col])]
        m = n_samples // 2

        left = self._build_kd_tree(xy[:m], curr_depth + 1)
        right = self._build_kd_tree(xy[m+1:], curr_depth + 1)
        return KDTreeNode(left, right, 
                          x=xy[m, :self.n_feats], 
                          y=xy[m, self.n_feats:],
                          depth=curr_depth)

    def _search_kd_tree(self, node, sample):
        if node is None:
            return 
        # step1: find leaf node
        col = node.depth % self.n_feats
        if sample[col] < node.x[col]: 
            self._search_kd_tree(node.left, sample)
        else:
            self._search_kd_tree(node.right, sample)
        # step2: update nodes and dists if needed
        dist = self._dist_func(node.x, sample)
        for i in range(self.k):
            if self.dists[i] == -float("inf") or dist < self.dists[i]:
                self.dists.insert(i, dist)
                self.dists = self.dists[:-1]
                self.nodes.insert(i, node)
                self.nodes = self.nodes[:-1]
                break
        # step3: go down if needed
        if abs(sample[col] - node.x[col]) < max(self.dists):
            if sample[col] < node.x[col]:
                self._search_kd_tree(node.right, sample)
            else:
                self._search_kd_tree(node.left, sample)


class KNNClassifier(KNN):

    def __init__(self, n_neighbors=5, weights="uniform", p=2, algorithm="kd_tree"):
        super().__init__(n_neighbors, weights, p, algorithm)

    @staticmethod
    def _agg_func(y, weights):
        score = dict()
        for cls in np.unique(y):
            score[cls] = weights[y == cls].sum()
        return max(score.items(), key=lambda kv: kv[1])[0]


class KNNRegressor(KNN):

    def __init__(self, n_neighbors=5, weights="uniform", p=2, algorithm="kd_tree"):
        super().__init__(n_neighbors, weights, p, algorithm)

    @staticmethod
    def _agg_func(y, weights):
        result = 0.0
        for sample, weight in zip(y, weights):
            result += weight * sample
        return result

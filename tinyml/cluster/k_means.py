from collections import defaultdict

import numpy as np


class KMeans:

    def __init__(self, 
                 n_clusters=3, 
                 init="k-means++", 
                 n_init=5, 
                 max_iter=300,
                 tol=1e-4):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        assert init in ("random", "k-means++")
        self.init = init

        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, x):
        best_centers, best_inertia = None, float("inf")
        for _ in range(self.n_init):
            centers, inertia = self._fit_one_time(x)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers

        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia

    def _fit_one_time(self, x):
        centers = self._get_init_centers(x)
        for i in range(self.max_iter):
            # E step
            cls_dict = defaultdict(list)
            for sample in x:
                cls = np.argmin([self._get_distance(sample, c) for c in centers])
                cls_dict[cls].append(sample)
            # M step
            new_centers = []
            for cls, samples in cls_dict.items():
                new_centers.append(self._get_center(samples))
            new_centers = np.array(new_centers)

            if self._get_distance(centers, new_centers) < self.tol:
                break
            centers = new_centers
        return centers, self._get_inertia(cls_dict, centers)

    def predict(self, x):
        if x.ndim == 1:
            return self._predict_sample(x)
        else:
            return np.array([self._predict_sample(sample) for sample in x])

    def _predict_sample(self, sample):
        return np.argmin([self._get_distance(sample, c) 
                          for c in self.cluster_centers_])

    def _get_center(self, samples):
        return np.mean(samples, axis=0)

    def _get_init_centers(self, x):
        if self.init == "random":
            center_idx = np.random.choice(range(len(x)), size=self.n_clusters)
            return x[center_idx]
        else:
            center_idx = list()
            center_idx.append(np.random.choice(len(x)))
            for i in range(1, self.n_clusters):
                dists = np.empty(len(x))
                for i, sample in enumerate(x):
                    d = np.min([self._get_distance(sample, x[c]) for c in center_idx])
                    dists[i] = d
                p = dists / dists.sum()
                new_center_idx = np.random.choice(len(x), p=p)
                center_idx.append(new_center_idx)
            return x[center_idx]

    @staticmethod
    def _get_inertia(cls_dict, centers):
        inertia = 0.0
        for cls, samples in cls_dict.items():
            inertia += sum([np.linalg.norm(centers[cls] - s) for s in samples])
        return inertia

    @staticmethod
    def _get_distance(x1, x2, ord=None):
        return np.linalg.norm(x1 - x2, ord=ord)


class KMedoids(KMeans):

    def _get_center(self, samples):
        distances = []
        for i in range(len(samples)):
            d = 0.0
            for j in range(len(samples)):
                d += self._get_distance(samples[i], samples[j], ord=1)
            distances.append(d)
        return samples[np.argmin(distances)]

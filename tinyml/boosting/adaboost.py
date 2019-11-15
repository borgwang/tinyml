import copy

import numpy as np
from tqdm import tqdm

from tinyml.linear_model import LinearRegressor
from tinyml.linear_model import LogisticRegressor
from tinyml.tree import DecisionTreeClassifier
from tinyml.tree import DecisionTreeRegressor


class AdaBoost:

    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        self.learners = None
        self.weights = None

    def fit(self, x, y):
        self.weights = np.zeros(self.n_estimators)
        data_dist = np.ones(len(x), dtype=float) / len(x)

        for i, learner in enumerate(tqdm(self.learners)):
            # fit model using data with distribution data_dist
            sample_idx = np.random.choice(a=range(len(x)), size=len(y),
                                          p=data_dist)
            learner.fit(x[sample_idx], y[sample_idx])
            pred = learner.predict(x)

            weight, data_dist = self._get_weights_and_dist(y, pred, data_dist)
            self.weights[i] = weight

    def predict(self, x):
        if x.ndim == 1:
            return self._predict_sample(x)
        else:
            return np.array([self._predict_sample(s) for s in x])

    def _predict_sample(self, sample):
        all_preds = np.array([l.predict(sample) for l in self.learners])
        return self._agg_func(all_preds, weights=self.weights)

    def _get_weights_and_dist(self, y, pred, dist):
        raise NotImplementedError

    @staticmethod
    def _agg_func(y, weights):
        raise NotImplementedError


class AdaBoostClassifier(AdaBoost):

    def __init__(self, base_estimator="tree", n_estimators=50):
        super().__init__(base_estimator, n_estimators)
        assert base_estimator in ("tree", "linear")
        if base_estimator == "tree":
            learner = DecisionTreeClassifier(max_depth=1)
        else:
            learner = LogisticRegressor()
        self.learners = [copy.deepcopy(learner) 
                         for _ in range(self.n_estimators)]

    @staticmethod
    def _agg_func(y, weights):
        score = dict()
        for cls in np.unique(y):
            score[cls] = weights[y == cls].sum()
        return max(score.items(), key=lambda kv: kv[1])[0]

    def _get_weights_and_dist(self, y, pred, dist):
        n_classes = len(np.unique(y))
        # calculate weights of current learner
        e = np.sum((y != pred) * dist)
        weight = np.log((1 - e) / (e + 1e-10)) + np.log(n_classes - 1)

        # update data weights distribution
        new_dist = dist * np.exp(weight * (y != pred))
        new_dist /= new_dist.sum()
        return weight, new_dist


class AdaBoostRegressor(AdaBoost):
    """AdaBoost regression model.
    Ref: H. Drucker. “Improving Regressors using Boosting Techniques”, 1997.
    """
    def __init__(self, 
                 loss="linear",
                 base_estimator="tree", 
                 n_estimators=50):
        super().__init__(base_estimator, n_estimators)
        assert base_estimator in ("tree", "linear")
        if base_estimator == "tree":
            learner = DecisionTreeRegressor(max_depth=1)
        else:
            learner = LinearRegressor()
        self.learners = [copy.deepcopy(learner) 
                         for _ in range(self.n_estimators)]

        assert loss in ("linear", "square", "exponential")
        self.loss = loss

    @staticmethod
    def _agg_func(y, weights):
        weights /= weights.sum()
        # weighted median
        accum = 0.0
        for i, idx in enumerate(np.argsort(weights)):
            accum += weights[idx]
            if accum >= 0.5:
                median_idx = [i] if accum > 0.5 else [i, i + 1]
                break
        return np.mean(y[median_idx], axis=0)

    def _get_weights_and_dist(self, y, pred, dist):
        loss = self._loss_func(y, pred)
        beta = loss.mean() / (1.0 - loss.mean())

        weights = np.log(1.0 - beta)
        dist = dist * beta ** loss
        dist /= dist.sum()
        return weights, dist

    def _loss_func(self, y, pred):
        pred = np.reshape(pred, y.shape)
        if self.loss == "linear":
            loss = np.abs(y - pred) / np.max(y - pred)
        elif self.loss == "square":
            loss = (y - pred) ** 2 / np.max((y - pred) ** 2)
        else:
            loss = 1.0 - np.exp(np.abs(y - pred) / np.max(y - pred))
        return loss

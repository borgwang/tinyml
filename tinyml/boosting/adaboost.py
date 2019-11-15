import copy
import numpy as np
from tqdm import tqdm

from tinyml.tree import DecisionTreeClassifier
from tinyml.tree import DecisionTreeRegressor


class AdaBoost:

    def __init__(self, base_estimator, n_estimators, learning_rate):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.learners = None

    def fit(self, x, y):
        self.weights = np.zeros(self.n_estimators)
        data_dist = np.ones(len(x), dtype=float) / len(x)
        eps = 1e-8
        n_classes = len(np.unique(y))

        for i, learner in enumerate(self.learners):
            # fit model using data with distribution data_dist
            resample = np.random.choice(a=range(len(x)), 
                                        size=len(y),
                                        p=data_dist)
            learner.fit(x[resample], y[resample])
            pred = learner.predict(x)

            # calculate weights of current learner
            e = np.sum((y != pred) * data_dist)
            weight = np.log((1 - e) / (e + eps)) + np.log(n_classes - 1)

            # update data weights distribution
            data_dist = data_dist * np.exp(weight * (y != pred))
            data_dist /= data_dist.sum()

            self.weights[i] = weight

    def predict(self, x):
        predictions = np.array(
            [learner.predict(x) for learner in self.learners]).T
        results = []
        for pred in predictions:
            res = self._agg_func(pred, weight=self.weights)
            results.append(res)
        return np.array(results)

    def _agg_func(self, y, weight):
        raise NotImplementedError
        

class AdaBoostClassifier(AdaBoost):

    def __init__(self,
                 base_estimator="decision_tree",
                 n_estimators=5,
                 learning_rate=0.1):
        super().__init__(base_estimator, n_estimators, learning_rate)
        assert base_estimator in ("decision_tree")
        if base_estimator == "decision_tree":
            self.learner = DecisionTreeClassifier(max_depth=1)
        else:
            raise ValueError
        self.learners = [copy.deepcopy(self.learner) 
                         for  _ in range(self.n_estimators)]

    @staticmethod
    def _agg_func(y, weight):
        score = dict()
        for cls in np.unique(y):
            score[cls] = weight[y == cls].sum()
        return max(score.items(), key=lambda kv:kv[1])[0]


class AdaBoostRegressor(AdaBoost):

    @staticmethod
    def _agg_func(y, weight):
        result = 0.0
        for sample, weight in zip(y, weights):
            result += weight * sample
        return result

from collections import Counter

import numpy as np
from tqdm import tqdm

from tinyml.tree import DecisionTreeClassifier
from tinyml.tree import DecisionTreeRegressor
from tinyml.bagging import RandomForest
from tinyml.bagging import RandomForestClassifier
from tinyml.bagging import RandomForestRegressor


class ExtraTreesClassifier(RandomForestClassifier):

    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_features="sqrt",
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=None):
        super().__init__(n_estimators, criterion,  
                         max_features, min_samples_split, 
                         min_impurity_split, max_depth)
        self.tree_params.update({"random_split": True})
        self.learners = [DecisionTreeClassifier(**self.tree_params)
                         for _ in range(self.n_estimators)]


class ExtraTreesRegressor(RandomForestRegressor):

    def __init__(self,
                 n_estimators=100,
                 criterion="mse",
                 max_features="sqrt",
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=None):
        super().__init__(n_estimators, criterion,  
                         max_features, min_samples_split, 
                         min_impurity_split, max_depth)
        self.tree_params.update({"random_split": True})
        self.learners = [DecisionTreeRegressor(**self.tree_params)
                         for _ in range(self.n_estimators)]

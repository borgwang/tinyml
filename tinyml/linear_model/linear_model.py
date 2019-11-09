import numpy as np


class LinearModel:
    pass


class LogisticRegression(LinearModel):
    pass


class LinearRegression(LinearModel):
    pass


class ElasticNet(LinearModel):
    pass


class ElasticNetClassifier(ElasticNet):
    pass


class ElasticNetRegressor(LinearModel):
    pass


class LassoClassifier(ElasticNetClassifier):
    pass


class LassoRegressor(ElasticNetRegressor):
    pass


class RidgeClassifier(ElasticNetCalssifier):
    pass


class RidgeRegressor(ElasticNetRegressor):
    pass


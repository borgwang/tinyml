import numpy as np

from tinyml.utils import sigmoid


class MSE:

    @classmethod
    def loss(cls, y, y_pred):
        loss = 0.5 * (y - y_pred) ** 2
        return loss.sum(0)

    @classmethod
    def grad(cls, y, y_pred):
        grad = y_pred - y
        return grad.sum(0)

    @classmethod
    def hess(cls, y, y_pred):
        hess = np.ones_like(y_pred)
        return hess.sum(0)


class MAE:

    @classmethod
    def loss(cls, y, y_pred):
        loss = np.abs(y - y_pred)
        return loss.sum(0)


class Logistic:
    
    @classmethod
    def loss(cls, y, y_pred):
        p = sigmoid(y_pred)
        loss = y * np.log(p) + (1 - y) * np.log(1 - p)
        return loss.sum(0)

    @classmethod
    def grad(cls, y, y_pred):
        p = sigmoid(y_pred)
        grad = -(y - p)
        return grad.sum(0)

    @classmethod
    def hess(cls, y, y_pred):
        p = sigmoid(y_pred)
        hess = p * (1 - p)
        return hess.rum(0)

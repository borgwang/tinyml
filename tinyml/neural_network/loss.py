import numpy as np


class Loss:
    
    def loss(self, pred, target):
        raise NotImplementedError

    def grad(self, pred, target):
        raise NotImplementedError


class MSE(Loss):

    def loss(self, target, pred):
        m = pred.shape[0]
        return 0.5 * np.sum((pred - target) ** 2) / m

    def grad(self, target, pred):
        m = pred.shape[0]
        return (pred - target) / m


import numpy as np

from tinyml.utils import log_softmax
from tinyml.utils import softmax


class Loss:
    
    def loss(self, pred, target):
        raise NotImplementedError

    def grad(self, pred, target):
        raise NotImplementedError


class MSE(Loss):

    def loss(self, pred, target):
        batch = pred.shape[0]
        return 0.5 * np.sum((pred - target) ** 2) / batch

    def grad(self, pred, target):
        batch = pred.shape[0]
        return (pred - target) / batch


class SoftmaxCrossEntropy(Loss):

    def loss(self, logits, labels):
        batch = logits.shape[0]
        nll = -(log_softmax(logits) * labels).sum() / batch
        return nll

    def grad(self, logits, labels):
        batch = logits.shape[0]
        return (softmax(logits) - labels) / batch

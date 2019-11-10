import numpy as np


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    exp_sum = np.sum(exps, axis=axis, keepdims=True)
    return x - x_max - np.log(exp_sum)


def is_numerical(x):
    return isinstance(x, int) or isinstance(x, float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize(x, epsilon=1e-8):
    return (x - x.mean(0)) / (x.std(0) + epsilon)

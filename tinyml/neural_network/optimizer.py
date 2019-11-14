import numpy as np


class Optimizer:

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, grads, params):
        # convert list of dicts to a numpy array
        grad_values = list()
        for grad_dict in grads:
            for grad in grad_dict.values():
                grad_values.append(grad)
        grad_values = np.array(grad_values)
        # compute step according to derived class method
        step_values = self._compute_step(grad_values)
        # apply update to params
        i = 0
        for param_dict in params:
            for name, param in param_dict.items():
                step = step_values[i]
                param_dict[name] += step
                i += 1

    def _compute_step(self, grad):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate, momentum=0.0):
        super().__init__(learning_rate)
        self._momentum = momentum
        self._acc = 0
    
    def _compute_step(self, grad):
        self._acc = self._momentum * self._acc + grad
        return -self.lr * self._acc


class Adam(Optimizer):

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self._b1 = beta1
        self._b2 = beta2
        self._eps = epsilon

        self._t = 0
        self._m = 0
        self._v = 0

    def _compute_step(self, grad):
        self._t += 1

        self._m += (1.0 - self._b1) * (grad - self._m)
        self._v += (1.0 - self._b2) * (grad ** 2 - self._v)

        # bias correction
        _m = self._m / (1 - self._b1 ** self._t)
        _v = self._v / (1 - self._b2 ** self._t)

        step = -self.lr * _m / (_v ** 0.5 + self._eps)
        return step

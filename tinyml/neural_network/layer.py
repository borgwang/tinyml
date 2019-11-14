import numpy as np


class Layer:
    
    def __init__(self):
        self.params = {}
        self.grads = {}

        self.is_init = False

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grads):
        raise NotImplementedError


class Dense(Layer):

    def __init__(self, num_out):
        super().__init__()
        self.shapes = {"w": [None, num_out], "b": [num_out]}
        self.inputs = None

    def forward(self, inputs):
        if not self.is_init:
            self.shapes["w"][0] = inputs.shape[1]
            self._init_params()
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T

    def _init_params(self):
        w_shape = self.shapes["w"]
        a = np.sqrt(6.0 / (w_shape[0] + w_shape[1]))
        self.params["w"] = np.random.uniform(-a, a, size=w_shape)
        self.params["b"] = np.zeros(self.shapes["b"], dtype=float)
        self.is_init = True


class Activation(Layer):

    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(Activation):

    def func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        return self.func(x) * (1.0 - self.func(x))


class Tanh(Activation):

    def func(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1.0 - self.func(x) ** 2


class ReLU(Activation):

    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative(self, x):
        return x > 0.0

import numpy as np


class Layer:
    
    def __init__(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grads):
        raise NotImplementedError


class Dense(Layer):
    
    def forward(self, inputs):
        pass

    def backward(self, grads):
        pass


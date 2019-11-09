import numpy as np


class Optimizer:

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self):
        pass

    def compute_step(self):
        raise NotImplementedError


class SGD(Optimizer):
    
    def compute_step(self):
        pass

import numpy as np

from tinyml.neural_network.loss import MSE


class LinearModel:
    
    @staticmethod
    def regularize(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def regularize_grad(*args, **kwargs):
        raise NotImplementedError


class LogisticRegressor(LinearModel):
    pass


class LinearRegressor(LinearModel):

    def __init__(self, n_iterations=100, learning_rate=0.01):
        self.n_iters = n_iterations
        self.lr = learning_rate

        self.w, self.b = None, None

    def fit(self, x, y):
        # BUG
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        # init w and b
        batch, in_dim, out_dim = x.shape[0], x.shape[1], y.shape[1]
        self.w = np.random.uniform(-0.02, 0.02, (in_dim, out_dim))
        self.b = np.zeros((out_dim,))

        for n in range(self.n_iters):
            pred = x @ self.w + self.b

            loss = np.mean(0.5 * (pred - y) ** 2)
            grad = (pred - y) / batch

            d_w = x.T @ grad 
            d_b = grad.sum(axis=0)

            self.w -= self.lr * d_w
            self.b -= self.lr * d_b
            print(loss)

    def predict(self, x):
        return x @ self.w + self.b 

    @staticmethod
    def regularize(w):
        return 0

    @staticmethod
    def regularize_grad(*args, **kwargs):
        return 0


class ElasticNet(LinearModel):
    pass


class ElasticNetClassifier(ElasticNet):
    pass


class ElasticNetRegressor(ElasticNet):
    pass


class LassoClassifier(ElasticNetClassifier):
    pass


class LassoRegressor(ElasticNetRegressor):
    pass


class RidgeClassifier(ElasticNetClassifier):
    pass


class RidgeRegressor(ElasticNetRegressor):
    pass


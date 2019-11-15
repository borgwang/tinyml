import numpy as np

from tinyml.neural_network.loss import MSE
from tinyml.neural_network.loss import SoftmaxCrossEntropy
from tinyml.utils import get_one_hot
from tinyml.utils import normalize


class LinearModel:

    def __init__(self, n_iterations, learning_rate, normalize):
        self.n_iterations = n_iterations
        self.lr = learning_rate
        self.normalize = normalize

        self.w, self.b = None, None
        self.loss = None

    def fit(self, x, y):
        if self.normalize:
            x = normalize(x)

        batch, in_dim, out_dim = x.shape[0], x.shape[1], y.shape[1]
        self.w = np.random.uniform(-0.02, 0.02, (in_dim, out_dim))
        self.b = np.zeros((out_dim,))

        for n in range(self.n_iterations):
            logit = x @ self.w + self.b

            loss = self.loss.loss(logit, y) + self.regularize(self.w)
            grad_from_loss = self.loss.grad(logit, y)

            d_w = x.T @ grad_from_loss + self.regularize_grad(self.w)
            d_b = grad_from_loss.sum(axis=0)

            self.w -= self.lr * d_w
            self.b -= self.lr * d_b

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        if self.normalize:
            x = normalize(x)
        return x @ self.w + self.b

    @staticmethod
    def regularize(*args, **kwargs):
        return 0

    @staticmethod
    def regularize_grad(*args, **kwargs):
        return 0


class LogisticRegressor(LinearModel):

    def __init__(self, n_iterations=100, learning_rate=0.1, normalize=False):
        super().__init__(n_iterations, learning_rate, normalize)
        self.loss = SoftmaxCrossEntropy()

    def fit(self, x, y):
        # categorize
        y = get_one_hot(y, len(np.unique(y)))
        super().fit(x, y)

    def predict(self, x):
        logits = super().predict(x)
        return np.argmax(logits, axis=1)


class LinearRegressor(LinearModel):

    def __init__(self, n_iterations=100, learning_rate=0.1, normalize=False):
        super().__init__(n_iterations, learning_rate, normalize)
        self.loss = MSE()

    def fit(self, x, y):
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        super().fit(x, y)


class ElasticNetRegressor(LinearRegressor):
    def __init__(self,
                 alpha=1.0,
                 l1_ratio=0.5,
                 n_iterations=100,
                 learning_rate=0.1,
                 normalize=False):
        super().__init__(n_iterations, learning_rate, normalize)
        self.l1_coef = alpha * l1_ratio
        self.l2_coef = alpha * (1 - l1_ratio)

    def regularize(self, w):
        l1 = np.abs(w).sum()
        l2 = 0.5 * (w ** 2).sum()
        return self.l1_coef * l1 + self.l2_coef * l2

    def regularize_grad(self, w):
        return self.l1_coef * np.sign(w) + self.l2_coef * w


class LassoRegressor(ElasticNetRegressor):

    def __init__(self,
                 alpha=1.0,
                 n_iterations=100,
                 learning_rate=0.1,
                 normalize=False):
        super().__init__(alpha, 1.0, n_iterations, learning_rate, normalize)


class RidgeRegressor(ElasticNetRegressor):

    def __init__(self,
                 alpha=1.0,
                 n_iterations=100,
                 learning_rate=0.1,
                 normalize=False):
        super().__init__(alpha, 0.0, n_iterations, learning_rate, normalize)

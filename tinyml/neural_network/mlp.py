import numpy as np

from tinyml.neural_network.layer import Dense
from tinyml.neural_network.layer import ReLU
from tinyml.neural_network.layer import Sigmoid
from tinyml.neural_network.layer import Tanh
from tinyml.neural_network.loss import MSE
from tinyml.neural_network.loss import SoftmaxCrossEntropy
from tinyml.neural_network.net import Net
from tinyml.neural_network.optimizer import SGD
from tinyml.neural_network.optimizer import Adam
from tinyml.utils import get_one_hot

activation_dict = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}


class MLP:

    def __init__(self, 
                 hidden_layer_sizes, 
                 activation, 
                 solver, 
                 batch_size, 
                 learning_rate, 
                 max_iter,
                 momentum,
                 beta1,
                 beta2,
                 epsilon,
                 verbose):
        assert solver in ("sgd", "adam")
        if solver == "sgd":
            self.optimizer = SGD(learning_rate, momentum=momentum)
        else:
            self.optimizer = Adam(learning_rate, beta1=beta1, beta2=beta2,
                                  epsilon=epsilon)
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.activation = activation_dict[activation]
        self.verbose = verbose
        self.layer_sizes = hidden_layer_sizes

        self.loss = None
        self.net = None

    def _build_net(self):
        layers = []
        for i, n_units in enumerate(self.layer_sizes):
            layers.append(Dense(n_units))
            if i < len(self.layer_sizes) - 1:
                layers.append(self.activation())
        return Net(layers)

    def fit(self, x, y):
        self.layer_sizes += (y.shape[1],)
        self.net = self._build_net()

        running_loss = None
        for e in range(self.max_iter):
            for i in range(len(x) // self.batch_size + 1):
                batch_idx = np.random.randint(0, len(x), size=self.batch_size)
                batch_x, batch_y = x[batch_idx], y[batch_idx]

                pred = self.net.forward(batch_x)
                loss = self.loss.loss(pred, batch_y)
                grad = self.loss.grad(pred, batch_y)
                grad = self.net.backward(grad)
                self.optimizer.step(grad, self.net.params)
                if not running_loss:
                    running_loss = loss
                else:
                    running_loss = running_loss * 0.9 + loss + 0.1
            if self.verbose:
                print("iter: %d running loss: %.4f" % (e, running_loss))

    def predict(self, x):
        return self.net.forward(x)


class MLPClassifier(MLP):

    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation="relu",
                 solver="sgd",
                 batch_size=64,
                 learning_rate=1e-2,
                 max_iter=200,
                 momentum=0.9,
                 beta1=0.9,
                 beta2=0.99,
                 epsilon=1e-8,
                 verbose=True):
        super().__init__(hidden_layer_sizes,
                         activation,
                         solver,
                         batch_size,
                         learning_rate,
                         max_iter,
                         momentum,
                         beta1,
                         beta2,
                         epsilon,
                         verbose)
        self.loss = SoftmaxCrossEntropy()

    def fit(self, x, y):
        # categorize
        y = get_one_hot(y, len(np.unique(y)))
        super().fit(x, y)

    def predict(self, x):
        logits = super().predict(x)
        return np.argmax(logits, axis=1)


class MLPRegressor(MLP):

    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation="relu",
                 solver="adam",
                 batch_size=64,
                 learning_rate=1e-2,
                 max_iter=200,
                 momentum=0.9,
                 beta1=0.9,
                 beta2=0.99,
                 epsilon=1e-8,
                 verbose=True):
        super().__init__(hidden_layer_sizes,
                         activation,
                         solver,
                         batch_size,
                         learning_rate,
                         max_iter,
                         momentum,
                         beta1,
                         beta2,
                         epsilon,
                         verbose)
        self.loss = MSE()

    def fit(self, x, y):
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        super().fit(x, y)

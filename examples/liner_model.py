import runtime_path  # isort:skip

from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from tinyml.linear_model import LogisticRegressor
from tinyml.linear_model import LinearRegressor

from tinynn.core.layer import Dense
from tinynn.core.net import Net
from tinynn.core.loss import MSE
from tinynn.core.optimizer import SGD
from tinynn.core.model import Model


def main():
    """
    print("LogisticRegressor on Digits dataset.")
    dataset = load_digits()
    x, y = dataset.data, dataset.target
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    model = LogisticRegressor()
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)
    print("accuracy: %.4f" % accuracy_score(test_y, test_pred))

    print("LinearRegressor on Boston dataset.")
    dataset = load_boston()
    x, y = dataset.data, dataset.target
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    model = LinearRegressor()
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)
    print("mse: %.4f" % mean_squared_error(test_y, test_pred))
    """

    dataset = load_boston()
    x, y = dataset.data, dataset.target
    y = y.reshape((-1, 1))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
    net = Net([Dense(1)])
    loss = MSE()
    optimizer = SGD(lr=0.1)
    model = Model(net, loss=loss, optimizer=optimizer)
    for n in range(100):
        pred = model.forward(train_x)
        loss, grads = model.backward(pred, train_y)
        model.apply_grads(grads)
        print(loss)


if __name__ == '__main__':
    main()

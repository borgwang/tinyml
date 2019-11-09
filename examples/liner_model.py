import runtime_path  # isort:skip

from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from tinyml.linear_model import LogisticRegressor
from tinyml.linear_model import LinearRegressor


def main():
    print("LogisticRegressor on Digits dataset.")
    dataset = load_digits()
    x, y = dataset.data, dataset.target
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    model = LogisticRegressor(n_estimators=20, max_depth=5)
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)
    print("accuracy: %.4f" % accuracy_score(test_y, test_pred))

    print("LinearRegressor on Boston dataset.")
    dataset = load_boston()
    x, y = dataset.data, dataset.target
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    model = LinearRegressor(n_estimators=20)
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)
    print("mse: %.4f" % mean_squared_error(test_y, test_pred))


if __name__ == '__main__':
    main()

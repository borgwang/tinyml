import runtime_path  # isort:skip

from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from tinyml.tree import DecisionTreeClassifier
from tinyml.tree import DecisionTreeRegressor


def main():
    print("DecisionTreeClassifier on Digits dataset.")
    dataset = load_digits()
    x, y = dataset.data, dataset.target
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    model = DecisionTreeClassifier(max_depth=10)
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)
    print("accuracy: %.4f" % accuracy_score(test_y, test_pred))

    print("DecisionTreeRegressor on Boston dataset.")
    dataset = load_boston()
    x, y = dataset.data, dataset.target
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    model = DecisionTreeRegressor()
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)
    print("mse: %.4f" % mean_squared_error(test_y, test_pred))


if __name__ == '__main__':
    main()

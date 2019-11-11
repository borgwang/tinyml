import runtime_path  # isort:skip

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from tinyml.gaussian_process import GaussianProcessRegressor
from tinyml.utils import standardize


def main():
    print("GaussianProcessRegressor on Boston dataset.")
    dataset = load_boston()
    x, y = dataset.data, dataset.target
    x = standardize(x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    model = GaussianProcessRegressor()
    model.fit(train_x, train_y)
    test_pred, _ = model.predict(test_x)
    print("mse: %.4f" % mean_squared_error(test_y, test_pred))


if __name__ == '__main__':
    main()

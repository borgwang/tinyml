import runtime_path  # isort:skip

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from tinyml.cluster import KMeans


def main():
    print("KMeans on isotropic Gaussian blobs.")
    x, y = make_blobs(200)

    model = KMeans(n_clusters=3)
    model.fit(x)
    print("Inertia: %.2f" % model.inertia_)

    y_pred = model.predict(x)
    for label in np.unique(y_pred):
        data = x[y_pred == label]
        plt.scatter(data[:, 0], data[:, 1], label=str(label))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

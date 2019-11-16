import runtime_path  # isort:skip

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from tinyml.cluster import KMeans
from tinyml.cluster import KMedoids


def main():
    print("KMeans on isotropic Gaussian blobs.")
    x, y = make_blobs(200)

    model = KMeans(n_clusters=3)
    model.fit(x)
    centers = model.cluster_centers_
    print("Inertia: %.2f" % model.inertia_)
    print("Cluster centers: \n",  centers)

    """
    model = KMedoids(n_clusters=3)
    model.fit(x)
    centers = model.cluster_centers_
    print("Inertia: %.2f" % model.inertia_)
    print("Cluster centers: \n",  centers)
    """

    y_pred = model.predict(x)
    for label in np.unique(y_pred):
        data = x[y_pred == label]
        plt.scatter(data[:, 0], data[:, 1], label=str(label))
    plt.scatter(centers[:, 0], centers[:, 1], label="cluster centers")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

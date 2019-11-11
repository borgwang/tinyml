import numpy as np
from scipy.optimize import minimize


class GaussianProcessRegressor:

    def __init__(self):
        self.params = {"l": 0.5, "sigma_f": 0.2}

        self.train_x, self.train_y = None, None

    def fit(self, x, y):
        self.params = self._optimize_kernel_params(x, y, self.params)
        self.train_x, self.train_y = x, y

    def predict(self, x):
        l, sigma_f = self.params["l"], self.params["sigma_f"]
        Kff = self.gaussian_kernel(l, sigma_f, x, x)
        Kyy = self.gaussian_kernel(l, sigma_f, self.train_x, self.train_x)
        Kfy = self.gaussian_kernel(l, sigma_f, x, self.train_x)
        Kyy_inv = np.linalg.inv(Kyy + 1e-10 * np.eye(len(self.train_x)))

        mu = Kfy @ Kyy_inv @ self.train_y
        cov = Kff - Kfy @ Kyy_inv @ Kfy.T
        return mu, cov

    @staticmethod
    def gaussian_kernel(l, sigma_f, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * (x1 @ x2.T)
        return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)

    def _optimize_kernel_params(self, x, y, params):
        def nll_loss(params):
            l, sigma_f = params
            Kyy = self.gaussian_kernel(l, sigma_f, x, x)
            Kyy_inv = np.linalg.inv(Kyy + 1e-10 * np.eye(len(x)))
            return (0.5 * y.T @ Kyy_inv @ y + 
                    0.5 * np.linalg.slogdet(Kyy)[1] + 
                    0.5 * len(x) * np.log(2 * np.pi))
        res = minimize(nll_loss, [self.params["l"], self.params["sigma_f"]],
                       bounds=((1e-5, None), (1e-5, None)), method="L-BFGS-B")
        return {"l": res.x[0], "sigma_f": res.x[1]}


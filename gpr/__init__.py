import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def make_grid(x):
    n = np.prod([y.size for y in x])
    d = len(x)
    return np.array(np.meshgrid(*tuple(x))).reshape(d, n).T


def kron_prod(mat_list, b):
    x = b.copy()
    n = [x.shape[1] for x in mat_list]
    for j in range(len(mat_list) - 1, -1, -1):
        d = n[j]
        x = x.reshape(int(x.size / d), d)
        x = mat_list[j].dot(x.T)
        x = x.ravel()
    return x


def kron_prod_naive(mat_list, b):
    res = 1
    for x in mat_list:
        res = np.kron(res, x)
    return res.dot(b)


class GPR:
    def __init__(self):
        self.nr_dimensions = 0
        self.noise = 1e-6
        self.alpha = 1
        self.length_scale = []
        self.minus_log_likelihood = np.inf
        self.weights = np.array(0.0)
        self.weights_are_fitted = False

    def init_kernel(self, d, noise, length_scale):
        if length_scale is None:
            self.length_scale = [1] * d
        else:
            self.length_scale = length_scale
        self.nr_dimensions = d
        self.noise = noise

    def gaussian_kernel(self, x_train, x_eval):
        res = []
        for i, x1, x2 in zip(range(self.nr_dimensions), x_train, x_eval):
            dx = (x1[:, None] - x2[None, :]) / self.length_scale[i]
            _g = np.exp(-0.5 * dx * dx)
            res.append(_g)
        return res

    def fit_weights(self, x, y, noise=1e-6, length_scale=None):
        self.init_kernel(len(x), noise, length_scale)
        self._x_train = x
        gausss_kern = self.gaussian_kernel(x, x)
        self._eig_vec_kern = []
        self._kron_eig_val = 1
        for k in gausss_kern:
            l, o = np.linalg.eigh(k)
            o = o.T
            self._eig_vec_kern.append(o)
            self._kron_eig_val = np.kron(self._kron_eig_val, np.maximum(l, 0))
        self.weights = kron_prod(self._eig_vec_kern, y)
        self.weights = self.weights / (self._kron_eig_val + self.noise ** 2)
        self.weights = kron_prod([x.T for x in self._eig_vec_kern], self.weights) / self.alpha
        self.weights_are_fitted = True
        self.minus_log_likelihood = y.dot(self.weights)
        self.minus_log_likelihood += np.log(self._kron_eig_val + self.noise ** 2).sum()
        self.minus_log_likelihood += np.log(self.alpha)

    def predict(self, x):
        if self.weights_are_fitted:
            gauss_kern = self.gaussian_kernel(x, self._x_train)
            return kron_prod(gauss_kern, self.weights) * self.alpha

    def fit(self, x_train, y_train):
        dx = [(np.diff(x).mean() * 2, (x.max() - x.min())) for x in x_train]

        def target_fun(x):
            self.fit_weights(x_train, y_train, noise=x[-1], length_scale=x[:-1])
            return self.minus_log_likelihood
        x0=np.array([1]*self.nr_dimensions+[1e-6])
        sol = minimize(target_fun,x0)


if __name__ == '__main__':
    d = 2
    x_train = [np.linspace(0, 1, 1000) for i in range(d)]
    x_grid = make_grid(x_train)
    n = x_grid.shape[0]
    y_train = x_grid.sum(axis=1) ** 2 + np.random.normal(0, 0.1, n)

    gpr = GPR()
    gpr.fit_weights(x_train, y_train)
    # plt.plot(gpr.weights)
    #plt.plot(x_grid.ravel(), y_train, '.')
    x_test = [np.sort(np.random.random(100)) for i in range(d)]
    x_grid_test = make_grid(x_test)

    #plt.plot(x_grid_test.ravel(), gpr.predict(x_test))
    #plt.show()
    dx = np.array([(np.diff(x).mean() * 2, (x.max() - x.min())) for x in x_train])


    def target_fun(x):
        q = np.exp(x[:-1])
        q = q / (1 + q)
        length_scale = q * dx[:, 0] + (1 - q) * dx[:, 1]
        print(q, length_scale)
        gpr.fit_weights(x_train, y_train, noise=x[-1], length_scale=length_scale)
        return gpr.minus_log_likelihood


    x0 = np.array([1] * gpr.nr_dimensions + [1e-6])

    #sol = minimize(target_fun, x0)
    y0 = target_fun(x0)
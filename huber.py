import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

matplotlib.use("tkagg")


def LASSORegressionUsingCVX(A, Y, alpha):
    Y = Y.reshape(-1, 1)
    n = np.shape(A)[1]
    x = cp.Variable((n, 1))
    objective = cp.Minimize(
        cp.sum_squares(A @ x - Y) + alpha * cp.norm(x[1 : n - 1], 1)
    )
    prob = cp.Problem(objective)
    prob.solve()
    theta = x.value
    return theta


class HuberLinearRegression(object):
    """
    Linear regression model: Y = Ax, fit by minimizing the reguarized loss function
    with Huber regularization
    """

    def __init__(self, A, Y, x_init=None, mu=0.01):
        """
        @params A: samples
        @param Y: targets
        @param x_init: the initial coefficients
        @param mu: regularization parameter
        """
        self.mu = mu
        self.x = None
        self.x_init = x_init
        self.A = A
        self.Y = Y

    def predict(self, A):
        prediction = np.matmul(A, self.x)
        return prediction

    def regularized_loss(self, x):
        self.x = x
        x_norm = np.linalg.norm(x, 1)
        if x_norm <= self.mu:
            regularization = np.dot(x, x) / (2 * self.mu)
        else:
            regularization = np.linalg.norm(x, 1) - self.mu / 2
        residual = self.predict(self.A) - self.Y
        loss = np.dot(residual, residual) + regularization
        print(loss)
        return loss

    def fit(self, maxiter=100):
        res = minimize(
            self.regularized_loss,
            self.x_init,
            method="BFGS",
            options={"maxiter": maxiter, "disp": True},
        )
        self.x = res.x


def main():
    # Load data from Q1.csv
    df = pd.read_csv("./Q1.csv", names=["X", "Y"])
    X = df.X.values.reshape(-1, 1)
    Y = df.Y.values.reshape(-1, 1).flatten()

    # Transform data to polynomial features
    deg = 50
    poly = PolynomialFeatures(degree=deg)
    A = poly.fit_transform(X)

    # Find LASSO regression parameters and plot trendline
    alpha1 = 1e-6
    theta = LASSORegressionUsingCVX(A, Y, alpha1)
    z = np.dot(A, theta)

    plt.plot(X, Y, "b.", label="Ground truth")
    plt.plot(X, z, "r", label=r"LASSO Regression with $\alpha$={}".format(alpha1))

    # Huber regression
    colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i, mu in enumerate([1, 0.1, 0.01, 0.001, 0.0001]):
        x_init = np.random.rand(A.shape[1])
        color = colors[i%8]
        model = HuberLinearRegression(A, Y, x_init, mu)
        model.fit()
        plt.plot(
            X, model.predict(A), color=color, label=r"Huber Regression with $\mu$={}".format(mu)
        )

    plt.legend()
    plt.show()


main()

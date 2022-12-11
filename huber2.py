import random
import warnings
from functools import partial

import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize, rosen
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

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


class MyLinearRegression(object):
    """
    Linear regression model: Y = Ax, fit by minimizing the reguarized loss function
    with Huber regularization
    """

    def __init__(self, regularization_function):
        """
        @param regularization_function: regularization function, which has arguments (coeffs)
        """
        self.regularization_function = regularization_function
        self.x = None

    def predict(self, A):
        prediction = np.matmul(A, self.x)
        return prediction

    def objective_function(self, x):
        self.x = x
        residual = self.predict(self.A) - self.Y
        loss = np.dot(residual, residual) + self.regularization_function(x)
        return loss

    def fit(self, A, Y, x_init=None, method="BFGS", maxiter=None):
        """
        @params A: samples
        @param Y: targets
        @param x_init: the initial coefficients
        """
        # A[:,0] = np.random.rand(A.shape[0])  # random intercept
        self.A = A
        self.Y = Y
        if x_init is None:
            self.x_init = np.random.rand(A.shape[1])
        else:
            self.x_init = x_init
        res = minimize(
            self.objective_function,
            self.x_init,
            method=method,
            options={"maxiter": maxiter, "disp": True},
        )
        self.x = res.x


def huber_regularization(mu, x):
    x_norm = np.linalg.norm(x[1:], 1)
    if x_norm <= mu:
        regularization = np.dot(x[1:], x[1:]) / (2 * mu)
    else:
        regularization = x_norm - mu / 2
    return regularization


def lasso_regularization(alpha, x):
    regularization = alpha * np.linalg.norm(x[1:], 1)
    return regularization


def ridge_regularization(alpha, x):
    regularization = alpha * np.linalg.norm(x[1:], 2) ** 2
    return regularization


def main():
    # Load data
    df = pd.read_csv("./insurance.csv")
    df = df.drop_duplicates()

    df_dummies = df.copy()

    for i in df_dummies.columns:
        if df_dummies[i].dtype == "object":
            dummies = pd.get_dummies(df_dummies[i], prefix=f"{i}")
            df_dummies = pd.concat([df_dummies, dummies], axis=1)
            df_dummies = df_dummies.drop(i, axis=1)

    X = df_dummies.drop("charges", axis=1)
    Y = df_dummies["charges"]
    deg = 2

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    alpha = 0.1
    pipeline1 = make_pipeline(
        PolynomialFeatures(degree=deg),
        MyLinearRegression(partial(lasso_regularization, alpha)),
    )
    pipeline1.fit(X_train, Y_train)
    Y_min_pred = pipeline1.predict(X_test)

    pipeline2 = make_pipeline(PolynomialFeatures(degree=deg), Lasso(alpha))
    pipeline2.fit(X_train, Y_train)
    Y_sklearn_pred = pipeline2.predict(X_test)

    mu = 100
    pipeline3 = make_pipeline(
        PolynomialFeatures(degree=deg),
        MyLinearRegression(partial(huber_regularization, mu)),
    )
    pipeline3.fit(X_train, Y_train)
    Y_huber_pred = pipeline3.predict(X_test)

    plt.plot(
        [Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()],
        "y",
    )
    plt.plot(
        Y_test,
        Y_min_pred,
        "r.",
        label=r"Lasso regression (using minimize) with $\alpha$={}".format(alpha),
    )
    print("Lasso regression score (using minimize)", r2_score(Y_test, Y_min_pred))

    plt.plot(
        Y_test,
        Y_sklearn_pred,
        "g.",
        label=r"Lasso regression (using sklearn) with $\alpha$={}".format(alpha),
    )
    print("Lasso regression score (using sklearn)", r2_score(Y_test, Y_sklearn_pred))

    plt.plot(
        Y_test,
        Y_huber_pred,
        "b.",
        label=r"Huber regression with $\mu$={}".format(mu),
    )
    print("Huber regression score", r2_score(Y_test, Y_huber_pred))

    plt.legend()
    plt.show()

    # Huber regression grid search
    # colors = [
    #    "tab:orange",
    #    "tab:green",
    #    "tab:purple",
    #    "tab:brown",
    #    "tab:pink",
    #    "tab:gray",
    #    "tab:olive",
    #    "tab:cyan",
    # ]
    ## for i, mu in enumerate([1, 0.1, 0.01, 0.001, 0.0001]):
    # for i, mu in enumerate([100]):
    #    color = colors[i % 8]
    #    model = MyLinearRegression(A, Y, partial(huber_regularization, mu))
    #    model.fit(method="Powell")
    #    plt.plot(
    #        X,
    #        model.predict(A),
    #        color=color,
    #        label=r"Huber Regression with $\mu$={}".format(mu),
    #    )

    # plt.legend()
    # plt.show()


main()

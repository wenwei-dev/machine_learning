from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

matplotlib.use("tkagg")


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
        self._coef = None

    def predict(self, A):
        prediction = np.matmul(A, self._coef)
        return prediction

    def objective_function(self, _coef):
        self._coef = _coef
        residual = self.predict(self.A) - self.Y
        loss = np.dot(residual, residual) + self.regularization_function(_coef)
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
        self._coef = res.x


def huber_regularization(mu, x):
    x_norm = np.linalg.norm(x, 1)
    if x_norm <= mu:
        regularization = np.dot(x, x) / (2 * mu)
    else:
        regularization = x_norm - mu / 2
    return regularization


def lasso_regularization(alpha, x):
    regularization = alpha * np.linalg.norm(x, 1)
    return regularization


def ridge_regularization(alpha, x):
    regularization = alpha * np.linalg.norm(x, 2) ** 2
    return regularization


class Pipeline(object):
    def __init__(self, deg, model):
        self.polynormial_features_estimator = PolynomialFeatures(degree=deg)
        self.model = model

    def fit(self, X, Y):
        features = self.polynormial_features_estimator.fit_transform(X)
        self.model.fit(features, Y)

    def predict(self, X):
        features = self.polynormial_features_estimator.fit_transform(X)
        y_pred = self.model.predict(features)
        return y_pred

    @property
    def coef(self):
        return self.model.coef_


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
    deg = 3

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    alpha = 0.01

    pipeline_lasso = make_pipeline(PolynomialFeatures(degree=deg), Lasso(alpha))
    pipeline_lasso.fit(X_train, Y_train)
    Y_lasso_sklearn_pred = pipeline_lasso.predict(X_test)

    pipeline_ridge = make_pipeline(PolynomialFeatures(degree=deg), Ridge(alpha))
    pipeline_ridge.fit(X_train, Y_train)
    Y_ridge_sklearn_pred = pipeline_ridge.predict(X_test)

    pipeline_enet = make_pipeline(
        PolynomialFeatures(degree=deg), ElasticNet(alpha, l1_ratio=0.4)
    )
    pipeline_enet.fit(X_train, Y_train)
    Y_enet_sklearn_pred = pipeline_enet.predict(X_test)

    plt.plot(
        [Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()],
        "y",
    )

    plt.plot(
        Y_test,
        Y_lasso_sklearn_pred,
        "r.",
        label=r"Lasso regression with $\alpha$={}".format(alpha),
    )
    print("Lasso regression R2", r2_score(Y_test, Y_lasso_sklearn_pred))
    print("Lasso regression MAE", mean_absolute_error(Y_test, Y_lasso_sklearn_pred))

    plt.plot(
        Y_test,
        Y_ridge_sklearn_pred,
        "g.",
        label=r"Ridge regression with $\alpha$={}".format(alpha),
    )
    print("Ridge regression R2", r2_score(Y_test, Y_ridge_sklearn_pred))
    print("Ridge regression MAE", mean_absolute_error(Y_test, Y_ridge_sklearn_pred))

    plt.plot(
        Y_test,
        Y_enet_sklearn_pred,
        "b.",
        label=r"ElasticNet regression with $\alpha$={}".format(alpha),
    )
    print("ElasticNet regression R2", r2_score(Y_test, Y_enet_sklearn_pred))
    print("ElasticNet regression MAE", mean_absolute_error(Y_test, Y_enet_sklearn_pred))

    # Huber regression grid search
    colors = [
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    for i, mu in enumerate([1, 0.1, 0.01, 0.001]):
        color = colors[i % len(colors)]
        pipeline_x = make_pipeline(
            PolynomialFeatures(degree=deg),
            MyLinearRegression(partial(huber_regularization, mu)),
        )
        pipeline_x.fit(X_train, Y_train)
        Y_huber_pred_x = pipeline_x.predict(X_test)
        plt.plot(
            Y_test,
            Y_huber_pred_x,
            color=color,
            marker="o",
            alpha=0.3,
            linestyle="None",
            label=r"Huber Regression with $\mu$={}".format(mu),
        )
        print("Huber regression R2 %s, mu %s" % (r2_score(Y_test, Y_huber_pred_x), mu))
        print(
            "Huber regression MAE %s, mu %s"
            % (mean_absolute_error(Y_test, Y_huber_pred_x), mu)
        )

    plt.legend()
    plt.show()


main()

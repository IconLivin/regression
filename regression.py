from abc import ABC, abstractmethod
from numpy import matrix, array, insert, sum, power, log, linspace, identity, zeros_like
from numpy.linalg import pinv


class ModelException(Exception):
    def __init__(self):
        super().__init__("The length for X and Y must be equal")


class Model(ABC):
    def __init__(self, X, Y):
        if len(matrix(X)) != len(Y):
            raise ModelException
        self.X = insert(matrix(X, dtype=float).T, 0, 1, axis=0).T
        self.Y = array(Y, dtype=float)
        self.b = []

    @abstractmethod
    def fit(self, respConv=False):
        pass

    def __getattribute__(self, name):
        return super().__getattribute__(name)


class LinearRegression(Model):
    def __init__(self, X, Y):
        super().__init__(X, Y)

    def fit(self, respConv=False):
        if respConv:
            self.Y, self.teta = responseConversion(self.X, self.Y)
        self.b = pinv(self.X).dot(matrix(self.Y).T).T.tolist()[0]
        self.yHat = (self.X * matrix(self.b).T).T.tolist()[0]
        self.yAv = sum(self.Y) / len(self.Y)
        self.R2 = 1 - (sum(power(self.Y - self.yHat, 2))) / (sum(power(self.Y - self.yAv, 2)))
        self.S2 = sum(power(self.Y - self.yHat, 2)) / (len(self.Y) - len(self.X.T) - 1)
        self.F = (
            (matrix(self.b) * self.X.T * matrix(self.Y).T - (sum(self.Y) ** 2) / len(self.Y))
            / (len(self.X.T) - 1)
            / self.S2
        ).tolist()[0][0]
        return self.b


class PowerRegression(Model):
    def __init__(self, X, Y):
        super().__init__(X, Y)

    def fit(self, pow=2, respConv=False):
        if respConv:
            self.Y, self.teta = responseConversion(self.X, self.Y)
        self.b = (
            pinv(power(self.X, pow, out=zeros_like(self.X), where=(self.X != 0)))
            .dot(matrix(self.Y).T)
            .T.tolist()[0]
        )
        self.yHat = (
            power(self.X, pow, out=zeros_like(self.X), where=(self.X != 0)) * matrix(self.b).T
        ).T.tolist()[0]
        self.yAv = sum(self.Y) / len(self.Y)
        self.R2 = 1 - (sum(power(self.Y - self.yHat, 2))) / (sum(power(self.Y - self.yAv, 2)))
        self.S2 = sum(power(self.Y - self.yHat, 2)) / (len(self.Y) - len(self.X.T) - 1)
        self.F = (
            (
                matrix(self.b)
                * power(self.X.T, pow, out=zeros_like(self.X.T), where=(self.X.T != 0))
                * matrix(self.Y).T
                - (sum(self.Y) ** 2) / len(self.Y)
            )
            / (len(self.X.T) - 1)
            / self.S2
        ).tolist()[0][0]
        return self.b


def responseConversion(X, Y):
    if abs(max(Y) / min(Y)) > 10:
        lambdas = linspace(-2, 2, num=1000)
        if min(Y) < 0:
            print("Negative value is found on Y, use Y without conversion")
            return Y, 1
        W = [log(Y) if la == 0 else (power(Y, la) - 1) / la for la in lambdas]
        middle = identity(len(Y)) - X * pinv(X)
        lambda_max = [
            (
                -0.5 * len(Y) * log(matrix(w) * middle * matrix(w).T / len(Y))
                + (la - 1) * sum(log(Y))
            ).tolist()[0][0]
            for w, la in zip(W, lambdas)
        ]
        teta = (
            round(lambdas[lambda_max.index(max(lambda_max))] * 2.0, 1) / 2.0
            if abs(lambdas[lambda_max.index(max(lambda_max))]) >= 5e-1
            else 0
        )
        Y = log(Y) if teta == 0 else (power(Y, teta) - 1) / teta
        return Y, teta
    else:
        print(
            f"The max to min ratio is too small to use response conversion: \
            {max(Y)} / {min(Y)} = {max(Y) / min(Y)}"
        )
        return Y, 1

from abc import ABC, abstractmethod
from numpy import (
    matrix,
    array,
    insert,
    sum,
    power,
    log,
    linspace,
    identity,
    zeros_like,
    cov,
    vstack,
    finfo,
)
from numpy.linalg import pinv
from typing import *


class ModelException(Exception):
    def __init__(self):
        super().__init__("The length for X and Y must be equal")


class Model(ABC):
    def __init__(self, X: List[List], Y: List) -> None:
        if len(matrix(X)) != len(Y):
            raise ModelException
        self.X = insert(matrix(X, dtype=float).T, 0, 1, axis=0).T
        self.Y = array(Y, dtype=float)
        self.b = []

    @abstractmethod
    def fit(self, respConv: bool = False):
        pass

    def __getattribute__(self, name: Any):
        return super().__getattribute__(name)

    def __str__(self) -> str:
        return f"X:\n{array(self.X.tolist())[:,1:].tolist()}]\n\nY:\n{list(self.Y)}\n\nb:{self.b}"


class LinearRegression(Model):
    def __init__(self, X: List[List], Y: List) -> None:
        super().__init__(X, Y)

    def fit(self, respConv: bool = False) -> List[float]:
        if respConv:
            self.Y, self.teta = responseConversion(self.X, self.Y)
        self.b = pinv(self.X).dot(matrix(self.Y).T).T.tolist()[0]
        self.yHat = (self.X * matrix(self.b).T).T.tolist()[0]
        yAv = sum(self.Y) / len(self.Y)
        self.R2 = 1 - (sum(power(self.Y - self.yHat, 2))) / (
            sum(power(self.Y - yAv, 2))
        )
        self.S2 = sum(power(self.Y - self.yHat, 2)) / (len(self.Y) - len(self.X.T) - 1)
        self.F = (
            (
                matrix(self.b) * self.X.T * matrix(self.Y).T
                - (sum(self.Y) ** 2) / len(self.Y)
            )
            / (len(self.X.T) - 1)
            / self.S2
        ).tolist()[0][0]
        return self.b


class PowerRegression(Model):
    def __init__(self, X: List[List], Y: List) -> None:
        super().__init__(X, Y)

    def fit(
        self,
        power_: List[float] = [2.0],
        respConv: bool = False,
    ) -> List[float]:
        if respConv:
            self.Y, self.teta = responseConversion(self.X, self.Y)
        self.R2 = 0
        if type(power_) != list:
            power_ = [power_]
        for pow_ in power_:
            b = (
                pinv(power(self.X, pow_, out=zeros_like(self.X), where=(self.X != 0)))
                .dot(matrix(self.Y).T)
                .T.tolist()[0]
            )
            yHat = (
                power(self.X, pow_, out=zeros_like(self.X), where=(self.X != 0))
                * matrix(b).T
            ).T.tolist()[0]
            yAv = sum(self.Y) / len(self.Y)
            R2 = 1 - (sum(power(self.Y - yHat, 2))) / (sum(power(self.Y - yAv, 2)))
            if self.R2 < R2:
                self.b = b
                self.yHat = yHat
                self.S2 = sum(power(self.Y - self.yHat, 2)) / (
                    len(self.Y) - len(self.X.T) - 1
                )
                self.F = (
                    (
                        matrix(self.b)
                        * power(
                            self.X.T,
                            pow_,
                            out=zeros_like(self.X.T),
                            where=(self.X.T != 0),
                        )
                        * matrix(self.Y).T
                        - (sum(self.Y) ** 2) / len(self.Y)
                    )
                    / (len(self.X.T) - 1)
                    / self.S2
                ).tolist()[0][0]
                self.R2 = R2
                self.power = pow_
        return self.b


class MultipleRegression(Model):
    def __init__(self, X: List[List], Y: List) -> None:
        super().__init__(X, Y)

    def fit(self, respConv: bool = False):
        self.teta = 1
        if respConv:
            self.Y, self.teta = responseConversion(self.X, self.Y)
        X_ = []
        self.powers = []
        for count, x in enumerate(self.X.T[1:]):
            c = cov(vstack((x, self.Y)))[0][1]
            if abs(c) < 1e-2:
                print(f"{count} parameter has {c} covariance with Y")
            else:
                model = PowerRegression([[elem] for elem in x.tolist()[0]], self.Y)
                model.fit(power_=linspace(-3, 3).tolist())
                X_.append(power(array(x), model.power).tolist()[0])
                self.powers.append(round(model.power, 1))
        self.X = insert(matrix(X_, dtype=float), 0, 1, axis=0).T
        self.b = pinv(self.X).dot(matrix(self.Y).T).T.tolist()[0]
        self.model = f"Y = {round(self.b[0],2)}"
        self.yHat = (self.X * matrix(self.b).T).T.tolist()[0]
        yAv = sum(self.Y) / len(self.Y)
        self.R2 = 1 - (sum(power(self.Y - self.yHat, 2))) / (
            sum(power(self.Y - yAv, 2))
        )
        self.S2 = sum(power(self.Y - self.yHat, 2)) / (len(self.Y) - len(self.X.T) - 1)
        self.F = (
            (
                matrix(self.b) * self.X.T * matrix(self.Y).T
                - (sum(self.Y) ** 2) / len(self.Y)
            )
            / (len(self.X.T) - 1)
            / self.S2
        ).tolist()[0][0]

        for count, value in enumerate(self.powers):
            self.model += f" + {round(self.b[count+1],4)} * X{count+1}^{value}"
        return self.b


def responseConversion(X: matrix, Y: array) -> Union[array, float]:
    Y += abs(min(Y)) + 1
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
            {max(Y)} / {min(Y)} = {abs(max(Y) / min(Y))}"
        )
        return Y, 1

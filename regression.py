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


class hashabledict(dict, MutableMapping):
    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


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
    def fit(self, respConv: bool = False) -> List[float]:
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
        self.R2 = 1 - (sum(power(self.Y - self.yHat, 2))) / (sum(power(self.Y - yAv, 2)))
        self.S2 = sum(power(self.Y - self.yHat, 2)) / (len(self.Y) - len(self.X.T) - 1)
        self.F = (
            (matrix(self.b) * self.X.T * matrix(self.Y).T - (sum(self.Y) ** 2) / len(self.Y))
            / (len(self.X.T) - 1)
            / self.S2
        ).tolist()[0][0]
        return self.b


class PowerRegression(Model):
    def __init__(self, X: List[List], Y: List) -> None:
        super().__init__(X, Y)

    def fit(
        self,
        power_: Union[List[float], float] = [2.0],
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
                power(self.X, pow_, out=zeros_like(self.X), where=(self.X != 0)) * matrix(b).T
            ).T.tolist()[0]
            yAv = sum(self.Y) / len(self.Y)
            R2 = 1 - (sum(power(self.Y - yHat, 2))) / (sum(power(self.Y - yAv, 2)))
            if self.R2 < R2:
                self.b = b
                self.yHat = yHat
                self.S2 = sum(power(self.Y - self.yHat, 2)) / (len(self.Y) - len(self.X.T) - 1)
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

    def fit(self, respConv: bool = False) -> List[float]:
        if respConv:
            self.Y, self.teta = responseConversion(self.X, self.Y)
        X_ = []
        self.powers = []
        for count, x in enumerate(self.X.T[1:]):
            model = PowerRegression([[elem] for elem in x.tolist()[0]], self.Y)
            model.fit(power_=linspace(-3, 3).tolist())
            X_.append(power(array(x), model.power).tolist()[0])
            self.powers.append(round(model.power, 1))
        self.X = insert(matrix(X_, dtype=float), 0, 1, axis=0).T
        self.b = pinv(self.X).dot(matrix(self.Y).T).T.tolist()[0]
        self.model = f"Y = {round(self.b[0],2)}"
        self.yHat = (self.X * matrix(self.b).T).T.tolist()[0]
        yAv = sum(self.Y) / len(self.Y)
        self.R2 = 1 - (sum(power(self.Y - self.yHat, 2))) / (sum(power(self.Y - yAv, 2)))
        self.S2 = sum(power(self.Y - self.yHat, 2)) / (len(self.Y) - len(self.X.T) - 1)
        self.F = (
            (matrix(self.b) * self.X.T * matrix(self.Y).T - (sum(self.Y) ** 2) / len(self.Y))
            / (len(self.X.T) - 1)
            / self.S2
        ).tolist()[0][0]

        for count, value in enumerate(self.powers):
            self.model += f" + {round(self.b[count+1],4)} * X{count+1}^{value}"
        return self.b


class FullLinearRegression(Model):
    def __init__(self, X: List[List], Y: List) -> None:
        super().__init__(X, Y)

    def fit(self, respConv: bool = False) -> List[float]:
        if respConv:
            self.Y, self.teta = responseConversion(self.X, self.Y)
        X_full = set(
            (tuple(x), hashabledict({count + 1: 1}))
            for count, x in enumerate(self.X.T[1:].tolist())
        )

        for count, x in enumerate(self.X.T[1:].tolist()):
            X_temp = X_full.copy()
            for x_ in X_full:
                if count + 1 in x_[1].keys():
                    continue
                else:
                    dict_ = hashabledict(x_[1])
                    dict_[count + 1] = 1
                    x_new = [x1 * x2 for x1, x2 in zip(x, list(x_[0]))]
                    new_cell = (tuple(x_new), dict_)
                    X_temp.add(new_cell)
            X_full = X_temp

        self.X = insert(matrix([list(x[0]) for x in X_full]), 0, 1, axis=0).T
        indexes = [k[1] for k in X_full]
        self.b = pinv(self.X).dot(matrix(self.Y).T).T.tolist()[0]
        self.model = f"Y = {round(self.b[0],2)}"
        self.yHat = (self.X * matrix(self.b).T).T.tolist()[0]
        yAv = sum(self.Y) / len(self.Y)
        self.R2 = 1 - (sum(power(self.Y - self.yHat, 2))) / (sum(power(self.Y - yAv, 2)))
        self.S2 = sum(power(self.Y - self.yHat, 2)) / (len(self.Y) - len(self.X.T) - 1)
        self.F = (
            (matrix(self.b) * self.X.T * matrix(self.Y).T - (sum(self.Y) ** 2) / len(self.Y))
            / (len(self.X.T) - 1)
            / self.S2
        ).tolist()[0][0]

        for count, value in enumerate(indexes):
            pp_x = ""
            for key in sorted(value.keys()):
                pp_x += f"X{key} * "
            pp_x = pp_x[:-3]
            self.model += f" + {round(self.b[count+1],4)} * " + pp_x
        return self.b


class FullofKRegression(Model):
    def __init__(self, X: List[List], Y: List) -> None:
        super().__init__(X, Y)

    def fit(self, power_: int = 2, respConv: bool = False) -> List[float]:
        if power_ < 1:
            raise "THe power must be greater than zero."

        if respConv:
            self.Y, self.teta = responseConversion(self.X, self.Y)
        if power_ == 1:
            linear = FullLinearRegression(self.X.T[1:].T, self.Y)
            linear.fit()
            self.b = linear.b
            self.R2 = linear.R2
            self.S2 = linear.S2
            self.F = linear.F
            self.model = linear.model
            del linear

        else:
            X_full = set(
                (tuple(x), hashabledict({count + 1: 1}))
                for count, x in enumerate(self.X.T[1:].tolist())
            )
            for i in range(power_):
                for count, x in enumerate(self.X.T[1:].tolist()):
                    X_temp = X_full.copy()
                    for x_ in X_full:
                        dict_ = hashabledict(x_[1])
                        if count + 1 in dict_.keys():
                            if dict_[count + 1] == power_:
                                continue
                            dict_[count + 1] += 1
                        else:
                            dict_[count + 1] = 1
                        x_new = [x1 * x2 for x1, x2 in zip(x, list(x_[0]))]
                        new_cell = (tuple(x_new), dict_)
                        X_temp.add(new_cell)
                    X_full = X_temp

            self.X = insert(matrix([list(x[0]) for x in X_full]), 0, 1, axis=0).T
            indexes = [k[1] for k in X_full]
            self.b = pinv(self.X).dot(matrix(self.Y).T).T.tolist()[0]
            self.model = f"Y = {round(self.b[0],2)}"
            self.yHat = (self.X * matrix(self.b).T).T.tolist()[0]
            yAv = sum(self.Y) / len(self.Y)
            self.R2 = 1 - (sum(power(self.Y - self.yHat, 2))) / (sum(power(self.Y - yAv, 2)))
            self.S2 = sum(power(self.Y - self.yHat, 2)) / (len(self.Y) - len(self.X.T) - 1)
            self.F = (
                (matrix(self.b) * self.X.T * matrix(self.Y).T - (sum(self.Y) ** 2) / len(self.Y))
                / (len(self.X.T) - 1)
                / self.S2
            ).tolist()[0][0]

            for count, value in enumerate(indexes):
                pp_x = ""
                for key in sorted(value.keys()):
                    pp_x += f"X{key}"
                    if value[key] != 1:
                        pp_x += f"^{value[key]}"
                    pp_x += " * "
                pp_x = pp_x[:-3]
                self.model += f" + {round(self.b[count+1],4)} * " + pp_x
        return self.b


def responseConversion(X: matrix, Y: array) -> Tuple[array, float]:
    if min(Y) < 0:
        print("Negative value is found on Y, use Y without conversion")
        return Y, 1
    Y_ratio = Y + abs(min(Y)) + 1
    if abs(max(Y_ratio) / min(Y_ratio)) > 10:
        lambdas = linspace(-2, 2, num=1000)
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

from abc import ABC, abstractmethod
from numpy import matrix, array, dot, insert
from numpy.linalg import pinv


class ModelException(Exception):
    def __init__(self):
        super().__init__("The length for X and Y must be equal")


class Model(ABC):
    def __init__(self, X, Y):
        if len(matrix(X)) != len(Y):
            raise ModelException
        self.X = insert(matrix(X).T, 0, 1, axis=0).T
        self.Y = array(Y)
        self.b = []

    @abstractmethod
    def calculate_b(self):
        pass

    def __getattribute__(self, name):
        return super().__getattribute__(name)


class LinearRegression(Model):
    def __init__(self, X, Y):
        super().__init__(X, Y)

    def calculate_b(self):
        self.b = pinv(self.X).dot(matrix(self.Y).T).T.tolist()[0]
        return self.b

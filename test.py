from regression import LinearRegression
from numpy import matrix


l = LinearRegression([[0, 1, 2], [1, 1, 1]], [5, 2])

print(l.calculate_b())
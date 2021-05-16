from regression import LinearRegression, PowerRegression
from random import randint
from matplotlib import pyplot as plt
import numpy as np

X = []
Y = []

"""with open(r"C:\Course\regression\Data.txt", "r") as f:
    for line in f.read().split("\n"):
        val = line.split(" ")
        X.append(val[:-1])
        Y.append(val[-1])"""

for p1 in np.linspace(1, 10):
    Y.append(14 + 11 / (p1 ** 2) + randint(-1, 1))
    X.append([p1])


l = PowerRegression(X, Y)
b = l.fit(respConv=True, pow=-2)

fig, ax = plt.subplots(figsize=(8, 6))


ax.scatter(X, Y, label="Experiments")
ax.plot(X, l.yHat)
plt.show()

import pandas as pd, matplotlib.pyplot as plt, numpy as np
from regression import *

data = pd.read_excel("data.xlsx")

X = [[x[0]] for x in data.values]
Y = [x[1] for x in data.values]

model = PowerRegression(X, Y)

b = model.fit(pow=3)
print(model)
x = np.linspace(np.amin(X), np.amax(X))
y = b[0] + b[1] * (pow(x, 3))

plt.scatter(np.array(data.values, dtype=float)[:, 0], np.array(data.values, dtype=float)[:, 1])
plt.plot(x, y, label="model", color="green")
plt.show()
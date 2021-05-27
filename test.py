import pandas as pd, matplotlib.pyplot as plt, numpy as np
from regression import *

data = pd.read_excel("data.xlsx")

X = [[x[0]] for x in data.values]
Y = [x[1] for x in data.values]

model = MultipleRegression(X, Y)

power = np.linspace(3, 4)
b = model.fit()
print(model.R2, model.model)
x = np.linspace(np.amin(X), np.amax(X))
# y = b[0] + b[1] * (pow(x, model.power))

plt.scatter(
    np.array(data.values, dtype=float)[:, 0], np.array(data.values, dtype=float)[:, 1]
)
# plt.plot(x, y, label="model", color="green")
plt.show()
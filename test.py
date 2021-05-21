import pandas as pd

data = pd.read_excel("cen-god.xls")
for line in data[" ПОКАЗАТЕЛИ "]:
    print(line.split(" "))

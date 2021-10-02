import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
data = pd.read_csv("Iris.csv")

formatter = {}
irises = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


for i in data:
    if i != "class":
        for iris in irises:
            if iris not in formatter:
                formatter[iris] = []

            x = list(data[i][data["class"]==iris])
            avg = sum(x)/len(x)
            formatter[iris].append(avg)

print(formatter)
filtered_data = []
for i in formatter:
    formatter[i].append(i)
    filtered_data.append(formatter[i])

print(filtered_data)
new_data = pd.DataFrame(filtered_data, columns=[i for i in data])
new_data.set_index("class").plot(kind="bar")
plt.show()

from random import randint
import math
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')
df_filtered = df[df['class'] != "Iris-virginica"]
x_full = df_filtered.drop(columns=["class"])
y_full = df_filtered["class"]

x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=5, stratify=y_full)

n = 100

activation_function = lambda x: 1 if 1/(1+pow(math.e, -x)) > 0.5 else 0

w = [0, 0, 0, 0]
b = 0
l = 0.1

epochs = 1000

classes = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1
}

for i in range(epochs):
    for idx in range(len(x_train)):
        row = x_train.iloc[idx]
        
        x = row[0]
        y = row[1]
        z = row[2]
        r = row[3]

        a = activation_function(x * w[0] + y * w[1] + z * w[2] + r * w[3] + b)
        e = classes[y_train.iloc[idx]] - a

        if e != 0:
            done = False
            w[0] += e * l * x
            w[1] += e * l * y
            w[2] += e * l * z
            w[3] += e * l * r

            b += e * l

# test perceptron
count = 0
correct = 0

for idx in range(len(x_test)):
    row = x_test.iloc[idx]
    
    x = row[0]
    y = row[1]
    z = row[2]
    r = row[3]

    a = activation_function(x * w[0] + y * w[1] + z * w[2] + r * w[3] + b)
    actual = classes[y_test.iloc[idx]]

    if a == actual:
        correct += 1
    count += 1

print("My Implementation Accuracy", str(correct / count * 100) + "%")


# classify x_train using sklearn Perceptron
from sklearn.linear_model import Perceptron
model = Perceptron(max_iter=1000, tol=1e-3)
model.fit(x_train, y_train)
output = pd.Series(model.predict(x_test))
li_output = [classes[output.iloc[i]] for i in range(len(output))]
li_original = [classes[y_test.iloc[i]] for i in range(len(y_test))]

correct = 0

for i in range(len(li_output)):
    if li_output[i] == li_original[i]:
        correct += 1 

print("SKLearn Accuracy", str(correct / len(li_output) * 100) + "%")


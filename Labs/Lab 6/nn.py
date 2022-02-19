from random import randint
import math
n = 100
points = [(randint(-10, 10), randint(-10, 10)) for i in range(n)]
# line y = x
actual = [1 if i[0] == i[1] else 0 for i in points]

activation_function = lambda x: 1 if 1/(1+pow(math.e, -x)) > 0.5 else 0

w = [0, 0]
b = 0
l = 0.1

epochs = 1000

for i in range(epochs):
    #train perceptron with random points
    count = 0
    done = True
    for point in points:
        x, y = point
        a = activation_function(x * w[0] + y * w[1] + b)
        e = actual[count] - a
        if e != 0:
            done = False
            w[0] += e * l * x
            w[1] += e * l * y
            b += e * l

        count += 1

# test perceptron
count = 0
test_points = [(randint(-10, 10), randint(-10, 10)) for i in range(n)]
actual = [1 if i[0] == i[1] else 0 for i in test_points]

correct = 0
for point in test_points:
    x, y = point
    a = activation_function(x * w[0] + y * w[1] + b)
    if a == actual[count]:
        correct += 1
    count += 1

print("Accuracy", str(correct / len(test_points) * 100) + "%")
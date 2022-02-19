from random import randint
import math

points = [(0, 0), (0, 1), (1, 0), (1, 1)]
# line y = x

actuals = [
    [1 if (i[0] and i[1]) else 0 for i in points],
    [1 if (i[0] or i[1]) else 0 for i in points],
    [1 if not (i[0] and i[1]) else 0 for i in points]
]

weights = []
biases = []

l = 1
epochs = 10

activation_function = lambda x: 1 if x > 0 else 0

for actual in actuals:

    w = [0, 0]
    b = 0


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
    correct = 0
    for point in points:
        x, y = point
        a = activation_function(x * w[0] + y * w[1] + b)
        if a == actual[count]:
            correct += 1
        count += 1
    

    print("Accuracy", str(correct / len(points) * 100) + "%")
    weights.append(w)
    biases.append(b)

print("and function weights and biases:", weights[0], biases[0])
print("or function weights and biases:", weights[1], biases[1])
print("nand function weights and biases:", weights[2], biases[2])

xor_actuals = [1 if i[0] != i[1] else 0 for i in points]

# test XOR perceptron
count = 0
correct = 0
for point in points:
    x, y = point
    nand_output = activation_function(x * weights[2][0] + y * weights[2][1] + biases[2])
    or_output = activation_function(x * weights[1][0] + y * weights[1][1] + biases[1])

    xor_output = activation_function(nand_output * weights[0][0] + or_output * weights[0][1] + biases[0])

    if xor_output == xor_actuals[count]:
        correct += 1
    count += 1

print("Accuracy", str(correct / len(points) * 100) + "%")

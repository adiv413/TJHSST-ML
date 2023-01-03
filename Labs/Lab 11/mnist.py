import sys; args = sys.argv[1:]
# Aditya Vasantharao, pd. 4
import math
import random
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random 
import math

def backprop(x_values, weights, transfer_function_dx, expected_outputs, alpha):
    errors = [[0.0 for j in range(len(x_values[i]))] for i in range(len(x_values))] # initialize errors to empty list with same shape as x_values
    gradients = [[[0.0 for k in j] for j in i] for i in weights] # stores negative gradient * alpha values, same shape as weights

    for layer in range(len(x_values) - 2, -1, -1): # layer
        for i in range(len(x_values[layer])): # each node in layer
            if layer == len(x_values) - 2: # if this is the last layer: special case
                first_neg_gradient = (expected_outputs[i] - x_values[layer + 1][i]) * x_values[layer][i]
                errors[layer][i] = (expected_outputs[i] - x_values[layer + 1][i]) * weights[layer - 1][i][0] * transfer_function_dx(x_values[layer][i])
                gradients[layer - 1][i][0] = alpha * first_neg_gradient # negative gradient for the very last weight 
            else:
                # compute error first, then the negative gradient
                sum_errors = 0.0

                for j in range(len(errors[layer + 1])): # iterate through all of the errors in the next layer
                    sum_errors += weights[layer][i][j] * errors[layer + 1][j]
                    neg_gradient = x_values[layer][i] * errors[layer + 1][j]
                    gradients[layer - 1][i][j] = alpha * neg_gradient

                errors[layer][i] = sum_errors * transfer_function_dx(x_values[layer][i])

    for layer in range(len(gradients)):
        for i in range(len(gradients[layer])):
            for j in range(len(gradients[layer][i])): # update all weights with the respective negative gradient * alpha value
                weights[layer][i][j] += gradients[layer][i][j]
    print(errors[-1][-1])

def forward_prop(x_values, weights, transfer_function):
    for layer in range(len(x_values) - 2): # layer = index of current layer
        curr_layer = x_values[layer]
        curr_weights = weights[layer]
        next_layer = [0.0 for layer in x_values[layer + 1]] # used to accumulate all of the node * weight values

        # outer: next layer, inner: curr layer
        for i in range(len(curr_layer)):
            for j in range(len(next_layer)):
                curr_weight = curr_weights[i][j]
                next_layer[j] += float(curr_layer[i] * curr_weight)
        
        for i in range(len(next_layer)):
            next_layer[i] = float(transfer_function(next_layer[i]))

            

        for i in range(len(next_layer)): # set the next layer of x_values
            x_values[layer + 1][i] = next_layer[i]

    curr_layer = x_values[-2]
    next_layer = [max([(curr_layer[k], k) for k in range(len(curr_layer))])[1]]

    for i in range(len(next_layer)): # set the next layer of x_values
        x_values[-1][i] = next_layer[i]

def linear(inp):
    return inp

def relu(inp):
    return inp if inp > 0 else 0

def logistic(inp):
    try:
        return 1 / (1 + math.e ** (-inp))
    except Exception as e:
        if inp > 0:
            return 1
        return 0


def scaled_logistic(inp):
    return 2 * logistic(inp) - 1

def logistic_dx(inp):
    return inp * (1 - inp)

(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Flatten the training and testing data

new_train_x = []
for sample in train_x:
    to_add = []

    for row in sample:
        for pixel in row:
            to_add.append(pixel)

    new_train_x.append(np.array(to_add))

train_x = np.array(new_train_x)

new_test_x = []
for sample in test_x:
    to_add = []

    for row in sample:
        for pixel in row:
            to_add.append(pixel)

    new_test_x.append(np.array(to_add))

test_x = np.array(new_test_x)

num_samples = len(train_x)
print(1)
raw_transfer_function = "logistic"
transfer_function_map = {"linear" : linear, "relu" : relu, "logistic" : logistic, "scaled_logistic" : scaled_logistic}
transfer_function = transfer_function_map[raw_transfer_function.lower()]
transfer_function_dx = logistic_dx

alpha = 0.1 # learning rate
epochs = 100

n = len(train_x[0])
node_counts = [n, 512, 10, 1]
weights = [[[random.random() for k in range(node_counts[i + 1])] for j in range(node_counts[i])] for i in range(len(node_counts) - 2)]
# print(inputs)
# print(expected_outputs)
# print(inequality)
# print(ineq_type)
# print(radius)
print("Layer counts:", *node_counts)

# run forward and backprop

for current_epoch in range(epochs):
    for current_input_idx in range(len(train_x)):
        inp_list = train_x[current_input_idx]
        # print(inp_list)
        x_values = [[0.0 for j in range(node_counts[i])] for i in range(len(node_counts))] # node values for all nodes in the network

        for i in range(len(x_values[0])):
            x_values[0][i] = inp_list[i]

        forward_prop(x_values, weights, transfer_function)
        # print(x_values[-1])
        # print()
        # print(inp, expected_outputs)
        backprop(x_values, weights, transfer_function_dx, test_x[current_input_idx], alpha)

        # if current_epoch > epochs - 6:
        #     print(expected_outputs[current_input_idx], x_values[-1])

        # if current_epoch % int(epochs / 10) == 0:
        #     print(expected_outputs[current_input_idx], x_values[-1])

    if current_epoch % int(epochs / 10) == 0:
        for layer in weights:
            for length in range(len(layer[0])):
                for node in layer:
                    print(node[length], end=" ")
            print()
        

# weights go from curr layer to next layer, 3d array
print()
print()
for layer in weights:
    for length in range(len(layer[0])):
        for i in layer:
            print(i[length], end=" ")
    print()  


# test the network

# test_inputs = []
# test_expected_outputs = []
# goof_count = 0
# num_test = 500

# for i in range(num_test):
#     x = (random.random() - 0.5) * 3
#     y = (random.random() - 0.5) * 3
#     test_inputs.append([x, y, 1])
#     output = None

#     if ineq_type == ">":
#         output = (x * x + y * y > radius)
#     elif ineq_type == ">=":
#         output = (x * x + y * y >= radius)
#     elif ineq_type == "<":
#         output = (x * x + y * y < radius)
#     else:
#         output = (x * x + y * y <= radius)

#     test_expected_outputs.append([1 if output else 0])

# for idx in range(len(test_inputs)):
#     inp_list = test_inputs[idx]
#     x_values = [[0.0 for j in range(node_counts[i])] for i in range(len(node_counts))] # node values for all nodes in the network

#     for i in range(len(x_values[0])):
#         x_values[0][i] = inp_list[i]

#     forward_prop(x_values, weights, transfer_function)
#     print(x_values[-1], test_expected_outputs[idx])
#     val = 1 if x_values[-1][0] > 0.5 else 0

#     if val != test_expected_outputs[idx][0]:
#         goof_count += 1

# print(goof_count, goof_count/num_test)


# from tensorflow import keras
# from keras.datasets import mnist
# from keras import layers, models
# import matplotlib.pyplot as plt
# import numpy as np
# import random 
# import math
# import pandas as pd
# from sklearn.model_selection import train_test_split

# f = open("weights.txt", "w")
# (train_x, train_y), (test_x, test_y) = mnist.load_data()

# # Flatten the training and testing data

# new_train_x = []
# for sample in train_x:
#     to_add = []

#     for row in sample:
#         for pixel in row:
#             to_add.append(pixel)

#     new_train_x.append(np.array(to_add))

# train_x = np.array(new_train_x)

# new_test_x = []
# for sample in test_x:
#     to_add = []

#     for row in sample:
#         for pixel in row:
#             to_add.append(pixel)

#     new_test_x.append(np.array(to_add))

# test_x = np.array(new_test_x)

# n = 100
# data_size = len(train_x[0])

# activation_function = lambda x: 1 if 1 / (1 + pow(math.e, -x)) > 0.5 else 0
# layer_1_num = 784
# layer_2_num = 512
# w = [[random.random() for i in range(layer_1_num)], [random.random() for i in range(layer_2_num)]]
# b = [random.random(), random.random()]
# l = 0.1

# epochs = 1000

# for i in range(epochs):
#     # forward
#     for idx in range(len(train_x)):
#         row = train_x[idx]
        
#         weight_sum_1 = 0

#         for i in row:
#             weight_sum_1 += i * w[0][i]

#         a_1 = activation_function(weight_sum_1 + b[0])
#         e = classes[y_train.iloc[idx]] - a

#         if e != 0:
#             done = False
#             w[0] += e * l * x
#             w[1] += e * l * y
#             w[2] += e * l * z
#             w[3] += e * l * r

#             b += e * l

# # test perceptron
# count = 0
# correct = 0

# for idx in range(len(x_test)):
#     row = x_test.iloc[idx]
    
#     x = row[0]
#     y = row[1]
#     z = row[2]
#     r = row[3]

#     a = activation_function(x * w[0] + y * w[1] + z * w[2] + r * w[3] + b)
#     actual = classes[y_test.iloc[idx]]

#     if a == actual:
#         correct += 1
#     count += 1

# print("My Implementation Accuracy", str(correct / count * 100) + "%")


# # classify x_train using sklearn Perceptron
# from sklearn.linear_model import Perceptron
# model = Perceptron(max_iter=1000, tol=1e-3)
# model.fit(x_train, y_train)
# output = pd.Series(model.predict(x_test))
# li_output = [classes[output.iloc[i]] for i in range(len(output))]
# li_original = [classes[y_test.iloc[i]] for i in range(len(y_test))]

# correct = 0

# for i in range(len(li_output)):
#     if li_output[i] == li_original[i]:
#         correct += 1 

# print("SKLearn Accuracy", str(correct / len(li_output) * 100) + "%")



# # Create model
# model = keras.Sequential()
# model.add(keras.layers.Input(shape=train_x.shape))
# model.add(keras.layers.Dense(512, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))
# model.summary()

# model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# history = model.fit(train_x, train_y, epochs=30, validation_data=(test_x, test_y))

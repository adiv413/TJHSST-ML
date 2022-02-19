# keras multilayer network
import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


iris = load_iris()

data = iris["data"]
classes = iris["target"]

encoder = OneHotEncoder()
classes = encoder.fit_transform(classes[:, np.newaxis]).toarray()

scaler = StandardScaler()
data = scaler.fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(keras.layers.InputLayer(x_train.shape))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])

epochs = 50

weights = []
mse = []

for i in range(epochs):
    history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
    weight = model.layers[0].get_weights()[0][0][0] # list of numpy arrays
    weights.append(weight)
    mse.append(history.history['mse'][0])

plt.plot(mse)
plt.plot(weights)
plt.xlabel('Epoch')
plt.ylabel('Weight or Error')
plt.legend(['MSE', 'Weight'], loc='upper right')
plt.show()

# implement one-vs-all classification for iris dataset
y_train_0 = []
y_train_1 = []
y_train_2 = []

y_test_0 = []
y_test_1 = []
y_test_2 = []

for i in y_train:
    if i[0] == 1:
        y_train_0.append(np.array([1]))
    else:
        y_train_0.append(np.array([0]))
    
    if i[1] == 1:
        y_train_1.append(np.array([1]))
    else:
        y_train_1.append(np.array([0]))
    
    if i[2] == 1:
        y_train_2.append(np.array([1]))
    else:
        y_train_2.append(np.array([0]))

for i in y_test:
    if i[0] == 1:
        y_test_0.append(np.array([1]))
    else:
        y_test_0.append(np.array([0]))
    
    if i[1] == 1:
        y_test_1.append(np.array([1]))
    else:
        y_test_1.append(np.array([0]))
    
    if i[2] == 1:
        y_test_2.append(np.array([1]))
    else:
        y_test_2.append(np.array([0]))

y_train_0 = np.array(y_train_0)
y_train_1 = np.array(y_train_1)
y_train_2 = np.array(y_train_2)

y_test_0 = np.array(y_test_0)
y_test_1 = np.array(y_test_1)
y_test_2 = np.array(y_test_2)

model1 = keras.Sequential()
model1.add(keras.layers.InputLayer(x_train.shape))
model1.add(keras.layers.Dense(50, activation='relu'))
model1.add(keras.layers.Dense(1, activation='sigmoid'))
model1.summary()

model1.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse'])
history = model1.fit(x_train, y_train_0, epochs=100, validation_data=(x_test, y_test_0))


model2 = keras.Sequential()
model2.add(keras.layers.InputLayer(x_train.shape))
model2.add(keras.layers.Dense(50, activation='relu'))
model2.add(keras.layers.Dense(1, activation='sigmoid'))
model2.summary()

model2.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse'])
history = model2.fit(x_train, y_train_1, epochs=100, validation_data=(x_test, y_test_1))


model3 = keras.Sequential()
model3.add(keras.layers.InputLayer(x_train.shape))
model3.add(keras.layers.Dense(50, activation='relu'))
model3.add(keras.layers.Dense(1, activation='sigmoid'))
model3.summary()

model3.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse'])
history = model3.fit(x_train, y_train_2, epochs=100, validation_data=(x_test, y_test_2))

count = 0
correct = 0

for i in x_test:
    predicted_value = [model1.predict(i.reshape(1, 4))[0][0], model2.predict(i.reshape(1, 4))[0][0], model3.predict(i.reshape(1, 4))[0][0]]
    out = [0, 0, 0]
    out[np.argmax(predicted_value)] = 1
    
    if out[0] == y_test[count][0] and out[1] == y_test[count][1] and out[2] == y_test[count][2]:
        correct += 1

    count += 1

print("OVA Model Accuracy: ", str(round(correct / count * 100, 3)) + "%")
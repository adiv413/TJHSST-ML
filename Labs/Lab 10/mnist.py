import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(512, activation='relu', input_shape=x_train.shape))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy', 'mse'])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['accuracy'])
plt.plot(history.history['mse'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy or Error')
plt.legend(['Accuracy', 'MSE'], loc='upper right')
plt.show()
# import tensorflow and other libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# implement Glaucoma Detection based on Deep Convolutional Neural Network paper

model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=3, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(filters=96, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu'))

# add fully connected layers
model.add(Flatten())
model.add(Dense(units=2048, activation='relu'))
# add dropout layer
model.add(Dropout(0.5))
model.add(Dense(units=2, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# dot fit that bih
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))


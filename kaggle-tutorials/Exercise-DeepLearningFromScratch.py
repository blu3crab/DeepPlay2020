__author__ = "blu3crab"
__license__ = "Apache License 2.0"
__version__ = "0.0.1"

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

img_rows, img_cols = 28, 28
num_classes = 10


def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)

    x = raw[:, 1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y


fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)

# Set up code checking
from learntools.core import binder

binder.bind(globals())
from learntools.deep_learning.exercise_7 import *

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

print("Setup Complete")
fashion_model = Sequential()

fashion_model.add(Conv2D(12, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
fashion_model.add(Flatten())
fashion_model.add(Dense(100, activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

fashion_model.fit(x, y,
          batch_size=100,
          epochs=4,
          validation_split = 0.2)

# Epoch 4/4
# 48000/48000 [==============================] - 37s 779us/sample - loss: 0.2087 - accuracy: 0.9230
# #- val_loss: 0.2491 - val_accuracy: 0.9128

print("second_fashion_model-->")
second_fashion_model = Sequential()

second_fashion_model.add(Conv2D(24, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

second_fashion_model.add(Conv2D(24, kernel_size=(3, 3), activation='relu'))
second_fashion_model.add(Conv2D(24, kernel_size=(3, 3), activation='relu'))
second_fashion_model.add(Flatten())
second_fashion_model.add(Dense(256, activation='relu'))
second_fashion_model.add(Dense(num_classes, activation='softmax'))

second_fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

second_fashion_model.fit(x, y,
          batch_size=256,
          epochs=8,
          validation_split = 0.2)


# Don't remove this line (ensures comptibility with tensorflow 2.0)
second_fashion_model.history.history['val_acc'] = second_fashion_model.history.history['val_accuracy']
# Check your answer
q_6.check()
# Epoch 8/8
# 48000/48000 [==============================] - 52s 1ms/sample - loss: 0.0866 - accuracy: 0.9696
# #- val_loss: 0.3088 - val_accuracy: 0.9087


# # import TF.Learn
# import numpy
# import tensorflow as tf
# from keras.datasets import mnist
# from sklearn.model_selection import train_test_split
#
# from Classification import TEST_SIZE
# from Helper_functions import divide_chunks
#
#
# def dnn_default(x_data, label):
#     # init = tf.global_variables_initializer()
#     # saver = tf.train.Saver()
#
#     # Training section
#     n_epochs = 40
#     batch_size = 50
#
#     x_train, x_test, y_train, y_test = train_test_split(x_data, label, test_size=TEST_SIZE, random_state=42)
#     (x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()
#
#     x_train = numpy.asarray(x_train)
#     y_train = numpy.asarray(y_train)
#     x_test = numpy.asarray(x_test)
#     y_test = numpy.asarray(y_test)
#
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(8, 50)),
#         tf.keras.layers.Dense(128, activation='elu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(13, activation='softmax')
#     ])
#
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     print("Training")
#     model.fit(x_train, y_train, epochs=50)
#
#     print("Evaluate")
#     model.evaluate(x_test, y_test)
#
#
#     print("x")
#
#


# CNN
from collections import Counter

import numpy
import sklearn
from keras.utils import to_categorical
from numpy.polynomial.tests.test_laguerre import L2
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.datasets import mnist
import tensorflow as tf

from Classification import TEST_SIZE
import matplotlib.pyplot as plt


def cnn(x, y):
    classes = len(Counter(y).keys())

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=TEST_SIZE)

    x_train = numpy.asarray(x_train)
    x_train = numpy.array(x_train)[:, :, :, numpy.newaxis]
    x_test = numpy.array(x_test)[:, :, :, numpy.newaxis]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu',
                     input_shape=(x_train.shape[1], x_train.shape[2], 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    # sgd = optimizers.SGD(lr=0.1, momentum=0.95, nesterov=True)
    # sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test))

    # predict first 4 images in the test set
    # print(model.predict(x_test[:4]))

    print("finish")

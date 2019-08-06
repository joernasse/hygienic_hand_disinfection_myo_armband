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
import numpy
from numpy.polynomial.tests.test_laguerre import L2
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.datasets import mnist

from Classification import TEST_SIZE


def cnn(x, y):
    # xa=numpy.reshape(x,(9,10))
    train = int(len(x) * (1 - TEST_SIZE))
    x_train = x[:train]
    y_train = y[:train]
    y_test = y[train:]
    x_test = x[train:]

    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=3,
                     activation='relu',
                     input_shape=(x_train[0].shape[0], x_train[0].shape[1], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(13, activation='softmax'))

    sgd = optimizers.SGD(lr=0.1, momentum=0.95, nesterov=True)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=256, epochs=25, verbose=1, validation_data=(x_test, y_test))

    print("")

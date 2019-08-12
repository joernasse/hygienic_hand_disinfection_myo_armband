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
import os
from collections import Counter

import numpy
import sklearn
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from numpy.polynomial.tests.test_laguerre import L2
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import layers

from Classification import TEST_SIZE
import matplotlib.pyplot as plt


def dnn(x, y):
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation=tf.nn))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(10, activation='softmax'))


def cnn_2(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features['x'], [-1, 25, 9, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 6 * 2 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=13)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn(x, y):
    classes = len(Counter(y).keys())

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=TEST_SIZE)

    x_train = numpy.asarray(x_train)
    x_train = numpy.array(x_train)[:, :, :, numpy.newaxis]
    x_test = numpy.array(x_test)[:, :, :, numpy.newaxis]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()

    # layer 1
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(Flatten())
    model.add(Dense(256, activation='elu'))
    model.add(Dense(classes, activation='softmax'))

    sgd = optimizers.SGD(lr=0.1, momentum=0.95)
    # sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    checkpoint_path = "G:/Masterarbeit/deep_learning/training_emg_3/model.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     verbose=2,
                                                     monitor='acc',
                                                     save_best_only=True,
                                                     mode='max')

    model.fit(x_train, y_train,
              batch_size=4000,
              epochs=60,
              validation_data=(x_test, y_test),
              callbacks=[cp_callback],
              verbose=2)

    model.save('myFirstCNN.model')
    model.summary()

    # predict first 4 images in the test set
    # print(model.predict(x_test[:4]))

    print("finish")

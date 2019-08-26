# # import TF.Learn
# import np
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
#     x_train = np.asarray(x_train)
#     y_train = np.asarray(y_train)
#     x_test = np.asarray(x_test)
#     y_test = np.asarray(y_test)
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
import random
from collections import Counter
from time import time

import numpy as np
import sklearn
import Constant
import keras
import tensorflow
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization, Activation
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import Helper_functions


def dnn(save_path, users_data, batch, epochs):
    # for n in range(len(Constant.USERS_cross)):
    test_user = users_data[0].copy()
    train_users = users_data.copy()
    train_users.pop(0)

    x_train, y_train = flat_data_user_cross_val(train_users)
    x_test, y_test = flat_data_user_cross_val([test_user])

    classes = 13

    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=TEST_SIZE)
    model = tf.keras.Sequential()
    model.add(layers.Dense(2048, activation='elu'))
    model.add(layers.Dense(1024, activation='elu'))
    model.add(layers.Dense(512, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(layers.Dense(classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(save_path + "/dnn_feature_extraction_model.h5",
                                                     verbose=1, monitor='val_acc',
                                                     save_best_only=True, mode='max')

    print("batch_size", batch, "epochs", epochs)

    history = model.fit(x_train, y_train,
                        batch_size=batch, epochs=epochs,
                        validation_data=(x_test, y_test), callbacks=[cp_callback], verbose=1)

    y_predict = model.predict_classes(x_test, batch)
    # model.save('myFirstCNN.model')

    model.summary()
    # visualization_history(history)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plot_confusion_matrix(y_test, y_predict, classes=Constant.sub_label,
    #                       title='Confusion matrix, without normalization')
    #
    # # Plot normalized confusion matrix
    # plot_confusion_matrix(y_test, y_predict, classes=Constant.sub_label, normalize=True,
    #                       title='Normalized confusion matrix')

    plt.show()
    print("finish")


def cnn_kaggle(x, y, save_path, batch, epochs):
    x_train, x_test, y_train, y_test_one_hot, classes, y_test = pre_process_cnn(x, y)
    model = keras.Sequential()

    # layer 1
    model.add(keras.layers.Conv2D(64,
                                  kernel_size=3,
                                  activation='relu',
                                  input_shape=(x_train.shape[1], x_train.shape[2], 1),
                                  padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Layer 2
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.25))

    # Layer 3
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.25))

    # Layer 4
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(500,
                                 use_bias=False,
                                 activation='elu'))
    model.add(keras.layers.BatchNormalization())

    # Output Layer
    model.add(keras.layers.Dense(13, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    cp_callback = [
        keras.callbacks.EarlyStopping(
            monitor='val_acc',
            patience=10,
            mode='max',
            verbose=2),
        keras.callbacks.ModelCheckpoint(
            save_path + "/cnn_kaggle_model.h5",
            verbose=2,
            monitor='val_acc',
            save_best_only=True,
            mode='max')
    ]

    print("CNN pattern kaggle", "batch_size", batch, "epochs", epochs)
    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=batch,
                        epochs=epochs,
                        validation_data=(x_test, y_test_one_hot),
                        shuffle=True,
                        callbacks=cp_callback,
                        verbose=2)

    eval = model.evaluate(x_test, to_categorical(y_test))
    print("EVAL", eval)
    y_predict = model.predict_classes(x_test, batch)

    Helper_functions.visualization(history=history, y_predict=y_predict, y_test=y_test)


def pre_process_cnn(x, y, adapt_model=False):
    classes = len(Counter(y).keys())
    if adapt_model:
        x = np.array(x)[:, :, :, np.newaxis]
        y = to_categorical(y)
        return x, y, classes
    # x_train, y_train, x_test, y_test = train_test_split_self(x, y, Constant.TEST_SIZE)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=Constant.TEST_SIZE)
    x_train = np.asarray(x_train)
    x_train = np.array(x_train)[:, :, :, np.newaxis]
    x_test = np.array(x_test)[:, :, :, np.newaxis]
    y_train = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    return x_train, x_test, y_train, y_test_one_hot, classes, y_test


def cnn(x, y, save_path, batch, epochs):
    x_train, x_test, y_train, y_test_one_hot, classes, y_test = pre_process_cnn(x, y)
    model = Sequential()

    # layer 1
    model.add(Conv2D(32, kernel_size=3,
                     activation='relu',
                     input_shape=(x_train.shape[1],
                                  x_train.shape[2], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(32, kernel_size=3,
                     activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # Layer 3
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    cp_callback = [
        keras.callbacks.EarlyStopping(
            monitor='val_acc',
            patience=10,
            mode='max',
            verbose=1),
        keras.callbacks.ModelCheckpoint(
            save_path + "/cnn_model.h5",
            verbose=1,
            monitor='val_acc',
            save_best_only=True,
            mode='max')
    ]

    print("batch_size", batch, "epochs", epochs)

    history = model.fit(x_train, y_train,
                        batch_size=batch,
                        epochs=epochs,
                        validation_data=(x_test, y_test_one_hot),
                        callbacks=cp_callback,
                        verbose=2)

    y_predict = model.predict_classes(x_test, batch)

    model.summary()

    Helper_functions.visualization(history, y_test, y_predict)
    plt.show()
    print("finish")


def adapt_model_for_user(x_train, y_train, save_path, batch, epochs, user_name, x_test, y_test, save_label=None,
                         model=None):
    x_train, y_train, classes, = pre_process_cnn(x_train, y_train, adapt_model=True)
    save_path_1 = save_path + "/cnn_kaggle_adapt_model.h5"
    if model is None:
        print("No model loaded, use new model")
        model = keggle_model(save_path, x_train.shape[1], x_train.shape[2])
        save_path_1 = save_path + "/cnn_kaggle_only_user_short_train_model.h5"
        epochs = epochs * 5  # Da untrainiertes Netz
    cp_callback = [
        keras.callbacks.EarlyStopping(
            monitor='val_acc',
            patience=10,
            mode='max',
            verbose=1),
        keras.callbacks.ModelCheckpoint(
            save_path_1,
            verbose=1,
            monitor='val_acc',
            save_best_only=True,
            mode='max')
    ]
    print("batch_size", batch, "epochs", epochs)
    start = time()
    history = model.fit(x_train, y_train,
                        batch_size=batch,
                        epochs=epochs,
                        validation_data=(x_train, y_train),
                        callbacks=cp_callback,
                        verbose=1)
    print("--- %s seconds ---" % (time() - start))
    model.summary()

    x_test, tmp_y, tmp = pre_process_cnn(x_test, y_test, adapt_model=True)
    y_predict = model.predict_classes(x_test, batch)

    Helper_functions.visualization(history, y_test, y_predict)
    plt.show()
    print("finish")


def predict_for_load_model(x, y, model, batch_size):
    # nur um visualisierung zu generieren
    # x_train, x, y_train, y = train_test_split(x, y, test_size=Constant.TEST_SIZE, random_state=42)
    # print("x",len(x))
    #ENDE

    x = np.array(x)[:, :, :, np.newaxis]
    y_predict = model.predict_classes(x, batch_size)
    eval = model.evaluate(x, to_categorical(y))
    print("EVAL", eval)
    Helper_functions.visualization(0, y, y_predict)


def flat_data_user_cross_val(users_data):
    x, y = [], []
    for user in users_data:
        for n in range(len(user['data'])):
            x.append(user['data'][n])
            y.append(user['label'][n])
    return x, y


def train_test_split_self(x, y, test_size, seed=42):
    c = list(zip(x, y))
    l = len(x)
    test_l = int(l * test_size)
    random.seed(seed)
    random.shuffle(c)
    x, y = zip(*c)

    x_train = x[:(l - test_l)]
    y_train = y[:(l - test_l)]
    x_test = x[(l - test_l):]
    y_test = y[(l - test_l):]

    # x_test, y_test = [], []
    # indices = []
    #
    # while len(x_test) < test_l:
    #     i = random.randint(0, l - 1)
    #     if not indices.__contains__(i):
    #         indices.append(i)
    #         x_test.append(x[i])
    #         y_test.append(y[i])
    #         del x[i]
    #         del y[i]
    return x_train, y_train, x_test, y_test


def keggle_model(save_path, shape_x, shape_y):
    model = keras.Sequential()

    # layer 1
    model.add(keras.layers.Conv2D(64,
                                  kernel_size=3,
                                  activation='relu',
                                  input_shape=(shape_x, shape_y, 1),
                                  padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Layer 2
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.25))

    # Layer 3
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.25))

    # Layer 4
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(500,
                                 use_bias=False,
                                 activation='elu'))
    model.add(keras.layers.BatchNormalization())

    # Output Layer
    model.add(keras.layers.Dense(13, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

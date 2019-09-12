# CNN
import gc
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
from Helper_functions import flat_users_data

import Helper_functions


def dnn(save_path, users_data, batch, epochs):
    """

    :param save_path:
    :param users_data:
    :param batch:
    :param epochs:
    :return:
    """
    # for n in range(len(Constant.USERS_cross)):
    test_user = users_data[0].copy()
    train_users = users_data.copy()
    train_users.pop(0)

    x_train, y_train = flat_users_data(train_users)
    x_test, y_test = flat_users_data([test_user])

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
    print("finish")


def cnn_kaggle(x, y, save_path, batch, epochs, config, early_stopping=20):
    """

    :param x:
    :param y:
    :param save_path:
    :param batch:
    :param epochs:
    :param config:
    :param x_test_in:
    :param y_test_in:
    :param early_stopping:
    :return:
    """
    x_train, x_test, y_train, classes, y_test, x_val, y_val_one_hot = pre_process_cnn(x, y, validation_size=0.1)
    print("Training", len(x_train),
          "\nValidation", len(x_val),
          "\nTest", len(x_test))
    model = kaggle_model(x_train.shape[1], x_train.shape[2])

    cp_callback = [
        keras.callbacks.EarlyStopping(
            monitor='val_acc',
            patience=early_stopping,
            mode='max',
            verbose=1),
        keras.callbacks.ModelCheckpoint(
            save_path + "/" + config + "_cnn_kaggle.h5",
            verbose=1,
            monitor='val_acc',
            save_best_only=True,
            mode='max')]

    history = model.fit(x_train, y_train,
                        batch_size=batch,
                        epochs=epochs,
                        validation_data=(x_val, y_val_one_hot),
                        callbacks=cp_callback,
                        verbose=1,
                        shuffle=True,
                        use_multiprocessing=True)

    print("CNN pattern kaggle", "batch_size", batch, "epochs", epochs)

    y_predict = model.predict_classes(x_test,batch_size=batch)
    acc = Helper_functions.visualization(history, y_test, y_predict, save_path=save_path, config=config)

    print("Train for CNN_Kaggle done")
    del x_train, y_train, x_val, x_test
    return model, "CNN_Kaggle", acc


def pre_process_cnn(x, y, adapt_model=False, validation_size=0.1, calc_test_set=True):
    classes = len(Counter(y).keys())
    if adapt_model:
        x = np.array(x)[:, :, :, np.newaxis]
        y = to_categorical(y)
        return x, y, classes

    samples_number_val_size = int(len(x) * validation_size)
    x, x_val, y, y_val = train_test_split(x, y, random_state=42, test_size=validation_size)
    y_val_one_hot = to_categorical(y_val)
    x_val = np.array(x_val)[:, :, :, np.newaxis]
    if not calc_test_set:
        x_train = np.array(x)[:, :, :, np.newaxis]
        y_train = to_categorical(y)
        return x_train, [], y_train, classes, [], x_val, y_val_one_hot

    test_size = samples_number_val_size / len(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=test_size)
    x_train = np.asarray(x_train)
    x_train = np.array(x_train)[:, :, :, np.newaxis]
    x_test = np.array(x_test)[:, :, :, np.newaxis]
    y_train = to_categorical(y_train)
    return x_train, x_test, y_train, classes, y_test, x_val, y_val_one_hot


def cnn_1(x, y, save_path, batch, epochs, config=""):
    """

    :param x:
    :param y:
    :param save_path:
    :param batch:
    :param epochs:
    :param config:
    :return:
    """
    print("batch_size", batch, "epochs", epochs)
    x_train, x_test, y_train, y_test_one_hot, classes, y_test, x_val, y_val_one_hot = pre_process_cnn(x, y,
                                                                                                      validation_size=0.1)

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
            patience=20,
            mode='max',
            verbose=1),
        keras.callbacks.ModelCheckpoint(
            save_path + "/" + config + "_cnn_1.h5",
            verbose=1,
            monitor='val_acc',
            save_best_only=True,
            mode='max')]

    history = model.fit(x_train, y_train,
                        batch_size=batch,
                        epochs=epochs,
                        validation_data=(x_val, y_val_one_hot),
                        callbacks=cp_callback,
                        verbose=1, shuffle=True)

    y_predict = model.predict_classes(x_test, batch)

    model.summary()

    acc = Helper_functions.visualization(history, y_test, y_predict, save_path=save_path, config=config)
    print("Train and predict CNN1 done")
    del model
    keras.backend.clear_session()
    gc.collect()
    del x_train, y_train, x_val, x_test
    return "CNN_1", acc


def adapt_model_for_user(x_train, y_train, save_path, batch, epochs,
                         user_name, x_test, y_test, save_label=None, model=None):
    """

    :param x_train:
    :param y_train:
    :param save_path:
    :param batch:
    :param epochs:
    :param user_name:
    :param x_test:
    :param y_test:
    :param save_label:
    :param model:
    :return:
    """
    x_train, y_train, classes, = pre_process_cnn(x_train, y_train, adapt_model=True)
    save_path_1 = save_path + "/cnn_kaggle_adapt_model.h5"
    if model is None:
        print("No model loaded, use new model")
        model = kaggle_model(x_train.shape[1], x_train.shape[2])
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
    # plt.show()
    print("finish")


def predict_for_load_model(x, y, model, batch_size):
    """

    :param x:
    :param y:
    :param model:
    :param batch_size:
    :return:
    """
    x = np.array(x)[:, :, :, np.newaxis]
    y_predict = model.predict_classes(x, batch_size)
    eval = model.evaluate(x, to_categorical(y))
    print("EVAL", eval)
    Helper_functions.visualization(0, y, y_predict, show_figures=True)


def kaggle_model(shape_x, shape_y):
    """
    :param shape_x:
    :param shape_y:
    :return:
    """
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
    model.add(keras.layers.Dense(12, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

#!/usr/bin/env python
"""
This script contains the creation, training and prediction of CNNs.
Also contains the adaptive approach
"""

from collections import Counter
from time import time
import numpy as np
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
import Constant
import Helper_functions

__author__ = "Joern Asse"
__copyright__ = ""
__credits__ = ["Joern Asse"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Joern Asse"
__email__ = "joernasse@yahoo.de"
__status__ = "Production"


def calculate_cnn(x, y, save_path="./", batch=32, epochs=10, config="", early_stopping=20, cnn_pattern=Constant.CNN_1,
                  validation_size=Constant.validation_set_size, test_size=Constant.test_set_size, perform_test=False,
                  visualization=False):
    """
    Train a CNN with given training data and save the model
    :param x: list
            The training data
    :param y:list
            The label for training data
    :param save_path:string
            The save path
    :param batch:int
            Batch size
    :param epochs:int
            Number of epochs
    :param config:string
            The configuration, indicates the CNN, will be added to model name
    :param early_stopping:int
            The parameter set the early stopping
    :param cnn_pattern:string
            Indicates which model structure will be used
    :param test_size:float
            Specifies the portion of data to be retained as test data
    :param validation_size:float
            Specifies the portion of data to be retained as validation data
    :param perform_test:boolean
            If True a prediction will performed on the test data
    :param visualization: boolean
            If True the results and history will be visualized
    :return:
    """

    x_train, x_test, x_val, y_train, y_test, y_val_one_hot, classes = prepare_data_for_cnn(x, y,
                                                                                           validation_size=validation_size,
                                                                                           test_size=test_size)
    acc = 0
    print("Training", len(x_train),
          "\nValidation", len(x_val),
          "\nTest", len(x_test),
          "\nBatch size", batch,
          "\nEpochs", epochs)

    print("Training for", cnn_pattern, " - Start")
    if cnn_pattern == Constant.CNN_KAGGLE:
        model = create_kaggle_model(x_train.shape[1], x_train.shape[2], classes)
    else:
        model = create_cnn_1_model(x_train.shape[1], x_train.shape[2], classes)

    cp_callback = [
        keras.callbacks.EarlyStopping(
            monitor='val_acc', patience=early_stopping,
            mode='max', verbose=1),
        keras.callbacks.ModelCheckpoint(
            save_path + "/" + config + "_cnn_" + cnn_pattern + ".h5", verbose=1,
            monitor='val_acc', save_best_only=True, mode='max')]

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch,
                        epochs=epochs,
                        validation_data=(x_val, y_val_one_hot),
                        callbacks=cp_callback,
                        verbose=1,
                        shuffle=True,
                        use_multiprocessing=True)
    print("Training for", cnn_pattern, " - Done")
    if visualization:
        Helper_functions.history_visualization(history, save_path=save_path, config=config, show_results=visualization)

    if perform_test:
        print("Prediction for", cnn_pattern, " - Start")
        y_predict = model.predict_classes(x_test, batch_size=batch)
        print("Prediction for", cnn_pattern, " - Done")
        acc = Helper_functions.result_visualization(y_test, y_predict, save_path=save_path, config=config + cnn_pattern)
        del x_test, y_test

    print("Train for " + cnn_pattern + " done")
    del x_train, y_train, x_val, y_val_one_hot
    return model, cnn_pattern, acc


def prepare_data_for_cnn(x, y, validation_size=Constant.validation_set_size,
                         test_size=Constant.test_set_size, calc_test_set=False):
    """
    Prepare the preprocessed data for the usage in a CNN. Splits the data into test, training and validation data.
    :param x:matrix
            The windowed and preprocessed data
    :param y: list
            The labels for the input matrix x
    :param test_size:float
            Specifies the portion of data to be retained as test data
    :param validation_size:float
            Specifies the portion of data to be retained as validation data
    :param calc_test_set:boolean
            If True a test data set will be calculated from the input data set
    :return: x_train, x_test, x_val, y_train, y_test, y_val, classes
            Return the matrices of the test,training and validation data
            Return the lists of the labels for the test, training and validation matrices
            Return the number of classes
    """
    if test_size > 0 and calc_test_set:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=test_size)
        x_test = np.array(x_test)[:, :, :, np.newaxis]
    else:
        x_train = x
        y_train = y
        x_test, y_test = [], []

    validation_size = (int(len(x) * validation_size)) / len(x)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42, test_size=validation_size)

    y_val = to_categorical(y_val)
    x_val = np.array(x_val)[:, :, :, np.newaxis]

    x_train = np.asarray(x_train)
    x_train = np.array(x_train)[:, :, :, np.newaxis]

    y_train = to_categorical(y_train)
    classes = len(Counter(y).keys())
    return x_train, x_test, x_val, y_train, y_test, y_val, classes


def adapt_model_for_user(x_train, y_train, save_path, batch, epochs, file_name, x_test_in, y_test_in, model,
                         calc_test_set=False, cnn_pattern=Constant.CNN_KAGGLE):
    """
    Trains a generalized model with a data set of an unknown user
    :param x_train:matrix
            Data with which the CNN is to be trained
    :param y_train:list
            labels for the training data
    :param save_path:string
            The save path
    :param batch:int
            Batch size
    :param epochs:int
            Number of epochs
    :param cnn_pattern:string
            Indicates which model structure will be used
    :param x_test_in: matrix
             Data with which the CNN is to be tested
    :param y_test_in: list
            labels for the test data
    :param model:
            An existing CNN
    :param calc_test_set:boolean
            If True, perform a prediction on the test date
    :return:
    """
    x_train, x_test, x_val, y_train, y_test, y_val, classes = prepare_data_for_cnn(x_train, y=y_train,
                                                                                   calc_test_set=calc_test_set)
    if len(x_test_in) > 0:
        x_test = np.array(x_test_in)[:, :, :, np.newaxis]
    print("Training", len(x_train),
          "\nValidation", len(x_val),
          "\nTest", len(x_test),
          "\nBatch size", batch,
          "\nEpochs", epochs)

    cp_callback = [
        keras.callbacks.EarlyStopping(
            monitor='val_acc', patience=5,
            mode='max', verbose=0),
        keras.callbacks.ModelCheckpoint(
            save_path + "/" + file_name + "_cnn_" + cnn_pattern + "_adapt.h5", verbose=0,
            monitor='val_acc', save_best_only=True, mode='max')]

    start = time()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=cp_callback,
                        verbose=2)
    print("--- %s seconds ---" % (time() - start))
    if len(x_test_in) > 0:
        y_predict = model.predict_classes(x_test, batch)
        Helper_functions.history_visualization(history, "./", file_name, True)
        Helper_functions.result_visualization(y_test_in, y_predict, True, Constant.label_display_without_rest, file_name,
                                              "./")
    print("finish")
    return model


def predict_for_model(x_test, y_test, model, batch=32):
    """
    Perform a prediction for given data on the given model.
    :param x_test: matrix
             Data with which the CNN is to be tested
    :param y_test: list
            labels for the test data
    :param model:
            An existing CNN
    :param batch:int
            The batch size
    :return:
    """
    x_test = np.array(x_test)[:, :, :, np.newaxis]
    y_predict = model.predict_classes(x_test, batch)
    evaluation = model.evaluate(x_test, to_categorical(y_test))
    print("EVAL", evaluation)
    accuracy_score = Helper_functions.result_visualization(y_test, y_predict, show_figures=True, save_path="./")
    return evaluation, accuracy_score


def create_rehman_model(shape_x, shape_y, output):
    """
    Creates a CNN based on the structure of Rehman's scientific work.
    :param shape_x:int
            Input shape of y for the CNN
    :param shape_y:int
            Input shape of y for the CNN
    :param output:int
            Output of the CNN (normally the number of classes)
    :return:
    """
    model = Sequential()
    # layer 1
    model.add(Conv2D(32, kernel_size=3,
                     activation='relu',
                     input_shape=(shape_x, shape_y, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(Dense(512, activation='elu'))
    model.add(Dense(output, activation='softmax'))
    return model


def create_cnn_1_model(shape_x, shape_y, output):
    """
        Creates an CNN based on a simple structure
    :param shape_x:int
            Input shape of y for the CNN
    :param shape_y:int
            Input shape of y for the CNN
    :param output:int
            Output of the CNN (normally the number of classes)
    """
    model = Sequential()

    # layer 1
    model.add(Conv2D(32, kernel_size=3,
                     activation='relu',
                     input_shape=(shape_x, shape_y, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(32, kernel_size=3,
                     activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # Layer 3
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dense(output, activation='softmax'))
    return model


def create_kaggle_model(shape_x, shape_y, output):
    """
    Creates a CNN based on the structure of entry on kaggle.com.
    :param shape_x:int
            Input shape of y for the CNN
    :param shape_y:int
            Input shape of y for the CNN
    :param output:int
            Output of the CNN (normally the number of classes)
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
    model.add(keras.layers.Dense(output, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model

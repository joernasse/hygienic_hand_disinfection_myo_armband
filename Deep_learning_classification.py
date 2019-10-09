from collections import Counter
from time import time

import numpy as np
import keras
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
import tensorflow as tf
import Constant
import Helper_functions


def calculate_cnn(x, y, save_path="./", batch=32, epochs=10, config="", early_stopping=20, cnn_pattern=Constant.CNN_1,
                  validation_size=Constant.validation_set_size, test_size=Constant.test_set_size, perform_test=False,
                  visualization=False):
    """

    :param x:
    :param y:
    :param save_path:
    :param batch:
    :param epochs:
    :param config:
    :param early_stopping:
    :param cnn_pattern:
    :param test_size:
    :param validation_size:
    :param perform_test:
    :param visualization:
    :return:
    """

    x_train, x_test, x_val, y_train, y_test, y_val_one_hot, classes = prepare_data_for_cnn(x, y,
                                                                                           validation_size=validation_size,
                                                                                           test_size=test_size,
                                                                                           calc_test_set=perform_test)
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


def dnn(save_path, users_data, batch, epochs):
    """

    :param save_path:
    :param users_data:
    :param batch:
    :param epochs:
    :return:
    """
    test_user = users_data[0].copy()
    train_users = users_data.copy()
    train_users.pop(0)

    x_train, y_train = Helper_functions.flat_users_data(train_users)
    x_test, y_test = Helper_functions.flat_users_data([test_user])

    classes = 13

    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=TEST_SIZE)
    model = tf.keras.Sequential()
    model.add(layers.Dense(2048, activation='elu'))
    model.add(layers.Dense(1024, activation='elu'))
    model.add(layers.Dense(512, activation='elu'))
    model.add(Dropout(0.5))
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


def prepare_data_for_cnn(x, y, validation_size=Constant.validation_set_size,
                         test_size=Constant.test_set_size, calc_test_set=False):
    """

    :param x:
    :param y:
    :param adapt_model:
    :param validation_size:
    :param calc_test_set:
    :param test_size:
    :return:
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


# def cnn_1(x, y, save_path, batch, epochs, config=""):
#     """
#
#     :param x:
#     :param y:
#     :param save_path:
#     :param batch:
#     :param epochs:
#     :param config:
#     :return:
#     """
#     print("batch_size", batch, "epochs", epochs)
#     x_train, x_test, y_train, y_test_one_hot, classes, y_test, x_val, y_val_one_hot = pre_process_cnn(x, y,
#                                                                                                       validation_size=0.1)
#
#     # model = Sequential()
#     #
#     # # layer 1
#     # model.add(Conv2D(32, kernel_size=3,
#     #                  activation='relu',
#     #                  input_shape=(x_train.shape[1],
#     #                               x_train.shape[2], 1)))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     #
#     # # Layer 2
#     # model.add(Conv2D(32, kernel_size=3,
#     #                  activation='relu'))
#     # # model.add(MaxPooling2D(pool_size=(2, 2)))
#     # #
#     # # Layer 3
#     # model.add(Flatten())
#     # model.add(Dense(512, activation='elu'))
#     # model.add(Dense(classes, activation='softmax'))
#
#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer='adam',
#                   metrics=['accuracy'])
#
#     cp_callback = [
#         keras.callbacks.EarlyStopping(
#             monitor='val_acc',
#             patience=20,
#             mode='max',
#             verbose=1),
#         keras.callbacks.ModelCheckpoint(
#             save_path + "/" + config + "_cnn_1.h5",
#             verbose=1,
#             monitor='val_acc',
#             save_best_only=True,
#             mode='max')]
#
#     history = model.fit(x_train, y_train,
#                         batch_size=batch,
#                         epochs=epochs,
#                         validation_data=(x_val, y_val_one_hot),
#                         callbacks=cp_callback,
#                         verbose=1, shuffle=True)
#
#     y_predict = model.predict_classes(x_test, batch)
#
#     model.summary()
#
#     acc = Helper_functions.result_visualization(history, y_test, y_predict, save_path=save_path, config=config)
#     print("Train and predict CNN1 done")
#     del model
#     keras.backend.clear_session()
#     gc.collect()
#     del x_train, y_train, x_val, x_test
#     return "CNN_1", acc


def adapt_model_for_user(x_train, y_train, save_path, batch, epochs, file_name, x_test_in, y_test_in, model,
                         calc_test_set=False, cnn_pattern=Constant.CNN_KAGGLE):
    """

    :param x_train:
    :param y_train:
    :param save_path:
    :param batch:
    :param epochs:
    :param file_name:
    :param x_test_in:
    :param y_test_in:
    :param model:
    :param calc_test_set:
    :param cnn_pattern:
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


def predict_for_load_model(x_test, y_test, model, batch_size=32):
    """

    :param x_test:
    :param y_test:
    :param model:
    :param batch_size:
    :return:
    """
    x_test = np.array(x_test)[:, :, :, np.newaxis]
    y_predict = model.predict_classes(x_test, batch_size)
    evaluation = model.evaluate(x_test, to_categorical(y_test))
    print("EVAL", evaluation)
    accuracy_score = Helper_functions.result_visualization(y_test, y_predict, show_figures=True, save_path="./")
    return evaluation, accuracy_score


def create_rehman_model(shape_x, shape_y, output):
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
    :param shape_x:
    :param shape_y:
    :param output:
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

#!/usr/bin/env python
"""
This script contains all functions about the classic classification algorithms.
Includes user dependent, user independent, calculations
Includes grid Search for given classifier and parameter set
"""

import csv
import matplotlib.pyplot as plt
import numpy
from sklearn import clone
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn
from sklearn.metrics import accuracy_score
import Constant
import Helper_functions
from Helper_functions import flat_users_data
from Save_Load import save_classifier

__author__ = "Joern Asse"
__copyright__ = ""
__credits__ = ["Joern Asse"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Joern Asse"
__email__ = "joernasse@yahoo.de"
__status__ = "Production"


def train_user_independent(training_data, test_data, config, classifiers_name, classifiers, save_path,
                           save_model=False, visualization=False):
    """
    Training a user independent classic classifier by a given set of training_data. Also performs a prediction on given test set
    :param training_data: list of dict
            dict:{'data':list,'label':list}
    :param test_data: list of dict
            dict:{'data':list,'label':list}
    :param config:string
            Configuration to identifier the CNN
    :param classifiers_name: string
            Name of the classifier
    :param classifiers:
            The given classifier which should be trained
    :param save_path:string
            Path to the file where the classifier is stored
    :param save_model:boolean
            If True, the model will be saved
    :param visualization: boolean
            If True, the results are visualized
    :return:
    """
    print("User independent - Start")
    best, best_clf = 0, None
    for i in range(len(classifiers)):
        print("Start for", classifiers_name[i])
        x_train, y_train = flat_users_data(training_data)
        x_test, y_test = flat_users_data(test_data)

        print("Training number", len(x_train), "\nTest number", len(x_test))

        classifiers[i].fit(x_train, y_train)
        y_predict = classifiers[i].predict(x_test)
        accuracy = accuracy_score(y_test, y_predict)
        print(classifiers_name[i], accuracy)

        f = open(save_path + "/Overview_user_independent_" + config + ".csv", 'a', newline='')
        with f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([classifiers_name[i], str(accuracy), config])
        f.close()

        if accuracy > best:
            best_clf = classifiers[i]
        best = accuracy
        name = classifiers_name[i]

        if visualization:
            Helper_functions.result_visualization(y_true=y_test,
                                                  y_predict=y_predict,
                                                  show_figures=visualization,
                                                  labels=Constant.labels_without_rest,
                                                  config=config,
                                                  save_path=save_path)
            plt.show()
        if save_model:
            save = save_path + "/" + name + config + '.joblib'
            print(save)
            save_classifier(best_clf, save)
    print("User independent - Done")
    return True


def grid_search(classifier, x_train, y_train, x_test, y_test):
    """
    Training a user dependent classic classifier and perform a grid search to tune the hyperparameter.
    After grid search perform another training and prediction
    :param classifier:
            Already trained classifier (LDA, QDA, SVM, KNN, Bayes, Random_Forest)
    :param x_train: list
            List of training data
    :param y_train:list
            List of labels for training data
    :param x_test: list
            List of test data
    :param y_test: list
            List of label for test data
    :return: classifier,float,float,,list
            classifier: the result from the grid search,
            acc_before_gs: The accuracy of the prediction before grid search,
            acc_after_gs: The accuracy of the prediction after grid search,
            y_predict: The predicted classes (labels) after grid search
            Returns the classifier, the accuracy, the true label and predicted label
    """
    print("Grid search - Start")

    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test, y_test)
    acc_before_gs = accuracy_score(y_test, y_predict)

    classifier = GridSearchCV(estimator=classifier, param_grid=Constant.rf_parameter, n_jobs=-1,
                              verbose=1, cv=10)
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test, y_test)
    acc_after_gs = accuracy_score(y_test, y_predict)
    return classifier, acc_before_gs, acc_after_gs, y_predict


def train_user_dependent(user_data, config, user_name, classifiers, classifiers_name,
                         save_path, save_model=False, visualization=False):
    """
    Training a user dependent classic classifier with a given training set
    :param user_data:list of dict
            dict:{'data':list,'label':list}
    :param config:string
            The configuration to identify the classification
    :param user_name: string
            Name of the user for which the classifier will be trained
    :param classifiers:
            Classifier
    :param classifiers_name: string
            Name of the classifier
    :param save_path:string
            Path to the file where the results and the classifier will be saved
    :param save_model:boolean
            If True, the model will be saved
    :param visualization:boolean
            If True, the results are visualized
    :return:
    """
    print("User dependent - Start ")
    best, best_clf, name = 0, None, "no_name"
    x_train, x_test, y_train, y_test = train_test_split(user_data['data'], user_data['label'],
                                                        test_size=Constant.test_set_size, random_state=42, shuffle=True)

    for i in range(len(classifiers)):
        clf = clone(classifiers[i])
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)

        accuracy = accuracy_score(y_test, y_predict)
        mae = sklearn.metrics.mean_absolute_error(y_test, y_predict)
        print([user_name, classifiers_name[i], accuracy, mae, config])

        if accuracy > best:
            best_clf = clf
            best = accuracy
            name = classifiers_name[i]

        f = open(save_path + "/Overview_all_users.csv", 'a', newline='')
        with f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([user_name, classifiers_name[i], accuracy, mae, config])
        f.close()

        if visualization:
            Helper_functions.result_visualization(y_test, y_predict, True, Constant.label_display_with_rest, config,
                                                  save_path)

    if save_model:
        save_classifier(best_clf, save_path + name + user_name + "-" + config + ".joblib")
    print("User dependent - Done")


def predict_for_unknown_user(model, unknown_user_data):
    """
    Predict results by a given trained model for unknown user (user independent category)
    :param model: Classifier
            Given (trained) classifier for prediction
    :param unknown_user_data:
    :return: No returns
    """
    x, y = flat_users_data(unknown_user_data)
    y_predict = model.predict(x)
    Helper_functions.result_visualization(y, y_predict)
    plt.show()

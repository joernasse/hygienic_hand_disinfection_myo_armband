#!/usr/bin/env python
"""
This script contains all functions about the classic classification algorithms.
Includes user dependent, user independent, calculations
Includes grid Search for given classifier and parameter set
"""

import csv
import pickle
import matplotlib.pyplot as plt
import numpy
from sklearn import clone
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn
from sklearn.metrics import accuracy_score
import Constant
import Helper_functions
from Helper_functions import flat_users_data

__author__ = "Joern Asse"
__copyright__ = ""
__credits__ = ["Joern Asse"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Joern Asse"
__email__ = "joernasse@yahoo.de"
__status__ = "Production"


def train_user_independent_cross_validation(data, classifier, classifier_name, save_path, config):
    """
    This function will cross validate a set of user data.
    Given is a set of Data ordered by users.
    For each iteration another user is used as test data record. The training data is formed from the other users.
    :param data: list of dict
            dict:{'data':list,'label':list}
    :param classifier: Classifier by Scikit
            Describe the classifierfor training
    :param classifier_name: string
            Describe the Name of the classifier
    :param save_path: string
            Describe the path for saving, without the filename
    :param config: string
            Describe the used configuration
    :return: No returns
    """
    print("Cross validation - Start")
    offset = 1
    scores = []
    for n in range(len(data)):
        clf_copy = clone(classifier)
        x_val, y_val = flat_users_data([data[n]])
        reduced_data = data.copy()
        reduced_data.pop(n)
        x_train_cv, y_train_cv = flat_users_data(reduced_data)

        print("Train length", len(x_train_cv),
              "\nValidation length", len(x_val))

        clf_copy.fit(x_train_cv, y_train_cv)
        accuracy = accuracy_score(y_val, clf_copy.predict(x_val))
        scores.append(accuracy)
        print("User" + str(n + offset), accuracy, classifier_name)
        del clf_copy

        f = open(save_path + "/Overview_CV_" + config + ".csv", 'a', newline='')
        with f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["User" + str(n + offset), accuracy, classifier_name])
        f.close()

    f = open(save_path + "/Overview_CV_" + config + ".csv", 'a', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Mean", str(numpy.mean(scores)), "Std", str(numpy.std(scores)), classifier_name])
    f.close()
    print("Cross validation - Done")


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


def train_user_dependent_grid_search(classifier, training_data, test_data):
    """
    Training a user dependent classic classifier and perform a grid search to tune the hyperparameter.
    :param classifier:
            Classifier (LDA, QDA, SVM, KNN, Bayes, Random_Forest)
    :param training_data: list of dict
            dict:{'data':list,'label':list}
    :param test_data: list of dict
            dict:{'data':list,'label':list}
    :return: classifier,float, list,list
            Returns the classifier, the accuracy, the true label and predicted label
    """
    print("User dependent grid search - Start")
    x_train, y_train = flat_users_data(training_data)
    x_test, y_test = flat_users_data(test_data)

    print("Train length", len(x_train),
          "\nValidation length", len(x_test))

    classifier = GridSearchCV(estimator=classifier, param_grid=Constant.rf_parameter, n_jobs=-1, verbose=1, cv=10)

    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    return classifier, accuracy, y_test, y_predict


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


def save_classifier(classifier, path):
    """
    Save the given classifier at the given path
    :param classifier: Classifier
            Classifier to save
    :param path: sting
            Path at which the classifier should be saved
    :return: No returns
    """
    with open(path, 'wb') as file:
        pickle.dump(classifier, file)


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

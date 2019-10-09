import csv
import pickle
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics, clone
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
import sklearn
from sklearn.metrics import accuracy_score
import Constant
import Helper_functions
from Helper_functions import flat_users_data


def train_user_independent_cross_validation(data, classifier, classifier_name, norm, save_path, config):
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
    :param norm: boolean
            If True, data will be normed
            If False data will not be normed
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

        if norm:
            x_train_cv = norm_data(x_train_cv)
            x_val = norm_data(x_val)

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
                           save_model=False, norm=False, visualization=False):
    """

    :param training_data:
    :param test_data:
    :param config:
    :param classifiers_name:
    :param classifiers:
    :param save_path:
    :param save_model:
    :param norm:
    :param visualization:
    :return:
    """
    print("User independent - Start")
    best, best_clf = 0, None
    for i in range(len(classifiers)):
        print("Start for", classifiers_name[i])
        x_train, y_train = flat_users_data(training_data)
        x_test, y_test = flat_users_data(test_data)

        # Norm
        if norm:
            x_train = norm_data(x_train)
            x_test = norm_data(x_test)
            config = config + "_norm"

        print("Training number", len(x_train),
              "\nTest number", len(x_test))

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


def train_user_dependent_grid_search(classifier, training_data, test_data, norm):
    print("User dependent grid search - Start")
    x_train, y_train = flat_users_data(training_data)
    x_test, y_test = flat_users_data(test_data)

    # Norm
    if norm:
        x_train = norm_data(x_train)

    print("Train length", len(x_train),
          "\nValidation length", len(x_test))

    classifier = GridSearchCV(estimator=classifier, param_grid=Constant.rf_parameter, n_jobs=-1, verbose=1, cv=10)

    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    return classifier, accuracy, y_test, y_predict


def train_user_dependent(user_data, config, user_name, classifiers, classifiers_name, save_path, save_model=False,
                         visualization=False, norm=False):
    """

    :param user_data:
    :param config:
    :param user_name:
    :param classifiers:
    :param classifiers_name:
    :param save_path:
    :param save_model:
    :param visualization:
    :return:
    """
    print("User dependent - Start ")
    best, best_clf, name = 0, None, "no_name"
    x_train, x_test, y_train, y_test = train_test_split(user_data['data'], user_data['label'],
                                                        test_size=Constant.test_set_size, random_state=42, shuffle=True)
    if norm:
        config += "_norm"

    for i in range(len(classifiers)):
        if norm:
            x_train = norm_data(x_train)
            x_test = norm_data(x_test)

        clf=clone(classifiers[i])
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


def predict_for_unknown_user(model, unknown_user_data, norm=False):
    """
    Predict results by a given trained model for unknown user (user independent category)
    :param model: Classifier
            Given (trained) classifier for prediction
    :param unknown_user_data:
    :param norm: list of dict
            dict:{'data':list,'label':list}
    :return: No returns
    """
    x, y = flat_users_data(unknown_user_data)
    if norm:
        x = norm_data(x)
    y_predict = model.predict(x)
    Helper_functions.result_visualization(y, y_predict)
    plt.show()


def norm_data(data):
    """
    Norm the input data by the sciKit-Learn StdandardScaler function
    :param data: array
            Data which should be normed
    :return: array
            Normed data
    """
    print("Normed by sklearn.preprocessing.StandardScaler")
    sc = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    sc.fit(data)
    return sc.transform(data)

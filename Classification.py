import csv
import multiprocessing
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
def train_user_independent(training_data, config, clf_name="", cv=False,
                           clf=None, save_model=False, norm=False):
    """

    :param training_data:
    :param test_data:
    :param config:
    :param clf_name:
    :param cv:
    :param clf:
    :param save_model:
    :param norm:
    :return:
    """
    if norm:
        config = "norm_" + config
    x_train, y_train = flat_users_data(training_data)
    # x_test, y_test = flat_users_data(test_data)
    best = 0
    if clf is None:
        print("Load default set of Classifier")
        # clf = [Constant.random_forest, Constant.gauss, Constant.knn, Constant.lda, Constant.qda, Constant.svc]
        # classifier_name = ["randomForest", "bayers", "KNN", "LDA", "QDA", "SVM"]
        clf = [Constant.gauss, Constant.knn, Constant.lda, Constant.qda, Constant.random_forest]
        classifier_name = ["Bayers", "KNN", "LDA", "QDA", "Random_Forest"]
    else:
        clf = [clf]
        classifier_name = [clf_name]
    for i in range(len(clf)):
        print("Start for", classifier_name[i])

        # Cross validation
        if cv:
            print("Cross validation is activ")
            offset = 1
            scores = []
            for n in range(len(training_data)):
                clf_copy = clone(clf[i])
                x_val, y_val = flat_users_data([training_data[n]])
                reduced_data = training_data.copy()
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
                print("User" + str(n + offset), accuracy, classifier_name[i])
                del clf_copy

                f = open("G:Masterarbeit/user_independent_detail/Overview_CV_" + config + ".csv", 'a', newline='')
                with f:
                    writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["User" + str(n + offset), accuracy, classifier_name[i]])
                f.close()

            f = open("./Overview_CV_" + config + ".csv", 'a', newline='')
            with f:
                writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Mean", str(numpy.mean(scores)), "Std", str(numpy.std(scores))])
            f.close()
        else:

            # save = 'G:/Masterarbeit/classic_clf/CV_' + clf_name + config + '.joblib'
            # scores = cross_val_score(clf, x_train, y_train, cv=10, n_jobs=-1)  # 10 fold-CV
            # print(scores)
            # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

            # Norm
            if norm:
                x_train = norm_data(x_train)
                x_test = norm_data(x_test)

            print("Train length", len(x_train),
                  "\nValidation length", len(x_test))
            clf[i].fit(x_train, y_train)
            y_predict = clf[i].predict(x_test)
            accuracy = accuracy_score(y_test, y_predict)
            print(classifier_name[i], accuracy)
            f = open("./Overview_user_independent_" + config + ".csv", 'a', newline='')
            with f:
                writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([classifier_name[i], str(accuracy), config])
            f.close()
            if accuracy > best:
                best_clf = clf[i]
                best = accuracy
                name = classifier_name[i]

            Helper_functions.visualization(0, y_test, y_predict, show_figures=False,
                                           labels=Constant.labels_without_rest, config=config)
            if save_model:
                save = 'G:/Masterarbeit/user_independent_detail/' + name + config + '.joblib'
                print(save)
                save_classifier(best_clf, save)
            plt.show()
    return


# Save Best classifier for each configuration per user
def train_user_dependent(user_data, config, user_name, cv=False, save_model=False):
    """

    :param user_data:
    :param config:
    :param user_name:
    :param cv:
    :param save_model:
    :return:
    """
    print("Start user dependent training")
    classifier = [Constant.random_forest, Constant.gauss, Constant.knn, Constant.lda, Constant.qda, Constant.svc]
    classifier_name = ["randomForest", "bayers", "KNN", "LDA", "QDA", "SVM"]
    best, best_clf = 0, None

    x_train, x_test, y_train, y_test = train_test_split(user_data['data'], user_data['label'], test_size=0.2,
                                                        random_state=42, shuffle=True)
    for i in range(len(classifier)):
        if cv:
            scores = cross_val_score(classifier[i], x_train, y_train, cv=3, n_jobs=-1, verbose=1)

        classifier[i].fit(x_train, y_train)
        y_prediction = classifier[i].predict(x_test)
        accuracy = accuracy_score(y_test, y_prediction)
        mae = sklearn.metrics.mean_absolute_error(y_test, y_prediction)
        report = sklearn.metrics.classification_report(y_test, y_prediction)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_prediction)
        # print(report)
        # print(confusion_matrix)
        if accuracy > best:
            best_clf = classifier[i]
            best = accuracy
            name = classifier_name[i]

        f = open("G:/Masterarbeit/user_dependent_detail/Overview_all_users.csv", 'a', newline='')
        with f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([user_name, classifier_name[i], accuracy, mae, config])
        f.close()
        print([user_name, classifier_name[i], accuracy, mae, config])

    if save_model:
        save_classifier(best_clf,
                        "I:" + name + user_name + "-" + config + ".joblib")


def save_classifier(clf, path):
    """

    :param clf:
    :param path:
    :return:
    """
    with open(path, 'wb') as file:
        pickle.dump(clf, file)
    return


def predict_for_unknown_user(model, unknown_data, norm=False):
    x, y = flat_users_data(unknown_data)
    if norm:
        x = norm_data(x)

    sc = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    print("Normiert, sklearn.preprocessing.StandardScaler")
    sc.fit(x)
    x = sc.transform(x)
    y_predict = model.predict(x)
    Helper_functions.visualization(0, y, y_predict)
    plt.show()


def norm_data(data):
    print("Normed by sklearn.preprocessing.StandardScaler")
    sc = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    sc.fit(data)
    return sc.transform(data)

import multiprocessing
import os
import pickle

import numpy
from sklearn import metrics, clone
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GAUSS
from sklearn.metrics import accuracy_score
# from sklearn.multiclass import OneVsRestClassifier
# from hmmlearn import hmm

from Constant import USERS, USERS_cross

counter = 0
TEST_SIZE = 0.2
rfc_list, results = [], []
min_forest = 32
times = 4
max_forest = 128
process_number = multiprocessing.cpu_count()
border = int(times / process_number)


def flat_data_user_cross_val(users_data):
    x, y = [], []
    for user in users_data:
        for n in range(len(user['data'])):
            x.append(user['data'][n])
            y.append(user['label'][n])
    return x, y


def eval_best_cross(cross_val_results, config):
    best = 0
    for res in cross_val_results:
        if res["mean"] > best:
            best = res["mean"]
            classifier_name = res["classifier_name"]
            cross_val_res = res["predict"]
            best_clf = res["classifier"]
    best_variante = str(best) + classifier_name + config
    print(best, classifier_name, config)
    with open("E:/Masterarbeit/user_cross_val.txt", 'a') as file:
        file.write(best_variante + "\n")
        path = "E:/Masterarbeit/user_cross_val/" + classifier_name + config + ".joblib"
    save_classifier(best_clf, path)


def train_user_independent(users_data, config, name):
    sc_tr = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    sc_test = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

    classifier = RandomForestClassifier(n_jobs=-1, criterion='gini', n_estimators=1000, min_samples_split=2,
                                        bootstrap=True, max_depth=16, max_features=3)
    # classifier = GAUSS()
    # classifier = SVC(C=2,gamma='scale', kernel='rbf',shrinking=True)
    # classifier = OneVsRestClassifier(svc,n_jobs=-1)
    # classifier = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)

    rf_p = {'criterion': ['gini'],
            'max_depth': [16, 64],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [3, 4],
            "max_features": [3, 10],
            'bootstrap': [True, False]}
    # lda_best = LDA(solver='lsqr',shrinkage='auto')
    # params = LDA().get_params()
    # lda_p = {'solver': ['lsqr', 'eigen'],
    #           'n_components': [1, 2, 5, 10, 20],
    #           'shrinkage': ['auto', 0.1, 0.5, 0.9]}
    # classifier = LDA(n_components=1)
    # classifier = QDA()

    accuracy = []
    print("not_norm-" + name + "-" + config)
    print(classifier.get_params())

    # grid_search = GridSearchCV(classifier, rf_p,  cv=5, iid=False)
    for n in range(len(USERS_cross)):
        test_user = users_data[n].copy()
        train_users = users_data.copy()
        train_users.pop(n)

        x_train, y_train = flat_data_user_cross_val(train_users)
        x_test, y_test = flat_data_user_cross_val([test_user])
        # Norm
        # sc_tr.fit(x_train)
        # sc_test.fit(x_test)
        # X = sc_tr.transform(x_train)
        # X_test = sc_test.transform(x_test)
        #
        X = x_train
        X_test = x_test

        clf = clone(classifier)
        clf.fit(X, y_train)

        # grid search
        # grid_search.fit(X, y_train)
        # result=grid_search.cv_results_
        # clf=grid_search.best_estimator_

        # sorted(clf.cv_results_.keys())
        acc = accuracy_score(y_test, clf.predict(X_test))
        accuracy.append(acc)
        print("Accuracy for", USERS_cross[n], acc)
    mean_acc = numpy.mean(accuracy)
    print("Mean Accuracy", mean_acc)

    path = os.getcwd()
    save_classifier(classifier, path + "\\user_independent" + name + str(mean_acc) + config + ".joblib")

    # classifier_cross_val["predict"].append(accuracy_score(y_test, clf.predict(x_test)))
    # classifier_cross_val["classifier"].append(clone(clf))

    # classifier_cross_val["mean"] = numpy.mean(classifier_cross_val["predict"])
    # print(classifier_cross_val["classifier_name"], "mean", classifier_cross_val["mean"], "cross val",
    #       classifier_cross_val["predict"], config)

    # cross_val_results.append(classifier_cross_val)
    # eval_best_cross(cross_val_results, config)


def train_user_dependent(users_data, config):
    # sc = sklearn.preprocessing.StandardScaler()
    randomForest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    svm = SVC(gamma='scale', kernel='rbf')
    # svm = OneVsRestClassifier(svc, n_jobs=-1)
    knn = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
    lda = LDA(n_components=1)
    qda = QDA()
    naive_bayers = GAUSS()
    classifier = [randomForest, naive_bayers, knn, lda, qda, svm]
    classifier_name = ["randomForest", "bayers", "KNN", "LDA", "QDA", "SVM"]
    config_results = []

    for i in range(len(classifier)):
        best, best_clf = 0, None
        classifier_result_val = {"classifier_name": classifier_name[i], "classifier": None, "predict": [], "mean": 0}
        for user in users_data:
            x_train, x_test, y_train, y_test = train_test_split(user['data'], user['label'], test_size=TEST_SIZE,
                                                                random_state=42)

            new_clf = clone(classifier[i])
            new_clf.fit(x_train, y_train)
            accuracy = accuracy_score(y_test, new_clf.predict(x_test))
            if accuracy > best:
                best_clf = clone(new_clf)
                best = accuracy

            classifier_result_val["predict"].append(accuracy)
        classifier_result_val["mean"] = numpy.mean(classifier_result_val["predict"])
        classifier_result_val["classifier"] = best_clf
        config_results.append(classifier_result_val)
        print(classifier_result_val["classifier_name"], ";",
              "mean", ";", classifier_result_val["mean"], ";",
              "predictions", classifier_result_val["predict"], config)


def eval_best_classifier(classifiers, x_test, y_test, config, save_path):
    best = 0
    for clf in classifiers:
        y_predict = clf[0].predict(x_test)
        score = accuracy_score(y_test, y_predict)
        if score >= best:
            best = score
            best_clf = clf
    print(score, best_clf[1], config)
    save_classifier(best_clf, save_path + "/" + best_clf[1] + "-" + config + ".joblib")

    # except :
    #     print(sys.exc_info()[0],config)


def save_classifier(clf, path):
    with open(path, 'wb') as file:
        pickle.dump(clf, file)

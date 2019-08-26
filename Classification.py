import multiprocessing
import os
import pickle
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics, clone
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GAUSS
from sklearn.metrics import accuracy_score
from Constant import USERS, USERS_cross, TEST_SIZE
import Helper_functions
from sklearn.neighbors import KNeighborsClassifier

counter = 0
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


def train_user_independent(users_data, config, mixed_user_data=False, clf_name="", cv=False, clf=None, user_name=""):
    sc_tr = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    sc_test = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

    accuracy = []
    # print("not_norm-" + clf_name + "-" + config)
    # print(random_forest.get_params())

    if mixed_user_data:
        # clf = lda
        save = 'G:/Masterarbeit/user_dependent_detail/' + user_name + clf_name + config + '.joblib'
        x, y = flat_data_user_cross_val(users_data)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=TEST_SIZE)

        # Norm
        print("Normiert, sklearn.preprocessing.StandardScaler")
        sc_tr.fit(x_train)
        sc_test.fit(x_test)
        x_train = sc_tr.transform(x_train)
        x_test = sc_test.transform(x_test)
        # if cv:
        #     save = 'G:/Masterarbeit/classic_clf/CV_' + clf_name + config + '.joblib'
        #     scores = cross_val_score(clf, x_train, y_train, cv=10, n_jobs=-1)  # 10 fold-CV
        #     print(scores)
        #     print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # else:
        print("Train data", len(x_train))
        print("Test data", len(x_test))
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x_test)
        print(clf)
        print(save)
        save_classifier(clf, save)
        Helper_functions.visualization(0, y_test, y_predict, skip_confusion=True)
        plt.show()
        return

    # grid_search = GridSearchCV(random_forest, rf_p,  cv=5, iid=False)
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

        clf = clone(clf)
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
    save_classifier(clf, path + "\\user_independent" + name + str(mean_acc) + config + ".joblib")

    # classifier_cross_val["predict"].append(accuracy_score(y_test, clf.predict(x_test)))
    # classifier_cross_val["random_forest"].append(clone(clf))

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


def save_classifier(clf, path):
    with open(path, 'wb') as file:
        pickle.dump(clf, file)
    return


def predict_for_unknown(model, data):
    x, y = flat_data_user_cross_val(data)
    sc = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    print("Normiert, sklearn.preprocessing.StandardScaler")
    sc.fit(x)
    x = sc.transform(x)
    y_predict = model.predict(x)
    Helper_functions.visualization(0, y, y_predict)
    plt.show()

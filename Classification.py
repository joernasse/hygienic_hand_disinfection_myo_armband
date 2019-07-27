import datetime
import multiprocessing
import pickle
import sys
import timeit
from multiprocessing import Process, Queue
import time
from tkinter import filedialog

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GAUSS
from sklearn.metrics import accuracy_score

from Data_transformation import transform_data_collection
from Helper_functions import countdown, cls
from Collect_data import collect_raw_data
from Save_Load import load_feature_csv, load_classifier

counter = 0
TEST_SIZE = 0.2
rfc_list, results = [], []
min_forest = 32
times = 4
max_forest = 128
process_number = multiprocessing.cpu_count()
border = int(times / process_number)


# class Classifier:
#     def __init__(self):
#         self.load_path = ""
#         self.que = Queue()
#         self.x_train = []
#         self.x_test = []
#         self.y_train = []
#         self.y_test = []
#         self.results = []
#         self.rfc_list = []
#         self.grid_search_res = []
#
#         for i in np.linspace(min_forest, max_forest, times, dtype=int):
#             self.rfc_list.append(RandomForestClassifier(n_estimators=i, criterion="gini", bootstrap=True))
#
#         self.grid_parameter = {"max_features": [10, 30, None],
#                                "min_samples_split": [5, 10, 20, 30, 40]}
#         # "bootstrap": [True, False],
#         # "criterion": ["gini", "entropy"]}
#         # self.grid_parameter = {"max_depth": [10, None],
#         #                        "max_features": [10, 20, 30, 40, 50],
#         #                        "min_samples_split": [2, 3, 10],
#         #                        "bootstrap": [True, False],
#         #                        "criterion": ["gini", "entropy"]}
#
#     def train_classifier(self, x, y):
#         randomForest = RandomForestClassifier()
#         svm = SVC()
#         lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
#         qda = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
#         knn = sklearn.neighbors.KNeighborsClassifier()
#         naiveBayers = sklearn.naive_bayes.GaussianNB()
#
#         print("Start -- Train Classifier ")
#         self.x_train, self.x_test, \
#         self.y_train, self.y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
#         processes = []
#         manager = multiprocessing.Manager()
#         return_dict = manager.dict()
#
#         print("Start -- Grid search")
#         start = timeit.default_timer()
#         for j in range(process_number):
#             s = j * border
#             t = (j + 1) * border
#             process = Process(name="GridSearch", target=self.grid_search, args=(self.rfc_list[s:t], return_dict))
#             processes.append(process)
#             process.start()
#
#         for p in processes:
#             p.join()
#
#         print("debug")
#         res = return_dict.values()
#
#         stop = timeit.default_timer()
#         print('Time: ', datetime.timedelta(seconds=(stop - start)))
#
#         print(return_dict.values())
#         print("debug")
#
#         #  Pool part
#         # for c in rfc_list:
#         #     results = [pool.apply_async(self.grid_search, args=(c,))]
#         # pool.close()
#         # pool.join()
#
#         # process = Process(name="GridSearch", target=self.grid_search, args=(Classifier,))
#         # processes.append(process)
#         # process.start()
#
#         # for
#
#         # for process in processes:
#         #     process.start()
#         #
#         # for process in processes:
#         #     process.join()
#
#         # while 1:
#         #     if any(p.is_alive() for p in processes):
#         #         if que.qsize() == times:
#         #             print("All processes are done")
#         #             for p in processes:
#         #                 p.terminate()
#         #                 print("terminate",p)
#         #                 p.join()
#         #                 print("join", p)
#         #             break
#
#         # res = [que.get() for p in processes]
#         # try:
#         #     res = que.get_nowait()
#         # except:
#         #     print("nothing happen")
#
#         # output = [p.get() for p in results]
#
#         best_score = 0
#         best_estimator = None
#         for x in res:
#             if x[1] > best_score:
#                 best_score = x[1]
#                 best_estimator = x[0]
#
#         filename = "Classifier" + str(time.time()) + ".joblib"
#         # save
#         with open(filename, 'wb') as file:
#             pickle.dump(best_estimator, file)
#
#         # y_i = rfc.predict(x_test)
#         # y_i2 = rfc.predict(x_test)
#         # y_i3 = rfc_25.predict(x_test)
#
#         # print('SkLearn : ', metrics.accuracy_score(y_test, y_i2))
#         # print('SkLearn : ', metrics.accuracy_score(y_test, y_i3))
#         #
#
#     # def GridSearch_(classifiers, params, x, y, output):
#     #     best_results = []
#     #     for clf in classifiers:
#     #         grid_search = GridSearchCV(clf, param_grid=params, cv=5, iid=False)
#     #         grid_search.fit(x, y)
#     #         report(results=grid_search.cv_results_, best_results=best_results, n=clf.n_estimators)
#     #     output.put(best_results)
#
#     def grid_search(self, classifier, return_dict):
#
#         for classifier in classifier:
#             print("Start -- Grid search for tree size ", classifier.n_estimators)
#             grid_search = GridSearchCV(classifier, param_grid=self.grid_parameter, cv=5, iid=False)
#             result = grid_search.fit(self.x_train, self.y_train)
#             return_dict[classifier.n_estimators] = [result.best_estimator_, result.best_score_]
#             print("Done - Tree size ", classifier.n_estimators, " \n| Score", result.best_score_, " \n| Estimator",
#                   result.best_estimator_)
#
#
# # def report(results, best_results, n):
# #     candidates = np.flatnonzero(results['rank_test_score'] == 1)
# #     for candidate in candidates:
# #         print("Mean validation score: {0:.3f} (std: {1:.3f}) (n: {2:3d})".format(
# #             results['mean_test_score'][candidate],
# #             results['std_test_score'][candidate],
# #             n))
# #         print("Parameters: {0}".format(results['params'][candidate]))
# #         best_results.append([n, results['params'][candidate]])
#
#
# def predict(label_information, prediction_time=1):
#     global hub
#     global listener
#     global status
#
#     classifier = load_classifier()
#     status = 0
#
#     with hub.run_in_background(listener.on_event):
#         print("\nGesture prediction will begin")
#         input("Press Enter to continue...")
#         countdown(2)
#         print("Start")
#         cls()
#         while 1:
#             windowed_data, raw_data = collect_raw_data(prediction_time)
#             transformed_data_collection = transform_data_collection(windowed_data)
#
#             cls()
#             prediction = classifier.predict(transformed_data_collection)
#
#             diversity = np.bincount(prediction)
#             marjory = np.argmax(diversity)
#             n = len(prediction)
#
#             print("prediction diversity ", diversity)
#             print("prediction marjory ", marjory)
#
#             if marjory >= n / 2:
#                 print("\n", label_information[marjory])
#
#
# def train_simple_classifier(x, y):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
#
#     rfc = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=2, n_jobs=2)
#     rfc.fit(x_train, y_train)
#     predict_val = rfc.predict_proba(x_test)
#
#     tmp = rfc.predict_proba(x_test)[0:10]
#     y_pred = rfc.predict(x_test)
#     errors = abs(y_pred - y_test)
#
#     print('SkLearn : ', metrics.accuracy_score(y_test, y_pred))
#
#
# def main():
#     global g_load_path
#     g_load_path = filedialog.askdirectory(title="Select feature folder in User directory")
#     emg, imu, label = load_feature_csv(g_load_path)
#
#     # train_simple_classifier(emg, label)
#     classifier = Classifier()
#     classifier.train_classifier(x=emg, y=label)
#
#     # emg try
#     # train_classifier(emg, label)


def train_classifier_1(x, y, path, config):
    # try:
    config = config.split(".csv")[0]
    sc = sklearn.preprocessing.StandardScaler()
    randomForest = RandomForestClassifier(n_estimators=100)
    svm = SVC(gamma='scale')
    knn = sklearn.neighbors.KNeighborsClassifier()
    lda = LDA(n_components=1)
    qda = QDA()
    naiveBayers = GAUSS()

    # print("Start -- Train Classifier ")
    x_train, x_test, \
    y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

    # debug
    if "EMGIMU" in config:
        print("")
    classifier_list = []
    classifier_list.append([randomForest.fit(x_train, y_train), "randomForest"])
    classifier_list.append([svm.fit(x_train, y_train), "SVM"])
    classifier_list.append([lda.fit(x_train, y_train), "LDA"])
    classifier_list.append([qda.fit(x_train, y_train), "QDA"])
    classifier_list.append([knn.fit(x_train, y_train), "KNN"])
    classifier_list.append([naiveBayers.fit(x_train, y_train), "naiveBayers"])

    best = 0
    for clf in classifier_list:
        y_predict = clf[0].predict(x_test)
        score = accuracy_score(y_test, y_predict)
        if score >= best:
            best = score
            print(score, clf[1], config)
            save_classifier(clf, path + "/" + clf[1]+"-" + config + ".joblib")


    # except :
    #     print(sys.exc_info()[0],config)


def save_classifier(clf, path):
    with open(path, 'wb') as file:
        pickle.dump(clf, file)

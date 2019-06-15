import pickle
import multiprocessing as mplib
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from Data_transformation import transform_data_collection
from Helper_functions import countdown, cls
from Myo_communication import collect_raw_data
from Save_Load import load_csv, load_classifier

counter = 0
TEST_SIZE = 0.2
rfc_list, results = [], []

min_forest = 2
times = 2
max_forest = 4

process_number = 1
border = int(times / process_number)
n = np.linspace(min_forest, max_forest, times, dtype=int)
for i in n:
    rfc_list.append(RandomForestClassifier(n_estimators=i))

grid_parameter = {"max_depth": [10, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}


def train_classifier():
    # clf = AdaBoostClassifier(n_estimators=7, learning_rate=1)  # , random_state=np.random.randint(0,9))
    # rfc = RandomForestClassifier(n_estimators=20)
    # rfc_25 = RandomForestClassifier(n_estimators=25)
    # svm = SVC(gamma='auto')
    global results
    results = []
    processes = []
    que = mplib.Queue()
    x, y, name_extension = load_csv()
    print("Start -- training models ")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

    for j in range(process_number):
        s = j * border
        t = (j + 1) * border
        classifier = rfc_list[s:t]
        process = mplib.Process(name="GridSearch", target=GridSearch,
                                args=(classifier, grid_parameter, x_train, y_train, que))
        processes.append(process)

    print("Start -- Grid Search")
    start = time.time()
    for process in processes:
        process.start()

    while 1:
        if any(p.is_alive() for p in processes):
            if que.qsize() == times:
                print("All processes are done")
                for p in processes:
                    # p.terminate()
                    # print("terminate",p)
                    p.join()
                    print("join", p)
                break
    end = time.time()
    print("duration: ", end - start)

    # print("merge results")
    # print(results)
    res = [que.get() for p in processes]

    input("debug")
    best_score = 0
    for x in res:
        if x.best_score > best_score:
            best_score = x.best_score
            best_estimator = x

    filename = "classifier" + name_extension + ".joblib"
    # save
    with open(filename, 'wb') as file:
        pickle.dump(best_estimator, file)

    # y_i = rfc.predict(x_test)
    # y_i2 = rfc.predict(x_test)
    # y_i3 = rfc_25.predict(x_test)
    # print('SkLearn : ', metrics.accuracy_score(y_test, y_i))
    # print('SkLearn : ', metrics.accuracy_score(y_test, y_i2))
    # print('SkLearn : ', metrics.accuracy_score(y_test, y_i3))
    #


# def GridSearch_(classifiers, params, x, y, output):
#     best_results = []
#     for clf in classifiers:
#         grid_search = GridSearchCV(clf, param_grid=params, cv=5, iid=False)
#         grid_search.fit(x, y)
#         report(results=grid_search.cv_results_, best_results=best_results, n=clf.n_estimators)
#     output.put(best_results)


def GridSearch(classifiers, params, x, y, que):
    global results
    best_score = 0
    for clf in classifiers:
        # print("Start -- tree size ", clf.n_estimators)
        grid_search = GridSearchCV(clf, param_grid=params, cv=5, iid=False)
        tmp = grid_search.fit(x, y)
        # results.append(tmp.best_estimator_)
        que.put(tmp.best_estimator_)
        print("Done -- ", clf.n_estimators, " | Score", tmp.best_score_, " | Estimator", tmp.best_estimator_)
    return
    # return True


# def report(results, best_results, n):
#     candidates = np.flatnonzero(results['rank_test_score'] == 1)
#     for candidate in candidates:
#         print("Mean validation score: {0:.3f} (std: {1:.3f}) (n: {2:3d})".format(
#             results['mean_test_score'][candidate],
#             results['std_test_score'][candidate],
#             n))
#         print("Parameters: {0}".format(results['params'][candidate]))
#         best_results.append([n, results['params'][candidate]])


def predict(label_information, prediction_time=1):
    global hub
    global listener
    global status

    classifier = load_classifier()
    status = 0

    with hub.run_in_background(listener.on_event):
        print("\nGesture prediction will begin")
        input("Press Enter to continue...")
        countdown(2)
        print("Start")
        cls()
        while 1:
            windowed_data, raw_data = collect_raw_data(prediction_time)
            transformed_data_collection = transform_data_collection(windowed_data)

            cls()
            prediction = classifier.predict(transformed_data_collection)

            diversity = np.bincount(prediction)
            marjory = np.argmax(diversity)
            n = len(prediction)

            print("prediction diversity ", diversity)
            print("prediction marjory ", marjory)

            if marjory >= n / 2:
                print("\n", label_information[marjory])

from __future__ import print_function

# ordentliche prediction Ergebnisse
# PREDICT_TIME: float = 0.5
import datetime
import os
import timeit
from multiprocessing import Process
from tkinter import filedialog

import numpy

from Classification import train_classifier_1
from Constant import *
from Deep_learning import dnn_default
from Helper_functions import list_list_to_matrix

from Process_data import process_raw_data, window_data
from Save_Load import *


def main():
    deep_learning = True
    classifier = True
    featureExtraction = False

    process_number = 1
    # process_number = multiprocessing.cpu_count()
    processes = []
    split = numpy.array_split(numpy.asarray(USERS), process_number)

    start = timeit.default_timer()

    for j in range(process_number):
        s = j * process_number
        t = (j + 1) * process_number
        if classifier:
            process = Process(name="Classify", target=train_classifier, args=(split[j], deep_learning))
        elif featureExtraction:
            process = Process(name="Feature extraction", target=feature_extraction, args=(split[j],))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()

    stop = timeit.default_timer()
    print('Time: ', datetime.timedelta(seconds=(stop - start)))
    print("Finish")
    return True


def train_classifier(users, deep_learning):
    for user in users:
        if not os.path.isdir(collections_default_path + user + "/classifier"):
            os.mkdir(collections_default_path + user + "/classifier")
        files = os.listdir(collections_default_path + user + "/features")
        for file in files:
            if deep_learning:
                train_dnn(user, 50, 0)
            # x, y = load_feature_csv(open(collections_default_path + user + "/features/" + file))

            # dnn_default(x, y)
            # train_classifier_1(x, y, collections_default_path + user + "/classifier", file)
    return True


def train_dnn(user, window, overlap):
    emg_window_collection, imu_window_collection = [], []
    # emg and imu, sep and conti
    directories = [os.listdir(collections_default_path + user + SEPARATE_PATH), \
                   os.listdir(collections_default_path + user + CONTINUES_PATH)]
    path_add = [SEPARATE_PATH, CONTINUES_PATH]
    for i in range(len(directories)):
        matrix_collection, matrix_collection_imu, labels = [], [], []
        for steps in directories[i]:
            s_path = collections_default_path + user + path_add[i] + "/" + steps
            emg_raw, imu_raw = load_raw_csv(emg_path=s_path + "/emg.csv", imu_path=s_path + "/imu.csv")
            emg_window, imu_window = window_data(emg_raw, imu_raw, window=window,
                                                 degree_of_overlap=overlap, skip_timestamp=1)

            for j in range(len(emg_window)):
                merged = [emg for emg in emg_window[j][:-1]+[imu for imu in imu_window[j][:-1]]]
                matrix_collection.extend(list_list_to_matrix(merged))
            labels.extend([int(emg_raw['label'][0])] * len(emg_window))
        dnn_default(matrix_collection,labels)


def feature_extraction(users):
    for user in users:
        for dataset in level_1:
            for sensor in level_2:
                for window in level_3:
                    for overlap in level_4:
                        for feature in level_5:
                            process_raw_data(user, dataset=dataset, overlap=overlap,
                                             sensor=sensor, window=window, feature=feature)
                            # if classifier:
                            #     Classifier.train_classifier()
    return True


if __name__ == '__main__':
    main()

from __future__ import print_function

# ordentliche prediction Ergebnisse
# PREDICT_TIME: float = 0.5
import datetime
import timeit
from multiprocessing import Process

import numpy

from Classification import train_classifier_user_cross_validation, train_classifier_1
# from Deep_learning import dnn_default

from Process_data import process_raw_data, window_data, window_data_matrix
from Save_Load import *


def main():
    train_dep = train_user_dependent()
    if train_dep:
        return True
    # calculation_config_statistics()
    # deep_learning = False
    # classifier = False
    # featureExtraction = False
    # user_cross_val = True
    # n_job = True
    # processes = []
    #
    # if n_job:
    #     prepare_data_for_cross_validation_over_user(user_cross_val_feature_selection)
    #
    # if user_cross_val:
    #     process_number = 1
    #     # process_number = 3
    #     split = numpy.array_split(numpy.asarray(user_cross_val_feature_selection), process_number)
    # else:
    #     process_number = 1
    #     # process_number = multiprocessing.cpu_count()
    #     split = numpy.array_split(numpy.asarray(USERS), process_number)
    # start = timeit.default_timer()
    #
    # for j in range(process_number):
    #     s = j * process_number
    #     t = (j + "-" + 1) * process_number
    #     if classifier:
    #         process = Process(name="Classify", target=train_classifier, args=(split[j], deep_learning))
    #     elif featureExtraction:
    #         process = Process(name="Feature extraction", target=feature_extraction, args=(split[j],))
    #     elif user_cross_val:
    #         process = Process(name="user_cross_val", target=prepare_data_for_cross_validation_over_user,
    #                           args=(split[j],))
    #
    #     processes.append(process)
    #     process.start()
    #
    # for p in processes:
    #     p.join()
    #
    # stop = timeit.default_timer()
    # print('Time: ', datetime.timedelta(seconds=(stop - start)))
    # print("Finish")
    # return True


def prepare_data_for_cross_validation_over_user(level_1_local):
    for config in level_1_local:
        users_data = load_feature_csv_all_user(config)
        train_classifier_user_cross_validation(users_data, config)
    return True


# def train_classifier(users, deep_learning):
#     for user in users:
#         if not os.path.isdir(collections_default_path + "-" + user + "-" + "/classifier"):
#             os.mkdir(collections_default_path + "-" + user + "-" + "/classifier")
#         files = os.listdir(collections_default_path + "-" + user + "-" + "/features")
#         for file in files:
#             if deep_learning:
#                 train_dnn(user, 50, 0)
#             else:
#                 x, y = load_feature_csv(open(collections_default_path + "-" + user + "-" + "/features/" + "-" + file))
#                 train_classifier(x, y, collections_default_path + "-" + user + "-" + "/classifier", file)
#
#             # dnn_default(x, y)
#     return True


# def train_dnn(user, window, overlap):
#     emg_window_collection, imu_window_collection = [], []
#     # emg and imu, sep and conti
#     directories = [os.listdir(collections_default_path +"-"+ user +"-"+ SEPARATE_PATH), \
#                    os.listdir(collections_default_path +"-"+ user +"-"+ CONTINUES_PATH)]
#     path_add = [SEPARATE_PATH, CONTINUES_PATH]
#     windows, labels = [], []
#     for i in range(len(directories)):
#         matrix_collection, matrix_collection_imu = [], []
#         for steps in directories[i]:
#             s_path = collections_default_path +"-"+ user +"-"+ path_add[i] +"-"+ "/" +"-"+ steps
#             emg_raw, imu_raw = load_raw_csv(s_path +"-"+ "/emg.csv", s_path +"-"+ "/imu.csv")
#             emg_window, imu_window = window_data(emg_raw, imu_raw, window=window, degree_of_overlap=overlap,
#                                                  skip_timestamp=1)
#
#             merged = []
#             for j in range(len(emg_window)):
#                 windows.append([emg for emg in emg_window[j][:-1]])
#             labels.extend([int(emg_raw['label'][0])] * len(emg_window))
#
#     matrix = []
#     for item in windows:
#         matrix.append(numpy.asarray(item))
#     # matrix_merged = numpy.asmatrix(tmp)
#     dnn_default(matrix, labels)


def train_user_dependent():
    for dataset in level_1:
        for sensor in level_2:
            for window in level_3:
                for overlap in level_4:
                    for feature in level_5:
                        config = dataset + "-" + sensor + "-" + str(window) + "-" + str(overlap) + "-" + feature
                        users_data = load_feature_csv_all_user(config)
                        if not users_data:
                            continue
                        train_classifier_1(users_data, config)
    return True


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


def calculation_config_statistics():
    config_mean = []
    overview = load_overview()
    for dataset in level_1:
        for sensor in level_2:
            for window in level_3:
                for overlap in level_4:
                    for feature in level_5:
                        config = dataset + "-" + sensor + "-" + str(window) + "-" + str(overlap) + "-" + feature
                        config_items = []

                        for item in overview:
                            if item[3] == dataset and \
                                    item[4] == sensor and \
                                    item[5] == str(window) and \
                                    item[6] == str(overlap) and \
                                    item[7] == feature:
                                config_items.append(item)
                        if config_items:
                            tmp = [config]
                            tmp.extend([numpy.mean([float(x[0]) for x in config_items])])
                            config_mean.append(tmp)

    f = open("E:/Masterarbeit/config_mean.csv", 'w', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in config_mean:
            writer.writerow(item)

    # index = ["accuracy", "classifier", "mode", "sensor", "window", "overlap", "feature_set"]
    # classifier_results = {
    #     'accuracy': [],
    #     'classifier': [],
    #     'mode': [],
    #     'sensor': [],
    #     'window': [],
    #     'overlap': [],
    #     'feature_set': []
    # }
    #
    # for item in user_cross_val_feature_selection:
    #     users_data = load_feature_csv_all_user(item)
    #     # reader=csv file
    #     # for column in reader:
    #     #     for n in range(len(column)):
    #     #         classifier_results[n]=column[n]
    #     # return classifier_results


if __name__ == '__main__':
    main()

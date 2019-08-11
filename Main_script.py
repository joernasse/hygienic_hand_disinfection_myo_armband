from __future__ import print_function
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sklearn
from numpy import newaxis
from sklearn.model_selection import train_test_split

import numpy as np
import Classification
import Deep_learning
from Constant import *

from Process_data import process_raw_data, window_data, window_data_matrix, window_only_one_sensor
from Save_Load import *
import tensorflow as tf

imu_dict = {
        'Step0': [],
        'Step1': [],
        'Step1_1': [],
        'Step1_2': [],
        'Step2': [],
        'Step2_1': [],
        'Step3': [],
        'Step4': [],
        'Step5': [],
        'Step5_1': [],
        'Step6': [],
        'Step6_1': [],
        'Rest': []
    }
emg_dict = {'Step0': [],
                'Step1': [],
                'Step1_1': [],
                'Step1_2': [],
                'Step2': [],
                'Step2_1': [],
                'Step3': [],
                'Step4': [],
                'Step5': [],
                'Step5_1': [],
                'Step6': [],
                'Step6_1': [],
                'Rest': []}


def main():
    # train_user_dependent()
    # calculation_config_statistics()
    # path = os.getcwd()
    # train_user_independent(best_config_rf)

    train_cnn(100, 0.75)
    ### Used all User Data,separate & continuous, 80% Training, 20% Test !IMU ONLY! ###
    # 12-0.5 T0.2   ->  46,64%      244417/244417 [==============================] - 50s 205us/sample - loss: 1.5746 - acc: 0.4664 - val_loss: 1.4192 - val_acc: 0.5255
    # 12-0.75 T.02  ->  49,87%      488831/488831 [==============================] - 98s 201us/sample - loss: 1.4751 - acc: 0.4987 - val_loss: 1.3235 - val_acc: 0.5478
    # 12-0 T0.2     ->  42,07%      122212/122212 [==============================] - 24s 200us/sample - loss: 1.7015 - acc: 0.4207 - val_loss: 1.5457 - val_acc: 0.4663
    # 25-0.5 T0.2   ->  51,11%      117323/117323 [==============================] - 52s 442us/sample - loss: 1.4507 - acc: 0.5111 - val_loss: 1.2352 - val_acc: 0.5850
    # 25-0.75 T0.2  ->  57,01%      234651/234651 [==============================] - 103s 437us/sample - loss: 1.2821 - acc: 0.5702 - val_loss: 1.0990 - val_acc: 0.6316
    # 25-0 T0.2     ->  47,64%      58660/58660 [==============================] - 26s 446us/sample - loss: 1.5607 - acc: 0.4765 - val_loss: 1.3220 - val_acc: 0.5630

    # 90% Training, 10% Test
    # 25-0.75 T0.1  ->  56,36%      263982/263982 [==============================] - 108s 408us/sample - loss: 1.3001 - acc: 0.5636 - val_loss: 1.1108 - val_acc: 0.6279

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    # 2 Variante
    # train_cnn(25, 0.75)

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


def train_user_independent(config):
    users_data = load_feature_csv_all_user(config)
    Classification.train_user_independent(users_data, config, "RandomForest")
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

def load_raw_data_for_nn():
    global imu_dict
    global emg_dict
    path_add = [SEPARATE_PATH, CONTINUES_PATH]
    for user in USERS_cross:
        path = collections_default_path + user
        directories = [os.listdir(path + SEPARATE_PATH), os.listdir(path + CONTINUES_PATH)]
        for i in range(len(directories)):
            for steps in directories[i]:
                index = steps[2:]
                path_data = path + path_add[i] + "/" + steps
                # imu_dict[index].extend(load_raw_2(path_data + "/imu.csv"))
                emg_dict[index].extend(load_raw_2(path_data + "/emg.csv"))
                # print(index,len(emg_dict[index]))
        print(user, "done")
    return imu_dict, emg_dict


def window_raw_data_for_nn(window, overlap, imu_dict, emg_dict):
    labels, imu_data, emg_data = [], [], []
    for key in save_label:
        # implementation for IMU
        # imu, label = window_only_one_sensor(imu_dict[key], window=window, degree_of_overlap=overlap)
        # imu_data.extend(imu)
        # s = numpy.vstack((imu_data, imu))

        # implementation for EMG
        emg, label = window_only_one_sensor(emg_dict[key], window=window, degree_of_overlap=overlap)
        emg_data.extend(emg)
        s = numpy.vstack((emg_data, emg))

        labels.extend(numpy.asarray(label))
    return imu_data, emg_data, labels


def train_cnn(window, overlap):
    imu, emg = load_raw_data_for_nn()
    imu_windows, emg_windows, labels = window_raw_data_for_nn(window, overlap, imu, emg)
    labels = [int(i) for i in labels]

    # cnn(numpy.asarray(imu_windows), numpy.asarray(labels))
    print(len(emg_windows), len(labels))
    Deep_learning.cnn(numpy.asarray(emg_windows), numpy.asarray(labels))


def train_user_dependent():
    for data_set in level_1:
        for sensor in level_2:
            for window in level_3:
                for overlap in level_4:
                    for feature in level_5:
                        config = data_set + "-" + sensor + "-" + str(window) + "-" + str(overlap) + "-" + feature
                        users_data = load_feature_csv_all_user(config)
                        if not users_data:
                            continue
                        Classification.train_user_dependent(users_data, config)
    return True


def feature_extraction(users):
    for user in users:
        for data_set in level_1:
            for sensor in level_2:
                for window in level_3:
                    for overlap in level_4:
                        for feature in level_5:
                            process_raw_data(user, dataset=data_set, overlap=overlap,
                                             sensor=sensor, window=window, feature=feature)
    return True


def calculation_config_statistics():
    config_mean = []
    overview = load_overview()
    for data_set in level_1:
        for sensor in level_2:
            for window in level_3:
                for overlap in level_4:
                    for feature in level_5:
                        config = data_set + "-" + sensor + "-" + str(window) + "-" + str(overlap) + "-" + feature
                        config_items = []

                        for item in overview:
                            if config == item[17]:
                                config_items.append(item)
                        if config_items:
                            tmp = [config]
                            tmp.extend([numpy.mean([float(x[0]) for x in config_items])])
                            config_mean.append(tmp)

    f = open("G:/Masterarbeit/config_mean.csv", 'w', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in config_mean:
            writer.writerow(item)


if __name__ == '__main__':
    main()

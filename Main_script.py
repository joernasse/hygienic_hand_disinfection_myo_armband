from __future__ import print_function

import tensorflow
from keras.utils import plot_model
import matplotlib.pyplot as plt
import Classic_classification
import Constant
import Deep_learning_classification
import Helper_functions
import Process_data
import Save_Load
import numpy as np
import csv
import multiprocessing as mp
import os
import sklearn
from tensorflow.python.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
path_add = [Constant.SEPARATE_PATH, Constant.CONTINUES_PATH]


def preprocess_raw_data(window, overlap, user_list, data_set, preprocess, sensor=Constant.IMU,
                        ignore_rest_gesture=True, norm_by_rest=False):
    """

    :param window:
    :param overlap:
    :param user_list:
    :param data_set:
    :param preprocess:
    :param sensor:
    :param ignore_rest_gesture:
    :param norm_by_rest:
    :return:
    """
    print("Preprocessing raw data - Start")
    print("Window", window, "Overlap", overlap)

    raw_data = Process_data.collect_data_for_single_sensor(user_list, sensor, data_set)
    if norm_by_rest:
        print("Start normalization")
        raw_data = Helper_functions.normalize_by_rest_gesture(data=raw_data, sensor=sensor)

    if ignore_rest_gesture:
        raw_data['Rest'] = []

    w_data, labels = Process_data.window_raw_data_for_nn(window, overlap, raw_data=raw_data,
                                                         ignore_rest_gesture=ignore_rest_gesture)

    if sensor == Constant.IMU:
        if preprocess == Constant.z_norm:
            ignore_return, w_data = Process_data.z_norm(emg=w_data, imu=w_data)
    elif sensor == Constant.EMG:
        if preprocess == Constant.filter_:
            w_data = Process_data.filter_emg_data(emg=w_data, filter_type="")
        elif preprocess == Constant.z_norm:
            w_data, ignore_return = Process_data.z_norm(emg=w_data, imu=w_data)

    # TODO check if working in windowing function above

    print("Window number", len(w_data),
          "\nLabel number", len(labels),
          "\nPreprocessing raw data - Done")
    return w_data, labels


# TODO überdenken
def pre_process_raw_data_adapt_model(window, overlap, user, sensor, ignore_rest_gesture=True,
                                     normalize_by_rest=False, collection_path=Constant.collections_path_default,
                                     data_set=Constant.SEPARATE + Constant.CONTINUES):
    """

    :param window:
    :param overlap:
    :param user:
    :param sensor:
    :param ignore_rest_gesture:
    :param normalize_by_rest:
    :param collection_path:
    :param data_set:
    :return:
    """
    print("Preprocessing raw data for adapt model - Start")
    print("Window", window, "Overlap", overlap)
    training_data, test_data = load_training_and_test_raw_data_for_adapt_model(user, sensor,
                                                                               collection_path=collection_path,
                                                                               data_set=data_set)

    if normalize_by_rest:
        training_data = Helper_functions.normalize_by_rest_gesture(data=training_data, sensor=sensor)
        test_data = Helper_functions.normalize_by_rest_gesture(data=test_data, sensor=sensor)

    window_data_train, labels_train = Process_data.window_raw_data_for_nn(window=window, overlap=overlap,
                                                                          raw_data=training_data,
                                                                          ignore_rest_gesture=ignore_rest_gesture)
    window_data_test, labels_test = Process_data.window_raw_data_for_nn(window, overlap, raw_data=test_data,
                                                                        ignore_rest_gesture=ignore_rest_gesture)

    print("Training number", len(window_data_train),
          "\nTest length", len(window_data_test),
          "\nPreprocessing raw data for adapt model - Done")
    return window_data_train, labels_train, window_data_test, labels_test


def calculate_total_raw_data(path="G:/Masterarbeit/Collections/"):
    raw_emg_length_by_user = []
    raw_imu_length_by_user = []
    optimal_emg_length_total = []
    optimal_imu_length_total = []
    file = open("./raw_data_summary.csv", 'a', newline='')
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["user", "emg_samples", "optimal_emg", "quote_emg", "imu_samples", "optimal_imu", "quote_imu"])

    for user in Constant.USERS:
        load_path = path + user
        path_add = [Constant.SEPARATE_PATH, Constant.CONTINUES_PATH]
        number_emg_sample, number_imu_sample, number_record = 0, 0, 0
        directories = [os.listdir(load_path + Constant.SEPARATE_PATH), os.listdir(load_path + Constant.CONTINUES_PATH)]
        for i in range(len(directories)):  # go through all directories
            for steps in directories[i]:  # go through all Steps
                raw_emg, raw_imu = Save_Load.load_raw_data_for_both_sensors(
                    emg_path=load_path + path_add[i] + "/" + steps + "/emg.csv",
                    imu_path=load_path + path_add[i] + "/" + steps + "/imu.csv")
                number_emg_sample += len(raw_emg['ch0'])
                number_imu_sample += len(raw_imu['x_ori'])
            number_record += len(directories[i])

        raw_emg_length_by_user.append(number_emg_sample)
        raw_imu_length_by_user.append(number_imu_sample)

        optimal_emg_length = number_record * 5 * 400
        optimal_imu_length = number_record * 5 * 100

        optimal_emg_length_total.append(optimal_emg_length)
        optimal_imu_length_total.append(optimal_imu_length)

        quote_emg = number_emg_sample / optimal_emg_length
        quote_imu = number_imu_sample / optimal_imu_length

        writer.writerow(
            [user, str(number_emg_sample), optimal_emg_length, quote_emg, number_imu_sample, optimal_imu_length,
             quote_imu])
    total_emg = np.sum(raw_emg_length_by_user)
    total_imu = np.sum(raw_imu_length_by_user)
    total_optimum_emg = np.sum(optimal_emg_length_total)
    total_optimum_imu = np.sum(optimal_imu_length_total)
    writer.writerow(["User1_to_15", total_emg, total_optimum_emg, total_emg / total_optimum_emg, total_imu,
                     total_optimum_imu, total_imu / total_optimum_imu])
    file.close()


def main():
    calc_missing_config("G:/Masterarbeit/Results/User_dependent_classic/Overview_all_users_update.csv")
    return True
    # calculation_config_statistics("G:/Masterarbeit/")
    #
    # calculate_total_raw_data()
    # return True

    # feature_extraction(["User001"])
    # n_jobs = 8
    # partitioned_configs = np.array_split(Constant.missing_config_calc, n_jobs)
    # processes = [mp.Process(target=train_user_dependent_classic,
    #                         args=([Constant.USERS_SUB, "G:/Masterarbeit/feature_sets_filter/", True,
    #                                partitioned_configs[i]])) for i in range(n_jobs)]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()

    # train_user_dependent_classic(user_list=["User001"],
    #                              feature_set_path="G:/Masterarbeit/feature_sets_filter/",
    #                              ignore_rest_gesture=True,
    #                              predefine_configs=Constant.missing_config_calc,
    #                              model_save_path="./",
    #                              save_model=True,
    #                              visualization=False,
    #                              classifiers=Constant.classifiers,
    #                              classifier_names=Constant.classifier_names,
    #                              norm=True)
    return True
    # --------------------------------------------Train user independent classic - START-------------------------------#
    # train_user_independent_classic("no_pre_pro-separate-EMGIMU-100-0.9-rehman", True,
    #                                "//192.168.2.101/g/Masterarbeit/feature_sets_filter/", Constant.USERS_SUB, "User007",
    #                                "./", [Constant.random_forest], ["Random Forest"], True, True, True)
    # return True
    # --------------------------------------------Train user independent classic - END---------------------------------#

    # --------------------------------------------Adapt CNN for Unknown User START-------------------------------------#
    # model_l_path = "//192.168.2.101/g/Masterarbeit/Results/User_independent_cnn/" \
    #                "User007_Unknown/no_pre_pro-separate-EMG-25-0.9-NA_cnn_model.h5"
    # x_train, y_train, x_test, y_test = \
    #     pre_process_raw_data_adapt_model(window=25, overlap=0.9, user="User007",
    #                                      sensor=Constant.EMG, data_set=Constant.SEPARATE,
    #                                      collection_path="//192.168.2.101/g/Masterarbeit/Collections/")
    #
    # Deep_learning.adapt_model_for_user(x_train=x_train, y_train=y_train, save_path="./", batch=32, epochs=10,
    #                                    x_test_in=x_test, y_test_in=y_test, model=None,
    #                                    config="no_pre_pro-separate-IMU-25-0.9-norm-NA")
    # return True
    # --------------------------------------------Adapt CNN for Unknown User END---------------------------------------#

    # --------------------------------------------Plot CNN-------------------------------------------------------------#
    # tensorflow.keras.utils.plot_model(Deep_learning.create_cnn_1_model(8, 100, 12),
    #                                   to_file='G:/Masterarbeit/CNN_1_structure.svg', show_shapes=True,
    #                                   show_layer_names=True)
    # --------------------------------------------Plot CNN-------------------------------------------------------------#

    # --------------------------------------------Predict user independent CNN - START---------------------------------#
    # load_model_path = "G:/Masterarbeit/Results/User_independent_cnn/User002_Unknown/no_pre_pro-separatecontinues-EMG-100-0.9-NA_cnn_CNN_Kaggle.h5"
    # predict_for_unknown_user_cnn(load_model_path, "User002", "no_pre_pro-separatecontinues-EMG-100-0.9-NA")
    # return True
    # --------------------------------------------Predict user independent CNN - END-----------------------------------#

    # --------------------------------------------Train user dependent CNN - START-------------------------------------#
    # save_path = "G:/Masterarbeit/Results/User_dependent_cnn/"
    # for user in Constant.USERS_SUB:
    #     train_user_dependent_cnn(["no_pre_pro-separate-EMG-100-0.9-NA"], user,
    #                              False, save_path, True, Constant.CNN_KAGGLE, True)
    # return True
    # --------------------------------------------Train user dependent CNN - END---------------------------------------#

    # --------------------------------------------Grid search ---------------------------------------------------------#
    # user_independent_grid_search(Constant.random_forest, "Random Forest", "./",
    #                              "no_pre_pro-separate-EMGIMU-100-0.9-georgi", True, True, Constant.USERS_cross,
    #                              "G:/Masterarbeit/feature_sets_filter/", "User007", True, True)
    # return True
    # --------------------------------------------Grid search ---------------------------------------------------------#

    # --------------------------------------------Train user independent CNN - START-----------------------------------#
    train_user_independent_cnn(Constant.USERS_SUB, ["no_pre_pro-continues-IMU-25-0.9-NA"], "User002", True,
                               "./", False, Constant.CNN_KAGGLE, True, 32, 50, 2)
    return True
    # --------------------------------------------Train user independent CNN - END-----------------------------------#

    #

    # train_user_independent("no_pre_pro-separate-IMU-100-0.9-rehman",trainings_data=Constant.USERS,feature_sets_path="G:/Masterarbeit/feature_sets_filter/")
    # train_user_dependent(Constant.USERS, skip_rest=True)  # feature_extraction_complete()
    # calculation_config_statistics()
    # return True

    # calculation_config_statistics()
    # path = os.getcwd()
    # for config in Constant.top_ten_user_dependent_configs[1:]:
    #     train_user_independent(config=config, ignore_rest_gesture=True,
    #                            trainings_data=Constant.USERS, feature_sets_path="G:/Masterarbeit/feature_sets_filter/")
    # return True
    # load_model_from = "G:/Masterarbeit/deep_learning/CNN_final_results/training_kaggle_imu_0"

    # --------------------------------------------Train CNN 1----------------------------------------------#
    # save_path = save_path + "/CNN_1/"
    # if not os.path.isdir(save_path):  # Collection dir
    #     os.mkdir(save_path)
    # # name, acc = Deep_learning.cnn_1(x, labels, save_path, batch=50, epochs=500, config=config)
    #
    # f = open("G:/Masterarbeit/deep_learning_filter/Overview_CNN.csv", 'a', newline='')
    # with f:
    #     writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow([user, name, str(acc), config])
    # f.close()
    # del x
    # del labels

    # --------------------------------------------Train CNN kaggle-----------------------------------------------------#

    # save_path = "G:/Masterarbeit/deep_learning_filter/" + "User007_Unknown"
    # if not os.path.isdir(save_path):  # Collection dir
    #     os.mkdir(save_path)
    # save_path = save_path + "/CNN_Kaggle"
    # if not os.path.isdir(save_path):  # Collection dir
    #     os.mkdir(save_path)
    # # For all Users, can be changed
    # x_train, y_train = preprocess_raw_data(window=int(config_split[3]), overlap=float(config_split[4]),
    #                                        user_list=USERS_cross, preprocess=config_split[0],
    #                                        data_set=config_split[1],
    #                                        sensor=config_split[2], ignore_rest_gesture=True, norm_by_rest=False)
    #
    # x_test, y_test = preprocess_raw_data(window=int(config_split[3]), overlap=float(config_split[4]),
    #                                      user_list=["User007"], preprocess=config_split[0], data_set=config_split[1],
    #                                      sensor=config_split[2], ignore_rest_gesture=True, norm_by_rest=False)
    #
    # Deep_learning.predict_for_load_model(x_test, y_test, load_model(
    #     "G:/Masterarbeit/deep_learning_filter/User007_Unknown/CNN_Kaggle/no_pre_pro-separate-EMG-100-0.9-NA_cnn_model.h5"),
    #                                      batch_size=32)
    # return True
    #
    # model_name, acc = Deep_learning.calculate_cnn(x_train, y_train, save_path, batch=32, epochs=10, config=config,
    #                                               x_test_in=x_test, y_test_in=y_test, early_stopping=3)
    #
    # f = open("G:/Masterarbeit/deep_learning_filter/Overview_CNN.csv", 'a', newline='')
    # with f:
    #     writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(["User007_unknown", model_name, str(acc), config])
    # f.close()

    # --------------------------------------------Predict from loaded model for unknown user---------------------------#
    # Deep_learning.predict_for_load_model(x, labels, load_model(load_model_from + "/cnn_kaggle_model.h5"),
    #                                      batch_size=200)

    # --------------------------------------------Adapt model for User-------------------------------------------------#
    # skip_s0_rest = False
    # x_train_window, y_train, x_test_window, y_test = pre_process_raw_data_adapt_model(25, 0.75, ["User007"], sensor,
    #                                                                                   skip_s0_rest=skip_s0_rest)
    # if skip_s0_rest:
    #     labels = sub_label
    # else:
    #     labels = save_label
    #
    # # x_test_window= [t.transpose() for t in x_test_window]
    # # x_train_window = [t.transpose() for t in x_train_window]
    #
    # model = load_model(load_model_from + "/cnn_kaggle_model.h5")
    # Deep_learning.adapt_model_for_user(x_train_window, y_train,
    #                                    save_path, 200, 100, "User007",
    #                                    x_test_window, y_test, save_label=labels, model=model)

    # --------------------------------------------Train DNN with feature extraction data-------------------------------#
    # users_data = load_feature_csv_all_user(best_config_rf)
    # Deep_learning.dnn(save_path=save_path, batch=50, epochs=10, users_data=users_data)

    # ----------------------------------Train user independent classic-------------------------------------------------#
    # config = best_emg_rf
    # users_data = load_feature_from_many_users(config, USERS_cross)
    # Classification.train_user_independent(users_data=users_data,
    #                                       config=config,
    #                                       mixed_user_data=True,
    #                                       clf_name="OneVsRest",
    #                                       clf=Constant.one_vs_Rest,
    #                                       cv=False)
    # return True

    # ----------------------------------Train user dependent classic--------------------------------------------------#
    # config_split = best_config_rf
    # print("Config", config_split)
    # # Predict for all Users
    # for user in USERS:
    #     print(user)
    #     users_data = load_feature_for_user(config_split, user)
    #     Classification.train_user_independent(training_data=users_data,
    #                                           config=config_split,
    #                                           mixed_user_data=True,
    #                                           classifiers_name="Random_Forest_User_dependent",
    #                                           classifiers=Constant.random_forest,
    #                                           cv=False,
    #                                           user_name=user)

    # print("All done")
    # return True

    # ----------------------------------Predict classic ML against User007-------------------------------------------#
    # config = "separate-EMG-100-0.75-georgi"
    # classic_clf = "G:/Masterarbeit/classic_clf/"
    # users_data = load_feature_csv_one_user(config, "User007")
    # with open(classic_clf + 'SVM_Only_EMGseparate-EMG-100-0.75-georgi.joblib', 'rb') as pickle_file:
    #     model = pickle.load(pickle_file)
    # Classification.predict_for_unknown(model=model,
    #                                    data=users_data)
    #
    # return True


def train_user_independent_classic(config, ignore_rest_gesture=True, feature_sets_path="",
                                   training_users=Constant.USERS_cross, test_user="User007", save_path="./",
                                   classifier=Constant.classifiers, classifier_names=Constant.classifier_names,
                                   norm=False, save_model=False, visualization=False):
    """

    :param config:
    :param ignore_rest_gesture:
    :param feature_sets_path:
    :param training_users:
    :param test_user:
    :param save_path:
    :param classifier:
    :param classifier_names:
    :param norm:
    :param save_model:
    :param visualization:
    :return:
    """
    print("Current config", config)
    training_data = Save_Load.load_raw_data(config=config, user_list=training_users, path=feature_sets_path)
    test_data = Save_Load.load_raw_data(config=config, user_list=[test_user], path=feature_sets_path)
    print(len(training_data))

    if ignore_rest_gesture:
        test_data[0] = Process_data.remove_rest_gesture_data(user_data=test_data[0])
        for i in range(len(training_data)):
            training_data[i] = Process_data.remove_rest_gesture_data(user_data=training_data[i])

    Classic_classification.train_user_independent(training_data=training_data,
                                                  test_data=test_data,
                                                  classifiers=classifier,
                                                  classifiers_name=classifier_names,
                                                  save_path=save_path,
                                                  config=config,
                                                  norm=norm,
                                                  save_model=save_model,
                                                  visualization=visualization)


def load_training_and_test_raw_data_for_adapt_model(user, sensor, data_set,
                                                    collection_path=Constant.collections_path_default, session='s0'):
    global path_add

    training_dict = {
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
        'Rest': []}
    test_dict = {
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
        'Rest': []}

    directories = []
    path = collection_path + user

    if Constant.SEPARATE in data_set:
        directories.append(os.listdir(path + Constant.SEPARATE_PATH))
        path_add.append(Constant.SEPARATE_PATH)

    if Constant.CONTINUES in data_set:
        directories.append(os.listdir(path + Constant.CONTINUES_PATH))
        path_add.append(Constant.SEPARATE_PATH)

    for i in range(len(directories)):
        for steps in directories[i]:
            index = steps[2:]
            path_data = path + path_add[i] + "/" + steps

            if sensor == Constant.IMU:
                if steps.__contains__(session):
                    training_dict[index].extend(Save_Load.load_raw_for_single_sensor(path_data + "/imu.csv"))
                else:
                    test_dict[index].extend(Save_Load.load_raw_for_single_sensor(path_data + "/imu.csv"))
            else:
                if steps.__contains__(session):
                    training_dict[index].extend(Save_Load.load_raw_for_single_sensor(path_data + "/emg.csv"))
                else:
                    test_dict[index].extend(Save_Load.load_raw_for_single_sensor(path_data + "/emg.csv"))

    print(user, "done")
    return training_dict, test_dict


def train_user_dependent_classic(user_list, feature_set_path, ignore_rest_gesture=True, predefine_configs=None,
                                 model_save_path="./", save_model=False, visualization=False,
                                 classifiers=Constant.classifiers, classifier_names=Constant.classifier_names,
                                 norm=True):
    """
    Go over all config steps and prepare data for each combination of configuration.
    Each level in Config. can be changed in the Constant.py
    Load feature set described by config.
    Ignore Rest gesture if "skip_rest" is True
    :param user_list: list
            List of string which represent the users
    :param feature_set_path: string
            Path to folder with the feature sets
    :param ignore_rest_gesture: boolean
            If True, skip the "Rest" gesture
            If False, don´t skip "Rest" gesture
    :param predefine_configs: string
            Can be used to use only the predefine config, instead of iterate over all all possible configurations.
    :param model_save_path: string
            The path where the classifier/model should be saved
    :return:
    """
    for config in predefine_configs:
        if predefine_configs is not None:
            for user in user_list:
                print(user)
                users_data = Save_Load.load_raw_data(config=config, path=feature_set_path,
                                                     user_list=[user])

                if not users_data:
                    continue

                if ignore_rest_gesture:
                    users_data = Process_data.remove_rest_gesture_data(users_data[0])

                Classic_classification.train_user_dependent(user_data=users_data,
                                                            config=config,
                                                            user_name=user,
                                                            classifiers=classifiers,
                                                            classifiers_name=classifier_names,
                                                            save_path=model_save_path,
                                                            save_model=save_model,
                                                            visualization=visualization,
                                                            norm=norm)

        #     TODO remove after use
    return True

    for pre in Constant.level_0:
        for data_set in Constant.level_1:
            for sensor in Constant.level_2:
                for window in Constant.level_3:
                    for overlap in Constant.level_4:
                        for feature in Constant.level_5:
                            config = pre + "-" + data_set + "-" + sensor + "-" + str(window) + \
                                     "-" + str(overlap) + "-" + feature
                            if predefine_configs:
                                config = predefine_configs

                            users_data = Save_Load.load_raw_data(config=config, path=feature_set_path,
                                                                 user_list=[user])

                            if not users_data:
                                continue

                            if ignore_rest_gesture:
                                users_data = Process_data.remove_rest_gesture_data(users_data[0])

                            Classic_classification.train_user_dependent(user_data=users_data,
                                                                        config=config,
                                                                        user_name=user,
                                                                        classifiers=classifiers,
                                                                        classifiers_name=classifier_names,
                                                                        save_path=model_save_path,
                                                                        save_model=save_model,
                                                                        visualization=visualization,
                                                                        norm=norm)
    return True


def feature_extraction_complete(n_jobs=4):
    """

    :return:
    """
    partitioned_users = np.array_split(Constant.USERS, n_jobs)
    processes = [mp.Process(target=feature_extraction, args=([partitioned_users[i]])) for i in range(n_jobs)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def user_dependent_complete(feature_set_path, save_path="./", n_jobs=4):
    """
    Train the classifier SVM, KNN, LDA, QDA, Random Forest Bayers
    for all Users (User001 to User015)
    for all possible combinations of configurations
    ech level of configuration can be changed in Constant.py
    :param feature_set_path: string
             Path to folder with the feature sets
    :param save_path: string

    :param n_jobs: int
            The processors to use
    :return: No returns
    """
    partitioned_users = np.array_split(Constant.USERS, n_jobs)
    processes = [
        mp.Process(target=train_user_dependent_classic,
                   args=([partitioned_users[i], feature_set_path, True, save_path]))
        for i in range(n_jobs)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def feature_extraction(user_list):
    """
    Extract the features from a given set of users
    :param user_list:  array, shape = [n_users] The array of users for feature extraction
    :return: No returns
    """
    for user in user_list:
        for preprocessing in Constant.level_0:
            for data_set in Constant.level_1:
                for sensor in Constant.level_2:
                    for window in Constant.level_3:
                        for overlap in Constant.level_4:
                            for feature in Constant.level_5:
                                Process_data.process_raw_data(user, data_set=data_set, overlap=overlap,
                                                              sensor=sensor, window=window, feature=feature,
                                                              pre=preprocessing)


# TODO: Löschen?!
def calculation_config_statistics(load_path):
    """

    :param load_path:
    :return:
    """
    config_mean = []
    overview = Save_Load.load_prediction_summary(path=load_path)
    for pre in Constant.level_0:
        for data_set in Constant.level_1:
            for sensor in Constant.level_2:
                for window in Constant.level_3:
                    for overlap in Constant.level_4:
                        for feature in Constant.level_5:
                            config = pre + "-" + data_set + "-" + sensor + "-" + str(window) + "-" + str(
                                overlap) + "-" + feature
                            config_items = []

                            for item in overview:
                                if config == item[4]:
                                    config_items.append(item)
                            if not config_items:
                                continue
                            if len(config_items) == 90:  # Für jeden Nutzer ein Eintrag
                                print(config_items[0][2])
                                config_mean.append([config,
                                                    # [x[1] for x in config_items],
                                                    np.mean([float(x[2]) for x in config_items]),
                                                    [x[2] for x in config_items]])
                            else:
                                print("")

    f = open("G:Masterarbeit/user_dependent_detail/Overview_by_config.csv", 'w', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in config_mean:
            writer.writerow(item)


def calc_missing_config(load_path):
    missing_config=[]
    overview = Save_Load.load_prediction_summary(path=load_path)
    for pre in Constant.level_0:
        for data_set in Constant.level_1:
            for sensor in Constant.level_2:
                for window in Constant.level_3:
                    for overlap in Constant.level_4:
                        for feature in Constant.level_5:
                            config = pre + "-" + data_set + "-" + sensor + "-" + str(window) + "-" + str(
                                overlap) + "-" + feature
                            config_items = []

                            for item in overview:
                                try:
                                    if config == item[4]:
                                        config_items.append(item)
                                except:
                                    continue
                            if not config_items:
                                missing_config.append(config)
    f = open("G:Masterarbeit/Results/User_dependent_classic/missing_config"+str(len(missing_config))+".csv", 'w', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in missing_config:
            writer.writerow([item])



def train_user_independent_cnn(train_user_list, config_list, user, norm=False, save_path="./",
                               perform_test=False, cnn_pattern=Constant.CNN_1, ignore_rest_gesture=True, batch=32,
                               epochs=100, early_stopping=3):
    """

    :param train_user_list:
    :param config_list:
    :param user:
    :param norm:
    :param save_path:
    :param perform_test:
    :param cnn_pattern:
    :param ignore_rest_gesture:
    :param batch:
    :param epochs:
    :param early_stopping:
    :return:
    """
    for config in config_list:
        config_split = config.split('-')
        save_path = save_path + "/" + user
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        preprocess = config_split[0]
        data_set = config_split[1]
        sensor = config_split[2]
        window = int(config_split[3])
        overlap = float(config_split[4])

        x, labels = preprocess_raw_data(window=window, overlap=overlap, user_list=train_user_list,
                                        preprocess=preprocess, data_set=data_set, sensor=sensor,
                                        ignore_rest_gesture=ignore_rest_gesture, norm_by_rest=False)
        if norm:
            for i in range(len(x)):
                sc = sklearn.preprocessing.StandardScaler(copy=True, with_std=True)
                sc.fit(x[i])
                x[i] = sc.transform(x[i])
            config += "_norm"

        model, model_name, acc = Deep_learning_classification.calculate_cnn(x=x, y=labels, save_path=save_path,
                                                                            batch=batch, epochs=epochs, config=config,
                                                                            early_stopping=early_stopping, cnn_pattern=cnn_pattern,
                                                                            perform_test=perform_test)

        f = open(save_path + "/Results" + cnn_pattern + "_UI.csv", 'a', newline='')
        with f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([user, model_name, str(acc), config])
        f.close()


def train_user_dependent_cnn(config_list, user, norm=False, save_path="./", perform_test=False,
                             cnn_pattern=Constant.CNN_1, ignore_rest_gesture=True):
    """

    :param config_list:
    :param user:
    :param norm:
    :param save_path:
    :param perform_test:
    :param cnn_pattern:
    :param ignore_rest_gesture:
    :return:
    """
    overview_path = save_path
    save_path = save_path + "/" + user
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for config in config_list:
        config_split = config.split('-')
        preprocess = config_split[0]
        sensor = config_split[2]
        data_set = config_split[1]
        window = int(config_split[3])
        overlap = float(config_split[4])

        x, labels = preprocess_raw_data(window=window, overlap=overlap, user_list=[user], preprocess=preprocess,
                                        data_set=data_set, sensor=sensor, ignore_rest_gesture=ignore_rest_gesture,
                                        norm_by_rest=False)
        if norm:
            for i in range(len(x)):
                sc = sklearn.preprocessing.StandardScaler(copy=True, with_std=True)
                sc.fit(x[i])
                x[i] = sc.transform(x[i])
                config += "norm"

        model, model_name, acc = Deep_learning_classification.calculate_cnn(x=x, y=labels, save_path=save_path,
                                                                            batch=32, epochs=100, config=config,
                                                                            early_stopping=5, cnn_pattern=cnn_pattern,
                                                                            perform_test=perform_test)

        f = open(overview_path + "/Results" + cnn_pattern + config + "_UD.csv", 'a', newline='')
        with f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([user, model_name, str(acc), config])
        f.close()


def predict_for_unknown_user_cnn(model_path, user, config):
    model = load_model(model_path)

    config_split = config.split('-')
    preprocess = config_split[0]
    data_set = config_split[1]
    sensor = config_split[2]
    window = int(config_split[3])
    overlap = float(config_split[4])
    norm = bool(config_split[5])
    if norm == "norm":
        norm = True
    else:
        norm = False
    x, labels = preprocess_raw_data(window=window, overlap=overlap, user_list=[user], data_set=data_set,
                                    preprocess=preprocess,
                                    sensor=sensor, ignore_rest_gesture=True, norm_by_rest=False)
    if norm:
        print("Norm data")
        for i in range(len(x)):
            sc = sklearn.preprocessing.StandardScaler(copy=True, with_std=True)
            sc.fit(x[i])
            x[i] = sc.transform(x[i])
            config += "norm"
    evaluation, accuracy_score = Deep_learning_classification.predict_for_load_model(x, labels, model, batch_size=32)


def user_independent_grid_search(classifier, name, save_path, config, visualization, save_model, training_user_list,
                                 feature_sets_path, test_user, ignore_rest_gesture, norm):
    training_data = Save_Load.load_raw_data(config=config, user_list=training_user_list, path=feature_sets_path)
    test_data = Save_Load.load_raw_data(config=config, user_list=[test_user], path=feature_sets_path)

    if ignore_rest_gesture:
        test_data[0] = Process_data.remove_rest_gesture_data(user_data=test_data[0])
        for i in range(len(training_data)):
            training_data[i] = Process_data.remove_rest_gesture_data(user_data=training_data[i])

    classifier, accuracy, y_test, y_predict = Classic_classification.train_user_dependent_grid_search(classifier=classifier,
                                                                                                      training_data=training_data,
                                                                                                      test_data=test_data,
                                                                                                      norm=norm)

    f = open(save_path + "/Overview_user_independent_" + config + ".csv", 'a', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow([name, str(accuracy), config])
    f.close()

    if visualization:
        Helper_functions.result_visualization(y_true=y_test, y_predict=y_predict, show_figures=False,
                                              labels=Constant.labels_without_rest, config=config)
        plt.show()
    if save_model:
        save = save_path + "/" + name + config + '.joblib'
        Classic_classification.save_classifier(classifier, save)
    print("User independent - Done")
    print("User dependent grid search - Done")
    return True


if __name__ == '__main__':
    main()

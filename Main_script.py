from __future__ import print_function

import csv
import multiprocessing as mp
import os
import sklearn
from tensorflow.python.keras.models import load_model
import Classification
import Constant
import Deep_learning
import Process_data
from Process_data import process_raw_data, window_data_for_both_sensor, window_for_one_sensor
import Save_Load
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
path_add = [Constant.SEPARATE_PATH, Constant.CONTINUES_PATH]


def normalize_by_rest_gesture(data, sensor, mode='rest_mean'):
    """

    :param data:
    :param sensor:
    :param mode:
    :return:
    """
    print("Normalization by Rest gesture - Start")
    if sensor == Constant.IMU:
        element = 10
    else:
        element = 9
    rest_data = data['Rest']
    channel, mean = [], []
    if mode == 'rest_mean':
        for ch in range(1, element):
            for i in range(len(rest_data)):
                channel.append(rest_data[i][ch])
            mean.append(np.mean(channel))  # Mean of base REST over all channels (1-8)

        try:
            for d in data:
                if d == 'Rest':
                    continue
                for ch in range(1, element):  # Channel/ IMU entries
                    for i in range(len(data[d])):  # Elements
                        data[d][i][ch] = data[d][i][ch] / mean[ch - 1]
        except:
            print("Not expected exception in normalization by rest function")
            raise

    if mode == 'max_value_channel':
        for d in data:
            items = []
            for ch in range(1, element):
                for i in range(len(data[d])):
                    items.append(data[d][i][ch])
                max_val = max(items)
                items = items / max_val
    print("Normalization by Rest gesture - Done")
    return data


def pre_process_raw_data(window, overlap, user_list, data_set, preprocess, sensor=Constant.IMU,
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

    raw_data = load_data_for_single_sensor(user_list, sensor, data_set)
    print(len(raw_data['Step1']))
    if norm_by_rest:
        print("Start normalization")
        raw_data = normalize_by_rest_gesture(data=raw_data, sensor=sensor)

    if ignore_rest_gesture:
        raw_data['Rest'] = []

    w_data, labels = window_raw_data_for_nn(window, overlap, raw_data=raw_data, ignore_rest_gesture=ignore_rest_gesture)

    if sensor == Constant.IMU:
        if preprocess == Constant.z_norm:
            ignore_return, w_data = Process_data.z_norm(emg=w_data, imu=w_data)
    elif sensor == Constant.EMG:
        if preprocess == Constant.filter_:
            w_data = Process_data.filter_emg_data(emg=w_data, filter_type="")
        elif preprocess == Constant.z_norm:
            w_data, ignore_return = Process_data.z_norm(emg=w_data, imu=w_data)

    # TODO check if working in windowing function above
    # labels = [int(i) for i in labels]

    print("Window length", len(w_data),
          "\nLabel length", len(labels),
          "\nPreprocessing raw data - Done")
    return w_data, labels

#TODO überdenken
def pre_process_raw_data_adapt_model(window, overlap, users, sensor, skip_steps_list=None,
                                     ignore_rest_gesture=True, normalize_by_rest=False):
    """

    :param window:
    :param overlap:
    :param users:
    :param sensor:
    :param skip_steps_list:
    :param ignore_rest_gesture:
    :param normalize_by_rest:
    :return:
    """
    print("Preprocessing raw data for adapt model - Start")
    print("Window", window, "Overlap", overlap)
    training_data, test_data = load_raw_data_for_adapt_model(users, sensor)

    if normalize_by_rest:
        training_data = normalize_by_rest_gesture(data=training_data, sensor=sensor)
        test_data = normalize_by_rest_gesture(data=test_data, sensor=sensor)

    if skip_steps_list is not None:
        for step in skip_steps_list:
            training_data[step] = []
            test_data[step] = []

    window_data_train, labels_train = window_raw_data_for_nn(window=window, overlap=overlap, raw_data=training_data,
                                                             ignore_rest_gesture=ignore_rest_gesture)
    window_data_test, labels_test = window_raw_data_for_nn(window, overlap, raw_data=test_data,
                                                           ignore_rest_gesture=ignore_rest_gesture)
    # TODO chekc if is done yet in function windowing above...
    # labels_train = [int(x) for x in labels_train]
    # labels_test = [int(x) for x in labels_test]

    # correct the order and indices
    if skip_steps_list:
        labels_train = [i - 1 for i in labels_train]
        labels_test = [i - 1 for i in labels_test]

    print("Data length", len(window_data_train),
          "\nTraining length", len(window_data_test),
          "\nPreprocessing raw data for adapt model - Done")
    return window_data_train, labels_train, window_data_test, labels_test


def main():
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

    # --------------------------------------------Train CNN User dependent for each user by given config---------------#
    for config in ["no_pre_pro-separate-EMG-100-0.9-NA"]:
        config_split = config.split('-')
        # for user in USERS:
        save_path = "G:/Masterarbeit/deep_learning_filter/User001_unknown"
        if not os.path.isdir(save_path):  # Collection dir
            os.mkdir(save_path)
        x, labels = pre_process_raw_data(window=int(config_split[3]), overlap=float(config_split[4]),
                                         user_list=Constant.USERS_cross, preprocess=config_split[0],
                                         data_set=config_split[1],
                                         sensor=config_split[2],
                                         ignore_rest_gesture=True, norm_by_rest=False)
        for i in range(len(x)):
            sc = sklearn.preprocessing.StandardScaler(copy=True, with_std=True)
            sc.fit(x[i])
            x[i] = sc.transform(x[i])

        model, model_name, acc = Deep_learning.cnn_kaggle(x, labels, save_path, batch=32, epochs=100, config=config,
                                                          early_stopping=5)

        f = open("G:/Masterarbeit/deep_learning_filter/Overview_CNN_Kaggle_UI_normStd.csv", 'a', newline='')
        with f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["User001_unknown", model_name, str(acc), config])
        f.close()
    return True

    # x = [t.transpose() for t in x]

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
    save_path = "G:/Masterarbeit/deep_learning_filter/" + "User007_Unknown"
    if not os.path.isdir(save_path):  # Collection dir
        os.mkdir(save_path)
    save_path = save_path + "/CNN_Kaggle"
    if not os.path.isdir(save_path):  # Collection dir
        os.mkdir(save_path)
    # For all Users, can be changed
    x_train, y_train = pre_process_raw_data(window=int(config_split[3]), overlap=float(config_split[4]),
                                            user_list=USERS_cross, preprocess=config_split[0],
                                            data_set=config_split[1],
                                            sensor=config_split[2], ignore_rest_gesture=True, norm_by_rest=False)

    x_test, y_test = pre_process_raw_data(window=int(config_split[3]), overlap=float(config_split[4]),
                                          user_list=["User007"], preprocess=config_split[0], data_set=config_split[1],
                                          sensor=config_split[2], ignore_rest_gesture=True, norm_by_rest=False)

    Deep_learning.predict_for_load_model(x_test, y_test, load_model(
        "G:/Masterarbeit/deep_learning_filter/User007_Unknown/CNN_Kaggle/no_pre_pro-separate-EMG-100-0.9-NA_cnn_model.h5"),
                                         batch_size=32)
    return True

    model_name, acc = Deep_learning.cnn_kaggle(x_train, y_train, save_path, batch=32, epochs=10, config=config,
                                               x_test_in=x_test, y_test_in=y_test, early_stopping=3)

    f = open("G:/Masterarbeit/deep_learning_filter/Overview_CNN.csv", 'a', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["User007_unknown", model_name, str(acc), config])
    f.close()

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
    config_split = best_config_rf
    print("Config", config_split)
    # Predict for all Users
    for user in USERS:
        print(user)
        users_data = load_feature_for_user(config_split, user)
        Classification.train_user_independent(training_data=users_data,
                                              config=config_split,
                                              mixed_user_data=True,
                                              classifiers_name="Random_Forest_User_dependent",
                                              classifiers=Constant.random_forest,
                                              cv=False,
                                              user_name=user)
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


def train_user_independent(config, ignore_rest_gesture=True, feature_sets_path="",
                           training_users=Constant.USERS_cross, test_user="User007", save_path="./",
                           classifier=Constant.classifiers, classifier_names=Constant.classifier_names):
    """

    :param config:
    :param ignore_rest_gesture:
    :param feature_sets_path:
    :param training_users:
    :param test_user:
    :param save_path:
    :param classifier:
    :param classifier_names:
    :return:
    """
    print("Current config", config)
    training_data = Save_Load.load_features(config=config, user_list=training_users, path=feature_sets_path)
    test_data = Save_Load.load_features(config=config, user_list=[test_user], path=feature_sets_path)

    if ignore_rest_gesture:
        test_data[0] = remove_rest_gesture_data(user_data=test_data[0])
        for i in range(len(training_data)):
            training_data[i] = remove_rest_gesture_data(user_data=training_data[i])

    Classification.train_user_independent(training_data=training_data,
                                          test_data=test_data,
                                          classifiers=classifier,
                                          classifiers_name=classifier_names,
                                          save_path=save_path,
                                          config=config,
                                          norm=True,
                                          save_model=False)


def load_data_for_single_sensor(user_list, sensor, data_set, collection_path=Constant.collections_path_default):
    """

    :param user_list:
    :param sensor:
    :param data_set:
    :param collection_path:
    :return:
    """
    global path_add
    raw_data = {'Step0': [],
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

    for user in user_list:
        directories = []
        path = collection_path + user

        if Constant.SEPARATE in data_set:
            directories.append(os.listdir(path + Constant.SEPARATE_PATH))

        if Constant.CONTINUES in data_set:
            directories.append(os.listdir(path + Constant.CONTINUES_PATH))

        for i in range(len(directories)):
            for steps in directories[i]:
                index = steps[2:]
                raw_data[index].extend(Save_Load.load_raw_for_single_sensor(path + path_add[i] + "/" + steps + "/"
                                                                            + sensor + ".csv"))

        print("Load raw data for", user, " - Done")
    print("Raw data length", len(raw_data['Step0']))
    return raw_data


def load_raw_data_for_adapt_model(users, sensor):
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

    for user in users:
        path = Constant.collections_path_default + user
        directories = [os.listdir(path + Constant.SEPARATE_PATH), os.listdir(path + Constant.CONTINUES_PATH)]
        for i in range(len(directories)):
            for steps in directories[i]:
                index = steps[2:]
                path_data = path + path_add[i] + "/" + steps
                if sensor == Constant.IMU:
                    if steps.__contains__('s0'):
                        training_dict[index].extend(Save_Load.load_raw_for_single_sensor(path_data + "/imu.csv"))
                    else:
                        test_dict[index].extend(Save_Load.load_raw_for_single_sensor(path_data + "/imu.csv"))
                else:
                    if steps.__contains__('s0'):
                        training_dict[index].extend(Save_Load.load_raw_for_single_sensor(path_data + "/emg.csv"))
                    else:
                        test_dict[index].extend(Save_Load.load_raw_for_single_sensor(path_data + "/emg.csv"))

        print(user, "done")
    return training_dict, test_dict


def window_raw_data_for_nn(window, overlap, raw_data, ignore_rest_gesture=True):
    """

    :param window:
    :param overlap:
    :param raw_data:
    :param ignore_rest_gesture:
    :return:
    """
    labels, window_data, emg_data = [], [], []
    if ignore_rest_gesture:
        label = Constant.labels_without_rest
    else:
        label = Constant.label
    for key in label:
        window_tmp, label = window_for_one_sensor(input_data=raw_data[key], window=window, degree_of_overlap=overlap)
        window_data.extend(window_tmp)
        np.vstack((window_data, window_tmp))
        labels.extend(np.asarray(label))
    return window_data, labels


def train_cnn(window, overlap, save_path="./", load_model_path="", sensor=Constant.IMU,cnn_model_name="CNN_1"):
    print("window", window, "overlap", overlap)
    if not load_model_path == "":
        imu, emg = load_data_for_single_sensor(["User007"], sensor)
        imu_windows, emg_windows, labels = window_raw_data_for_nn(window, overlap, imu, emg, sensor)
        labels = [int(i) for i in labels]
        print(len(imu_windows), len(labels))
        if sensor == Constant.IMU:
            Deep_learning.predict_for_load_model(np.asarray(imu_windows), np.asarray(labels),
                                                 load_model(load_model_path + "/cnn_imu_model.h5"))
        else:
            Deep_learning.predict_for_load_model(np.asarray(emg_windows), np.asarray(labels),
                                                 load_model(load_model_path + "/cnn_imu_model.h5"))
    else:
        imu, emg = load_data_for_single_sensor(["User001"], sensor)
        imu_windows, emg_windows, labels = window_raw_data_for_nn(window, overlap, imu, emg, sensor)
        labels = [int(i) for i in labels]
        print(len(emg_windows), len(labels))
        if sensor == Constant.IMU:
            Deep_learning.cnn_1(np.asarray(imu_windows), np.asarray(labels), save_path)
        else:
            Deep_learning.cnn_1(np.asarray(emg_windows), np.asarray(labels), save_path)

        # Deep_learning.cnn_1(np.asarray(emg_windows), np.asarray(labels), save_path)
        # Deep_learning.cnn_rehman(np.asarray(emg_windows), np.asarray(labels), save_path)


def train_user_dependent(users, feature_set_path, ignore_rest_gesture=True, predefine_config=None,
                         model_save_path="./"):
    """
    Go over all config steps and prepare data for each combination of configuration.
    Each level in Config. can be changed in the Constant.py
    Load feature set described by config.
    Ignore Rest gesture if "skip_rest" is True
    :param users: list
            List of string which represent the users
    :param feature_set_path: string
            Path to folder with the feature sets
    :param ignore_rest_gesture: boolean
            If True, skip the "Rest" gesture
            If False, don´t skip "Rest" gesture
    :param predefine_config: string
            Can be used to use only the predefine config, instead of iterate over all all possible configurations.
    :param model_save_path: string
            The path where the classifier/model should be saved
    :return:
    """
    for user in users:
        print(user)
        for pre in Constant.level_0:
            for data_set in Constant.level_1:
                for sensor in Constant.level_2:
                    for window in Constant.level_3:
                        for overlap in Constant.level_4:
                            for feature in Constant.level_5:
                                config = pre + "-" + data_set + "-" + sensor + "-" + str(window) + \
                                         "-" + str(overlap) + "-" + feature
                                if predefine_config:
                                    config = predefine_config

                                users_data = Save_Load.load_features(config=config, path=feature_set_path,
                                                                     user_list=[user])

                                if not users_data:
                                    continue

                                if ignore_rest_gesture:
                                    users_data = remove_rest_gesture_data(users_data)

                                Classification.train_user_dependent(user_data=users_data,
                                                                    config=config,
                                                                    user_name=user,
                                                                    classifiers=Constant.classifiers,
                                                                    classifiers_name=Constant.classifier_names,
                                                                    save_path=model_save_path,
                                                                    save_model=False,
                                                                    visualization=False)
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
        mp.Process(target=train_user_dependent, args=([partitioned_users[i], feature_set_path, True, save_path]))
        for i in range(n_jobs)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def remove_rest_gesture_data(user_data):
    """
    Remove the "Rest" Gessture data from a given data set
    :param user_data: list<dict{'data':array,'label':array}>
            The data set from which the date of "Reest" gesture shoul be removed
    :return: list<dict{'data':array,'label':array}>
    """
    data, label_ = [], []
    for i in range(len(user_data['label'])):
        if user_data['label'][i] == 12:
            continue
        data.append(user_data['data'][i])
        label_.append(user_data['label'][i])
    user_data['data'] = data
    user_data['label'] = label_
    return user_data


def feature_extraction(users):
    """
    Extract the features from a given set of users
    :param users:  array, shape = [n_users] The array of users for feature extraction
    :return: No returns
    """
    for user in users:
        for preprocessing in Constant.level_0:
            for data_set in Constant.level_1:
                for sensor in Constant.level_2:
                    for window in Constant.level_3:
                        for overlap in Constant.level_4:
                            for feature in Constant.level_5:
                                process_raw_data(user, data_set=data_set, overlap=overlap,
                                                 sensor=sensor, window=window, feature=feature, pre=preprocessing)


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
                            # print(len(config_items))
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


if __name__ == '__main__':
    main()

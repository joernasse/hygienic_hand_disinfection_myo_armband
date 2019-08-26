from __future__ import print_function
import os
import pickle

from tensorflow.python.keras.models import load_model
import Classification
import Constant
import Deep_learning
from Process_data import process_raw_data, window_data, window_data_matrix, window_only_one_sensor
from Save_Load import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
path_add = [SEPARATE_PATH, CONTINUES_PATH]


def normalize_data(data, sensor, mode='rest_mean'):
    if sensor == IMU:
        element = 10
    else:
        element = 9
    rest_data = data['Rest']
    channel, mean = [], []
    if mode == 'rest_mean':
        for ch in range(1, element):
            for i in range(len(rest_data)):
                channel.append(rest_data[i][ch])
            mean.append(numpy.mean(channel))  # Mean of base REST over all channels (1-8)

        try:
            for d in data:
                if d == 'Rest':
                    continue
                for ch in range(1, element):  # Channel/ IMU entries
                    for i in range(len(data[d])):  # Elements
                        data[d][i][ch] = data[d][i][ch] / mean[ch - 1]
        except:
            print("Not expected exception")

    if mode == 'max_value_channel':
        for d in data:
            items = []
            for ch in range(1, element):
                for i in range(len(data[d])):
                    items.append(data[d][i][ch])
                max_val = max(items)
                items = items / max_val
    return data


def pre_process_raw_data(window, overlap, users, sensor=IMU, skip_s0_rest=True, normalize=False):
    print("window", window, "overlap", overlap)
    raw_data = load_raw_data_for_nn(users, sensor)
    if normalize:
        print("Start normalization")
        print(raw_data['Step0'][0])
        raw_data = normalize_data(data=raw_data, sensor=sensor)
        print(raw_data['Step0'][0])

    if skip_s0_rest:
        raw_data['Step0'] = []
        raw_data['Rest'] = []
    window_data, labels = window_raw_data_for_nn(window, overlap, raw_data=raw_data, sensor=sensor,
                                                 skip_s0_rest=skip_s0_rest)
    labels = [int(i) for i in labels]
    if skip_s0_rest:
        labels = [i - 1 for i in labels]
    print(len(window_data), len(labels))
    return window_data, labels


def pre_process_raw_data_adapt_model(window, overlap, users, sensor, skip_s0_rest, normalize=False):
    print("window", window, "overlap", overlap)
    training_data, test_data = load_raw_data_for_adapt_model(users, sensor)
    if normalize:
        print("Start normalization")
        training_data = normalize_data(data=training_data, sensor=sensor)
        test_data = normalize_data(data=test_data, sensor=sensor)

    if skip_s0_rest:
        training_data['Step0'] = []
        training_data['Rest'] = []
        test_data['Step0'] = []
        test_data['Rest'] = []

    window_data_train, labels_train = window_raw_data_for_nn(window, overlap, raw_data=training_data, sensor=sensor,
                                                             skip_s0_rest=skip_s0_rest)
    window_data_test, labels_test = window_raw_data_for_nn(window, overlap, raw_data=test_data, sensor=sensor,
                                                           skip_s0_rest=skip_s0_rest)

    labels_train = [int(x) for x in labels_train]
    labels_test = [int(x) for x in labels_test]

    if skip_s0_rest:
        labels_train = [i - 1 for i in labels_train]
        labels_test = [i - 1 for i in labels_test]
    print(len(window_data_train), len(window_data_test))
    return window_data_train, labels_train, window_data_test, labels_test


def main():
    # train_user_dependent()
    # calculation_config_statistics()
    # path = os.getcwd()
    # train_user_independent(best_config_rf)

    save_path = "G:/Masterarbeit/deep_learning/CNN_final_results/training_user_dependent_emg_0_user001"
    load_model_from = "G:/Masterarbeit/deep_learning/CNN_final_results/training_kaggle_imu_0"

    # if not os.path.isdir(save_path):  # Collection dir
    #     os.mkdir(save_path)
    sensor = IMU

    # --------------------------------------------Pre process Data-----------------------------------------------------#
    # x, labels = pre_process_raw_data(100, 0.75, ["User007"], sensor, skip_s0_rest=False, normalize=False)

    # x = [t.transpose() for t in x]

    # --------------------------------------------Train CNN 1----------------------------------------------------------#
    # Deep_learning.cnn(x, labels, save_path, batch=200, epochs=200)

    # --------------------------------------------Train CNN kaggle-----------------------------------------------------#
    # # For all Users, can be changed
    # for user in USERS:
    #     save_path = "G:/Masterarbeit/deep_learning/CNN_final_results/kaggle_dependent/dependent_"+user
    #     if not os.path.isdir(save_path):  # Collection dir
    #         os.mkdir(save_path)
    #     x, labels = pre_process_raw_data(100, 0.75, [user], sensor, skip_s0_rest=False, normalize=False)
    # Deep_learning.cnn_kaggle(x, labels, save_path, batch=200, epochs=10)

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

    # ----------------------------------Train user dependent classic-------------------------------------------------#
    config = best_config_rf
    print("Config",config)
    # Predict for all Users
    for user in USERS:
        print(user)
        users_data = load_feature_csv_one_user(config, user)
        Classification.train_user_independent(users_data=users_data,
                                              config=config,
                                              mixed_user_data=True,
                                              clf_name="Random_Forest_User_dependent",
                                              clf=Constant.random_forest,
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


def train_user_independent(config):
    users_data = load_feature_from_many_users(config, USERS_cross)
    Classification.train_user_independent(users_data, config, "RandomForest", mixed_user_data=True)
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

def load_raw_data_for_nn(users, sensor):
    global imu_dict
    global emg_dict
    global path_add
    for user in users:
        path = collections_default_path + user
        directories = [os.listdir(path + SEPARATE_PATH), os.listdir(path + CONTINUES_PATH)]
        for i in range(len(directories)):
            for steps in directories[i]:
                index = steps[2:]
                path_data = path + path_add[i] + "/" + steps
                if sensor == IMU:
                    imu_dict[index].extend(load_raw_2(path_data + "/imu.csv"))
                else:
                    emg_dict[index].extend(load_raw_2(path_data + "/emg.csv"))
        print("Load raw data", user, "done")
    if sensor == IMU:
        return imu_dict
    else:
        return emg_dict
    # return imu_dict, emg_dict


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
        path = collections_default_path + user
        directories = [os.listdir(path + SEPARATE_PATH), os.listdir(path + CONTINUES_PATH)]
        for i in range(len(directories)):
            for steps in directories[i]:
                index = steps[2:]
                path_data = path + path_add[i] + "/" + steps
                if sensor == IMU:
                    if steps.__contains__('s0'):
                        training_dict[index].extend(load_raw_2(path_data + "/imu.csv"))
                    else:
                        test_dict[index].extend(load_raw_2(path_data + "/imu.csv"))
                else:
                    if steps.__contains__('s0'):
                        training_dict[index].extend(load_raw_2(path_data + "/emg.csv"))
                    else:
                        test_dict[index].extend(load_raw_2(path_data + "/emg.csv"))

        print(user, "done")
    return training_dict, test_dict


def window_raw_data_for_nn(window, overlap, raw_data, sensor, skip_s0_rest):
    labels, window_data, emg_data = [], [], []
    if skip_s0_rest:
        label = sub_label
    else:
        label = save_label
    for key in label:
        window_tmp, label = window_only_one_sensor(input_data=raw_data[key], window=window, degree_of_overlap=overlap)
        window_data.extend(window_tmp)
        s = numpy.vstack((window_data, window_tmp))

        # if sensor == IMU:
        #     # implementation for IMU
        # elif sensor == EMG:
        #     # implementation for EMG
        #     emg, label = window_only_one_sensor(emg_dict[key], window=window, degree_of_overlap=overlap)
        #     emg_data.extend(emg)
        #     s = numpy.vstack((emg_data, emg))

        labels.extend(numpy.asarray(label))
    return window_data, labels
    # return window_data, emg_data, labels


def train_cnn(window, overlap, save_path, load_model_path="", sensor=IMU):
    print("window", window, "overlap", overlap)
    if not load_model_path == "":
        imu, emg = load_raw_data_for_nn(["User007"], sensor)
        imu_windows, emg_windows, labels = window_raw_data_for_nn(window, overlap, imu, emg, sensor)
        labels = [int(i) for i in labels]
        print(len(imu_windows), len(labels))
        if sensor == IMU:
            Deep_learning.predict_for_load_model(numpy.asarray(imu_windows), numpy.asarray(labels),
                                                 load_model(load_model_path + "/cnn_imu_model.h5"))
        else:
            Deep_learning.predict_for_load_model(numpy.asarray(emg_windows), numpy.asarray(labels),
                                                 load_model(load_model_path + "/cnn_imu_model.h5"))
    else:
        imu, emg = load_raw_data_for_nn(["User001"], sensor)
        imu_windows, emg_windows, labels = window_raw_data_for_nn(window, overlap, imu, emg, sensor)
        labels = [int(i) for i in labels]
        print(len(emg_windows), len(labels))
        if sensor == IMU:
            Deep_learning.cnn(numpy.asarray(imu_windows), numpy.asarray(labels), save_path)
        else:
            Deep_learning.cnn(numpy.asarray(emg_windows), numpy.asarray(labels), save_path)

        # Deep_learning.cnn_1(numpy.asarray(emg_windows), numpy.asarray(labels), save_path)
        # Deep_learning.cnn_rehman(numpy.asarray(emg_windows), numpy.asarray(labels), save_path)


def train_user_dependent():
    for data_set in level_1:
        for sensor in level_2:
            for window in level_3:
                for overlap in level_4:
                    for feature in level_5:
                        config = data_set + "-" + sensor + "-" + str(window) + "-" + str(overlap) + "-" + feature
                        users_data = load_feature_from_many_users(config)
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

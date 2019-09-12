import os
from scipy.stats import zscore
from Constant import *
from Feature_extraction import *
from Save_Load import load_raw_data, save_features
import numpy as np

np.seterr(divide='ignore')


# Select user directory --  load all emg and imu data, window it, extract features
def process_raw_data(user, overlap, window, data_set, sensor, feature, pre):
    """
    Load raw data for user, window the data, pre process the data,
    extract  features from data, save extracted features to file
    :param user: string
        The user from which the features should be extracted
    :param overlap: float
        The size of overlap
    :param window: int
        The window size
    :param data_set: string
        The data set from user study: separate, continues,separatecontinues
    :param sensor: string
        The sensor data: EMG,IMU,EMGIMU
    :param feature: string
        The set of features  to extract
    :param pre: string
        The pre processing setting: filter, z-normalization, no pre processing
    """
    features = []
    save_path = collections_path_default + user + "/features"
    try:
        load_path = collections_path_default + user
        directories, path_add = [], []

        if SEPARATE in data_set:
            directories.append(os.listdir(load_path + SEPARATE_PATH))
            path_add.append(SEPARATE_PATH)
        if CONTINUES in data_set:
            directories.append(os.listdir(load_path + CONTINUES_PATH))
            path_add.append(CONTINUES_PATH)

        for i in range(len(directories)):
            tmp_features = []
            for steps in directories[i]:
                features = []
                s_path = load_path + path_add[i] + "/" + steps

                raw_emg, raw_imu = load_raw_data(emg_path=s_path + "/emg.csv", imu_path=s_path + "/imu.csv")
                w_emg, w_imu = window_data_for_both_sensor(raw_emg, raw_imu, window=window, degree_of_overlap=overlap,
                                                           skip_timestamp=1)

                # Pre process each window
                if pre == Constant.filter_:
                    w_emg = filter_emg_data(emg=w_emg, f_type=feature)
                elif pre == Constant.z_norm:
                    w_emg, w_imu = z_norm(w_emg, w_imu)

                if EMG + IMU in sensor:
                    features.append(feature_extraction(w_emg, mode=feature, sensor=EMG))
                    features.append(feature_extraction(w_imu, mode=feature, sensor=IMU))
                    tmp = []
                    for j in range(len(features[0])):
                        merged_feature = features[0][j]['fs'] + features[1][j]['fs']
                        if features[0][j]['label'] == features[1][j]['label']:
                            tmp.append({"fs": merged_feature, "label": features[1][j]['label']})
                        else:
                            print("ERROR! Should not happen!")
                    tmp_features.append(tmp)
                    continue
                if EMG in sensor:
                    tmp_features.append(feature_extraction(w_emg, mode=feature, sensor=EMG))
                if IMU in sensor:
                    tmp_features.append(feature_extraction(w_imu, mode=feature, sensor=IMU))
            features = tmp_features
        if pre:
            save_path = save_path + "_filter"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        filename = user + "-" + pre + "-" + data_set + "-" + sensor + "-" + str(window) + "-" + str(
            overlap) + "-" + feature
        save_features(features, save_path + "/" + filename + ".csv")
        print(filename + " done")
        return True
    except:
        print("ERROR!", user, data_set, sensor, ValueError)
        raise


def window_for_one_sensor(input_data, window, degree_of_overlap=0):
    """

    :param input_data:
    :param window:
    :param degree_of_overlap:
    :return:
    """
    try:
        offset = window * degree_of_overlap
        data_cutted = [np.asarray(x[1:-1]) for x in input_data]
        w_data, labels = [], []
        label = input_data[0][-1]
        length = len(input_data)
        blocks = int(length / abs(window - offset))
        first = 0
        for i in range(blocks):
            last = int(first + window)
            data = data_cutted[first:last]
            if not len(data) == window:
                print(label, " - first/last", first, last)
                first += int(window - offset)
                break
            w_data.append(np.asarray(data))
            labels.append(label)
            first += int(window - offset)
        return np.asarray(w_data), labels
    except:
        print("")


def window_data_for_both_sensor(emg_data, imu_data, window, degree_of_overlap, skip_timestamp):
    """

    :param emg_data:
    :param imu_data:
    :param window:
    :param degree_of_overlap:
    :param skip_timestamp:
    :return:
    """
    emg_window, imu_window = [], []
    emg_length, imu_length = len(emg_data['label']), len(imu_data['label'])

    window_imu = int(window / (emg_length / imu_length))
    offset_imu = window_imu * degree_of_overlap
    offset_emg = window * degree_of_overlap

    # define blocks (should be equal, for imu and emg) for calculation emg data used
    blocks = int(emg_length / abs(window - offset_emg))
    label = emg_data['label'][0]

    first_emg, first_imu = 0, 0
    for i in range(blocks):
        last_emg = first_emg + window
        last_imu = int(first_imu + window_imu)
        emg, imu = [], []

        if not len(emg_data['label'][first_emg:last_emg]) == window or not len(imu_data['label'][first_imu:last_imu]) == window_imu:
            first_emg += int(window - offset_emg)
            first_imu += int(window_imu - offset_imu)
            continue

        # if len(emg_data['label'][first_emg:last_emg]) == window:
        for n in identifier_emg[skip_timestamp:]:
            emg.append([j for j in emg_data[n][first_emg:last_emg]])
        emg.append(label)
        emg_window.append(emg)
        first_emg += int(window - offset_emg)

        # if len(imu_data['label'][first_imu:last_imu]) == window_imu:
        for k in identifier_imu[skip_timestamp:]:
            imu.append([j for j in imu_data[k][first_imu:last_imu]])
        imu.append(label)
        imu_window.append(imu)
        first_imu += int(window_imu - offset_imu)

    if len(emg_window) > len(imu_window):
        emg_window = emg_window[:len(imu_window)]
    elif len(imu_window) > len(emg_window):
        imu_window = imu_window[:len(emg_window)]
    return emg_window, imu_window


def filter_emg_data(emg, f_type):
    """

    :param emg:
    :param f_type:
    :return:
    """
    try:
        f_emg = []
        if f_type == Constant.rehman:
            b_emg, a_emg = Constant.rehman_b_emg, Constant.rehman_a_emg
        else:
            b_emg, a_emg = Constant.benalcazar_b_emg, Constant.benalcazar_a_emg
        for item in emg:
            tmp = [signal.filtfilt(b_emg, a_emg, x) for x in item[:-1]]
            tmp.append(item[-1])
            f_emg.append(tmp)
        return f_emg
    except:
        print("ERROR! filter_data")


def z_norm(emg, imu):
    """"""
    try:
        z_emg, z_imu = [], []
        for item in emg:
            tmp = [zscore(x) for x in item[:-1]]
            tmp.append(item[-1])
            z_emg.append(tmp)
        for item in imu:
            tmp = [zscore(x) for x in item[:-1]]
            tmp.append(item[-1])
            z_imu.append(tmp)
        return z_emg, z_imu
    except:
        print("Error! z_norm")

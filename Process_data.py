import os
import time

from Constant import *
from Save_Load import load_raw_csv, save_feature_csv
from Feature_extraction import *


# Select user directory --  load all emg and imu data, window it, feature extraction
def process_raw_data(user, overlap=None, window=None, dataset=None, sensor=None, feature=None):
    features = []
    save_path = collections_default_path + user

    try:
        load_path = collections_default_path + user
        directories, path_add = [], []
        if SEPARATE in dataset:
            directories.append(os.listdir(load_path + SEPARATE_PATH))
            path_add.append(SEPARATE_PATH)
        if CONTINUES in dataset:
            directories.append(os.listdir(load_path + CONTINUES_PATH))
            path_add.append(CONTINUES_PATH)

        for i in range(len(directories)):
            tmp_features = []
            for steps in directories[i]:
                features = []
                s_path = load_path + path_add[i] + "/" + steps

                emg_raw, imu_raw = load_raw_csv(emg_path=s_path + "/emg.csv", imu_path=s_path + "/imu.csv")
                emg_window, imu_window = window_data(emg_raw, imu_raw, window=window, degree_of_overlap=overlap)
                current_label = int(emg_raw['label'][0])

                if EMG in sensor:
                    res = feature_extraction(emg_window, label=current_label, mode=feature)
                    features.append(res)
                if IMU in sensor:
                    res = feature_extraction(imu_window, label=current_label, mode=feature)
                    features.append(res)
                if EMG + IMU in sensor:
                    tmp = []
                    for j in range(len(features[0])):
                        merged_feature = features[0][j]['fs'] + features[1][j]['fs']
                        if features[0][j]['label'] == features[1][j]['label']:
                            tmp.append({"fs": merged_feature, "label": features[1][j]['label']})
                        else:
                            print("ERROR! Should not happen!")
                    tmp_features.append(tmp)
            features = tmp_features

        filename = user + "-" + dataset + "-" + sensor + "-" + str(window) + "-" + str(overlap) + "-" + feature
        save_path = save_path + "/features"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        save_feature_csv(features, save_path + "/" + filename + ".csv")
        print(filename)
        return features
    except:
        print("ERROR!", user, steps, dataset, sensor, ValueError)
        raise


def window_data(emg_data, imu_data, window=20, degree_of_overlap=0.5,skip_timestamp=0):
    emg_window, imu_window = [], []
    emg_length, imu_length = len(emg_data['label']), len(imu_data['label'])

    window_imu = window / (emg_length / imu_length)
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
        for n in identifier_emg[skip_timestamp:]:
            emg.append([j for j in emg_data[n][first_emg:last_emg]])
        emg.append(label)
        emg_window.append(emg)
        first_emg += int(window - offset_emg)

        for k in identifier_imu[skip_timestamp:]:
            imu.append([j for j in imu_data[k][first_imu:last_imu]])
        imu.append(label)
        imu_window.append(imu)
        first_imu += int(window_imu - offset_imu)
    return emg_window, imu_window

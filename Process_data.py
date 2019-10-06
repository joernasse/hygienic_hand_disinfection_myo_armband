import os
import scipy.signal as signal
from scipy.stats import zscore
import Constant
import Feature_extraction as fe
import Save_Load
import numpy as np

np.seterr(divide='ignore')


def process_raw_data(user, overlap, window, data_set, sensor, feature, pre,
                     save_path_for_featureset="./", load_path=Constant.collections_path_default):
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
    :param save_path_for_featureset: string, default Constant.collection_path_default('G:/Masterarbeit/Collections/')
            Describes the save path for features. If empty the default path "./" will be used
    :param load_path: string
            Path to the Collection folder which contains the raw data
    """
    features = []
    save_path_for_featureset += user + "/features"
    try:
        load_path += user
        directories, path_add = [], []

        if Constant.SEPARATE in data_set:
            directories.append(os.listdir(load_path + Constant.SEPARATE_PATH))
            path_add.append(Constant.SEPARATE_PATH)
        if Constant.CONTINUES in data_set:
            directories.append(os.listdir(load_path + Constant.CONTINUES_PATH))
            path_add.append(Constant.CONTINUES_PATH)

        for i in range(len(directories)):  # go through all directories
            tmp_features = []
            for steps in directories[i]:  # go through all Steps
                features = []
                raw_emg, raw_imu = Save_Load.load_raw_data_for_both_sensors(
                    emg_path=load_path + path_add[i] + "/" + steps + "/emg.csv",
                    imu_path=load_path + path_add[i] + "/" + steps + "/imu.csv")

                w_emg, w_imu = window_data_for_both_sensor(raw_emg, raw_imu, window=window, degree_of_overlap=overlap,
                                                           skip_timestamp=1)

                # Preprocess each window
                if pre == Constant.filter_:
                    w_emg = filter_emg_data(emg=w_emg, filter_type=feature)
                elif pre == Constant.z_norm:
                    w_emg, w_imu = z_norm(w_emg, w_imu)

                if Constant.EMG + Constant.IMU in sensor:
                    features.append(fe.feature_extraction(w_emg, mode=feature, sensor=Constant.EMG))
                    features.append(fe.feature_extraction(w_imu, mode=feature, sensor=Constant.IMU))
                    tmp = []
                    for j in range(len(features[0])):
                        merged_feature = features[0][j]['fs'] + features[1][j]['fs']
                        if features[0][j]['label'] == features[1][j]['label']:
                            tmp.append({"fs": merged_feature, "label": features[1][j]['label']})
                        else:
                            print("ERROR! Should not happen!")
                    tmp_features.append(tmp)
                    continue
                if Constant.EMG in sensor:
                    tmp_features.append(fe.feature_extraction(w_emg, mode=feature, sensor=Constant.EMG))
                if Constant.IMU in sensor:
                    tmp_features.append(fe.feature_extraction(w_imu, mode=feature, sensor=Constant.IMU))
            features = tmp_features
        if pre:
            save_path_for_featureset = save_path_for_featureset + "_filter"
        if not os.path.isdir(save_path_for_featureset):
            os.mkdir(save_path_for_featureset)
        filename = user + "-" + pre + "-" + data_set + "-" + sensor + "-" + str(window) + "-" + str(
            overlap) + "-" + feature
        Save_Load.save_features(features, save_path_for_featureset + "/" + filename + ".csv")
        print(filename + " done")
        return True
    except:
        print("ERROR!", user, data_set, sensor, ValueError)
        raise


def window_for_one_sensor(input_data, window, degree_of_overlap=0):
    """
    TODO
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
                first += int(window - offset)
                break
            w_data.append(np.asarray(data))
            labels.append(int(label))
            first += int(window - offset)
        return np.asarray(w_data), labels
    except:
        print("")


def window_data_for_both_sensor(emg_data, imu_data, window, degree_of_overlap, skip_timestamp):
    """
    Window the EMG and IMU data, that the time interval is equal
    Window size will be different for IMU and EMG data
    :param emg_data: dict{}
    TODO:
    :param imu_data: dict{}
    TODO
    :param window: int
            Descibes the windows size
    :param degree_of_overlap: float
            Describes the degree of overlap. Example 0.5 means 50% of overlapping windows
    :param skip_timestamp: int 0 or 1
            Describes if the timestamp shoul be skipped by generating the window.
            1 skip
            0 don´t skip
    :return: array, array
            array of windows for EMG data
            array of windows for IMU data
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

        if not len(emg_data['label'][first_emg:last_emg]) == window \
                or not len(imu_data['label'][first_imu:last_imu]) == window_imu:
            first_emg += int(window - offset_emg)
            first_imu += int(window_imu - offset_imu)
            continue

        for n in Constant.identifier_emg[skip_timestamp:]:
            emg.append([j for j in emg_data[n][first_emg:last_emg]])
        emg.append(label)
        emg_window.append(emg)
        first_emg += int(window - offset_emg)

        for k in Constant.identifier_imu[skip_timestamp:]:
            imu.append([j for j in imu_data[k][first_imu:last_imu]])
        imu.append(label)
        imu_window.append(imu)
        first_imu += int(window_imu - offset_imu)

    if len(emg_window) > len(imu_window):
        emg_window = emg_window[:len(imu_window)]
    elif len(imu_window) > len(emg_window):
        imu_window = imu_window[:len(emg_window)]
    return emg_window, imu_window


def filter_emg_data(emg, filter_type):
    """
    Filters the EMG date with the with the filter, which is given by the filter type
    Two diffrent filters can be found in Constant.py
    :param emg: array
            The EMG data which should be filtered
    :param filter_type: string
            The filter type. Basically described in Constand.py
    :return: array
            The filtered EMG data
    """
    try:
        f_emg = []
        if filter_type == Constant.rehman:
            b_emg, a_emg = Constant.rehman_b_emg, Constant.rehman_a_emg
        else:
            b_emg, a_emg = Constant.benalcazar_b_emg, Constant.benalcazar_a_emg
        for item in emg:
            tmp = [signal.filtfilt(b_emg, a_emg, x) for x in item[:-1]]
            tmp.append(item[-1])
            f_emg.append(tmp)
        return f_emg
    except:
        print("ERROR! Problem in filter data")
        raise


def z_norm(emg, imu):
    """
    Performs the Z-Normalization for EMG and IMU data
    :param emg: array
            Array of EMG data
    :param imu: array
            Array of IMU data
    :return: array,array
            Z-normalized EMG data
            z-normalized IMU data

    """

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
        print("Error! Problem in Z normalization")
        raise


def collect_data_for_single_sensor(user_list, sensor, data_set, collection_path=Constant.collections_path_default):
    """

    :param user_list:
    :param sensor:
    :param data_set:
    :param collection_path:
    :return:
    """
    path_add = []
    raw_data = {'Step0': [], 'Step1': [], 'Step1_1': [], 'Step1_2': [], 'Step2': [], 'Step2_1': [], 'Step3': [],
                'Step4': [], 'Step5': [], 'Step5_1': [], 'Step6': [], 'Step6_1': [], 'Rest': []}

    for user in user_list:
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
                raw_data[index].extend(Save_Load.load_raw_for_single_sensor(path + path_add[i] + "/" + steps + "/"
                                                                            + sensor + ".csv"))

        print("Load raw data for", user, " - Done")
    print("Raw data length", len(raw_data['Step0']))
    return raw_data


def window_raw_data_for_nn(window, overlap, raw_data, ignore_rest_gesture=True):
    """
    TODO
    :param window:
    :param overlap:
    :param raw_data:
    :param ignore_rest_gesture:
    :return:
    """
    labels, window_data, emg_data, raw = [], [], [], 0
    if ignore_rest_gesture:
        label = Constant.labels_without_rest
    else:
        label = Constant.label
    for key in label:
        raw += len(raw_data[key])
        print(raw)
        window_tmp, label = window_for_one_sensor(input_data=raw_data[key], window=window, degree_of_overlap=overlap)
        window_data.extend(window_tmp)
        np.vstack((window_data, window_tmp))
        labels.extend(np.asarray(label))
    return window_data, labels


def remove_rest_gesture_data(user_data):
    """
    Remove the "Rest" Gessture data from a given data set
    :param user_data: list<dict{'data':array,'label':array}>
            The data set from which the date of "Rest" gesture shoul be removed
    :return: list<dict{'data':array,'label':array}>
    """
    print("Remove data for rest gesture - START")
    data, label_ = [], []
    for i in range(len(user_data['label'])):
        if user_data['label'][i] == 12:
            continue
        data.append(user_data['data'][i])
        label_.append(user_data['label'][i])
    user_data['data'] = data
    user_data['label'] = label_
    print("Remove data for rest gesture - DONE")
    return user_data

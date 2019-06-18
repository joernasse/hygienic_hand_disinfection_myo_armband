import os
from tkinter import filedialog

import math
import numpy as np

from Save_Load import load_raw_csv, save_feature_csv

MAX = 127
threshold = 0.30 * MAX

identifier_emg = "timestamp", "ch0", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7"
identifier_imu = "timestamp", "x_ori", "y_ori", "z_ori", "x_gyr", "y_gyr", "z_gyr", "x_acc", "y_acc", "z_acc"


def placeholder(feature_extration_mode="default"):
    # Select user directory --  load all emg and imu data, window it, feature extraction
    path = filedialog.askdirectory()
    directories = os.listdir(path)
    emg_feature, imu_feature = [], []
    for dir_name in directories:
        full_path = path + "/" + dir_name
        files = os.listdir(full_path)
        emg_data, imu_data = load_raw_csv(full_path + "/" + files[0], full_path + "/" + files[1])
        current_label = int(emg_data['label'][0])
        emg_window, imu_window = window_data(emg_data, imu_data)

        emg_feature.append(feature_extraction_default(emg_window, 1, 1, current_label))
        imu_feature.append(feature_extraction_default(imu_window, 1, 1, current_label))

    path = path + "/feature_" + feature_extration_mode
    if not os.path.isdir(path):
        os.mkdir(path)
    save_feature_csv(emg_feature, path + "/emg.csv")


def window_data(emg_data, imu_data, window=20, degree_of_overlap=0.5):
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
        for n in identifier_emg:
            emg.append([j for j in emg_data[n][first_emg:last_emg]])
        emg.append(label)
        emg_window.append(emg)
        first_emg += int(window - offset_emg)

        for k in identifier_imu:
            imu.append([j for j in imu_data[k][first_imu:last_imu]])
        imu.append(label)
        imu_window.append(imu)
        first_imu += int(window_imu - offset_imu)
    return emg_window, imu_window


def feature_extraction_default(window_list, skip_first=0, skip_last=0, label=-1):
    features = []
    for window in window_list:
        data_area = len(window) - 1 - skip_last
        feature = []
        for i in range(skip_first, data_area):
            feature.extend([rms(window[i]),
                            iav(window[i]),
                            ssi(window[i]),
                            var(window[i]),
                            wl(window[i]),
                            aac(window[i])])
        features.append({"fs": feature,
                         "label": label})

    return features


def feat_trans_def(saving_list, features, options=0):
    if len(saving_list) == 0:
        return [rms(features), iav(features), ssi(features), var(features), wl(features), aac(features)]
    else:
        saving_list.append(rms(features))
        saving_list.append(iav(features))
        saving_list.append(ssi(features))
        saving_list.append(var(features))
        saving_list.append(wl(features))
        saving_list.append(aac(features))
        return saving_list
    # feat_transf.append(tm3(array))


def normalization(channel):
    channel_norm = []
    x_max = np.max(channel)
    for xi in channel:
        channel_norm.append((MAX / x_max) * xi)
    return channel_norm


def rms(array):  # root mean square
    sum = 0
    n = len(array)
    for a in array:
        sum += a * a
    return np.sqrt(1 / n * sum)


def mav(array):  # Mean Absolute Value
    sum = 0
    n = len(array)
    for a in array:
        sum += np.abs(a)
    return 1 / n * sum


def energy(array):  # Energy Ratio
    sum = 0
    for a in array:
        sum += a * a
    return sum


def var(array):  # Variance
    n = len(array)
    sum = 0
    for a in array:
        sum += np.abs(a)
    return 1 / (n - 1) * sum


def wamp(array):  # Willison Amplitude
    n = len(array)
    sum = 0
    for i in range(n - 1):
        if np.abs(array[i] - array[i + 1]) >= threshold:
            sum += 1
        else:
            sum += 0
    return sum


def zc(array):  # Zero Crossing
    n = len(array)
    sum = 0
    for i in range(n - 1):
        x = array[i]
        y = array[i + 1]
        if (x * y <= threshold) and (np.abs(x - y) >= threshold):
            sum += 1
        else:
            sum += 0
    return sum


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def iav(array):
    sum = 0
    for a in array:
        sum += np.abs(a)
    return sum


def ssi(array):
    sum = 0
    for a in array:
        sum += a * a
    return sum


def tm3(array):
    n = len(array)
    print('n : ', n)
    sum = 0
    for a in array:
        sum += a * a * a
    return np.power((1 / float(n)) * sum, 1 / float(3))


def wl(array):  # wavlet
    sum = 0
    for a in range(0, len(array) - 1):
        sum += array[a + 1] - array[a]
    return sum


def aac(array):
    n = len(array)
    sum = 0
    for a in range(0, n - 1):
        sum += array[a + 1] - array[a]
    return sum / float(n)


def toEuler(quat):
    magnitude = math.sqrt(quat.x ** 2 + quat.y ** 2 + quat.z ** 2 + quat.w ** 2)
    quat.x = quat.x / magnitude
    quat.y = quat.y / magnitude
    quat.z = quat.z / magnitude
    quat.w = quat.w / magnitude

    # Roll
    roll = math.atan2(2.0 * (quat.w * quat.x + quat.y * quat.z),
                      1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y))

    # Pitch
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * quat.w * quat.y - quat.z * quat.x)))

    # Yaw
    yaw = math.atan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                     1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z))
    return [pitch, roll, yaw]


# frequency domain
def ampSpec(array):  # Amplitude Spectrum
    freq_array = np.fft.fft(array)
    n = len(freq_array)
    sum = 0
    for a in freq_array:
        sum += np.abs(a)
    return sum

# def mmdf(array):  # Modified Median Frequency
#     sum=0
#     for a in array:
#         sum+=ampSpec([a])
#     return 0.5*sum

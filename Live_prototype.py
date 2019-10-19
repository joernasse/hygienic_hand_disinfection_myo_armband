#!/usr/bin/env python
"""
Ths Script contains the functionality to perform a live validation for a given set of machine learning models.
It´s also possible to run a live application to perform the gestures of hygienic hand disinfection and
get direct feedback on how the data has been classified
"""

import csv
import os
import pickle
import threading
import time
from operator import add
from myo import init, Hub, StreamEmg
import myo as libmyo
import logging as log
import tensorflow as tf
import Constant
import Feature_extraction
import Helper_functions
import Process_data
import numpy as np
from tensorflow.python.keras.models import load_model
import collections
from Deep_learning_classification import adapt_model_for_user

__author__ = "Joern Asse"
__copyright__ = ""
__credits__ = ["Joern Asse"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Joern Asse"
__email__ = "joernasse@yahoo.de"
__status__ = "Production"

cnn_emg_ud_path = "./Live_Prediction/Load_model/User002_UD_no_pre_pro-separate-EMG-100-0.9-NA_cnn_CNN_Kaggle.h5"
cnn_imu_ud_path = "./Live_Prediction/Load_model/User002_UD_no_pre_pro-separate-IMU-25-0.9-NA_cnn_CNN_Kaggle.h5"
cnn_emg_ui_path = "./Live_Prediction/Load_model/User002_UI_no_pre_pro-separatecontinues-EMG-100-0.9-NA_cnn_CNN_Kaggle.h5"
cnn_imu_ui_path = "./Live_Prediction/Load_model/User002_UI_no_pre_pro-separatecontinues-IMU-25-0.9-NA_cnn_CNN_Kaggle.h5"
classic_ud_path = "./Live_Prediction/Load_model/User002_UD_Random_forest_no_pre_pro-separate-EMGIMU-100-0.9-georgi.joblib"
classic_ui_path = "./Live_Prediction/Load_model/User002_UI_Random_Forest_no_pre_pro-separate-EMGIMU-100-0.9-georgi.joblib"

# cnn_imu_adapted_path = "C:/EMG_Recognition/live-adapt-IMU_cnn_CNN_Kaggle_adapt.h5"
# cnn_emg_adapted_path = "C:/EMG_Recognition/live-adapt-EMG_cnn_CNN_Kaggle_adapt.h5"

headline_summary = ["Session", "seq", "y_true", "number_samples", "prediction_seq",
                    "prediction_seq_number", "correct_samples", "correct_samples_percent"]

headline_classic_detail = ["prediction", "probability_distribution", "y_true"]
headline_cnn_detail = ["prediction", "predict_percent", "predict_distribution", "y_true"]

DEVICE_L, DEVICE_R = None, None
EMG = []  # emg
ORI = []  # orientation
GYR = []  # gyroscope
ACC = []  # accelerometer
emg_l, emg_r = [], []
tmp = []
status = 0
imu_load_data = {"timestamp": [], "x_ori": [], "y_ori": [], "z_ori": [], "x_gyr": [], "y_gyr": [], "z_gyr": [],
                 "x_acc": [], "y_acc": [], "z_acc": [], "label": []}
emg_dict = {"timestamp": [], "ch0": [], "ch1": [], "ch2": [], "ch3": [], "ch4": [], "ch5": [], "ch6": [], "ch7": [],
            "label": []}

x_train_classic = []
live_prediction_threshold = 0.85


class GestureListener(libmyo.DeviceListener):
    def __init__(self, queue_size=1):
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)
        self.ori_data_queue = collections.deque(maxlen=queue_size)

    def on_arm_synced(self, event):
        print("arm synced", event)

    def on_connected(self, event):
        event.device.stream_emg(StreamEmg.enabled)

    def on_emg(self, event):
        with self.lock:
            if status:
                emg_l.append(DEVICE_L.emg)
                emg_r.append(DEVICE_R.emg)
                EMG.append([event.timestamp, event.emg])

    def on_orientation(self, event):
        with self.lock:
            if status:
                ORI.append([event.timestamp, event.orientation])
                ACC.append([event.timestamp, event.acceleration])
                GYR.append([event.timestamp, event.gyroscope])

    def get_ori_data(self):
        with self.lock:
            return list(self.ori_data_queue)


init()
hub = Hub()
device_listener = libmyo.ApiDeviceListener()
gesture_listener = GestureListener()


def check_samples_rate(warm_up_iteration=5, record_time=1):
    """
    Check the sample rat of the myo armbands. Should be run before  collecting data.
    :param warm_up_iteration:int, default 5
                            The iterations fo processing the warm up
    :param record_time:int, default 1
                        The record time
    :return: None
    """
    print("Check samples rate - Start")
    sum_emg, sum_imu = 0, 0
    print("Warm up")
    Helper_functions.wait(2)
    for i in range(warm_up_iteration):
        emg, ori, acc, gyr = collect_raw_data(record_time=record_time)
        sum_emg += len(emg)
        sum_imu += len(ori)
        print("EMG length", len(emg),
              "\nIMU length", len(ori))
    max_emg = (warm_up_iteration * 2 * record_time * 200)
    max_imu = (warm_up_iteration * 2 * record_time * 50)
    sr_emg = (sum_emg / max_emg) * 100
    sr_imu = (sum_imu / max_imu) * 100
    print("DS EMG", sr_emg,
          "\nDS_IMU", sr_imu)
    if (warm_up_iteration * 2 * 200) * 0.85 > sum_emg:
        print("EMG sample rate is under 85%")

    if (warm_up_iteration * 2 * 50) * 0.85 > sum_imu:
        print("IMU sample rate is under 85%")

    print("Check samples rate - Done")
    return


def load_models_for_validation():
    """
    Load the models from the global paths.
    Std. load cnn_emg_ud, cnn_imu_ud, cnn_emg_ui, cnn_imu_ui, classic_ud, classic_ui
    ud: user dependent
    ui: user independent
    :return: cnn_emg_ud, cnn_imu_ud, cnn_emg_ui, cnn_imu_ui, classic_ud, classic_ui
                    returns the loaded models
    """
    cnn_emg_ud = load_model(cnn_emg_ud_path)
    cnn_imu_ud = load_model(cnn_imu_ud_path)
    cnn_emg_ui = load_model(cnn_emg_ui_path)
    cnn_imu_ui = load_model(cnn_imu_ui_path)
    with open(classic_ud_path, 'rb') as pickle_file:
        classic_ud = pickle.load(pickle_file)
    with open(classic_ui_path, 'rb') as pickle_file:
        classic_ui = pickle.load(pickle_file)
    return cnn_emg_ud, cnn_imu_ud, cnn_emg_ui, cnn_imu_ui, classic_ud, classic_ui


def init(live_prediction_path=Constant.live_prediction_path):
    """
    Initialization for the communication with the Myo Armbands. Also pair devices.
    Waiting time is required to ensure correct streaming rate
    :param live_prediction_path:string, default "./Live_Prediction"
            the path to the live prediction folder
    :return: None
    """
    global status
    if not os.path.isdir(live_prediction_path):  # Collection dir
        os.mkdir(live_prediction_path)

    status = 1
    print("Initialization - Start")
    Helper_functions.wait(3)
    pair_devices()
    status = 0
    print("Initialization - Done")


def preprocess_data(w_emg, w_imu, filter_type):
    """
    Preprocess the windowed raw data. Apply the filter or z-normalization
    :param w_emg: list
                List of windowed EMG data
    :param w_imu: list
                List of windowed IMU data
    :param filter_type: string,
                Specifies which filter is to be used.
    :return: list,list
                Return the preprocessed windowed list of EMG and IMU data
    """
    if filter_type == Constant.filter_:
        p_emg, p_imu = Process_data.filter_emg_data(w_emg, filter_type)
    elif filter_type == Constant.z_norm:
        p_emg = Process_data.z_norm(w_emg)
        p_imu = Process_data.z_norm(w_imu)
    else:
        p_emg, p_imu = w_emg, w_imu
    return p_emg, p_imu


def main():
    """

    :return:
    """

    # --------------------------------------------Model Validation START ----------------------------------------------#
    cnn_imu_adapt, cnn_emg_adapt = None, None
    # OPTIONAL:
    # Load models for the adaptive approach
    # Edit path to the correct models
    # cnn_imu_adapt = load_model(
    #     "C:/EMG_Recognition/Live_Prediction/User001_Live/User001_live-adapt-IMU_cnn_CNN_Kaggle_adapt.h5")
    # cnn_emg_adapt = load_model(
    #     "C:/EMG_Recognition/Live_Prediction/User001_Live/User001_live-adapt-EMG_cnn_CNN_Kaggle_adapt.h5")

    # config_cnn_emg = "no_pre_pro-separate-EMG-100-0.9-NA"
    # config_cnn_imu = "no_pre_pro-separate-IMU-25-0.9-NA"
    # config_classic = "no_pre_pro-separate-EMGIMU-100-0.9-georgi"
    # cnn_emg_ud, cnn_imu_ud, cnn_emg_ui, cnn_imu_ui, classic_ud, classic = load_models_for_validation()
    # validate_models(cnn_emg_ud=cnn_emg_ud, cnn_imu_ud=cnn_imu_ud, cnn_emg_ui=cnn_emg_ui, cnn_imu_ui=cnn_imu_ui,
    #                 classic_ud=classic_ud, classic_ui=classic, config_cnn_emg=config_cnn_emg,
    #                 config_cnn_imu=config_cnn_imu, config_classic=config_classic, session=2)
    # --------------------------------------------Model Validation END ------------------------------------------------#

    # --------------------------------------------Live prediction for CNN START ---------------------------------------#
    cnn_imu_user002_path = "C:/emg-recognition-with-myo/Live_Prediction/User002_live-adapt-IMU_cnn_CNN_Kaggle_adapt.h5"
    cnn_emg_user002_path = "C:/emg-recognition-with-myo/Live_Prediction/User002_live-adapt-EMG_cnn_CNN_Kaggle_adapt.h5"

    live_prediction_cnn(config_emg="no_pre_processing-separate-EMG-100-0.9-NA",
                        config_imu="no_pre_processing-separate-IMU-25-0.9-NA",
                        cnn_emg=load_model(cnn_emg_user002_path),
                        cnn_imu=load_model(cnn_imu_user002_path),
                        record_time=2)
    # --------------------------------------------Live prediction for CNN END -----------------------------------------#

    # --------------------------------------------Live prediction for classic START -----------------------------------#
    with open(classic_ui_path, 'rb') as pickle_file:
        classic_classifier = pickle.load(pickle_file)

    live_prediction_classic(config="no_pre_processing-separate-EMGIMU-100-0.9-NA",
                            classifier=classic_classifier,
                            record_time=2)
    # --------------------------------------------Live prediction for classic END -------------------------------------#


def eval_predictions(predict, proba, y_true, file_prefix, session, seq, classic=False):
    """
    Evaluation for a given prediction. Requires the predicted and correct labels.
    Saved the detailed and summarised results in files
    :param predict:list
            The predicted classes(labels)
    :param proba:list,
            The probability of the predicted classes (labels)
    :param y_true:list,
            The correct labels
    :param file_prefix:string,
            Indicates the model type. For example emg_cnn_ui means the user independent CNN for EMG data
    :param session:int,
            Current recording session
    :param seq:int,
            Current sequence, can also be considered as number of gesture
    :param classic:boolean,
            If True the model used is one of the classic ML Classifier
    :return: None
    """
    write_header = False
    if not os.path.isdir(Constant.live_prediction_path + file_prefix):  # Collection dir
        os.mkdir(Constant.live_prediction_path + file_prefix)

    path = Constant.live_prediction_path + file_prefix + "/"
    number_samples = len(predict)
    sum_predict_dict = collections.OrderedDict(sorted(collections.Counter(predict).items()))
    if y_true in sum_predict_dict.keys():
        correct_samples = sum_predict_dict[y_true]
    else:
        correct_samples = 0
    if classic:
        index_max = max(sum_predict_dict, key=sum_predict_dict.get)

        # Summary results
        save_path = path + file_prefix + "_summary.csv"
        if not os.path.isfile(save_path):
            write_header = True
        file = open(save_path, 'a', newline='')
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writerow(headline_summary)
        writer.writerow([session, seq, y_true, number_samples, index_max, sum_predict_dict[index_max], correct_samples,
                         correct_samples / number_samples])
        file.close()

        # Detailed results
        save_path = path + file_prefix + "_detail.csv"
        if not os.path.isfile(save_path):
            write_header = True
        file = open(save_path, 'a', newline='')
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writerow(headline_classic_detail)
        writer.writerow([index_max, sum_predict_dict[index_max], [x for x in sum_predict_dict], y_true])
        file.close()

    else:
        sum_proba, index_max = sum_sequence_proba(proba)

        # Summary results
        save_path = path + file_prefix + "_summary.csv"
        if not os.path.isfile(save_path):
            write_header = True
        file = open(save_path, 'a', newline='')
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writerow(headline_summary)

        writer.writerow([session, seq, y_true, number_samples, index_max, sum_proba[index_max], correct_samples,
                         correct_samples / number_samples])
        file.close()

        # Detailed results
        save_path = path + file_prefix + "_detail.csv"
        if not os.path.isfile(save_path):
            write_header = True
        file = open(save_path, 'a', newline='')
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writerow(headline_cnn_detail)
        for j in range(len(predict)):
            writer.writerow([predict[j], [x for x in proba[j]], y_true])
        file.close()

    return True


def feature_extraction_live(w_emg, w_imu, feature_set=Constant.rehman):
    """
    Extracts the features of a given feature-set from the given raw data
    :param w_emg:matrix
            The matrix of the windowed raw data for the EMG signals
    :param w_imu:matrix
            The matrix of the windowed raw data for the IMU signals
    :param feature_set:string
            From this feature-set the features are extracted for extraction
    :return:matrix
            Matrix of extracted features. By default the features for EMG and IMU data will be combined
    """
    feature_emg, feature_imu = [], []
    for x in w_emg:
        feature = []
        for n in range(8):
            if feature_set == Constant.georgi:
                feature.extend(Feature_extraction.georgi([y[n] for y in x], sensor=Constant.EMG))
            elif feature_set == Constant.rehman:
                feature.extend(Feature_extraction.rehman([y[n] for y in x]))
            elif feature_set == Constant.robinson:
                feature.extend(Feature_extraction.robinson([y[n] for y in x]))
            elif feature_set == Constant.mantena:
                feature.extend(Feature_extraction.mantena([y[n] for y in x]))
            else:
                print("Could not match given feature set")
        feature_emg.append(feature)
    for x in w_imu:
        feature = []
        for n in range(9):
            if feature_set == Constant.georgi:
                feature.extend(Feature_extraction.georgi([y[n] for y in x], sensor=Constant.IMU))
            elif feature_set == Constant.rehman:
                feature.extend(Feature_extraction.rehman([y[n] for y in x]))
            elif feature_set == Constant.robinson:
                feature.extend(Feature_extraction.robinson([y[n] for y in x]))
            elif feature_set == Constant.mantena:
                feature.extend(Feature_extraction.mantena([y[n] for y in x]))
            else:
                print("Could not match given feature set")
        feature_imu.append(feature)

    features = []
    for i in range(len(feature_imu)):
        f = []
        for x in np.asarray([feature_emg[i], feature_imu[i]]).flatten('F'):
            f.extend(x)
        features.append(f)
    return features


def pair_devices():
    """
    Pair the two Myo Armbands
    :return:
    """
    global DEVICE_R
    global DEVICE_L
    print("Pair devices - Start")
    with hub.run_in_background(device_listener):
        Helper_functions.wait(.5)
        for i in range(3):  # Three trials to pair
            devices = device_listener.devices
            for device in devices:
                if device.arm == Constant.LEFT:
                    DEVICE_L = device
                    DEVICE_L.stream_emg(True)
                elif device.arm == Constant.RIGHT:
                    DEVICE_R = device
                    DEVICE_R.stream_emg(True)
            if not (DEVICE_L is None) and not (DEVICE_R is None):
                DEVICE_R.vibrate(libmyo.VibrationType.short)
                DEVICE_L.vibrate(libmyo.VibrationType.short)
                print("paired")
                log.info("Devices paired")
                return DEVICE_L, DEVICE_R
            Helper_functions.wait(2)
    hub.stop()
    print("Pair devices - Done")
    return None, None


def reformat_raw_data(emg, ori, acc, gyr):
    """
    reformat the collected raw data for windowing, preprocessing, feature extractino
    :param emg:dict{}
            Dictionary of collected EMG data
    :param ori:dict{}
            Dictionary of collected orientation data
    :param acc:dict{}
            Dictionary of collected accelerometer data
    :param gyr:dict{}
            Dictionary of collected gyroscope data
    :return:list,list
            emg: List of EMG data
            imu: List of IMU data
    """
    o = [[c.x, c.y, c.z] for c in [b[0] for b in [a[1:] for a in ori]]]
    a = [[c.x, c.y, c.z] for c in [b[0] for b in [a[1:] for a in acc]]]
    g = [[c.x, c.y, c.z] for c in [b[0] for b in [a[1:] for a in gyr]]]

    length = len(o)
    if any(len(lst) != length for lst in [a, g]):
        length = min(len(o), len(a), len(g))
        o = o[:length]
        a = a[:length]
        g = g[:length]
    # at least one list has a different length - unknown reason

    imu = []
    for i in range(len(o)):
        tmp = o[i]
        tmp.extend([x for x in a[i]])
        tmp.extend([x for x in g[i]])
        imu.append(tmp)

    emg = [y[0] for y in [x[1:] for x in emg]]
    return emg, imu


def collect_raw_data(record_time):
    """
    Collect the raw data from the myo devices
    :param record_time:int
            Record time
    :return:list<array>, list<array>, list<array>, list<array>
            EMG: Contains the raw data for the emg sensors (ch0...ch7) (measurement data)
            ORI: Contains the raw data for the orientation sensor (X,Y,Z) (measurement data)
            ACC: Contains the raw data for the accelerometer sensor (X,Y,Z) (measurement data)
            GYR: Contains the raw data for the gyroscope sensor (X,Y,Z) (measurement data)
    """
    global EMG
    global ORI
    global ACC
    global GYR
    global status

    EMG, ORI, ACC, GYR = [], [], [], []
    dif, status = 0, 0
    start = time.time()
    while dif <= record_time:
        status = 1
        end = time.time()
        dif = end - start
    status = 0
    return EMG, ORI, ACC, GYR


def window_live_classic(emg, imu, window, overlap):
    """
    Window the emg and imu raw data for a given window size and degree of overlap.
    In this function the windows size is specified to the emg data, the window size for imu will be calculated in this function to ensure that the same timeframe is covered.
    :param emg:list<array>
            The emg raw data
    :param imu:list<array>
            The imu raw data
    :param window:int
            The window length
    :param overlap:float
            The degree of overlap
    :return:matrix, matrix
            w_emg: The windows emg raw data
            w_imu: The windows imu raw data
    """
    window_imu = int(window / (len(emg) / len(imu)))
    offset_imu = window_imu * overlap
    offset_emg = window * overlap
    blocks = int((len(emg) / abs(window - offset_emg)))

    first_emg, first_imu = 0, 0
    w_emg, w_imu = [], []
    for i in range(blocks):
        last_emg = first_emg + window
        last_imu = int(first_imu + window_imu)
        d_emg = emg[first_emg:last_emg]
        d_imu = imu[first_imu:last_imu]

        if not len(d_emg) == window:
            continue
        if not len(d_imu) == window_imu:
            continue
        first_emg += int(window - offset_emg)
        first_imu += int(window_imu - offset_imu)
        w_emg.append(np.asarray(d_emg))
        w_imu.append(np.asarray(d_imu))
    return w_emg, w_imu


def window_live_for_one_sensor(data, window, overlap):
    """
    Window the data for only one sensor. It´s not like the window_live_classic!
    :param data:list
            Data which should be windows
    :param window:int
            Window size
    :param overlap:float
            degree of overlap
    :return:list
            window_data: List of windowed data
    """
    window_data = []
    length = len(data)
    offset = window * overlap
    blocks = int(length / abs(window - offset))

    first = 0
    for i in range(blocks):
        last = int(first + window)
        data_ = data[first:last]
        if not len(data_) == window:
            continue
        first += int(window - offset)
        window_data.append(np.asarray(data_))
        first += int(window - offset)
    return window_data


def live_prediction_cnn(config_emg, config_imu, cnn_emg, cnn_imu, record_time=2):
    """
    Live prediction for two given CNNs (for EMG and IMU data)
    This function perform a endless live prediction. To abort type Ctrl+C in command line.
    :param config_emg:string
            Configuration for the CNN_EMG
    :param config_imu:string
            Configuration for the CNN_IMU
    :param cnn_emg:
            The CNN which predict the result for EMG data
    :param cnn_imu:
            The CNN which predict the result for IMU data
    :param record_time:int
            Record time to collect raw data (EMG and IMU)
    :return:
    """
    pair_devices()

    Helper_functions.wait(2)
    # ------------- Configuration split for CNN EMG -------------
    config_split = config_emg.split('-')
    preprocess_emg = config_split[0]
    w_emg = int(config_split[3])
    overlap_emg = float(config_split[4])

    # ------------- Configuration split for CNN IMU -------------
    config_split = config_imu.split('-')
    w_imu = int(config_split[3])
    overlap_imu = float(config_split[4])
    with hub.run_in_background(gesture_listener.on_event):
        Helper_functions.wait(2)
        Helper_functions.countdown(3)
        DEVICE_R.vibrate(libmyo.VibrationType.short)
        while 1:
            emg, ori, acc, gyr = collect_raw_data(record_time=record_time)
            emg, imu = reformat_raw_data(emg, ori, acc, gyr)
            img_emg = window_live_for_one_sensor(emg, window=w_emg, overlap=overlap_emg)
            img_imu = window_live_for_one_sensor(imu, window=w_imu, overlap=overlap_imu)
            x_emg = np.array(img_emg)[:, :, :, np.newaxis]
            x_imu = np.array(img_imu)[:, :, :, np.newaxis]

            # predict_emg = cnn_emg.predict(x_emg)
            # predict_imu = cnn_imu.predict(x_imu)
            both_pred, emg_pred, imu_pred = prediction_calculation_cnn(cnn_emg.predict_proba(x_emg),
                                                                       cnn_imu.predict_proba(x_imu))
            # print("CNN EMG: ", predict_emg)
            # print("CNN IMU: ", predict_imu)
            print("Both CNNs: ", both_pred,
                  "\nEMG CNNs: ", emg_pred,
                  "\nIMU CNNs: ", imu_pred)
            Helper_functions.wait(0.5)


def live_prediction_classic(config, classifier, record_time=2):
    """
    Live prediction for a given classic classifier
    This function perform a endless live prediction. To abort type Ctrl+C in command line.
    :param config:
    :param classifier:
    :param record_time:
    :return:
    """

    pair_devices()
    # ------------- Configuration split for classic -------------
    config_split = config.split('-')
    preprocess = config_split[0]
    window = int(config_split[3])
    overlap = float(config_split[4])
    feature_set = config_split[5]
    with hub.run_in_background(gesture_listener.on_event):
        Helper_functions.wait(2)
        Helper_functions.countdown(3)
        while 1:
            emg, ori, acc, gyr = collect_raw_data(record_time=record_time)
            emg, imu = reformat_raw_data(emg, ori, acc, gyr)
            w_emg, w_imu = window_live_classic(emg, imu, window, overlap)
            w_emg, w_imu = preprocess_data(w_emg, w_imu, preprocess)
            features = feature_extraction_live(w_emg=w_emg, w_imu=w_imu, feature_set=feature_set)
            prediction = classifier.predict(features)
            print(prediction)
            Helper_functions.wait(0.5)


def prediction_calculation_cnn(emg_prediction, imu_prediction):
    """
    Calculation of the common prediction for two CNNs based on EMG and IMU data.
    :param emg_prediction:
    :param imu_prediction:
    :return:string
            Returns the common prediction
    """
    gesture_both = "No Gesture"
    sum_emg, best_index_emg = sum_sequence_proba(emg_prediction)
    sum_imu, best_index_imu = sum_sequence_proba(imu_prediction)
    best_emg = sum_emg[best_index_emg]
    best_imu = sum_imu[best_index_imu]
    summed_emg_imu = [x + y for x, y in zip(sum_emg, sum_imu)]
    index = np.argmax(summed_emg_imu)
    max_ = max(summed_emg_imu)

    if best_emg >= live_prediction_threshold or best_imu >= live_prediction_threshold:
        if best_index_emg == best_index_imu:
            index = best_index_emg
        elif best_emg > best_imu + Constant.delta or 1 - Constant.delta2 <= best_emg:
            index = best_index_emg
        elif best_imu > best_emg + Constant.delta or 1 - Constant.delta2 <= best_imu:
            index = best_index_imu
        gesture_both = Constant.label_display_without_rest[index]
    elif max_ >= live_prediction_threshold:
        index = np.argmax(summed_emg_imu)
        gesture_both = Constant.label_display_without_rest[index]
    return gesture_both, Constant.labels_without_rest[best_index_emg], Constant.labels_without_rest[best_index_imu]


def sum_sequence_proba(proba):
    """
    Sum the probabilities of the current sequence
    :param proba:list<array>
            List of probabilities for each class (label)
    :return:list, int
            sum_proba: Returns the summed probabilities
            index: best prediction (index)

    """
    sum_proba = []
    for i in range(Constant.classes):
        sum_proba.append(np.mean([float(x[i]) for x in proba]))
        index = np.argmax(sum_proba)
    return sum_proba, index


def validate_models(cnn_emg_ud, cnn_imu_ud, cnn_emg_ui, cnn_imu_ui, classic_ud, config_cnn_emg, config_cnn_imu,
                    config_classic, classic_ui, cnn_imu_adapt=None, cnn_emg_adapt=None, session=2):
    """
    This function perform a live validation on the given classifier. All gestures will record separately for 5 to 2 seconds.
    Record time can be changed in the Constant.py
    :param cnn_emg_ud:
            The user dependent CNN for EMG data
    :param cnn_imu_ud:
            The user dependent CNN for IMU data
    :param cnn_emg_ui:
            The user independent CNN for EMG data
    :param cnn_imu_ui:
            The user independent CNN for IMU data
    :param classic_ud:
            Classic user dependent classifier
    :param classic_ui:
            Classic user independent classifier
    :param config_cnn_emg:
    :param config_cnn_imu:
    :param config_classic:
    :param cnn_imu_adapt:
            Trained adaptive CNN (ideal based on UI CNN) for IMU data
    :param cnn_emg_adapt:
            Trained adaptive CNN (ideal based on UI CNN) for EMG data
    :param session: int, default 2
            Number of sessions
    :return:
    """
    if cnn_imu_adapt is not None and cnn_emg_adapt is not None:
        collect_data_to_train_adapt, collect_data_to_train_classic_live = False, False
    else:
        cnn_imu_adapt = tf.keras.models.clone_model(cnn_imu_ui)
        cnn_emg_adapt = tf.keras.models.clone_model(cnn_emg_ui)
        collect_data_to_train_adapt, collect_data_to_train_classic_live = True, True

    adapt_cnn_emg_train, adapt_cnn_imu_train, y_train, y_train_emg, y_train_imu = [], [], [], [], []
    datum_cnn_emg_number, datum_cnn_imu_number, datum_classic_number, total_raw_emg, total_raw_imu = 0, 0, 0, 0, 0
    classic_ud.verbose, classic_ui.verbose = 0, 0
    classic_live = None

    # ------------- Configuration split for CNN EMG -------------
    config_split = config_cnn_emg.split('-')
    preprocess_emg = config_split[0]
    window_emg = int(config_split[3])
    overlap_emg = float(config_split[4])

    # ------------- Configuration split for CNN IMU -------------
    config_split = config_cnn_imu.split('-')
    preprocess_imu = config_split[0]
    window_imu = int(config_split[3])
    overlap_imu = float(config_split[4])

    # ------------- Configuration split for classic -------------
    config_split = config_classic.split('-')
    preprocess_classic = config_split[0]
    w_classic = int(config_split[3])
    overlap_classic = float(config_split[4])
    feature_set = config_split[5]
    init()

    with hub.run_in_background(gesture_listener.on_event):
        check_samples_rate()
        Helper_functions.wait(2)
        for s in range(session):
            for n in range(len(Constant.seq_duration)):
                input("Press enter to start")
                Helper_functions.cls()
                print(
                    "Start with sequence. \nThe recording time of the gesture is " + str(
                        Constant.seq_duration[n]) + " seconds.")
                for label in range(len(Constant.label_display_without_rest)):
                    Helper_functions.countdown(3)
                    # ----------------------------------Record data----------------------------------------------------#
                    print(Constant.label_display_without_rest[label], "Start")
                    DEVICE_R.vibrate(libmyo.VibrationType.short)
                    emg, ori, acc, gyr = collect_raw_data(record_time=Constant.seq_duration[n])
                    total_raw_emg += len(emg)
                    total_raw_imu += len(ori)
                    DEVICE_L.vibrate(libmyo.VibrationType.short)
                    print("Stop")
                    emg, imu = reformat_raw_data(emg, ori, acc, gyr)

                    # ----------------------------------Perform classic prediction-------------------------------------#
                    # Classic windowing, EMG and IMU together
                    image_emg, image_imu = window_live_classic(emg, imu, w_classic, overlap_classic)

                    # Preprocessing
                    image_emg, image_imu = preprocess_data(image_emg, image_imu, preprocess_classic)

                    # Feature extraction
                    features = feature_extraction_live(w_emg=image_emg, w_imu=image_imu, feature_set=feature_set)
                    datum_classic_number += len(features)

                    if collect_data_to_train_classic_live:
                        x_train_classic.extend(features)
                        y_train.extend([label] * len(features))

                    predict_classic_ud = classic_ud.predict(features)
                    predict_classic_ui = classic_ui.predict(features)

                    # If live classifier is trained
                    if classic_live is not None:
                        predict_classic_live = classic_live.predict(features)

                    # ----------------------------------Perform CNN prediction-----------------------------------------#
                    # Separate windowing
                    image_emg = window_live_for_one_sensor(emg, window=window_emg, overlap=overlap_emg)
                    image_imu = window_live_for_one_sensor(imu, window=window_imu, overlap=overlap_imu)

                    w_emg = preprocess_data(image_emg, preprocess_emg, preprocess_emg)
                    w_imu = preprocess_data(image_imu, preprocess_imu, preprocess_emg)

                    datum_cnn_emg_number += len(image_emg)
                    datum_cnn_imu_number += len(image_imu)

                    if collect_data_to_train_adapt:
                        adapt_cnn_emg_train.extend(image_emg)
                        adapt_cnn_imu_train.extend(image_imu)
                        y_train_emg.extend([label] * len(image_emg))
                        y_train_imu.extend([label] * len(image_imu))

                    x_emg = np.array(image_emg)[:, :, :, np.newaxis]
                    x_imu = np.array(image_imu)[:, :, :, np.newaxis]

                    proba_emg_ud = cnn_emg_ud.predict_proba(x_emg)
                    proba_imu_ud = cnn_imu_ud.predict_proba(x_imu)
                    predict_emg_ud = cnn_emg_ud.predict_classes(x_emg)
                    predict_imu_ud = cnn_imu_ud.predict_classes(x_imu)

                    proba_emg_ui = cnn_emg_ui.predict_proba(x_emg)
                    proba_imu_ui = cnn_imu_ui.predict_proba(x_imu)
                    predict_emg_ui = cnn_emg_ui.predict_classes(x_emg)
                    predict_imu_ui = cnn_imu_ui.predict_classes(x_imu)

                    if cnn_imu_adapt is not None and cnn_emg_adapt is not None:
                        proba_emg_adapt = cnn_emg_adapt.predict_proba(x_emg)
                        proba_imu_adapt = cnn_imu_adapt.predict_proba(x_imu)

                        predict_emg_adapt = cnn_emg_adapt.predict_classes(x_emg)
                        predict_imu_adapt = cnn_imu_adapt.predict_classes(x_imu)

                    # ----------------------------------Evaluate results-----------------------------------------------#
                    print("EVAL...")
                    eval_predictions(predict=predict_emg_ud, proba=proba_emg_ud, y_true=label, file_prefix="cnn_emg_ud",
                                     session=s, seq=n, classic=False)
                    eval_predictions(predict=predict_imu_ud, proba=proba_imu_ud, y_true=label, file_prefix="cnn_imu_ud",
                                     session=s, seq=n, classic=False)
                    eval_predictions(predict=predict_emg_ui, proba=proba_emg_ui, y_true=label, file_prefix="cnn_emg_ui",
                                     session=s, seq=n, classic=False)
                    eval_predictions(predict=predict_imu_ui, proba=proba_imu_ui, y_true=label, file_prefix="cnn_imu_ui",
                                     session=s, seq=n, classic=False)
                    eval_predictions(predict=predict_classic_ud, proba=[], y_true=label, file_prefix="classic_ud",
                                     session=s, seq=n, classic=True)
                    eval_predictions(predict=predict_classic_ui, proba=[], y_true=label, file_prefix="classic_ui",
                                     session=s, seq=n, classic=True)

                    if not collect_data_to_train_adapt:
                        eval_predictions(predict=predict_emg_adapt, proba=proba_emg_adapt, y_true=label,
                                         file_prefix="cnn_emg_adapt", session=s, seq=n, classic=False)
                        eval_predictions(predict=predict_imu_adapt, proba=proba_imu_adapt, y_true=label,
                                         file_prefix="cnn_imu_adapt", session=s, seq=n, classic=False)

                    if not collect_data_to_train_classic_live:
                        eval_predictions(predict=predict_classic_live, proba=[], y_true=label,
                                         file_prefix="classic_live",
                                         session=s, seq=n, classic=True)

                if collect_data_to_train_classic_live:
                    print("Train classic")
                    print("Training Samples", len(x_train_classic))
                    classic_live = Constant.random_forest.fit(x_train_classic, y_train)
                    collect_data_to_train_classic_live = False

                if collect_data_to_train_adapt:
                    print("Train Adapt CNN EMG")
                    cnn_emg_adapt = adapt_model_for_user(model=cnn_emg_adapt, x_train=adapt_cnn_emg_train,
                                                         y_train=y_train_emg, x_test_in=[], y_test_in=[],
                                                         save_path="./", batch=8, epochs=10, file_name="live-adapt-EMG",
                                                         calc_test_set=False)
                    print("Train Adapt CNN IMU")
                    cnn_imu_adapt = adapt_model_for_user(model=cnn_imu_adapt, x_train=adapt_cnn_imu_train,
                                                         y_train=y_train_imu, x_test_in=[], y_test_in=[],
                                                         save_path="./", batch=8, epochs=10, file_name="live-adapt-IMU",
                                                         calc_test_set=False)
                    collect_data_to_train_adapt = False
    hub.stop()


if __name__ == '__main__':
    main()

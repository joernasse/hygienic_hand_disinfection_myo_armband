import collections
import csv
import os
import pickle
import threading
import time
from myo import init, Hub, StreamEmg
import myo as libmyo
import logging as log
import Constant
import Feature_extraction
import Helper_functions
import Process_data
import numpy as np
from tensorflow.python.keras.models import load_model
from itertools import groupby
import collections

from Classification import norm_data

classic_clf_paths = "G:/Masterarbeit/user_dependent_detail/User001Random_Forest_User_dependentseparate-EMGIMU-100-0.75-georgi.joblib"
imu_cnn_path = "G:/Masterarbeit/deep_learning/CNN_final_results/training_kaggle_imu_0"
emg_cnn_path = "G:/Masterarbeit/deep_learning/CNN_final_results/training_kaggle_emg_0"

cnn_emg_paths = ["./Live_Prediction/Load_model/User001_UD_no_pre_pro-separate-EMG-100-0.9-NA_cnn_CNN_Kaggle.h5",
                 "./Live_Prediction/Load_model/User001_UI_no_pre_pro-separate-EMG-100-0.9-norm-NA_cnn_kaggle.h5"]

cnn_imu_paths = ["./Live_Prediction/Load_model/User001_UD_no_pre_pro-separate-IMU-25-0.9-NA_cnn_CNN_Kaggle.h5",
                 "./Live_Prediction/Load_model/User001_UI_no_pre_pro-separate-IMU-25-0.9-norm-NA_cnn_kaggle.h5"]

classic_paths = [
    "./Live_Prediction/Load_model/User001_UD_Random Forest_no_pre_pro-separate-EMGIMU-100-0.9-rehman_norm.joblib",
    "./Live_Prediction/Load_model/User001_UI_Random Forest_no_pre_pro-separate-EMGIMU-100-0.9-rehman_norm.joblib"]

DEVICE_L, DEVICE_R = None, None
EMG = []  # emg
ORI = []  # orientation
GYR = []  # gyroscope
ACC = []  # accelerometer
emg_l, emg_r = [], []
tmp = []
status = 0
imu_load_data = {"timestamp": [],
                 "x_ori": [], "y_ori": [], "z_ori": [],
                 "x_gyr": [], "y_gyr": [], "z_gyr": [],
                 "x_acc": [], "y_acc": [], "z_acc": [],
                 "label": []}
emg_dict = {"timestamp": [],
            "ch0": [], "ch1": [], "ch2": [], "ch3": [], "ch4": [], "ch5": [], "ch6": [], "ch7": [],
            "label": []}

sequence_duration = [2, 5, 4, 3, 2]
cnn_emg_models = []
cnn_imu_models = []
classic_models = []
live_prediction_path = "./Live_Prediction/"
collect_for_train_random_forest = True
x_train = []


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


def check_samples_rate(n=5, record_time=1):
    print("Check samples rate - Start")
    sum_emg, sum_imu = 0, 0
    with hub.run_in_background(gesture_listener.on_event):
        print("Warm up")
        Helper_functions.wait(2)
        for i in range(n):
            emg, ori, acc, gyr = collect_raw_data(record_time=record_time)
            sum_emg += len(emg)
            sum_imu += len(ori)
            print("EMG length", len(emg),
                  "\nIMU length", len(ori))
        hub.stop()

    max_emg = (n * 2 * record_time * 200)
    max_imu = (n * 2 * record_time * 50)
    sr_emg = (sum_emg / max_emg) * 100
    sr_imu = (sum_imu / max_imu) * 100
    print("DS EMG", sr_emg,
          "\nDS_IMU", sr_imu)
    if (n * 2 * 200) * 0.85 > sum_emg:
        print("EMG sample rate under 85%")

    if (n * 2 * 50) * 0.85 > sum_imu:
        print("IMU sample rate under 85%")

    print("Check samples rate - Done")
    return


def init(live_prediction_path="./Live_Prediction"):
    global status
    global cnn_emg_models
    global cnn_imu_models
    global classic_models

    if not os.path.isdir(live_prediction_path):  # Collection dir
        os.mkdir(live_prediction_path)

    status = 1
    print("Initialization - Start")
    Helper_functions.wait(3)
    pair_devices()

    for path in cnn_emg_paths:
        cnn_emg_models.append(load_model(path))
    for path in cnn_imu_paths:
        cnn_imu_models.append(load_model(path))
    for path in classic_paths:
        with open(path, 'rb') as pickle_file:
            classic_models.append(pickle.load(pickle_file))
    # wait(4)
    status = 0
    check_samples_rate()
    print("Initialization - Done")


def preprocess_data(w_emg, w_imu, preprocess):
    """

    :param w_emg:
    :param w_imu:
    :param preprocess:
    :return:
    """
    if preprocess == Constant.filter_:
        return Process_data.filter_emg_data(w_emg, preprocess)
    elif preprocess == Constant.z_norm:
        return Process_data.z_norm(w_emg, w_imu)
    else:
        return w_emg, w_imu


def main():
    global cnn_emg_models
    global cnn_imu_models
    global classic_models
    global collect_for_train_random_forest
    global x_train
    y_train = []
    live_classifier = None

    preprocess = Constant.no_pre_processing
    init()
    Helper_functions.wait(2)

    with hub.run_in_background(gesture_listener.on_event):
        for seq in sequence_duration:
            for label in range(len(Constant.label_display_without_rest)):
                # ----------------------------------Record data--------------------------------------------------------#
                print("Start with sequence. \nThe recording time of the gesture is " + str(seq) + " seconds.")
                Helper_functions.countdown(3)
                print(Constant.label_display_without_rest[label], "Start")
                DEVICE_R.vibrate(libmyo.VibrationType.short)
                emg, ori, acc, gyr = collect_raw_data(record_time=seq)
                DEVICE_L.vibrate(libmyo.VibrationType.short)
                print("Stop")

                emg, imu = reformat_raw_data(emg, ori, acc, gyr)
                # ----------------------------------Perform classic prediction-----------------------------------------#
                window = 100
                overlap = 0.9
                # Classic windowing, EMG and IMU together
                w_emg, w_imu = window_live_classic(emg, imu, window, overlap)

                # Preprocessing
                w_emg, w_imu = preprocess_data(w_emg, w_imu, preprocess)

                # Feature extraction
                mode = Constant.rehman
                features = feature_extraction_live(w_emg=w_emg, w_imu=w_imu, mode=mode)

                # Norm
                norm = True
                if norm:
                    features = norm_data(features)
                    if collect_for_train_random_forest:
                        x_train.extend(features)
                        y_train.extend([label] * len(features))

                predict_classic = [clf.predict(features) for clf in classic_models]

                # IF live classifier is trained
                if not collect_for_train_random_forest and live_classifier is not None:
                    predict_live_classifier = live_classifier.predict(features)

                # ----------------------------------Perform CNN prediction---------------------------------------------#
                # Separate windowing

                w_emg = 100
                w_imu = 25
                overlap = 0.9
                w_emg = window_live_separate(emg, window=w_emg, overlap=overlap)
                w_imu = window_live_separate(imu, window=w_imu, overlap=overlap)

                x_emg = np.array(w_emg)[:, :, :, np.newaxis]
                x_imu = np.array(w_imu)[:, :, :, np.newaxis]

                proba_emg = [clf.predict_proba(x_emg) for clf in cnn_emg_models]
                proba_imu = [clf.predict_proba(x_imu) for clf in cnn_imu_models]

                predict_emg = [clf.predict_classes(x_emg) for clf in cnn_emg_models]
                predict_imu = [clf.predict_classes(x_imu) for clf in cnn_imu_models]

                # ----------------------------------Evaluate results---------------------------------------------------#
                evaluate_predictions_1(predict=predict_emg,
                                       proba=proba_emg,
                                       y_true=label,
                                       log_file_name="label_" + str(label) + "_emg_prediction.csv",
                                       clf_names=cnn_emg_paths)
                evaluate_predictions_1(predict=predict_imu,
                                       proba=proba_imu,
                                       y_true=label,
                                       log_file_name="label_" + str(label) + "_imu_prediction.csv",
                                       clf_names=cnn_imu_paths)
                evaluate_predictions_1(predict=predict_classic,
                                       proba=[],
                                       y_true=label,
                                       log_file_name="label_" + str(label) + "_classic_prediction.csv",
                                       classic=True,
                                       clf_names=classic_paths)
                if live_classifier is not None:
                    evaluate_predictions_1(predict=predict_live_classifier,
                                           proba=[], y_true=label,
                                           log_file_name="Live Classifier",
                                           clf_names=["Random Forest Live"],
                                           classic=True)

                Helper_functions.countdown(3)

            if collect_for_train_random_forest:
                collect_for_train_random_forest = False
                print("Training Samples", len(x_train))
                live_classifier = Constant.random_forest.fit(x_train, y_train)
                del x_train
                del y_train
    hub.stop()


def evaluate_predictions_1(predict, proba, y_true, log_file_name, clf_names, classic=False):
    save_path = live_prediction_path + log_file_name
    file = open(save_path, 'a', newline='')
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    with file:
        for n in range(len(predict)):
            if classic:
                writer.writerow([clf_names[n]])
                sum_pred_classic_dict = collections.OrderedDict(sorted(collections.Counter(predict[n]).items()))
                index_max_classic = max(sum_pred_classic_dict, key=sum_pred_classic_dict.get)
                writer.writerow(["label", "predict_sum"])
                for key in sum_pred_classic_dict.keys():
                    writer.writerow([key, sum_pred_classic_dict[key]])
                writer.writerow(["best_label", ])
                writer.writerow(
                    [index_max_classic, sum_pred_classic_dict[index_max_classic], [x for x in sum_pred_classic_dict]])
                writer.writerow([Constant.write_separater])

            else:
                sum_proba = []
                for i in range(Constant.classes):
                    sum_proba.append(np.mean([float(x[i]) for x in proba[n]]))
                writer.writerow([clf_names[n]])
                writer.writerow(["predict", "probabilities", "true_label"])
                for j in range(len(predict[n])):
                    writer.writerow([predict[n][j], [x for x in proba[n][j]], y_true])
                    # number_prediction = collections.Counter([x for x in predict])
                    index_max_emg = np.argmax(sum_proba)
                writer.writerow([index_max_emg, [x for x in sum_proba]])
                writer.writerow([Constant.write_separater])
    file.close()


def evaluate_predictions(cnn_proba_emg, cnn_pred_emg, cnn_proba_imu, cnn_pred_imu, classic_pred, y_true,
                         live_prediction_path):
    timestamp = str(time.time())
    cnn_emg_path = live_prediction_path + timestamp + "emg_prediction.csv"
    cnn_imu_path = live_prediction_path + timestamp + "imu_prediction.csv"
    classic_path = live_prediction_path + timestamp + "classic_prediction.csv"
    sequence_path = live_prediction_path + timestamp + "sequence_prediction.csv"

    for n in range(len(cnn_proba_emg)):
        # Save prediction results for each sample
        write_prediction_to_file(cnn_emg_path, cnn_pred_emg[n], cnn_proba_emg[n], y_true)
        write_prediction_to_file(cnn_imu_path, cnn_pred_imu[n], cnn_proba_imu[n], y_true)
        write_prediction_to_file(classic_path, classic_pred[n], [], y_true)

        sum_proba_emg, sum_proba_imu = [], []
        for i in range(Constant.classes):
            sum_proba_emg.append(np.mean([float(x[i]) for x in cnn_proba_emg[n]]))
            sum_proba_imu.append(np.mean([float(x[i]) for x in cnn_proba_imu[n]]))
        sum_pred_classic_dict = collections.Counter(classic_pred[n])
        sum_proba_classic = np.divide(list(sum_pred_classic_dict.values()), len(classic_pred[n]))

        index_max_emg = np.argmax(sum_proba_emg)
        index_max_imu = np.argmax(sum_proba_imu)
        index_max_classic = max(sum_pred_classic_dict, key=sum_pred_classic_dict.get)

        # Save combined results for each classifier
        write_prediction_to_file(cnn_emg_path, sum_proba_emg)
        write_prediction_to_file(cnn_imu_path, sum_proba_imu)
        file = open(classic_path, 'a', newline='')
        with file:
            for key in sum_pred_classic_dict.keys():
                writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([key, sum_pred_classic_dict[key]])
            writer.writerow([Constant.write_separater])
        file.close()

        print("EMG prediction:", Constant.label_display_without_rest[index_max_emg],
              "\nIMU prediction:", Constant.label_display_without_rest[index_max_imu],
              "\nClassic prediction", Constant.label_display_with_rest[index_max_classic])

        if not index_max_imu == index_max_emg:
            if np.abs(sum_proba_emg[index_max_emg] - sum_proba_imu[index_max_imu]) >= 0.1:
                if sum_proba_emg[index_max_emg] > sum_proba_imu[index_max_imu]:
                    final_choice = index_max_emg
                else:
                    final_choice = index_max_imu
        else:
            final_choice = index_max_emg
        print("Final choice:", Constant.label_display_without_rest[final_choice])

        write_prediction_to_file(sequence_path, ["CNN", final_choice], y_true)
        # write_prediction_to_file(sequence_path, ["Classic", index_max_classic], y_true)


def write_prediction_to_file(file_path, y_pred=None, y_prob=None, y_true=None):
    file = open(file_path, 'a', newline='')
    with file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["predict", "probabilities", "true_label"])
        for i in range(len(y_pred)):
            writer.writerow([y_pred[i], [x for x in y_prob[i]], y_true])
    file.close()


def most_common(lst):
    return max(set(lst), key=lst.count)


def feature_extraction_live(w_emg, w_imu, mode=Constant.mantena):
    feature_emg, feature_imu = [], []
    for x in w_emg:
        feature = []
        for n in range(8):
            if mode == Constant.georgi:
                feature.extend(Feature_extraction.georgi([y[n] for y in x], sensor=Constant.EMG))
            if mode == Constant.rehman:
                feature.extend(Feature_extraction.rehman([y[n] for y in x]))
            else:
                feature.extend(Feature_extraction.mantena([y[n] for y in x]))
        feature_emg.append(feature)
    for x in w_imu:
        feature = []
        for n in range(9):
            if mode == Constant.georgi:
                feature.extend(Feature_extraction.georgi([y[n] for y in x], sensor=Constant.IMU))
            if mode == Constant.rehman:
                feature.extend(Feature_extraction.rehman([y[n] for y in x]))
            else:
                feature.extend(Feature_extraction.mantena([y[n] for y in x]))
        feature_imu.append(feature)

    features = []
    for i in range(len(feature_imu)):
        f = []
        for x in np.asarray([feature_emg[i], feature_imu[i]]).flatten('F'):
            f.extend(x)
        features.append(f)
    return features


def pair_devices():
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


def reformat_raw_data(emg=[], ori=[], acc=[], gyr=[]):
    # print(len(emg))
    # print(len(ori))
    o = [[c.x, c.y, c.z] for c in [b[0] for b in [a[1:] for a in ori]]]
    a = [[f.x, f.y, f.z] for f in [e[0] for e in [d[1:] for d in acc]]]
    g = [[j.x, j.y, j.z] for j in [h[0] for h in [g[1:] for g in gyr]]]
    Helper_functions.wait(0.5)
    imu = []
    for i in range(len(o)):
        # todo check warum teilweise nicht gleichlang
        tmp = o[i]
        tmp.extend([x for x in a[i]])
        tmp.extend([x for x in g[i]])
        imu.append(tmp)

    emg = [y[0] for y in [x[1:] for x in emg]]
    return emg, imu


def collect_raw_data(record_time):
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
    log.info("EMG %d", len(EMG))
    log.info("IMU %d", len(ORI))

    # Only for count length
    # emg_count_list.append(len(EMG))
    # imu_count_list.append(len(ORI))
    return EMG, ORI, ACC, GYR


def window_live_classic(emg, imu, window, overlap):
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
            # print("first/last", first_emg, last_emg)
            continue
        if not len(d_imu) == window_imu:
            # print("first/last", first_imu, last_imu)
            continue
        first_emg += int(window - offset_emg)
        first_imu += int(window_imu - offset_imu)
        w_emg.append(np.asarray(d_emg))
        w_imu.append(np.asarray(d_imu))
    return w_emg, w_imu


def window_live_separate(raw_data, window, overlap):
    window_data = []
    # window EMG data
    length = len(raw_data)
    offset = window * overlap
    blocks = int(length / abs(window - offset))

    first = 0
    for i in range(blocks):
        data = []
        last = int(first + window)
        data = raw_data[first:last]
        if not len(data) == window:
            # print("first/last", first, last)
            continue
        first += int(window - offset)
        window_data.append(np.asarray(data))
        first += int(window - offset)
    return window_data


if __name__ == '__main__':
    main()

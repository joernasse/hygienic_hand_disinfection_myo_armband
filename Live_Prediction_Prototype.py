import collections
import csv
import os
import pickle
import threading
import time
from myo import init, Hub, StreamEmg
import myo as libmyo
import logging as log
# import Collect_data
import Constant
import Feature_extraction
import Process_data
from Helper_functions import wait
import numpy as np
from tensorflow.python.keras.models import load_model

classic_clf_path = "G:/Masterarbeit/user_dependent_detail/User001Random_Forest_User_dependentseparate-EMGIMU-100-0.75-georgi.joblib"
imu_cnn_path = "G:/Masterarbeit/deep_learning/CNN_final_results/training_kaggle_imu_0"
emg_cnn_path = "G:/Masterarbeit/deep_learning/CNN_final_results/training_kaggle_emg_0"

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
        wait(2)
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
    if (n * 2 * 200) * 0.9 > sum_emg:
        print("EMG sample rate under 90%")

    if (n * 2 * 50) * 0.9 > sum_imu:
        print("IMU sample rate under 90%")

    print("Check samples rate - Done")
    return


def init():
    global status
    status = 1
    print("Initialization - Start")
    wait(3)
    dev_l, dev_r = pair_devices()
    wait(4)
    status = 0
    print("Initialization - Done")
    check_samples_rate()
    # load models
    # if mode == 'classic':
    #     classic = True
    #     with open(classic_clf_path, 'rb') as pickle_file:
    #         model = pickle.load(pickle_file)
    # else:
    #     classic = False
    #     model_imu = load_model(imu_cnn_path)
    #     model_emg = load_model(emg_cnn_path)
    return dev_l, dev_r


def main():
    cnn_imu_model = load_model("./no_pre_pro-separate-IMU-25-0.9-NA_cnn_CNN_Kaggle.h5")
    cnn_emg_mode = load_model("./no_pre_pro-separate-EMG-100-0.9-NA_cnn_CNN_Kaggle.h5")
    live_prediction_path = "./Live_Prediction"
    if not os.path.isdir(live_prediction_path):  # Collection dir
        os.mkdir(live_prediction_path)

    mode = 'cnn'
    preprocess = Constant.no_pre_processing
    dev_l, dev_r = init()
    record_time = 1
    classic = False
    classic_model = None
    wait(2)
    with hub.run_in_background(gesture_listener.on_event):
        for label in range(len(Constant.label_display_without_rest)):
            print(Constant.label_display_without_rest[label], "Start")
            emg, ori, acc, gyr = collect_raw_data(record_time=record_time)
            print("Stop")

            emg, imu = reformat_raw_data(emg=emg, ori=ori, acc=acc, gyr=gyr)
            if classic:
                window = 100
                overlap = 0.9
                # Classic windowing, emg and imu together
                w_emg, w_imu = window_live_classic(emg, imu, window, overlap)

                # Feature extraction
                mode = Constant.georgi
                features = feature_extraction_live(w_emg=w_emg, w_imu=w_imu, mode=mode)
                y_predict = classic_model.predict(features)

            else:
                # Separate windowing
                w_emg = 100
                w_imu = 25
                overlap = 0.9
                w_emg = window_live_separate(emg, window=w_emg, overlap=overlap)
                w_imu = window_live_separate(imu, window=w_imu, overlap=overlap)

                x_emg = np.array(w_emg)[:, :, :, np.newaxis]
                x_imu = np.array(w_imu)[:, :, :, np.newaxis]

                proba_emg = cnn_emg_mode.predict_proba(x_emg)
                proba_imu = cnn_imu_model.predict_proba(x_imu)

                predict_emg = cnn_emg_mode.predict_classes(x_emg)
                predict_imu = cnn_imu_model.predict_classes(x_imu)

                evaluate_predictions(proba_emg, predict_emg, proba_imu, predict_imu, label, live_prediction_path)
                wait(3)
    hub.stop()


def evaluate_predictions(proba_emg, pred_emg, proba_imu, pred_imu, y_true, live_prediction_path):
    sum_proba_emg, sum_proba_imu = [], []
    for i in range(12):
        sum_proba_emg.append(np.mean([float(x[i]) for x in proba_emg]))
        sum_proba_imu.append(np.mean([float(x[i]) for x in proba_imu]))

    index_max_emg = np.argmax(sum_proba_emg)
    index_max_imu = np.argmax(sum_proba_imu)
    print("EMG prediction:", Constant.label_display_without_rest[index_max_emg],
          "\nIMU preddiction:", Constant.label_display_without_rest[index_max_imu])

    if not index_max_imu == index_max_emg:
        if np.abs(sum_proba_emg[index_max_emg] - sum_proba_imu[index_max_imu]) >= 0.1:
            if sum_proba_emg[index_max_emg] > sum_proba_imu[index_max_imu]:
                final_choice = index_max_emg
            else:
                final_choice = index_max_imu
    else:
        final_choice = index_max_emg
    print("Final choice:", Constant.label_display_without_rest[final_choice])

    emg_pred_correct, imu_pred_correct = 0, 0
    timestamp = str(time.time())
    f_emg = open(live_prediction_path + timestamp + "emg_prediction.csv", 'a', newline='')
    f_imu = open(live_prediction_path + timestamp + "emg_prediction.csv", 'a', newline='')
    for p in pred_emg:
        if p == y_true:
            emg_pred_correct += 1
            with f_emg:
                writer = csv.writer(f_emg, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([p, y_true])
        f_emg.close()

    for p in pred_imu:
        if p == y_true:
            imu_pred_correct += 1
            with f_imu:
                writer = csv.writer(f_emg, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([p, y_true])
    f_imu.close()


def most_common(lst):
    return max(set(lst), key=lst.count)


def feature_extraction_live(w_emg, w_imu, mode=Constant.mantena):
    feature_emg, feature_imu = [], []
    for x in w_emg:
        if mode == Constant.georgi:
            feature_emg.append(Feature_extraction.georgi(x, sensor=Constant.EMG))
        else:
            feature_emg.append(Feature_extraction.mantena(x))
    for x in w_imu:
        if mode == Constant.georgi:
            feature_imu.append(Feature_extraction.georgi(x, sensor=Constant.IMU))
        else:
            feature_imu.append(Feature_extraction.mantena(x))

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
        wait(.5)
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
            wait(2)
    hub.stop()
    print("Pair devices - Done")
    return None, None


def reformat_raw_data(emg=[], ori=[], acc=[], gyr=[]):
    # print(len(emg))
    # print(len(ori))
    o = [[c.x, c.y, c.z] for c in [b[0] for b in [a[1:] for a in ori]]]
    a = [[f.x, f.y, f.z] for f in [e[0] for e in [d[1:] for d in acc]]]
    g = [[j.x, j.y, j.z] for j in [h[0] for h in [g[1:] for g in gyr]]]
    imu = []
    for i in range(len(o)):
        #todo check warum teilweise nicht gleichlang
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
            print("first/last", first_emg, last_emg)
            continue
        if not len(d_imu) == window_imu:
            print("first/last", first_imu, last_imu)
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
            print("first/last", first, last)
            continue
        first += int(window - offset)
        window_data.append(np.asarray(data))
        first += int(window - offset)
    return window_data


if __name__ == '__main__':
    main()

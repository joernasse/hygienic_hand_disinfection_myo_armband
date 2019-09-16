import collections
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


def init():
    wait(3)
    dev_l, dev_r = pair_devices()
    wait(3)
    return dev_l, dev_r


def main(mode='classic'):
    window = 100
    overlap = 0.9
    preprocess = Constant.no_pre_processing
    dev_l, dev_r = init()
    record_time = 1

    if mode == 'classic':
        classic = True
        with open(classic_clf_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
    else:
        classic = False
        model_imu = load_model(imu_cnn_path)
        model_emg = load_model(emg_cnn_path)

    with hub.run_in_background(gesture_listener.on_event):
        emg, ori, acc, gyr = collect_raw_data(record_time=record_time)
    hub.stop()

    # Reformat & Windowing
    w_emg, w_imu = reformat_and_window_live(emg=emg, ori=ori, acc=acc, gyr=gyr, window=window, overlap=overlap,
                                            classic=classic)

    # Feature Extraction
    if classic:
        feature_emg, feature_imu = feature_extraction_live(w_emg=w_emg, w_imu=w_imu, mode=Constant.georgi)
        features = []
        for i in range(len(feature_imu)):
            tmp = np.asarray([feature_emg[i], feature_imu[i]]).flatten('F')
            f = []
            for x in np.asarray([feature_emg[i], feature_imu[i]]).flatten('F'):
                f.extend(x)
            features.append(f)
    else:
        x_emg = np.array(w_emg)[:, :, :, np.newaxis]
        x_imu = np.array(w_imu)[:, :, :, np.newaxis]


    y_predict = model.predict(features)
    print(y_predict)


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
    return feature_emg, feature_imu


def pair_devices():
    global DEVICE_R
    global DEVICE_L
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
    return None, None


def reformat_and_window_live(window, overlap, emg=[], ori=[], acc=[], gyr=[], classic=True):
    print(len(emg))
    print(len(ori))
    o = [[q.x, q.y, q.z] for q in [y[0] for y in [x[1:] for x in ori]]]
    a = [[q.x, q.y, q.z] for q in [y[0] for y in [x[1:] for x in acc]]]
    g = [[q.x, q.y, q.z] for q in [y[0] for y in [x[1:] for x in gyr]]]
    imu = []
    for i in range(len(o)):
        tmp = o[i]
        tmp.extend([x for x in a[i]])
        tmp.extend([x for x in g[i]])
        imu.append(tmp)

    emg = [y[0] for y in [x[1:] for x in emg]]

    if classic:
        # Classic windowing, emg and imu together
        window_emg, window_imu = window_live_classic(emg, imu, window, overlap)
    else:
        # Separate windowing
        window_emg = window_live_separate(emg, window=window, overlap=overlap)
        window_imu = window_live_separate(imu, window=window, overlap=overlap)

    return window_emg, window_imu


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
    print(dif)
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

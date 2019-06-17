import collections
import os
import threading
import time

import numpy as np

from myo import init, Hub, StreamEmg
import myo as libmyo

# const
from Data_transformation import transform_data_collection
from Helper_functions import countdown, cls
from Save_Load import save_csv, save_raw_csv

TRAINING_TIME: int = 2
PREDICT_TIME: float = 2.5
DATA_POINT_WINDOW_SIZE = 20
EMG_INTERVAL = 0.01
POSITION_INTERVAL = 0.04
COLLECTION_DIR = "Collections"

RIGHT = "right"
LEFT = "left"

WINDOW_EMG = 20
DEGREE_OF_OVERLAP = 0.5
OFFSET_EMG = WINDOW_EMG * DEGREE_OF_OVERLAP
SCALING_FACTOR_IMU_DESKTOP = 3.815  # calculated value at desktop PC, problems with Bluetooth connection 3.815821888279855
WINDOW_IMU = WINDOW_EMG / SCALING_FACTOR_IMU_DESKTOP
OFFSET_IMU = WINDOW_IMU * DEGREE_OF_OVERLAP

TIME_NOW = time.localtime()
TIMESTAMP = str(TIME_NOW.tm_year) + str(TIME_NOW.tm_mon) + str(TIME_NOW.tm_mday) + str(TIME_NOW.tm_hour) + str(
    TIME_NOW.tm_min) + str(TIME_NOW.tm_sec)

status = 0

DEVICE = []
EMG = []  # emg
ORI = []  # orientation
GYR = []  # gyroscope
ACC = []  # accelerometer


class GestureListener(libmyo.DeviceListener):
    def __init__(self, queue_size=1):
        super(GestureListener, self).__init__()
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)
        self.ori_data_queue = collections.deque(maxlen=queue_size)

    def on_connected(self, event):
        event.device.stream_emg(StreamEmg.enabled)

    def on_emg(self, event):
        with self.lock:
            if status:
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
listener = GestureListener()


def collect_raw_data(record_duration):
    global EMG
    global ORI
    global ACC
    global GYR
    global status
    global WINDOW_IMU
    global OFFSET_IMU
    EMG, ORI, ACC, GYR = [], [], [], []
    raw_data = {'EMG': EMG,
                'ORI': ORI,
                'GYR': GYR,
                'ACC': ACC}
    raw_data_window = {'EMG': [],
                       'ORI': [],
                       'GYR': [],
                       'ACC': []}
    dif = 0
    start = time.time()
    while dif <= record_duration:
        status = 1
        end = time.time()
        dif = end - start
    status = 0

    WINDOW_IMU = WINDOW_EMG / (len(EMG) / len(ORI))
    OFFSET_IMU = WINDOW_IMU * DEGREE_OF_OVERLAP

    # define blocks for EMG
    blocks = int(len(EMG) / abs(WINDOW_EMG - OFFSET_EMG))
    first = 0
    for i in range(blocks):
        last = first + WINDOW_EMG
        raw_data_window['EMG'].append(np.asarray(EMG[first:last]))
        first += int(WINDOW_EMG - OFFSET_EMG)

    # define blocks for IMU
    blocks = int(len(ORI) / abs(WINDOW_IMU - OFFSET_IMU))
    first = 0
    for i in range(blocks):
        last = int(first + WINDOW_IMU)
        raw_data_window['ORI'].append(np.asarray(ORI[first:last]))
        raw_data_window['GYR'].append(GYR[first:last])
        raw_data_window['ACC'].append(ACC[first:last])
        first += int(WINDOW_IMU - OFFSET_IMU)

    return raw_data_window, raw_data


def collect_training_data(label_display, probant="defaultUser", session=10):
    global status
    user_path = COLLECTION_DIR + "/" + probant
    if not os.path.isdir(COLLECTION_DIR):
        os.mkdir(COLLECTION_DIR)
    if not os.path.isdir(user_path):
        os.mkdir(user_path)

    time.sleep(1)
    status = 0
    label_window, label_raw = [], []
    raw_data_window = {'EMG': [],
                       'ORI': [],
                       'GYR': [],
                       'ACC': []}
    raw_data_original = {'EMG': [],
                         'ORI': [],
                         'GYR': [],
                         'ACC': []}

    with hub.run_in_background(listener.on_event):
        for j in range(session):
            for i in range(len(label_display)):
                print("\nGesture -- ", label_display[i], " : Ready?")
                input("Countdown start after press...")
                countdown(2)
                cls()
                print("Start")
                time.sleep(.5)
                tmp_data_window, tmp_data_raw = collect_raw_data(TRAINING_TIME)

                # WINDOW data
                entries = len(tmp_data_window['EMG'])
                raw_data_window['EMG'].extend(tmp_data_window['EMG'])
                raw_data_window['ACC'].extend(tmp_data_window['ACC'])
                raw_data_window['GYR'].extend(tmp_data_window['GYR'])
                raw_data_window['ORI'].extend(tmp_data_window['ORI'])
                label_window.extend(np.full((1, entries), i)[0])

                # RAW data
                entries = len(tmp_data_raw['EMG'])
                raw_data_original['EMG'].extend(tmp_data_raw['EMG'])
                raw_data_original['ACC'].extend(tmp_data_raw['ACC'])
                raw_data_original['GYR'].extend(tmp_data_raw['GYR'])
                raw_data_original['ORI'].extend(tmp_data_raw['ORI'])
                label_raw.extend(np.full((1, entries), i)[0])

                print("Stop")
                file_emg = user_path + "/" + "emg" + probant + str(j) + label_display[i] + ".csv"
                file_imu = user_path + "/" + "imu" + probant + str(j) + label_display[i] + ".csv"

                # Save raw data
                save_raw_csv(tmp_data_raw, i, file_emg, file_imu)

                print("Collected window data: ", len(raw_data_window))
                print("Collected raw data: ", len(raw_data_original))

            print("Saving collected data...")
            transformed_data_collection = transform_data_collection(raw_data_window)

            # Save processed window Data
            window_save = save_csv(transformed_data_collection, label_window,
                                   "hand_disinfection_collection_windowed" + TIMESTAMP + ".csv")
            # Save processed RAW data
            raw_save = save_csv(raw_data_original, label_window,
                                "hand_disinfection_collection_raw" + TIMESTAMP + ".csv")
            if window_save is not None & raw_save is not None:
                print("Saving succeed")

import collections
import os
import shutil
import threading
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from myo import init, Hub, StreamEmg
import myo as libmyo

# const
from Helper_functions import countdown, cls
from Save_Load import save_feature_csv, save_raw_csv

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
        # with self.lock:
        if status:
            EMG.append([event.timestamp, event.emg])

    def on_orientation(self, event):
        # with self.lock:
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


def check_sample_rate(runtime_s=100):
    emg_diagram, imu_diagram = [], []
    global EMG, ORI
    emg_samples, imu_samples = 0, 0
    with hub.run_in_background(listener.on_event):
        for i in range(runtime_s):
            collect_raw_data()
            emg_samples += len(EMG)
            imu_samples += len(ORI)
            emg_diagram.append(len(EMG))
            imu_diagram.append(len(ORI))
            print(i + 1)
    print("Total EMG samples ", emg_samples, " | ", emg_samples, "/", runtime_s * 200)
    print("Total IMU samples ", imu_samples, " | ", imu_samples, "/", runtime_s * 50)
    print("Mean EMG", emg_samples / runtime_s, "|", emg_samples / runtime_s, "/200")
    print("Mean IMU", imu_samples / runtime_s, "|", imu_samples / runtime_s, "/50")

    df_emg = pd.DataFrame({'x': emg_diagram,
                           'y': runtime_s})
    df_imu = df = pd.DataFrame({'x': imu_diagram,
                                'y': runtime_s})

    y_axis = [i for i in range(runtime_s)]
    plt.plot(y_axis, emg_diagram, 'ro')
    plt.show()
    input("x")


def collect_raw_data(record_duration=1):
    global EMG
    global ORI
    global ACC
    global GYR
    global status
    global WINDOW_IMU
    global OFFSET_IMU
    EMG, ORI, ACC, GYR = [], [], [], []
    dif = 0
    start = time.time()
    while dif <= record_duration:
        status = 1
        end = time.time()
        dif = end - start
    status = 0
    return


def collect_training_data(label_display, probant="defaultUser", session=10, delete_old=False):
    global status
    user_path = COLLECTION_DIR + "/" + probant
    raw_path = user_path + "/raw"

    if not os.path.isdir(COLLECTION_DIR):
        os.mkdir(COLLECTION_DIR)
    if not os.path.isdir(user_path):
        os.mkdir(user_path)
    if os.path.isdir(raw_path):
        if delete_old:
            shutil.rmtree(raw_path)
            os.mkdir(raw_path)
    else:
        os.mkdir(raw_path)
    time.sleep(1)

    with hub.run_in_background(listener.on_event):
        for j in range(session):
            for i in range(len(label_display)):
                print("\nGesture -- ", label_display[i], " : Ready?")
                input("Countdown starts after pressing enter...")
                countdown(2)
                cls()
                print("Start")
                time.sleep(.5)

                collect_raw_data(TRAINING_TIME)

                print("Stop")
                if not os.path.isdir(raw_path + "/" + label_display[i]):
                    os.mkdir(raw_path + "/" + label_display[i])

                save_raw_csv({"EMG": EMG, "ACC": ACC, "GYR": GYR, "ORI": ORI}, i,
                             raw_path + "/" + label_display[i] + "/emg.csv",
                             raw_path + "/" + label_display[i] + "/imu.csv")

                print("Collected emg data: ", len(EMG))
                print("Collected imu data: ", len(ORI))
            print("Session ", j + 1, "completed")
        print("Data collection completed")

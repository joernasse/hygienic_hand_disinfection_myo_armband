import collections
import os
import shutil
import statistics
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import logging as log

from myo import init, Hub, StreamEmg
import myo as libmyo

from Helper_functions import countdown, cls
from Save_Load import save_raw_csv

# const
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
    emg_samples, imu_samples, over_emg = 0, 0, 0
    with hub.run_in_background(listener.on_event):
        # warm start
        collect_raw_data(5)

        for i in range(runtime_s):
            collect_raw_data()
            emg_samples += len(EMG)
            if len(EMG) > 200:
                over_emg += 1
            imu_samples += len(ORI)
            emg_diagram.append(len(EMG))
            imu_diagram.append(len(ORI))
            print(i + 1)

    log.basicConfig(filename="log" + TIMESTAMP + str(runtime_s),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=log.DEBUG)
    log.info("Runtime" + str(runtime_s))
    log.info("Total EMG samples " + str(emg_samples) + " | " + str(emg_samples) + "/" + str(runtime_s * 200))
    log.info("Total IMU samples " + str(imu_samples) + " | " + str(imu_samples), "/" + str(runtime_s * 50))
    log.info("Mean EMG" + str(emg_samples / runtime_s), "|" + str(emg_samples / runtime_s) + "/200")
    log.info("Mean IMU" + str(imu_samples / runtime_s), "|" + str(imu_samples / runtime_s) + "/50")
    log.info("Std deviation EMG" + str(statistics.stdev(emg_diagram)))
    log.info("Std deviation IMU" + str(statistics.stdev(imu_diagram)))
    log.info("Over max EMG:" + str(over_emg))

    print("Runtime", runtime_s)
    print("Total EMG samples ", emg_samples, " | ", emg_samples, "/", runtime_s * 200)
    print("Total IMU samples ", imu_samples, " | ", imu_samples, "/", runtime_s * 50)
    print("Mean EMG", emg_samples / runtime_s, "|", emg_samples / runtime_s, "/200")
    print("Mean IMU", imu_samples / runtime_s, "|", imu_samples / runtime_s, "/50")
    print("Std deviation EMG", statistics.stdev(emg_diagram))
    print("Std deviation IMU", statistics.stdev(imu_diagram))
    print("Over max EMG:", over_emg)

    df_emg = pd.DataFrame({'x': emg_diagram,
                           'y': runtime_s})
    df_imu = df = pd.DataFrame({'x': imu_diagram,
                                'y': runtime_s})

    y_axis = [i for i in range(runtime_s)]

    # plt.subplot(121)
    plt.plot(y_axis, emg_diagram, 'r')
    plt.title('#Messungen EMG, #Messungen IMU')
    # plt.title('# Messungen EMG')

    # plt.subplot(122)
    plt.plot(y_axis, imu_diagram, 'b', )
    plt.legend(['EMG', 'IMU'])
    plt.show()


def collect_raw_data(record_duration=1):
    global EMG
    global ORI
    global ACC
    global GYR
    global status
    global WINDOW_IMU
    global OFFSET_IMU
    EMG, ORI, ACC, GYR = [], [], [], []
    dif, status = 0, 0
    start = time.time()
    while dif <= record_duration:
        status = 1
        end = time.time()
        dif = end - start
    status = 0
    return


def collect_separate_training_data(label_display, probant="defaultUser", session=10, delete_old=False,
                                   fixed_pause=False):
    global status
    user_path = COLLECTION_DIR + "/" + probant
    raw_path = user_path + "/raw"

    create_directories(probant, delete_old, raw_path)

    # if not os.path.isdir(COLLECTION_DIR):
    #     os.mkdir(COLLECTION_DIR)
    # if not os.path.isdir(user_path):
    #     os.mkdir(user_path)
    # if os.path.isdir(raw_path):
    #     if delete_old:
    #         shutil.rmtree(raw_path)
    #         os.mkdir(raw_path)
    # else:
    #     os.mkdir(raw_path)

    time.sleep(1)

    n = len(label_display)
    input("Start data collection, press enter...")
    countdown(3)
    with hub.run_in_background(listener.on_event):
        for j in range(session):
            for i in range(n):
                print("\nGesture -- ", label_display[i], " : be ready!")
                cls()
                print("Do Gesture!")
                time.sleep(.5)

                collect_raw_data(TRAINING_TIME)

                print("Stop!")
                if not os.path.isdir(raw_path + "/" + label_display[i]):
                    os.mkdir(raw_path + "/" + label_display[i])

                save_raw_csv({"EMG": EMG, "ACC": ACC, "GYR": GYR, "ORI": ORI}, i,
                             raw_path + "/" + label_display[i] + "/emg.csv",
                             raw_path + "/" + label_display[i] + "/imu.csv")

                log.info("Collected emg data: " + str(len(EMG)))
                log.info("Collected imu data:" + str(len(ORI)))
                print("Pause")
                countdown(5)

            log.info("Session " + str(j + 1) + "completed")
            print("Session ", j + 1, "completed")

        print("Data collection completed")
        log.info("Data collection completed")


def collect_continuous_trainings_data(label_display, proband="defaultUser", session=5, delete_old=False):
    global status
    user_path = COLLECTION_DIR + "/" + proband
    raw_path = user_path + "/raw"

    create_directories(proband, delete_old, raw_path)
    time.sleep(1)

    with hub.run_in_background(listener.on_event):
        for j in range(session):
            for i in range(6):
                print("Full motion sequence.\nSwitching to the next step is displayed visually")
                input("Countdown starts after pressing")
                countdown(2)
                print("Start")
                collect_raw_data(5)


def create_directories(proband, delete_old, raw_path):
    user_path = COLLECTION_DIR + "/" + proband
    if not os.path.isdir(COLLECTION_DIR):
        os.mkdir(COLLECTION_DIR)
        log.info("Create directory" + COLLECTION_DIR)
    if not os.path.isdir(user_path):
        os.mkdir(user_path)
        log.info("Create directory" + user_path)
    if os.path.isdir(raw_path):
        if delete_old:
            shutil.rmtree(raw_path)
            log.info("Remove directory" + raw_path)
            os.mkdir(raw_path)
            log.info("Create directory" + raw_path)
    else:
        os.mkdir(raw_path)
        log.info("Create directory" + raw_path)

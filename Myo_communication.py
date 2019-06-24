import collections
import os
import statistics
import threading
import time
from tkinter import Text, INSERT, Label, StringVar

import matplotlib.pyplot as plt
import logging as log

from myo import init, Hub, StreamEmg
import myo as libmyo

from Constant import emg_count_list, imu_count_list
from GUI import status_window, status_val
from Helper_functions import countdown, cls
from Save_Load import save_raw_csv, create_directories

DEVICE = []
EMG = []  # emg
ORI = []  # orientation
GYR = []  # gyroscope
ACC = []  # accelerometer
status = 0

TIME_NOW = time.localtime()
TIMESTAMP = str(TIME_NOW.tm_year) + str(TIME_NOW.tm_mon) + str(TIME_NOW.tm_mday) + str(TIME_NOW.tm_hour) + str(
    TIME_NOW.tm_min) + str(TIME_NOW.tm_sec)


class GestureListener(libmyo.DeviceListener):
    def __init__(self, queue_size=1):
        # super(GestureListener, self).__init__()
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

    # def devices(self):
    #     with self._cond:
    #         return list(self._devices.values())


init()
hub = Hub()
listener = GestureListener()


def check_sample_rate(runtime_s=100, warm_start=True):
    global EMG, ORI, GYR, ACC
    EMG, ORI, GYR, ACC = [], [], [], []
    emg_diagram, imu_diagram = [], []
    emg_samples, imu_samples, over_emg = 0, 0, 0
    with hub.run_in_background(listener.on_event):
        if warm_start:
            print("Warming up...")
            # collect_raw_data(5)
            time.sleep(5)
        for i in range(runtime_s):
            collect_raw_data()
            emg_samples += len(EMG)
            if len(EMG) > 200:
                over_emg += 1
            imu_samples += len(ORI)
            emg_diagram.append(len(EMG))
            imu_diagram.append(len(ORI))
            print(i + 1)

    # log.basicConfig(filename="log/log" + TIMESTAMP + str(runtime_s),
    #                 filemode='a',
    #                 format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                 datefmt='%H:%M:%S',
    #                 level=log.DEBUG)
    # log.info("Runtime" + str(runtime_s))
    # log.info("Total EMG samples " + str(emg_samples) + " | " + str(emg_samples) + "/" + str(runtime_s * 200))
    # log.info("Total IMU samples " + str(imu_samples) + " | " + str(imu_samples), "/" + str(runtime_s * 50))
    # log.info("Mean EMG" + str(emg_samples / runtime_s), "|" + str(emg_samples / runtime_s) + "/200")
    # log.info("Mean IMU" + str(imu_samples / runtime_s), "|" + str(imu_samples / runtime_s) + "/50")
    # log.info("Std deviation EMG" + str(statistics.stdev(emg_diagram)))
    # log.info("Std deviation IMU" + str(statistics.stdev(imu_diagram)))
    # log.info("Over max EMG:" + str(over_emg))

    imu_mean = imu_samples / runtime_s
    emg_mean = emg_samples / runtime_s
    print("Runtime", runtime_s)
    print("Total EMG samples ", emg_samples, " | ", emg_samples, "/", runtime_s * 200)
    print("Total IMU samples ", imu_samples, " | ", imu_samples, "/", runtime_s * 50)
    print("Mean EMG", emg_mean, "|", emg_mean, "/200")
    print("Mean IMU", imu_mean, "|", imu_mean / runtime_s, "/50")
    print("EMG ", runtime_s * 200 / emg_samples * 100, "%")
    print("EMG ", runtime_s * 50 / imu_samples * 100, "%")
    print("Std deviation EMG", statistics.stdev(emg_diagram))
    print("Std deviation IMU", statistics.stdev(imu_diagram))
    print("Over max EMG:", over_emg)

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

    EMG, ORI, ACC, GYR = [], [], [], []
    dif, status = 0, 0
    start = time.time()
    while dif <= record_duration:
        status = 1
        end = time.time()
        dif = end - start
    status = 0
    emg_count_list.append(len(EMG))
    imu_count_list.append(len(ORI))
    return


def collect_separate_training_data(display_label, save_label, raw_path, session=10, training_time=5):
    cls()
    time.sleep(1)
    print("Gesture set\n")
    print(*display_label, sep="\n")
    print("\nHold every gesture 5 seconds")
    n = len(display_label)

    with hub.run_in_background(listener.on_event):
        for s in range(session):
            session_display = "To start session " + str(s + 1) + ", press enter..."
            input(session_display)
            countdown(3)
            for i in range(n):
                print("Gesture -- ", save_label[i], " : be ready!")
                time.sleep(1)
                print("Do Gesture!")

                collect_raw_data(training_time)
                time.sleep(.5)
                dest_path = raw_path + "/" + "s" + str(s) + save_label[i]
                if not os.path.isdir(dest_path):
                    os.mkdir(dest_path)

                save_raw_csv({"EMG": EMG, "ACC": ACC, "GYR": GYR, "ORI": ORI}, i,
                             dest_path + "/emg.csv",
                             dest_path + "/imu.csv")

                log.info("Collected emg data: " + str(len(EMG)))
                log.info("Collected imu data:" + str(len(ORI)))
                cls()
                print("Pause")
                time.sleep(.5)
                countdown(5)
                cls()

            log.info("Session " + str(s + 1) + "completed")
            print("Session ", s + 1, "completed")

        print("Data collection completed")
        log.info("Data collection completed")
        return


def collect_continuous_trainings_data(display_label, save_label, raw_path, session=5, training_time=5):
    global status
    print("Prepare Application...")
    warm_start()

    print("Collect continuous training data")

    time.sleep(1)
    cls()
    print("Gesture set\n")
    print(*display_label, sep="\n")
    print("\nFull motion sequence.\nSwitching to the next step is displayed visually")

    with hub.run_in_background(listener.on_event):
        for s in range(session):
            session_display = "To start session " + str(s + 1) + ", press enter..."
            input(session_display)
            countdown(3)
            for i in range(len(save_label)):
                print("Do Gesture!")
                collect_raw_data(training_time)

                dest_path = raw_path + "/" + "s" + str(s) + save_label[i]
                if not os.path.isdir(dest_path):
                    os.mkdir(dest_path)

                save_raw_csv({"EMG": EMG, "ACC": ACC, "GYR": GYR, "ORI": ORI}, i,
                             dest_path + "/emg.csv",
                             dest_path + "/imu.csv")

                log.info("Collected emg data: " + str(len(EMG)))
                log.info("Collected imu data:" + str(len(ORI)))
                cls()
                print("NEXT!")
                time.sleep(.5)

            log.info("Session " + str(s + 1) + "completed")
            print("Session ", s + 1, "completed")

        print("Data collection completed")
        log.info("Data collection completed")
        return


def warm_start():
    collect_raw_data(5)
    return


def trial_round_separate(save_label, display_label):
    session = 1
    print("Gesture set\n")
    print(*display_label, sep="\n")
    print("\nHold every gesture 5 seconds")
    with hub.run_in_background(listener.on_event):
        for j in range(session):
            session_display = "To start session " + str(j + 1) + ", press enter..."
            input(session_display)
            countdown(3)
            cls()
            for i in range(len(display_label)):
                print("Gesture -- ", save_label[i], " be ready!")
                time.sleep(.5)
                print("Do Gesture!")
                cls()
                collect_raw_data(5)
                time.sleep(.5)
                print("Pause")
                time.sleep(.5)
                countdown(5)
                cls()
        log.info("Session " + str(j + 1) + "completed")
        print("Session ", j + 1, "completed")
    print("Trial round separated completed")
    log.info("Data collection completed")
    return


def trial_round_continuous(save_label, display_label):
    print("Gesture set\n")
    print(*display_label, sep="\n")
    print("\nFull motion sequence.\nSwitching to the next step is displayed visually")

    with hub.run_in_background(listener.on_event):
        for j in range(1):
            session_display = "To start session " + str(j + 1) + ", press enter..."
            input(session_display)
            countdown(3)
            for i in range(len(save_label)):
                print("Do Gesture!")
                collect_raw_data(5)
                cls()
                print("NEXT!")
                time.sleep(.5)

            log.info("Session " + str(j + 1) + "completed")
            print("Session ", j + 1, "completed")
        print("Trial round continuous completed")
    return

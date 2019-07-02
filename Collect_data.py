import collections
import os
import threading
import time
import logging

import logging as log
from multiprocessing.pool import ThreadPool

from myo import init, Hub, StreamEmg
import myo as libmyo
from Constant import *
from Helper_functions import countdown, wait
from Save_Load import save_raw_csv

DEVICE_L, DEVICE_R = None, None
EMG = []  # emg
ORI = []  # orientation
GYR = []  # gyroscope
ACC = []  # accelerometer
emg_l, emg_r = [], []
tmp = []
status = 0

TIME_NOW = time.localtime()
TIMESTAMP = str(TIME_NOW.tm_year) + str(TIME_NOW.tm_mon) + str(TIME_NOW.tm_mday) + str(TIME_NOW.tm_hour) + str(
    TIME_NOW.tm_min) + str(TIME_NOW.tm_sec)

# data collection shared variables
g_introduction_screen = None
g_files = []
g_training_time, g_mode = 0, 0
g_raw_path, g_img_path = "", ""
g_trial = False
emg_count_list, imu_count_list = [], []

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class GestureListener(libmyo.DeviceListener):
    def __init__(self, queue_size=1):
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)
        self.ori_data_queue = collections.deque(maxlen=queue_size)

    def on_arm_synced(self, event):
        print("x")

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


def pair_devices():
    global DEVICE_L
    global DEVICE_R
    with hub.run_in_background(device_listener):
        wait(.5)
        for i in range(3):
            devices = device_listener.devices
            for d in devices:
                if d.arm == LEFT:
                    DEVICE_L = d
                    DEVICE_L.stream_emg(True)
                elif d.arm == RIGHT:
                    DEVICE_R = d
                    DEVICE_R.stream_emg(True)
            if not (DEVICE_L is None) and not (DEVICE_R is None):
                DEVICE_R.vibrate(libmyo.VibrationType.short)
                DEVICE_L.vibrate(libmyo.VibrationType.short)
                logging.info("Devices paired")
                return True
            wait(2)
    hub.stop()
    return False


def collect_raw_data():
    global EMG
    global ORI
    global ACC
    global GYR
    global status
    global g_training_time

    EMG, ORI, ACC, GYR = [], [], [], []
    dif, status = 0, 0
    start = time.time()
    while dif <= g_training_time:
        status = 1
        end = time.time()
        dif = end - start
        g_introduction_screen.update_gesture_bar(dif)
    status = 0
    logging.info("EMG %d", len(EMG))
    logging.info("IMU %d", len(ORI))

    emg_count_list.append(len(EMG))
    imu_count_list.append(len(ORI))
    return


def collect_data(current_session):
    global g_introduction_screen
    global g_files
    global g_training_time
    global g_raw_path
    global g_img_path
    global DEVICE_R
    global emg_r
    global emg_l

    g_introduction_screen.set_session_text("Session " + str(current_session + 1))
    g_introduction_screen.set_countdown_text("")

    with hub.run_in_background(gesture_listener.on_event):
        countdown(g_introduction_screen, 3)
        for i in range(len(save_label)):
            path = g_raw_path + "/" + "s" + str(current_session) + save_label[i]
            emg_l, emg_r = [], []

            g_introduction_screen.set_gesture_description(hand_disinfection_description[i])
            g_introduction_screen.change_img(g_img_path + g_files[i])

            g_introduction_screen.set_countdown_text("")

            if g_mode == INDIVIDUAL:
                wait(0.7)

            DEVICE_R.vibrate(type=libmyo.VibrationType.short)
            g_introduction_screen.set_status_text("Start!")

            collect_raw_data()

            DEVICE_L.vibrate(type=libmyo.VibrationType.short)
            DEVICE_L.vibrate(type=libmyo.VibrationType.short)

            g_introduction_screen.set_status_text("Pause")
            g_introduction_screen.update_progressbars(1)

            if i < len(save_label) - 1:
                g_introduction_screen.set_gesture_description("Next: " + hand_disinfection_description[i + 1])
                g_introduction_screen.change_img(g_img_path + g_files[i + 1])

            if not g_trial:
                if not os.path.isdir(path):
                    os.mkdir(path)
                save_raw_csv({"EMG": EMG, "ACC": ACC, "GYR": GYR, "ORI": ORI}, i,
                             path + "/emg.csv",
                             path + "/imu.csv")
                log.info("Collected emg data: " + str(len(EMG)))
                log.info("Collected imu data:" + str(len(ORI)))

            wait(.5)

            if g_mode == INDIVIDUAL:
                if i < len(save_label) - 1:
                    countdown(g_introduction_screen, 5)

    hub.stop()
    g_introduction_screen.set_countdown_text("")
    g_introduction_screen.set_status_text("Session " + str(current_session + 1) + " done!")
    g_introduction_screen.set_gesture_description("")
    g_introduction_screen.change_img("intro_screen.jpg")
    logging.info("Data collection session %d complete", current_session)
    return


def update_gesture_bar():
    g_introduction_screen.update_gesture_bar()
    wait(1)

def init_data_collection(raw_path, introduction_screen, trial, mode, training_time=5):
    global g_introduction_screen
    global g_files
    global g_training_time
    global g_raw_path
    global g_img_path
    global g_mode
    global g_trial

    if pair_devices():
        g_training_time = training_time
        g_introduction_screen = introduction_screen
        g_introduction_screen.change_img("intro_screen.jpg")
        g_img_path = os.getcwd() + "/img/"
        g_files = os.listdir(g_img_path)
        g_introduction_screen.set_status_text("Hold every gesture for 5 seconds")
        g_raw_path = raw_path
        g_mode = mode
        g_trial = trial
        return True
    return False


def main():
    print("x")


if __name__ == '__main__':
    main()

import collections
import os
import statistics
import threading
import time

import matplotlib.pyplot as plt
import logging as log
from myo import init, Hub, StreamEmg
import myo as libmyo
from Constant import emg_count_list, imu_count_list, hand_disinfection_description, label_display, save_label, LEFT, \
    RIGHT, \
    INDIVIDUAL
from Helper_functions import countdown, cls, wait
from Save_Load import save_raw_csv

DEVICE_L, DEVICE_R = None, None
EMG = []  # emg
ORI = []  # orientation
GYR = []  # gyroscope
ACC = []  # accelerometer
status = 0

TIME_NOW = time.localtime()
TIMESTAMP = str(TIME_NOW.tm_year) + str(TIME_NOW.tm_mon) + str(TIME_NOW.tm_mday) + str(TIME_NOW.tm_hour) + str(
    TIME_NOW.tm_min) + str(TIME_NOW.tm_sec)

# data collection shared variables
g_introduction_screen = None
g_files = []
g_training_time = 0
g_raw_path, g_img_path = "", ""


class GestureListener(libmyo.ApiDeviceListener):
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

    # def devices(self):
    #     with self._cond:
    #         return list(self._devices.values())


init()
hub = Hub()
listener = GestureListener()


def pair_devices():
    global DEVICE_L
    global DEVICE_R
    with hub.run_in_background(listener):
        wait(2)
        devices = listener.devices
        for d in devices:
            if d.arm == LEFT:
                d.vibrate(libmyo.VibrationType.short)
                DEVICE_L = d
            elif d.arm == RIGHT:
                d.vibrate(libmyo.VibrationType.short)
                DEVICE_R = d
        print("x")
    hub.stop()
    return [DEVICE_L, DEVICE_R]


def check_sample_rate(runtime_s=100, warm_start=True):
    global EMG, ORI, GYR, ACC
    EMG, ORI, GYR, ACC = [], [], [], []
    emg_diagram, imu_diagram = [], []
    emg_samples, imu_samples, over_emg = 0, 0, 0
    with hub.run_in_background(listener.on_event):
        if warm_start:
            print("Warming up...")
            # collect_raw_data(5)
            wait(5)
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
        # g_introduction_screen.update_gesture_bar(1)
    status = 0
    emg_count_list.append(len(EMG))
    imu_count_list.append(len(ORI))
    return


def init_data_collection(raw_path, introduction_screen, session=10, training_time=5, mode="individually"):
    global g_introduction_screen
    global g_files
    global g_training_time
    global g_raw_path
    global g_img_path

    pair_devices()
    g_training_time = training_time
    g_introduction_screen = introduction_screen
    g_introduction_screen.change_img("intro_screen.jpg")
    g_img_path = os.getcwd() + "/img/"
    g_files = os.listdir(g_img_path)
    g_introduction_screen.set_description_text("Hold every gesture for 5 seconds")
    g_raw_path = raw_path


def collect_data(session, mode=INDIVIDUAL):
    global g_introduction_screen
    global g_files
    global g_training_time
    global g_raw_path
    global g_img_path
    global DEVICE_R

    with hub.run_in_background(listener.on_event):
        g_introduction_screen.init_sessionbar()
        countdown(g_introduction_screen, 3)
        for i in range(len(save_label)):
            path = g_raw_path + "/" + "s" + str(session) + save_label[i]
            g_introduction_screen.set_gesture_label(hand_disinfection_description[i])
            g_introduction_screen.set_description_val("")
            g_introduction_screen.change_img(g_img_path + g_files[i])
            # g_introduction_screen.set_description_text(hand_disinfection_description[i])

            wait(1)

            DEVICE_L.vibrate(type=libmyo.VibrationType.short)
            g_introduction_screen.set_description_text("Start!")
            collect_raw_data(g_training_time)
            g_introduction_screen.set_description_text("Pause")
            DEVICE_L.vibrate(type=libmyo.VibrationType.medium)
            DEVICE_L.vibrate(type=libmyo.VibrationType.short)

            if not os.path.isdir(path):
                os.mkdir(path)

            save_raw_csv({"EMG": EMG, "ACC": ACC, "GYR": GYR, "ORI": ORI}, i,
                         path + "/emg.csv",
                         path + "/imu.csv")
            log.info("Collected emg data: " + str(len(EMG)))
            log.info("Collected imu data:" + str(len(ORI)))

            wait(.5)
            if mode == INDIVIDUAL:
                countdown(g_introduction_screen, 5)
            else:
                DEVICE_R.vibrate(type=libmyo.VibrationType.short)
                # wait(.5)
            g_introduction_screen.update_session_bar(1)
    hub.stop()
    return


def collect_separate_training_data(raw_path, introduction_screen, session=10, training_time=5):
    display_label = hand_disinfection_description
    save_label = label_display
    introduction_screen.change_img("intro_screen.jpg")
    img_path = os.getcwd() + "/img/"
    files = os.listdir(img_path)
    files.sort()

    wait(1)
    introduction_screen.set_description_text("Gesture set")
    print(*display_label, sep="\n")
    print("\nHold every gesture 5 seconds")

    with hub.run_in_background(listener.on_event):
        for s in range(session):
            g_introduction_screen.init_gesturebar(training_time)
            countdown(3)
            for i in range(len(display_label)):
                introduction_screen.init_gesture_bar()
                introduction_screen.change_img(img_path + files[i])
                print("Gesture -- ", save_label[i], " : be ready!")
                wait(1)
                print("Do Gesture!")

                collect_raw_data(training_time)
                wait(.5)
                dest_path = raw_path + "/" + "s" + str(s) + save_label[i]
                if not os.path.isdir(dest_path):
                    os.mkdir(dest_path)

                save_raw_csv({"EMG": EMG, "ACC": ACC, "GYR": GYR, "ORI": ORI}, i,
                             dest_path + "/emg.csv",
                             dest_path + "/imu.csv")
                log.info("Collected emg data: " + str(len(EMG)))
                log.info("Collected imu data:" + str(len(ORI)))
                print("Pause")
                wait(.5)
                countdown(5)
                introduction_screen.update_session_bar(1)
            introduction_screen.update_total_bar(1)

            log.info("Session " + str(s + 1) + "completed")
            print("Session ", s + 1, "completed")

        print("Data collection completed")
        log.info("Data collection completed")
        return


def collect_continuous_trainings_data(raw_path, introduction_screen, session=5, training_time=5):
    display_label = hand_disinfection_description
    save_label = label_display
    global status
    print("Prepare Application...")
    warm_start()

    print("Collect continuous training data")

    wait(1)
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
                wait(.5)

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
                wait(.5)
                print("Do Gesture!")
                cls()
                collect_raw_data(5)
                wait(.5)
                print("Pause")
                wait(.5)
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
                wait(.5)

            log.info("Session " + str(j + 1) + "completed")
            print("Session ", j + 1, "completed")
        print("Trial round continuous completed")
    return


def main():
    print("x")


if __name__ == '__main__':
    main()

import os
import statistics
import time
from multiprocessing.pool import ThreadPool

import matplotlib.pyplot as plt
import logging as log
import numpy as np

from myo import init, Hub
import myo as libmyo


from Constant import emg_count_list, imu_count_list, hand_disinfection_display, label_display, save_label
from Helper_functions import countdown, cls
from Save_Load import save_raw_csv

DEVICE = []
EMG = []  # emg
ORI = []  # orientation
GYR = []  # gyroscope
ACC = []  # accelerometer
status = 0
g_device_right = 0
g_device_left = 0

EMG_INTERVAL = 0.01
POS_INTERVAL = 0.02

pool = ThreadPool(processes=2)

TIME_NOW = time.localtime()
TIMESTAMP = str(TIME_NOW.tm_year) + str(TIME_NOW.tm_mon) + str(TIME_NOW.tm_mday) + str(TIME_NOW.tm_hour) + str(
    TIME_NOW.tm_min) + str(TIME_NOW.tm_sec)

# data collection shared variables
g_introduction_screen = []
g_files = []
g_training_time = 0
g_raw_path = ""
g_img_path = ""

init()
hub = Hub()
listener_2 = libmyo.ApiDeviceListener()


def collect_independent_pos_data():
    global g_device_left
    global g_device_right
    ori_left, acc_left, gyr_left, ori_right, acc_right, gyr_right = [], [], [], [], [], []
    time.sleep(0.5)
    dif = 0
    start = time.time()
    while dif < 1:
        end = time.time()
        dif = end - start
        ori_left.append(np.asarray(g_device_left.orientation))
        acc_left.append(np.asarray(g_device_left.acceleration))
        gyr_left.append(np.asarray(g_device_left.gyroscope))

        ori_right.append(np.asarray(g_device_right.orientation))
        acc_right.append(np.asarray(g_device_right.acceleration))
        gyr_right.append(np.asarray(g_device_right.gyroscope))
        time.sleep(POS_INTERVAL)
    return {"ORI_L": ori_left, "ACC_L": acc_left, "GYR_L": gyr_left,
            "ORI_R": ori_right, "ACC_R": acc_right, "GYR_R": gyr_right}


def collect_independent_emg_data():
    global g_device_right
    global g_device_left
    emg_left, emg_right = [], []
    time.sleep(0.5)
    dif = 0
    start = time.time()
    g_device_left.stream_emg(True)
    g_device_right.stream_emg(True)
    while dif < 1:
        end = time.time()
        dif = end - start
        emg_left.append(np.asarray(g_device_left.emg))
        emg_right.append(np.asarray(g_device_right.emg))
        time.sleep(EMG_INTERVAL)

    g_device_right.stream_emg(False)
    g_device_left.stream_emg(False)

    return {"EMG_L": emg_left, "EMG_R": emg_right}


def pair_devices():
    global g_device_right
    global g_device_left
    with hub.run_in_background(listener_2.on_event):
        time.sleep(1)
        r, l = False, False
        devices = listener_2.devices
        for dev in devices:
            if dev.arm == "right":
                g_device_right = dev
                r = True
            elif dev.arm == "left":
                g_device_left = dev
                l = True
            if r and l:
                break
    # return [device_left, device_right]


def check_sample_rate(runtime_s=100, warm_start=True):
    global EMG, ORI, GYR, ACC
    EMG, ORI, GYR, ACC = [], [], [], []
    emg_diagram, imu_diagram = [], []
    emg_samples, imu_samples, over_emg = 0, 0, 0
    with hub.run_in_background(listener_2.on_event):
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


def init_data_collection(raw_path, introduction_screen, session=10, training_time=5):
    global g_introduction_screen
    global g_files
    global g_training_time
    global g_raw_path
    global g_img_path

    g_training_time = training_time
    g_introduction_screen = introduction_screen
    g_introduction_screen.change_img("intro_screen.jpg")
    g_img_path = os.getcwd() + "/img/"
    g_files = os.listdir(g_img_path)
    # g_introduction_screen.set_descr_text("Gesture set")
    g_introduction_screen.set_descr_text("Hold every gesture for 5 seconds")
    g_raw_path = raw_path


def collect_data(session):
    global pool
    global g_introduction_screen
    global g_files
    global g_training_time
    global g_raw_path
    global g_img_path

    pair_devices()
    with hub.run_in_background(listener_2.on_event):
        g_introduction_screen.init_sessionbar()
        time.sleep(3)
        countdown(g_introduction_screen, 3)

        for i in range(len(save_label)):
            g_introduction_screen.set_descr_val("")
            g_introduction_screen.change_img(g_img_path + g_files[i])
            g_introduction_screen.set_descr_text("Gesture -- " + label_display[i] + " : be ready!")
            time.sleep(1)
            g_introduction_screen.set_descr_text("Do Gesture!")

            async_result_emg = pool.apply_async(collect_independent_emg_data, args=())
            async_result_pos = pool.apply_async(collect_independent_pos_data, args=())

            return_val_emg = async_result_emg.get()
            return_val_pos = async_result_pos.get()

            time.sleep(.3)
            dest_path = g_raw_path + "/" + "s" + str(session) + save_label[i]

            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)

            save_raw_csv({"EMG": return_val_emg["EMG_L"],
                          "ACC": return_val_pos["ACC_L"],
                          "GYR": return_val_pos["GYR_L"],
                          "ORI": return_val_pos["ORI_L"]},
                         i, dest_path + "/l_emg.csv", dest_path + "/l_imu.csv")
            save_raw_csv({"EMG": return_val_emg["EMG_R"],
                          "ACC": return_val_pos["ACC_R"],
                          "GYR": return_val_pos["GYR_R"],
                          "ORI": return_val_pos["ORI_R"]},
                         i, dest_path + "/r_emg.csv", dest_path + "/r_imu.csv")

            # log.info("Collected emg data: " + str(len(EMG)))
            # log.info("Collected imu data:" + str(len(ORI)))
            g_introduction_screen.set_descr_text("Pause")
            time.sleep(.5)
            countdown(g_introduction_screen, 5)
            g_introduction_screen.update_session_bar(1)
    return


def collect_separate_training_data(raw_path, introduction_screen, session=10, training_time=5):
    display_label = hand_disinfection_display
    save_label = label_display
    introduction_screen.change_img("intro_screen.jpg")
    img_path = os.getcwd() + "/img/"
    files = os.listdir(img_path)
    files.sort()

    # cls()
    time.sleep(1)
    # print("Gesture set\n")
    introduction_screen.set_descr_text("Gesture set")
    print(*display_label, sep="\n")
    print("\nHold every gesture 5 seconds")
    n = len(display_label)

    with hub.run_in_background(listener_2.on_event):
        for s in range(session):
            introduction_screen.init_sessionbar()
            # session_display = "To start session " + str(s + 1) + ", press enter..."
            # input(session_display)
            countdown(3)
            for i in range(n):
                introduction_screen.change_img(img_path + files[i])
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
                introduction_screen.update_session_bar(1)
            introduction_screen.update_total_bar(1)

            log.info("Session " + str(s + 1) + "completed")
            print("Session ", s + 1, "completed")

        print("Data collection completed")
        log.info("Data collection completed")
        return


def collect_continuous_trainings_data(raw_path, introduction_screen, session=5, training_time=5):
    display_label = hand_disinfection_display
    save_label = label_display
    global status
    print("Prepare Application...")
    warm_start()

    print("Collect continuous training data")

    time.sleep(1)
    cls()
    print("Gesture set\n")
    print(*display_label, sep="\n")
    print("\nFull motion sequence.\nSwitching to the next step is displayed visually")

    with hub.run_in_background(listener_2.on_event):
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
    with hub.run_in_background(listener_2.on_event):
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

    with hub.run_in_background(listener_2.on_event):
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

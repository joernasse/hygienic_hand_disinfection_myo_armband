import collections
import os
import threading
import time
import logging

import logging as log
from tkinter import BOTH, StringVar, Label, HORIZONTAL, Entry, Button, IntVar, W, E, Tk, Checkbutton, VERTICAL, \
    DISABLED, NORMAL
from tkinter.ttk import Progressbar, Separator, Frame
from PIL import Image, ImageTk

from myo import init, Hub, StreamEmg
import myo as libmyo
from Constant import *
from Helper_functions import countdown, wait
from Save_Load import save_raw_csv, create_directories

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
g_training_time, g_mode, g_break = 0, 0, 0
g_raw_path, g_img_path = "", ""
g_trial = False
emg_count_list, imu_count_list = [], []
img_w, img_h = 450, 400

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

introduction_window = Tk()
collect_window = Tk()


class CollectDataWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.user_path = ""

        self.error_val = StringVar(self, value="Status")
        self.error_val.set("Status")
        self.sessions_label = Label(self, text="Durchgänge")
        self.record_time_label = Label(self, text="Zeit pro Geste")
        self.proband_label = Label(self, text="Proband Name")
        self.error_label = Label(self, textvariable=self.error_val)

        self.sep1 = Separator(self, orient=HORIZONTAL)
        self.sep2 = Separator(self, orient=HORIZONTAL)

        self.session_val = IntVar(self, value=10)
        self.record_time_val = IntVar(self, value=5)
        self.proband_val = StringVar(self, value="defaultUser")

        self.sessions_input = Entry(self, textvariable=self.session_val, width=3)
        self.record_time_input = Entry(self, textvariable=self.record_time_val, width=3)
        self.proband_input = Entry(self, textvariable=self.proband_val, width=17)

        self.collect_separate_btn = Button(master=self, text="Collect Separate",
                                           command=lambda: self.introduction_screen_ui(mode=INDIVIDUAL, trial=False))
        self.collect_continues_btn = Button(master=self, text="Collect Continues",
                                            command=lambda: self.introduction_screen_ui(mode=CONTINUES, trial=False))
        self.trial_separate_btn = Button(master=self, text="Trial Separate",
                                         command=lambda: self.introduction_screen_ui(mode=INDIVIDUAL, trial=True))
        self.trial_continues_btn = Button(master=self, text="Trial Continues",
                                          command=lambda: self.introduction_screen_ui(mode=CONTINUES, trial=True))
        self.close_btn = Button(self, text="Close", command=collect_window.withdraw)

        # Style
        self.sessions_label.grid(row=0, column=0, pady=4, padx=4, sticky=W)
        self.sessions_input.grid(row=0, column=1, padx=2, sticky=W)

        self.record_time_label.grid(row=1, column=0, pady=4, padx=4, sticky=W)
        self.record_time_input.grid(row=1, column=1, padx=2, sticky=W)

        self.proband_label.grid(row=2, column=0, pady=4, padx=4, sticky=W)
        self.proband_input.grid(row=2, column=1, padx=2, sticky=W)

        self.collect_separate_btn.grid(row=4, column=0, pady=8, padx=8)
        self.collect_continues_btn.grid(row=4, column=1, pady=4, padx=8)

        self.trial_separate_btn.grid(row=5, column=0, pady=4, padx=8)
        self.trial_continues_btn.grid(row=5, column=1, pady=4, padx=8)

        self.sep1.grid(row=6, column=0, sticky="ew", columnspan=3, padx=4, pady=8)

        self.error_label.grid(row=7, column=0, pady=8, padx=4)
        self.close_btn.grid(row=7, column=1, pady=8, padx=4)

    def introduction_screen_ui(self, mode, trial):
        global g_introduction_screen

        user_path = "Collections/" + self.proband_val.get()
        raw_path = user_path + "/raw"
        create_directories(proband=self.proband_val.get(), delete_old=False, raw_path=raw_path,
                           raw_sep=user_path + "/raw_separate",
                           raw_con=user_path + "/raw_continues")

        sessions = self.session_val.get()
        record_time = self.record_time_val.get()

        if mode == INDIVIDUAL:
            title = "Collect separate data"
            raw_path = user_path + "/raw_separate"
        else:
            title = "Collect continues data"
            raw_path = user_path + "/raw_continues"
        if trial:
            sessions = 1
            record_time = 5
            title += " TRIAL"

        g_introduction_screen = IntroductionScreen(introduction_window, record_time=record_time, sessions=sessions)
        introduction_window.title(title)

        if init_data_collection(raw_path=raw_path,
                                trial=trial,
                                mode=mode,
                                training_time=record_time):
            introduction_window.deiconify()
            introduction_window.mainloop()
        else:
            self.error_val.set("Paired failure")
            collect_window.update()
            print("Paired failure")


class IntroductionScreen(Frame):
    def __init__(self, master=None, record_time=5, sessions=10, ):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        load = Image.open("intro_screen.jpg")
        load = load.resize((img_w, img_h), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        self.img = Label(self, image=render)
        self.img.image = render
        self.status_text = StringVar()
        self.session_total = StringVar()
        self.countdown_value = StringVar()
        self.battery_value = StringVar()
        self.sessions = sessions
        self.record_time = record_time
        self.current_session = 0
        self.mode = ""

        self.status_label = Label(self, textvariable=self.status_text)  # Start, Pause
        self.gesture_countdown_label = Label(self, textvariable=self.countdown_value)
        self.session_total_label = Label(self, textvariable=self.session_total)
        self.battery_label = Label(self, textvariable=self.battery_value)

        self.start_session_btn = Button(self, text="Start Session", command=self.start_session)
        self.close_btn = Button(self, text="Close", command=self.close)

        self.deviating_time_val = IntVar(self, value=record_time)
        self.deviating_time_input = Entry(self, textvariable=self.deviating_time_val, width=5)

        self.progress_total = Progressbar(self, orient="horizontal", length=200, mode='determinate')
        self.progress_session = Progressbar(self, orient="horizontal", length=200, mode='determinate')
        self.progress_gesture = Progressbar(self, orient="horizontal", length=200, mode='determinate')
        self.progress_total["maximum"] = self.sessions * len(label_display)
        self.progress_gesture["maximum"] = self.record_time

        self.session_text = StringVar()
        self.session_text.set("Session 1")
        self.gesture_text = StringVar()

        self.session_label = Label(self, textvariable=self.session_text)
        self.gesture_label = Label(self, textvariable=self.gesture_text)
        self.total_label = Label(self, text="Total")

        # Style---------------------------------------------------------------------------
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)

        self.status_label.grid(row=1, column=1, pady=2, padx=2, sticky=W)
        self.gesture_countdown_label.grid(row=1, column=1, pady=4, sticky=E)

        self.session_total_label.grid(row=1, column=2, pady=4)
        self.gesture_label.grid(row=2, column=1, rowspan=2, columnspan=2, pady=4, padx=2, sticky=W)

        self.progress_gesture.grid(row=4, column=1, padx=4, sticky=W)
        self.deviating_time_input.grid(row=4, column=2, pady=8, padx=4)

        self.session_label.grid(row=5, column=0, pady=4, sticky=W)
        self.progress_session.grigd(row=5, column=1, padx=4, sticky=W)
        self.start_session_btn.grid(row=5, column=2, padx=4)

        self.total_label.grid(row=6, column=0, pady=4, sticky=W)
        self.progress_total.grid(row=6, column=1, padx=4, sticky=W)
        self.close_btn.grid(row=6, column=2, padx=4, pady=8)

    def start_session(self):
        global g_break
        if self.current_session < self.sessions:
            g_break = self.deviating_time_val.get()
            self.session_total.set(str(self.current_session + 1) + "/" + str(self.sessions) + " Sessions")
            self.init_sessionbar()

            self.deviating_time_input['state'] = DISABLED
            self.start_session_btn['state'] = DISABLED
            collect_data(self.current_session)
            self.start_session_btn['state'] = NORMAL
            self.deviating_time_input['state'] = NORMAL

            self.current_session += 1
            self.update_progressbars(1)
        if self.current_session == self.sessions:
            self.set_countdown_text("Data collection complete!")
            wait(3)
            self.close()
        return

    def close(self):
        self.destroy()
        introduction_window.withdraw()

    def change_img(self, path):
        load = Image.open(path)
        load = load.resize((img_w, img_h), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        self.img = Label(self, image=render)
        self.img.image = render
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)
        introduction_window.update()

    def init_sessionbar(self):
        self.progressbar_session_val = 0
        self.progress_session["value"] = self.progressbar_session_val
        self.progress_session["maximum"] = len(label_display)

    def set_status_text(self, text):
        self.status_text.set(text)
        introduction_window.update()

    def set_gesture_description(self, text):
        self.gesture_text.set(text)
        introduction_window.update()

    def set_countdown_text(self, text):
        self.countdown_value.set(text)
        introduction_window.update()

    def set_session_text(self, text):
        self.session_text.set(text)
        introduction_window.update()

    def update_progressbars(self, value):
        self.progress_session["value"] += value
        self.progress_total["value"] += value
        introduction_window.update()

    def update_gesture_bar(self, value):
        if value > self.record_time:
            value = self.record_time
        self.progress_gesture["value"] = value
        introduction_window.update()


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
            # Test ob ohne Ready besser
            # if g_mode == INDIVIDUAL:
            #     g_introduction_screen.set_status_text("Ready!")
            #     wait(1)

            DEVICE_R.vibrate(type=libmyo.VibrationType.short)
            g_introduction_screen.set_status_text("Start!")

            collect_raw_data()

            DEVICE_L.vibrate(type=libmyo.VibrationType.short)

            g_introduction_screen.set_status_text("Pause")
            g_introduction_screen.update_gesture_bar(0)
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

            if g_mode == INDIVIDUAL:
                wait(.5)
                if i < len(save_label) - 1:
                    countdown(g_introduction_screen, g_break)

    g_introduction_screen.set_countdown_text("")
    g_introduction_screen.set_status_text("Session " + str(current_session + 1) + " done!")
    g_introduction_screen.set_gesture_description("")
    g_introduction_screen.change_img("intro_screen.jpg")
    logging.info("Data collection session %d complete", current_session)
    return


def init_data_collection(raw_path, trial, mode, training_time=5):
    global g_files
    global g_training_time
    global g_raw_path
    global g_img_path
    global g_mode
    global g_trial

    if pair_devices():
        g_training_time = training_time
        g_introduction_screen.change_img("intro_screen.jpg")
        g_img_path = os.getcwd() + "/gestures/"
        # g_img_path = os.getcwd() + "/img/"
        g_files = os.listdir(g_img_path)
        g_introduction_screen.set_status_text("Hold every gesture for 5 seconds")
        g_raw_path = raw_path
        g_mode = mode
        g_trial = trial
        return True
    return False


def main():
    introduction_window.wm_title("Introduction Screen")
    introduction_window.withdraw()

    data_collect = CollectDataWindow(collect_window)
    collect_window.wm_title("Collect Data")
    collect_window.mainloop()


if __name__ == '__main__':
    main()

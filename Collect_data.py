#!/usr/bin/env python
"""
GestureListener class for communication with myo gesture control armband
IntroductionScreen and CollectDataWindow for the GUI
This script contains the GUI for the user study.
The script also includes functions to collect raw data from Myo wristbands and store it in the file systems.
"""

import collections
import os
import threading
import time
import Constant
import logging as log
import Live_prototype
import myo as libmyo
from tkinter import BOTH, StringVar, Label, HORIZONTAL, Entry, Button, IntVar, W, E, Tk, DISABLED, NORMAL
from tkinter.ttk import Progressbar, Separator, Frame
from PIL import Image, ImageTk
from myo import init, Hub
from Helper_functions import wait
from Save_Load import save_raw_data, create_directories_for_data_collection

__author__ = "Joern Asse"
__copyright__ = ""
__credits__ = ["Joern Asse"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Joern Asse"
__email__ = "joernasse@yahoo.de"
__status__ = "Production"

DEVICE_L, DEVICE_R = None, None
EMG = []  # Emg
ORI = []  # Orientation
GYR = []  # Gyroscope
ACC = []  # Accelerometer
TIME_NOW = time.localtime()
TIMESTAMP = str(TIME_NOW.tm_year) + str(TIME_NOW.tm_mon) + str(TIME_NOW.tm_mday) + str(TIME_NOW.tm_hour) + str(
    TIME_NOW.tm_min) + str(TIME_NOW.tm_sec)

# data collection shared variables
emg_l, emg_r = [], []
status = 0
g_introduction_screen = None
g_files = []
g_record_time, g_mode, g_break = 0, 0, 0
g_raw_path, g_img_path = "", ""
g_trial = False
emg_count_list, imu_count_list = [], []
img_w, img_h = 450, 400

log.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

introduction_window = Tk()
collect_window = Tk()


class GestureListener(libmyo.DeviceListener):
    def __init__(self, queue_size=1):
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


init()
hub = Hub()
device_listener = libmyo.ApiDeviceListener()
gesture_listener = GestureListener()


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
        self.test_person_label = Label(self, text="Proband Name")
        self.error_label = Label(self, textvariable=self.error_val)

        self.sep1 = Separator(self, orient=HORIZONTAL)
        self.sep2 = Separator(self, orient=HORIZONTAL)

        self.session_val = IntVar(self, value=10)
        self.record_time_val = IntVar(self, value=5)
        self.test_person_val = StringVar(self, value="defaultUser")

        self.sessions_input = Entry(self, textvariable=self.session_val, width=3)
        self.record_time_input = Entry(self, textvariable=self.record_time_val, width=3)
        self.test_person_input = Entry(self, textvariable=self.test_person_val, width=17)

        # --------------------------------------------Button-----------------------------------------------------------#

        self.collect_separate_btn = Button(master=self, text="Collect Separate",
                                           command=lambda: self.setup_introduction_screen(mode=Constant.SEPARATE,
                                                                                          trial=False))
        self.collect_continues_btn = Button(master=self, text="Collect Continues",
                                            command=lambda: self.setup_introduction_screen(mode=Constant.CONTINUES,
                                                                                           trial=False))
        self.trial_separate_btn = Button(master=self, text="Trial Separate",
                                         command=lambda: self.setup_introduction_screen(mode=Constant.SEPARATE,
                                                                                        trial=True))
        self.trial_continues_btn = Button(master=self, text="Trial Continues",
                                          command=lambda: self.setup_introduction_screen(mode=Constant.CONTINUES,
                                                                                         trial=True))
        self.live_prototype_btn = Button(master=self, text="Live Prototype",
                                         command=lambda: Live_prototype.main())
        self.close_btn = Button(self, text="Close", command=collect_window.withdraw)

        # --------------------------------------------Layout-----------------------------------------------------------#
        self.sessions_label.grid(row=0, column=0, pady=4, padx=4, sticky=W)
        self.sessions_input.grid(row=0, column=1, padx=2, sticky=W)

        self.record_time_label.grid(row=1, column=0, pady=4, padx=4, sticky=W)
        self.record_time_input.grid(row=1, column=1, padx=2, sticky=W)

        self.test_person_label.grid(row=2, column=0, pady=4, padx=4, sticky=W)
        self.test_person_input.grid(row=2, column=1, padx=2, sticky=W)

        self.collect_separate_btn.grid(row=4, column=0, pady=8, padx=8)
        self.collect_continues_btn.grid(row=4, column=1, pady=4, padx=8)

        self.trial_separate_btn.grid(row=5, column=0, pady=4, padx=8)
        self.trial_continues_btn.grid(row=5, column=1, pady=4, padx=8)

        self.sep1.grid(row=6, column=0, sticky="ew", columnspan=3, padx=4, pady=8)

        self.live_prototype_btn.grid(row=7, column=0, columnspan=2, pady=4, padx=4)

        self.sep2.grid(row=8, column=0, sticky="ew", columnspan=3, padx=4, pady=8)

        self.error_label.grid(row=9, column=0, pady=8, padx=4)
        self.close_btn.grid(row=9, column=1, pady=8, padx=4)

    def setup_introduction_screen(self, mode, trial):
        """
        In this function the introduction screen will be setup.
            - Save directories will be created.
            - Initialization of data collection.
        :param mode: string
                "individual" for data collection with break between each gesture
                "continues" for data collection without break between each gesture

        :param trial: boolean
                If True the trial mode is active.
                If False the trial mode is not active.
        :return: No returns
        """
        global g_introduction_screen

        user_path = "Collections/" + self.test_person_val.get()
        raw_path = user_path + "/raw"
        create_directories_for_data_collection(user=self.test_person_val.get(), delete_old=False, raw_path=raw_path,
                                               raw_sep=user_path + Constant.SEPARATE_PATH,
                                               raw_con=user_path + Constant.CONTINUES_PATH)

        sessions = self.session_val.get()
        record_time = self.record_time_val.get()

        if mode == Constant.SEPARATE:
            title = "Collect separate data"
            raw_path = user_path + Constant.SEPARATE_PATH
        else:
            title = "Collect continues data"
            raw_path = user_path + Constant.CONTINUES_PATH
        if trial:
            sessions = 1
            record_time = 5
            title += " TRIAL"

        g_introduction_screen = IntroductionScreen(introduction_window, record_time=record_time, sessions=sessions)
        introduction_window.title(title)

        if initialize_data_collection(raw_path=raw_path, trial=trial, mode=mode, record_time=record_time):
            introduction_window.deiconify()
            introduction_window.mainloop()
        else:
            self.error_val.set("Paired failure")
            collect_window.update()
            print("Paired failure")


class IntroductionScreen(Frame):
    def __init__(self, master=None, record_time=5, sessions=10):
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
        self.progress_total["maximum"] = self.sessions * len(Constant.label_display_with_rest)
        self.progress_gesture["maximum"] = self.record_time

        self.session_text = StringVar()
        self.session_text.set("Session 1")
        self.gesture_text = StringVar()

        self.session_label = Label(self, textvariable=self.session_text)
        self.gesture_label = Label(self, textvariable=self.gesture_text)
        self.total_label = Label(self, text="Total")

        # --------------------------------------------Layout-----------------------------------------------------------#
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)

        self.status_label.grid(row=1, column=1, pady=2, padx=2, sticky=W)
        self.gesture_countdown_label.grid(row=1, column=1, pady=4, sticky=E)

        self.session_total_label.grid(row=1, column=2, pady=4)
        self.gesture_label.grid(row=2, column=1, rowspan=2, columnspan=2, pady=4, padx=2, sticky=W)

        self.progress_gesture.grid(row=4, column=1, padx=4, sticky=W)
        self.deviating_time_input.grid(row=4, column=2, pady=8, padx=4)

        self.session_label.grid(row=5, column=0, pady=4, sticky=W)
        self.progress_session.grid(row=5, column=1, padx=4, sticky=W)
        self.start_session_btn.grid(row=5, column=2, padx=4)

        self.total_label.grid(row=6, column=0, pady=4, sticky=W)
        self.progress_total.grid(row=6, column=1, padx=4, sticky=W)
        self.close_btn.grid(row=6, column=2, padx=4, pady=8)

    def start_session(self):
        """
        Start a session for data recording
        :return: No returns
        """
        global g_break
        if self.current_session < self.sessions:
            g_break = self.deviating_time_val.get()
            self.session_total.set(str(self.current_session + 1) + "/" + str(self.sessions) + " Sessions")
            self.init_session_bar()

            self.deviating_time_input['state'] = DISABLED
            self.start_session_btn['state'] = DISABLED
            collect_data(self.current_session)
            self.start_session_btn['state'] = NORMAL
            self.deviating_time_input['state'] = NORMAL

            self.current_session += 1
            self.update_progress_bars(1)
        if self.current_session == self.sessions:
            self.set_countdown_text("Data collection complete!")
            wait(3)
            self.close()

    def countdown(self, t=5):
        """
        Countdown from a given value t. Also display the current time at the UI
        :param introduction_screen: instance of introduction screen from Gui.py. Set countdown text
        :param t: int, countdown time
        :return:
        """
        while t:
            min, secs = divmod(t, 60)
            time_format = '{:02d}:{:02d}'.format(min, secs)
            self.set_status_text("Pause! " + time_format)
            time.sleep(1)
            t -= 1

    def close(self):
        """
        Close the current window
        :return: No returns
        """
        self.destroy()
        introduction_window.withdraw()
        introduction_window.destroy()

    def change_image(self, path):
        """
        Change image to display
        :param path: string
                Path for the image
        :return: No returns
        """
        load = Image.open(path)
        load = load.resize((img_w, img_h), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        self.img = Label(self, image=render)
        self.img.image = render
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)
        introduction_window.update()

    def init_session_bar(self):
        """
        Initialization of the progress bars
        :return:
        """
        # self.progress_bar_session_val = 0
        self.progress_session["value"] = 0
        self.progress_session["maximum"] = len(Constant.label_display_with_rest)

    def set_status_text(self, text):
        """
        Set the Status text
        :param text: string
                Statustext to display
        :return: No returns
        """
        self.status_text.set(text)
        introduction_window.update()

    def set_gesture_description(self, text):
        """
        Set the description for the current gesture
        :param text: string
                Descriptiontext for gesture
        :return: No returns
        """
        self.gesture_text.set(text)
        introduction_window.update()

    def set_countdown_text(self, text):
        """
        Set the text for countdown
        :param text: string
                Countdowntext, noramly numbers from 0 to 5
        :return: No returns
        """
        self.countdown_value.set(text)
        introduction_window.update()

    def set_session_text(self, text):
        """
        Set the text for current session
        :param text: string
                Sessiontext, normaly the number of the current session
        :return:
        """
        self.session_text.set(text)
        introduction_window.update()

    def update_progress_bars(self, value):
        """
        Update the total- and session progress bar in this window
        :param value: int
                Update the progress bars by this value
        :return: No returns
        """
        self.progress_session["value"] += value
        self.progress_total["value"] += value
        introduction_window.update()

    def update_gesture_bar(self, value):
        """
        Update the gesture progress bar
        :param value: int
                Update the gesture progress bar by this value
        :return: No returns
        """
        if value > self.record_time:
            value = self.record_time
        self.progress_gesture["value"] = value
        introduction_window.update()


def pair_devices():
    """
    Pair the two Myo Armbands
    :return: boolean
            Returns True, if pairing was successful
            Returns False, if paring failed
    """
    global DEVICE_L
    global DEVICE_R
    with hub.run_in_background(device_listener):
        wait(.5)
        for i in range(3):
            devices = device_listener.devices
            for d in devices:
                if d.arm == Constant.LEFT:
                    DEVICE_L = d
                    DEVICE_L.stream_emg(True)
                elif d.arm == Constant.RIGHT:
                    DEVICE_R = d
                    DEVICE_R.stream_emg(True)
            if not (DEVICE_L is None) and not (DEVICE_R is None):
                DEVICE_R.vibrate(libmyo.VibrationType.short)
                DEVICE_L.vibrate(libmyo.VibrationType.short)
                log.info("Devices paired")
                return True
            wait(2)
    hub.stop()
    return False


def collect_raw_data():
    """
    Collect raw data from both myo devices.
    Collect the transmitted data is enabled via the variable status
    :return: No return
    """
    global EMG
    global ORI
    global ACC
    global GYR
    global status
    global g_record_time

    EMG, ORI, ACC, GYR = [], [], [], []
    dif, status = 0, 0
    start = time.time()
    while dif <= g_record_time:
        status = 1
        end = time.time()
        dif = end - start
        g_introduction_screen.update_gesture_bar(dif)
    status = 0
    log.info("EMG %d", len(EMG))
    log.info("IMU %d", len(ORI))

    emg_count_list.append(len(EMG))
    imu_count_list.append(len(ORI))


def collect_data(current_session):
    """
    The whole data collection process is handled by this function.
    Call functions to update UI.
    Handle vibrations for Myo armbands.
    handle breaks between two gestures.
    Save raw data, if not trial session.
    :param current_session: int
            The number of the current session, start by 0
    :return: No returns
    """
    global g_introduction_screen
    global g_files
    global g_record_time
    global g_raw_path
    global g_img_path
    global DEVICE_R
    global emg_r
    global emg_l

    g_introduction_screen.set_session_text("Session " + str(current_session + 1))
    g_introduction_screen.set_countdown_text("")

    with hub.run_in_background(gesture_listener.on_event):
        g_introduction_screen.countdown(3)
        for i in range(len(Constant.label)):
            path = g_raw_path + "/" + "s" + str(current_session) + Constant.label[i]
            emg_l, emg_r = [], []

            g_introduction_screen.set_gesture_description(Constant.hand_disinfection_description[i])
            g_introduction_screen.change_image(g_img_path + g_files[i])

            g_introduction_screen.set_countdown_text("")

            DEVICE_R.vibrate(type=libmyo.VibrationType.short)
            g_introduction_screen.set_status_text("Start!")

            collect_raw_data()

            DEVICE_L.vibrate(type=libmyo.VibrationType.short)

            g_introduction_screen.set_status_text("Pause")
            g_introduction_screen.update_gesture_bar(0)
            g_introduction_screen.update_progress_bars(1)

            if i < len(Constant.label) - 1:
                g_introduction_screen.set_gesture_description("Next: " + Constant.hand_disinfection_description[i + 1])
                g_introduction_screen.change_image(g_img_path + g_files[i + 1])

            if not g_trial:
                if not os.path.isdir(path):
                    os.mkdir(path)
                save_raw_data({"EMG": EMG, "ACC": ACC, "GYR": GYR, "ORI": ORI}, i, path + "/emg.csv", path + "/imu.csv")

                log.info("Collected emg data: " + str(len(EMG)))
                log.info("Collected imu data:" + str(len(ORI)))

            if g_mode == Constant.SEPARATE:
                wait(.5)
                if i < len(Constant.label) - 1:
                    g_introduction_screen.countdown(g_break)

    g_introduction_screen.set_countdown_text("")
    g_introduction_screen.set_status_text("Session " + str(current_session + 1) + " done!")
    g_introduction_screen.set_gesture_description("")
    g_introduction_screen.change_image("intro_screen.jpg")
    log.info("Data collection session %d complete", current_session)


def initialize_data_collection(raw_path, trial, mode, record_time=5):
    """
    Initialization the data collection
        - Try to pair Devices
        - Set global settings
    :param raw_path: string
            Describe the path to save raw data
    :param trial: boolean
           If True the the trial mode is active. Data will not be recorded
           If False the  the trial mode is deactivated. Data will be recorded
    :param mode: string
            The record mode can set by this parameter.
            "individual" for data collection with break between each gesture
            "continues" for data collection without break between each gesture
    :param record_time: int, default 5

    :return: boolean
            True, if device pairing was successful
            False, if device pairing was not successful
    """
    global g_files
    global g_record_time
    global g_raw_path
    global g_img_path
    global g_mode
    global g_trial

    if pair_devices():
        g_record_time = record_time
        g_introduction_screen.change_image("intro_screen.jpg")
        g_img_path = os.getcwd() + "/gestures/"
        g_files = os.listdir(g_img_path)
        g_introduction_screen.set_status_text("Hold every gesture for 5 seconds")
        g_raw_path = raw_path
        g_mode = mode
        g_trial = trial
        return True
    return False


def main():
    """
    Entry point for starting the Data collection.
    Setup the Configuration menu, display the UI.
    :return: No returns
    """
    introduction_window.wm_title("Introduction Screen")
    introduction_window.withdraw()

    data_collect = CollectDataWindow(collect_window)
    collect_window.wm_title("Collect Data")
    collect_window.mainloop()


if __name__ == '__main__':
    main()

import myo as libmyo
from myo import init, Hub, StreamEmg

DEVICE_R, DEVICE_L = None, None
init()
hub = Hub()
device_listener = libmyo.ApiDeviceListener()

with hub.run_in_background(device_listener):
    devices = device_listener.devices
    for d in devices:
        if d.arm == "left":
            DEVICE_L = d
            DEVICE_L.stream_emg(True)
        elif d.arm == "right":
            DEVICE_R = d
            DEVICE_R.stream_emg(True)
    if not (DEVICE_L is None) and not (DEVICE_R is None):
        DEVICE_R.vibrate(libmyo.VibrationType.short)
        DEVICE_L.vibrate(libmyo.VibrationType.lon)
hub.stop()

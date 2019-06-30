from tkinter import BOTH, StringVar, Label, HORIZONTAL, Entry, Button, IntVar, W, E, Tk, Checkbutton, VERTICAL
from tkinter.ttk import Progressbar, Separator, Frame

from PIL import Image, ImageTk

from Collect_data import trial_round_separate, trial_round_continuous, check_sample_rate, collect_data, \
    init_data_collection, pair_devices
from Constant import *
from Process_data import process_raw_data
from Save_Load import load_csv, create_directories

introduction_window = Tk()
collect_window = Tk()


class MainWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        self.p_val = StringVar(self, value="defaultUser")

        # main_window.title("EMG Recognition")
        self.collect_label = Label(self, text="Data Collection")
        self.proband_label = Label(self, text="Proband Name:")

        self.p_val = StringVar(self, value="defaultUser")
        self.proband = Entry(self, textvariable=self.p_val)

        self.sep1 = Separator(self, orient=HORIZONTAL)

        self.label3 = Label(self, text="Process data")

        self.sep3 = Separator(self, orient=HORIZONTAL)

        self.close_val = StringVar(self, value="close")
        self.close_btn = self.close = Button(self, textvariable=self.close_val, command=self.destroy)

        self.collect_data_btn = Button(self, text="Collect data",
                                       command=lambda: self.collect_data_ui(delete_old=False, session=2,
                                                                            proband=self.proband.get()))

        self.check_rate_btn = Button(self, text="Check sample rate",
                                     command=lambda: check_sample_rate(2, warm_start=False))

        self.process_data_btn = Button(self, text="Process data", command=process_raw_data)
        self.load_feature_btn = Button(self, text="Load feature file", command=load_csv)

        self.collect_label.grid(row=0, column=0, pady=4)

        self.collect_data_btn.grid(row=1, column=0, pady=8, padx=4, sticky=W)

        self.proband_label.grid(row=2, column=0, pady=4)
        self.proband.grid(row=2, column=1)

        self.label3.grid(row=3, pady=4)
        self.sep1.grid(row=3, column=0, sticky="ew", columnspan=5, padx=4)
        self.check_rate_btn.grid(row=5, column=0, pady=8, padx=4)
        self.process_data_btn.grid(row=5, column=1, pady=8, padx=4)
        self.load_feature_btn.grid(row=5, column=3, pady=8, padx=4)
        self.sep3.grid(row=9, column=0, sticky="ew", columnspan=5, padx=4)
        self.close_btn.grid(row=10, column=1, pady=8, padx=4)

    def collect_data_ui(self, delete_old=True, session=2, proband="defaultUser"):
        collect_window.deiconify()
        data_collect.user_path = "Collections/" + proband
        user_path = "Collections/" + proband
        raw_path = user_path + "/raw"
        create_directories(proband=proband, delete_old=delete_old, raw_path=raw_path,
                           raw_sep=user_path + "/raw_separate",
                           raw_con=user_path + "/raw_continues")


class CollectDataWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.user_path = ""

        self.sessions_label = Label(self, text="Durchgänge")
        self.record_time_label = Label(self, text="Zeit pro Geste")
        self.device_r_label = Label(self, text="Armband rechts")
        self.device_l_label = Label(self, text="Armband links")

        self.sep1 = Separator(self, orient=HORIZONTAL)
        self.sep2 = Separator(self, orient=HORIZONTAL)

        self.s_val = StringVar(self, value="10")
        self.r_val = StringVar(self, value="5")
        # self.trial_val = IntVar()
        # self.mode_val = IntVar()

        self.sessions_input = Entry(self, textvariable=self.s_val, width=3)
        self.record_time_input = Entry(self, textvariable=self.r_val, width=3)

        self.collect_separate_btn = Button(master=self, text="Collect Separate",
                                           command=lambda: self.collect_data(mode=INDIVIDUAL, trial=False))
        self.collect_continues_btn = Button(master=self, text="Collect Continues",
                                            command=lambda: self.collect_data(mode=CONTINUES, trial=False))
        self.trial_separate_btn = Button(master=self, text="Trial Separate",
                                         command=lambda: self.collect_data(mode=INDIVIDUAL, trial=True))
        self.trial_continues_btn = Button(master=self, text="Trial Continues",
                                          command=lambda: self.collect_data(mode=CONTINUES, trial=True))
        self.close_btn = Button(self, text="Close", command=self.destroy)

        self.sessions_label.grid(row=0, column=0, pady=4, padx=2)
        self.sessions_input.grid(row=0, column=1)

        self.record_time_label.grid(row=1, column=0, pady=4, padx=2)
        self.record_time_input.grid(row=1, column=1)

        self.collect_separate_btn.grid(row=4, column=0, pady=4, padx=8)
        self.collect_continues_btn.grid(row=4, column=2, pady=4, padx=8)
        Separator(self, orient=VERTICAL).grid(row=4, column=1, rowspan=3)
        self.trial_separate_btn.grid(row=5, column=0, pady=4, padx=8)
        self.trial_continues_btn.grid(row=5, column=2, pady=4, padx=8)

        self.sep1.grid(row=6, column=0, sticky="ew", columnspan=2, padx=4)

        self.close_btn.grid(row=7, column=1, pady=8, padx=4, sticky=E)

    def collect_data(self, mode, trial):
        sessions = int(self.s_val.get())
        if mode == INDIVIDUAL:
            raw_path = self.user_path + "/raw_separate"
        else:
            raw_path = self.user_path + "/raw_continues"

        introduction_window.deiconify()
        introduction_screen.init_totalbar(sessions)
        introduction_screen.init_sessionbar()
        init_data_collection(raw_path=raw_path,
                             trial=trial,
                             mode=mode,
                             introduction_screen=introduction_screen,
                             training_time=int(self.r_val.get()))
        introduction_screen.sessions = sessions
        # introduction_screen.mode = self.mode_val


class IntroductionScreen(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        load = Image.open("intro_screen.jpg")
        load = load.resize((450, 400), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        self.img = Label(self, image=render)
        self.img.image = render
        self.description_text = StringVar()
        self.description_text.set("Untertitel")
        self.countdown_value = StringVar()

        self.description_label = Label(self, textvariable=self.description_text)
        self.gesture_countdown_label = Label(self, textvariable=self.countdown_value)

        self.start_session_btn = Button(self, text="Start Session", command=self.start_session)

        self.progress_total = Progressbar(self, orient="horizontal", length=200, mode='determinate')
        self.progress_session = Progressbar(self, orient="horizontal", length=200, mode='determinate')

        self.description_label.grid(row=1, column=0, pady=4, columnspan=2)

        self.session_text = StringVar()
        self.gesture_text = StringVar()
        self.session_text.set("Session 1")

        self.session_label = Label(self, textvariable=self.session_text)
        self.gesture_label = Label(self, textvariable=self.gesture_text)
        self.total_label = Label(self, text="Total")

        # Style
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)

        self.gesture_countdown_label.grid(row=1, column=1, columnspan=2, pady=4)

        self.gesture_label.grid(row=2, column=0, columnspan=3, pady=2, padx=2, sticky=W)

        self.session_label.grid(row=3, column=0, padx=2, sticky=W)
        self.progress_session.grid(row=3, column=1, pady=2, sticky=W)
        self.start_session_btn.grid(row=3, column=2, padx=4, rowspan=2)

        self.total_label.grid(row=4, column=0, padx=2, sticky=W)
        self.progress_total.grid(row=4, column=1, pady=4, sticky=W)

        self.sessions = -1
        self.current_session = 0
        self.mode = ""

    def start_session(self):
        if self.current_session <= self.sessions:
            self.init_sessionbar()
            collect_data(self.current_session)
            self.current_session += 1
            self.update_progressbars(1)
            return
        else:
            self.set_description_val("Data collection complete!")
            # print("Data collection complete!")

    def change_img(self, path):
        load = Image.open(path)
        load = load.resize((450, 400), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        self.img = Label(self, image=render)
        self.img.image = render
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)
        introduction_window.update()

    def init_gesturebar(self, record_time):
        self.progressbar_gesture_val = 0
        # self.progress_gesture["value"] = self.progressbar_gesture_val
        # self.progress_gesture["maximum"] = record_time

    def init_sessionbar(self):
        self.progressbar_session_val = 0
        self.progress_session["value"] = self.progressbar_session_val
        self.progress_session["maximum"] = len(label_display)

    def init_totalbar(self, sessions):
        self.progressbar_total_val = 0
        self.progress_total["value"] = self.progressbar_total_val
        self.progress_total["maximum"] = sessions * len(label_display)

    def set_description_text(self, text):
        self.description_text.set(text)
        introduction_window.update()

    def set_gesture_label(self, text):
        self.gesture_text.set(text)
        introduction_window.update()

    def set_description_val(self, text):
        self.countdown_value.set(text)
        introduction_window.update()

    def update_progressbars(self, value):
        self.progress_session["value"] += value
        self.progress_total["value"] += value
        introduction_window.update()


introduction_screen = IntroductionScreen(introduction_window)
introduction_window.wm_title("Introduction Screen")
introduction_window.withdraw()

data_collect = CollectDataWindow(collect_window)
collect_window.wm_title("Collect Data")
collect_window.withdraw()

# from tkinter import *
# pip install pillow
from tkinter import BOTH, StringVar, Label, HORIZONTAL, Entry, Button, IntVar, W, E, Tk, Checkbutton
from tkinter.ttk import Progressbar, Separator, Frame

from PIL import Image, ImageTk

from Collect_data import trial_round_separate, trial_round_continuous, check_sample_rate, collect_data, \
    init_data_collection, pair_devices
from Constant import *
from Process_data import process_raw_data
from Save_Load import load_csv, create_directories


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

        # icons
        load_false = Image.open("icons8-false-100.png")
        # load_false = load_false.resize((50, 50), Image.ANTIALIAS)
        # self.icon_false = ImageTk.PhotoImage(load_false)
        #
        # load_true = Image.open("icons8-true-100.png")
        # load_true = load_true.resize((50, 50), Image.ANTIALIAS)
        # self.icon_true = ImageTk.PhotoImage(load_true)
        #
        # self.device_l_img = Label(self, image=self.icon_false)
        # self.device_l_img.image = self.icon_false
        # self.device_r_img = Label(self, image=self.icon_false)
        # self.device_r_img.image = self.icon_false

        self.sessions_label = Label(self, text="Durchgänge")
        self.record_time_label = Label(self, text="Zeit pro Geste")
        self.sep1 = Separator(self, orient=HORIZONTAL)
        self.sep2 = Separator(self, orient=HORIZONTAL)

        self.device_r_label = Label(self, text="Armband rechts")
        self.device_l_label = Label(self, text="Armband links")

        self.close_btn = Button(self, text="Close", command=self.destroy)
        s_val = StringVar(self, value="10")
        r_val = StringVar(self, value="5")
        self.sessions_input = Entry(self, textvariable=s_val, width=3)
        self.record_time_input = Entry(self, textvariable=r_val, width=3)

        # self.trial_separate_btn = Button(self, text="Trial round separated",command=lambda: trial_round_separate(save_label=save_label,display_label=hand_disinfection_description))
        # self.trial_continuous_btn = Button(self, text="Trial round",command=lambda: trial_round_continuous(save_label=save_label,display_label=hand_disinfection_description))
        # self.collect_continues_btn = Button(master=self, text="Durchgehender Ablauf",command=lambda: self.collect_data(sessions=int(self.sessions_input.get()),training_time=int(self.record_time_input.get()),raw_path=self.user_path + "/raw_separate",mode=CONTINUES))
        self.collect_data_btn = Button(master=self, text="Start data collection", command=self.collect_data)

        self.trial_val = IntVar()
        self.trial_cbox = Checkbutton(self, text="Testdurchgang", variable=self.trial_val)

        self.mode_val = IntVar()
        self.mode_cbox = Checkbutton(self, text="Separate Aufzeichnung", variable=self.mode_val)
        self.mode_cbox.var = self.mode_val
        # self.device_l_label.grid(row=0,column=2,padx=4)
        # self.device_r_label.grid(row=1, column=2, padx=4)
        # self.device_l_img.grid(row=0,column=3,padx=4)
        # self.device_r_img.grid(row=1, column=3, padx=4)

        # self.trial_separate_btn.grid(row=0, column=0, pady=4, padx=4)
        # self.trial_continuous_btn.grid(row=0, column=1, pady=4, padx=4)
        # self.sep1.grid(row=1, column=0, columnspan=2, sticky=E)
        self.sessions_label.grid(row=0, column=0, pady=4, padx=2)
        self.sessions_input.grid(row=0, column=1)

        self.record_time_label.grid(row=1, column=0, pady=4, padx=2)
        self.record_time_input.grid(row=1, column=1)

        self.mode_cbox.grid(row=2, column=0, pady=4, padx=4, sticky=W)
        self.trial_cbox.grid(row=2, column=1, pady=4, padx=4, sticky=W)
        self.collect_data_btn.grid(row=4, column=0, pady=4, padx=4)

        self.sep1.grid(row=5, column=0, sticky="ew", columnspan=2, padx=4)

        self.close_btn.grid(row=6, column=1, pady=8, padx=4, sticky=E)

    def collect_data(self):
        # introduction_screen.update()
        sessions = int(self.sessions_input.get())

        if self.mode_val == 1:
            raw_path = self.user_path + "/raw_separate"
        else:
            raw_path = self.user_path + "/raw_continues"

        mode = self.mode_val.get()
        trial = self.trial_val.get()

        introduction_window.deiconify()
        introduction_screen.init_totalbar(sessions)
        introduction_screen.init_sessionbar()
        init_data_collection(raw_path=raw_path,
                             trial=trial,
                             mode=mode,
                             introduction_screen=introduction_screen,
                             session=sessions,
                             training_time=int(self.record_time_input.get()))
        introduction_screen.sessions = sessions
        introduction_screen.mode = self.mode_val

class IntroductionScreen(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        load = Image.open("intro_screen.jpg")
        load = load.resize((650, 600), Image.ANTIALIAS)
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
        if self.current_session < self.sessions:
            self.init_sessionbar()
            collect_data(self.current_session, self.mode)
            self.current_session += 1
            self.update_total_bar(1)
            return
        else:
            print("Data collection complete!")

    def change_img(self, path):
        load = Image.open(path)
        load = load.resize((650, 600), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        self.img = Label(self, image=render)
        self.img.image = render
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)
        introduction_window.update()

    def init_gesturebar(self, record_time):
        self.progressbar_gesture_val = 0
        self.progress_gesture["value"] = self.progressbar_gesture_val
        self.progress_gesture["maximum"] = record_time

    def init_sessionbar(self):
        self.progressbar_session_val = 0
        self.progress_session["value"] = self.progressbar_session_val
        self.progress_session["maximum"] = len(label_display)

    def init_totalbar(self, sessions):
        self.progressbar_total_val = 0
        self.progress_total["value"] = self.progressbar_total_val
        self.progress_total["maximum"] = sessions

    def set_description_text(self, text):
        self.description_text.set(text)
        introduction_window.update()

    def set_gesture_label(self, text):
        self.gesture_text.set(text)
        introduction_window.update()

    def set_description_val(self, text):
        self.countdown_value.set(text)
        introduction_window.update()

    def update_total_bar(self, value):
        self.progressbar_total_val += value
        self.progress_total["value"] = self.progressbar_total_val
        introduction_window.update()

    def update_session_bar(self, value):
        self.progressbar_session_val += value
        self.progress_session["value"] = self.progressbar_session_val
        introduction_window.update()


introduction_window = Tk()
introduction_screen = IntroductionScreen(introduction_window)
introduction_window.wm_title("Introduction Screen")
introduction_window.withdraw()

collect_window = Tk()
data_collect = CollectDataWindow(collect_window)
collect_window.wm_title("Collect Data")
collect_window.withdraw()

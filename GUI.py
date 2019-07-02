from tkinter import BOTH, StringVar, Label, HORIZONTAL, Entry, Button, IntVar, W, E, Tk, Checkbutton, VERTICAL
from tkinter.ttk import Progressbar, Separator, Frame

from PIL import Image, ImageTk

from Collect_data import collect_data, init_data_collection
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
        self.proband_input = Entry(self, textvariable=self.p_val)

        self.sep1 = Separator(self, orient=HORIZONTAL)

        self.label3 = Label(self, text="Process data")

        self.sep3 = Separator(self, orient=HORIZONTAL)

        self.close_val = StringVar(self, value="close")
        self.close_btn = self.close = Button(self, textvariable=self.close_val, command=self.destroy)

        self.collect_data_btn = Button(self, text="Collect data",
                                       command=lambda: self.collect_data_ui(delete_old=True,
                                                                            proband=self.proband_input.get()))

        self.process_data_btn = Button(self, text="Process data", command=process_raw_data)
        self.load_feature_btn = Button(self, text="Load feature file", command=load_csv)

        self.collect_label.grid(row=0, column=0, pady=4, columnspan=2)

        self.proband_label.grid(row=1, column=0, pady=4)
        self.proband_input.grid(row=1, column=1, padx=8)

        self.collect_data_btn.grid(row=2, column=0, pady=8, padx=4, sticky=W)

        self.sep1.grid(row=3, column=0, sticky="ew", columnspan=5, padx=4, pady=8)
        self.label3.grid(row=4, pady=4, columnspan=2)

        self.process_data_btn.grid(row=5, column=0, pady=8, padx=4)
        self.load_feature_btn.grid(row=5, column=1, pady=8, padx=4)

        self.sep3.grid(row=6, column=0, sticky="ew", columnspan=5, padx=4)

        self.close_btn.grid(row=7, column=1, pady=8, padx=4)

    def collect_data_ui(self, delete_old=True, proband="defaultUser"):
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

        self.session_val = IntVar(self, value=10)
        self.record_time_val = IntVar(self, value=5)

        self.sessions_input = Entry(self, textvariable=self.session_val, width=3)
        self.record_time_input = Entry(self, textvariable=self.record_time_val, width=3)

        self.collect_separate_btn = Button(master=self, text="Collect Separate",
                                           command=lambda: self.introduction_screen_ui(mode=INDIVIDUAL, trial=False))
        self.collect_continues_btn = Button(master=self, text="Collect Continues",
                                            command=lambda: self.introduction_screen_ui(mode=CONTINUES, trial=False))
        self.trial_separate_btn = Button(master=self, text="Trial Separate",
                                         command=lambda: self.introduction_screen_ui(mode=INDIVIDUAL, trial=True))
        self.trial_continues_btn = Button(master=self, text="Trial Continues",
                                          command=lambda: self.introduction_screen_ui(mode=CONTINUES, trial=True))
        self.close_btn = Button(self, text="Close", command=collect_window.withdraw)

        self.sessions_label.grid(row=0, column=0, pady=4, padx=2)
        self.sessions_input.grid(row=0, column=1)

        self.record_time_label.grid(row=1, column=0, pady=4, padx=2)
        self.record_time_input.grid(row=1, column=1)

        self.collect_separate_btn.grid(row=4, column=0, pady=4, padx=8)
        self.collect_continues_btn.grid(row=4, column=2, pady=4, padx=8)
        self.trial_separate_btn.grid(row=5, column=0, pady=4, padx=8)
        self.trial_continues_btn.grid(row=5, column=2, pady=4, padx=8)

        self.sep1.grid(row=6, column=0, sticky="ew", columnspan=3, padx=4, pady=8)

        self.close_btn.grid(row=7, column=1, pady=8, padx=4, sticky=E)

    def introduction_screen_ui(self, mode, trial):
        introduction_screen = IntroductionScreen(introduction_window, record_time=self.record_time_val.get(),
                                                 sessions=self.session_val.get())
        introduction_window.deiconify()
        if mode == INDIVIDUAL:
            title = "Collect separate data"
            raw_path = self.user_path + "/raw_separate"
        else:
            title = "Collect continues data"
            raw_path = self.user_path + "/raw_continues"
        if trial:
            title += " TRIAL"
        introduction_window.title(title)

        introduction_screen.init_sessionbar()
        init_data_collection(raw_path=raw_path,
                             trial=trial,
                             mode=mode,
                             introduction_screen=introduction_screen,
                             training_time=self.record_time_val.get())
        introduction_window.mainloop()


class IntroductionScreen(Frame):
    def __init__(self, master=None, record_time=5, sessions=10):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        load = Image.open("intro_screen.jpg")
        load = load.resize((550, 500), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        self.img = Label(self, image=render)
        self.img.image = render
        self.status_text = StringVar()
        self.countdown_value = StringVar()
        self.sessions = sessions
        self.record_time = record_time
        self.current_session = 0
        self.mode = ""

        self.status_label = Label(self, textvariable=self.status_text)  # Start, Pause
        self.gesture_countdown_label = Label(self, textvariable=self.countdown_value)

        self.start_session_btn = Button(self, text="Start Session", command=self.start_session)
        self.close_btn = Button(self, text="Close", command=self.quit)

        self.progress_total = Progressbar(self, orient="horizontal", length=200, mode='determinate')
        self.progress_total["maximum"] = self.sessions * len(label_display)
        self.progress_session = Progressbar(self, orient="horizontal", length=200, mode='determinate')
        self.progress_gesture = Progressbar(self, orient="horizontal", length=200, mode='determinate')
        self.progress_gesture["maximum"] = self.record_time

        self.session_text = StringVar()
        self.session_text.set("Session 1")
        self.gesture_text = StringVar()

        self.session_label = Label(self, textvariable=self.session_text)
        self.gesture_label = Label(self, textvariable=self.gesture_text)
        self.total_label = Label(self, text="Total")

        # Style
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)

        self.status_label.grid(row=1, column=1, pady=2, padx=2, sticky=W)
        self.gesture_countdown_label.grid(row=1, column=1, pady=4, sticky=E)

        self.gesture_label.grid(row=2, column=1, columnspan=3, pady=4, padx=2, sticky=W)

        self.progress_gesture.grid(row=3, column=1, padx=4, sticky=W)

        self.session_label.grid(row=4, column=0, pady=4, sticky=W)
        self.progress_session.grid(row=4, column=1, padx=4, sticky=W)
        self.start_session_btn.grid(row=4, column=2, padx=4)

        self.total_label.grid(row=5, column=0, pady=4, sticky=W)
        self.progress_total.grid(row=5, column=1, padx=4, sticky=W)

        self.close_btn.grid(row=5, column=2, padx=4, pady=8)

    def start_session(self):
        if self.current_session <= self.sessions:
            self.init_sessionbar()
            collect_data(self.current_session)
            self.current_session += 1
            self.update_progressbars(1)
            return
        else:
            self.set_countdown_text("Data collection complete!")

    def change_img(self, path):
        load = Image.open(path)
        load = load.resize((550, 500), Image.ANTIALIAS)
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


introduction_window.wm_title("Introduction Screen")
introduction_window.withdraw()

data_collect = CollectDataWindow(collect_window)
collect_window.wm_title("Collect Data")
collect_window.withdraw()

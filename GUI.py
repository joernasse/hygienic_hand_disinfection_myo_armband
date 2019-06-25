from tkinter import *
# pip install pillow
from tkinter.ttk import Progressbar, Separator

from PIL import Image, ImageTk

from Collect_data import trial_round_separate, trial_round_continuous, check_sample_rate, \
    collect_continuous_trainings_data, collect_separate_training_data, collect_gui_sep_data, init_data_collection
from Constant import save_label, hand_disinfection_display, label_display, CONTINUES, SEPARATE
from Process_data import process_raw_data
from Save_Load import load_csv, create_directories


class MainWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        self.p_val = StringVar(self, value="defaultUser")

        # main_window.title("EMG Recognition")
        self.label1 = Label(self, text="Data Collection")
        self.label2 = Label(self, text="Proband Name:")

        self.p_val = StringVar(self, value="defaultUser")
        self.proband = Entry(self, textvariable=self.p_val)

        self.sep1 = Separator(self, orient=HORIZONTAL)

        self.sep2 = Separator(self, orient=VERTICAL)

        self.label3 = Label(self, text="Process data")

        self.sep3 = Separator(self, orient=HORIZONTAL)

        self.close_val = StringVar(self, value="close")
        self.close_btn = self.close = Button(self, textvariable=self.close_val, command=self.destroy)

        self.collect_data_btn = Button(self, text="Collect data",
                                       command=lambda: self.collect_data_ui(delete_old=False, session=2,
                                                                            proband=self.proband.get()))

        self.trial_separat_btn = Button(self, text="Trial round separated",
                                        command=lambda: trial_round_separate(save_label=save_label,
                                                                             display_label=hand_disinfection_display))

        self.trial_continuous_btn = Button(self, text="Trial round",
                                           command=lambda: trial_round_continuous(save_label=save_label,
                                                                                  display_label=hand_disinfection_display))

        self.check_rate_btn = Button(self, text="Check sample rate",
                                     command=lambda: check_sample_rate(2, warm_start=False))

        self.process_data_btn = Button(self, text="Process data", command=process_raw_data)
        self.load_feature_btn = Button(self, text="Load feature file", command=load_csv)

        self.label1.grid(row=0, column=0, pady=4)
        self.sep2.grid(row=0, column=3, sticky='ns', rowspan=3, padx=1, pady=4)

        self.collect_data_btn.grid(row=1, column=0, pady=8, padx=4)
        self.trial_separat_btn.grid(row=1, column=4, pady=8, padx=4)

        self.trial_continuous_btn.grid(row=2, column=4, pady=8, padx=4)
        self.label2.grid(row=2, column=0, pady=4)
        self.proband.grid(row=2, column=1)

        self.label3.grid(row=3, pady=4)
        self.sep1.grid(row=3, column=0, sticky="ew", columnspan=5, padx=4)
        self.check_rate_btn.grid(row=5, column=0, pady=8, padx=4)
        self.process_data_btn.grid(row=5, column=1, pady=8, padx=4)
        self.load_feature_btn.grid(row=5, column=3, pady=8, padx=4)
        self.sep3.grid(row=9, column=0, sticky="ew", columnspan=5, padx=4)
        self.close_btn.grid(row=10, column=1, pady=8, padx=4)

    def collect_data_ui(self, delete_old=True, session=2, proband="defaultUser"):
        # collect = Tk()
        # data_collect = Collect_Data_Window(collect)
        # collect.wm_title("Collect Data")
        # collect.geometry("400x400")

        collect.deiconify()
        data_collect.user_path = "Collections/" + proband
        user_path = "Collections/" + proband
        raw_path = user_path + "/raw"
        create_directories(proband=proband, delete_old=delete_old, raw_path=raw_path,
                           raw_sep=user_path + "/raw_separate",
                           raw_con=user_path + "/raw_continues")
        # data_collect.deiconify()


class CollectDataWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.user_path = ""

        self.label1 = Label(self, text="Durchgänge")
        self.label2 = Label(self, text="Zeit pro Geste")
        self.sep1 = Separator(self, orient=HORIZONTAL)
        self.close_btn = Button(self, text="Close", command=self.destroy)
        s_val = StringVar(self, value="10")
        r_val = StringVar(self, value="5")
        self.sessions = Entry(self, textvariable=s_val, width=3)
        self.record_time = Entry(self, textvariable=r_val, width=3)

        self.collect_separate_btn = Button(master=self, text="Collect separate data",
                                           command=lambda: self.collect_data(sessions=int(self.sessions.get()),
                                                                             training_time=int(self.record_time.get()),
                                                                             raw_path=self.user_path + "/raw_separate",
                                                                             mode=SEPARATE))
        self.collect_continues_btn = Button(master=self, text="Collect separate data",
                                            command=lambda: self.collect_data(sessions=int(self.sessions.get()),
                                                                              training_time=int(self.record_time.get()),
                                                                              raw_path=self.user_path + "/raw_separate",
                                                                              mode=CONTINUES))

        self.label1.grid(row=1, pady=4, padx=2)
        self.label2.grid(row=2, pady=4, padx=2)
        self.sep1.grid(row=5, column=0, sticky="ew", columnspan=5)
        self.close_btn.grid(row=6, column=1, pady=4, padx=4)
        self.sessions.grid(row=1, column=1)
        self.record_time.grid(row=2, column=1)
        self.collect_separate_btn.grid(row=3, column=0, pady=4, padx=4)
        self.collect_continues_btn.grid(row=3, column=2, pady=8, padx=4)

    def collect_data(self, sessions, training_time, raw_path, mode):
        introduction.deiconify()
        introduction_screen.init_totalbar(sessions)
        introduction_screen.init_sessionbar()
        init_data_collection(raw_path=raw_path, introduction_screen=introduction_screen, session=sessions,
                             training_time=training_time)
        introduction_screen.sessions = sessions

        # if mode == SEPARATE:
        #     collect_separate_training_data(raw_path=raw_path, session=sessions, training_time=training_time,
        #                                    introduction_screen=introduction_screen)
        # elif mode == CONTINUES:
        #     collect_continuous_trainings_data(raw_path=raw_path, session=sessions, training_time=training_time,
        #                                       introduction_screen=introduction_screen)


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
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)
        self.description_text = StringVar()
        self.description_text.set("Untertitel")
        self.description_value = StringVar()

        self.description_label = Label(self, textvariable=self.description_text)
        self.description_val_label = Label(self, textvariable=self.description_value)

        self.start_session_btn = Button(self, text="Start Session", command=self.start_session)
        self.start_session_btn.grid(row=2, column=2, padx=4, rowspan=2)

        # self.progressbar_total_val = 0
        # self.progressbar_session_val = 0
        self.progress_total = Progressbar(self, orient="horizontal", length=200, mode='determinate')
        self.progress_session = Progressbar(self, orient="horizontal", length=200, mode='determinate')

        self.description_label.grid(row=1, column=0, pady=4, columnspan=2, sticky=W)
        self.description_val_label.grid(row=1, column=2, pady=4, sticky=W)

        self.session_text = StringVar()
        self.session_text.set("Session 1")
        self.session_label = Label(self, textvariable=self.session_text)
        self.session_label = Label(self, textvariable=self.session_text)
        self.session_label.grid(row=2, column=0, padx=2, sticky=W)
        self.progress_session.grid(row=2, column=1, pady=2, sticky=W)

        self.total_label = Label(self, text="Total")
        self.total_label.grid(row=3, column=0, padx=2, sticky=W)
        self.progress_total.grid(row=3, column=1, pady=4, sticky=W)

        self.sessions = -1
        self.current_session = 0

    def start_session(self):
        if self.current_session < self.sessions:
            self.init_sessionbar()
            collect_gui_sep_data(self.current_session)
            self.current_session += 1
            self.update_total_bar(1)
            return
        else:
            print("Data collection complete!")

    def change_img(self, path):
        # full_path = "img/" + file
        load = Image.open(path)
        load = load.resize((450, 400), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        self.img = Label(self, image=render)
        self.img.image = render
        self.img.grid(row=0, column=0, padx=8, pady=8, columnspan=3)
        introduction.update()

    def init_sessionbar(self):
        self.progressbar_session_val = 0
        self.progress_session["value"] = self.progressbar_session_val
        self.progress_session["maximum"] = len(label_display)

    def init_totalbar(self, sessions):
        self.progressbar_total_val = 0
        self.progress_total["value"] = self.progressbar_total_val
        self.progress_total["maximum"] = sessions

    def set_descr_text(self, text):
        self.description_text.set(text)
        introduction.update()

    def set_descr_val(self, text):
        self.description_value.set(text)
        introduction.update()

    def update_total_bar(self, value):
        self.progressbar_total_val += value
        self.progress_total["value"] = self.progressbar_total_val
        introduction.update()

    def update_session_bar(self, value):
        self.progressbar_session_val += value
        self.progress_session["value"] = self.progressbar_session_val
        introduction.update()


introduction = Tk()
introduction_screen = IntroductionScreen(introduction)
introduction.wm_title("Introduction Screen")
introduction.withdraw()

collect = Tk()
data_collect = CollectDataWindow(collect)
collect.wm_title("Collect Data")
collect.geometry("300x300")
collect.withdraw()

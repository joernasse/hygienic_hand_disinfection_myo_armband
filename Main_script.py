from __future__ import print_function

from tkinter import *

from GUI import sessions, record_time, proband, close_val, main_window, data_collect_window
from Myo_communication import *
from Process_data import process_raw_data
from Save_Load import *

# ordentliche prediction Ergebnisse
# PREDICT_TIME: float = 0.5
from Constant import hand_disinfection, hand_disinfection_display


def main():
    Button(master=main_window, text="Collect data",
           command=lambda: collect_data_ui(delete_old=True, session=2, proband=proband.get())).grid(row=1, pady=8,
                                                                                                    padx=4)
    Button(master=main_window, text="Trial round separated",
           command=lambda: trial_round_separate(save_label=hand_disinfection,
                                                display_label=hand_disinfection_display)).grid(row=1, column=4, pady=8,
                                                                                               padx=4)
    Button(master=main_window, text="Trial round", command=lambda: trial_round_continuous(save_label=hand_disinfection,
                                                                                          display_label=hand_disinfection_display)).grid(
        row=2, column=4, pady=8, padx=4)

    Button(master=main_window, text="Check sample rate", command=lambda: check_sample_rate(2, warm_start=False)).grid(
        row=5, column=0, pady=8, padx=4)
    Button(master=main_window, text="Process data", command=process_raw_data).grid(row=5, column=1, pady=8, padx=4)
    Button(master=main_window, text="Load feature file", command=load_csv).grid(row=5, column=3, pady=8, padx=4)

    close_val.set("CLOSE!")

    # close.text="CLOSE!"
    main_window.mainloop()


def collect_data_ui(delete_old=True, session=2, proband="defaultUser"):
    user_path = "Collections/" + proband
    raw_path = user_path + "/raw"
    create_directories(proband=proband, delete_old=delete_old, raw_path=raw_path, raw_sep=user_path + "/raw_separate",
                       raw_con=user_path + "/raw_continues")

    Button(master=data_collect_window, text="Collect separate data",
           command=lambda: collect_separate_training_data(save_label=hand_disinfection,
                                                          display_label=hand_disinfection_display,
                                                          raw_path=user_path + "/raw_separate",
                                                          session=int(int(sessions.get())),
                                                          training_time=int(int(record_time.get())))).grid(row=3,
                                                                                                           column=0,
                                                                                                           pady=4,
                                                                                                           padx=4)
    Button(master=data_collect_window, text="Collect continues data",
           command=lambda: collect_continuous_trainings_data(save_label=hand_disinfection, session=session,
                                                             display_label=hand_disinfection_display,
                                                             raw_path=user_path + "/raw_separate",
                                                             training_time=5)).grid(row=3, column=2, pady=8, padx=4)

    data_collect_window.deiconify()
    data_collect_window.mainloop()


if __name__ == '__main__':
    main()

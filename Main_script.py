from __future__ import print_function

from tkinter import Tk, Button
from GUI import main_window, data_collect_window
from Myo_communication import check_sample_rate, collect_separate_training_data, \
    collect_continuous_trainings_data
from Process_data import process_raw_data
from Save_Load import load_csv, create_directories

# ordentliche prediction Ergebnisse
# PREDICT_TIME: float = 0.5

label_display = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6', 'Rest']

hand_disinfection = ['Step1', 'Step2', 'Step2_1', 'Step3', 'Step4', 'Step5', 'Step5_1', 'Step6', 'Step6_1',
                     'Rest']

hand_disinfection_display = ['Step 1    - palm and wrist',
                             'Step 2    - right palm on left back',
                             'Step 2.1  - left palm on right back',
                             'Step 3    - spread and interlocked fingers palm',
                             'Step 4    - Interlock fingers',
                             'Step 5    - Circular rubbing of the right thumb in the closed left palm of the hand',
                             'Step 5.1  - Circular rubbing of the left thumb in the closed right palm of the hand ',
                             'Step 6    - Circular movement of closed fingertips of the right hand on the left palm of the hand',
                             'Step 6.1  - Circular movement of closed fingertips of the left hand on the right palm of the hand',
                             'Rest      - Do nothing']


def main():
    Button(master=main_window, text="Collect data",
           command=lambda: collect_data_ui(delete_old=True, session=2, proband="defaultUser")).pack(pady=8)
    # Button(master=window, text="Train classifier", command=lambda: train_classifier()).pack(pady=8)
    Button(master=main_window, text="Check sample rate", command=lambda: check_sample_rate(100)).pack(pady=8)
    # Button(master=window, text="Predict live Gesture", command=predict).pack(pady=8)
    Button(master=main_window, text="Process data", command=process_raw_data).pack(pady=8)
    Button(master=main_window, text="Load feature file", command=load_csv).pack(pady=8)
    Button(master=main_window, text="Close", command=main_window.destroy).pack(pady=8)
    # Button(master=main_window,text="Collect Data",command=data_collect_window.show())

    main_window.mainloop()
    data_collect_window.mainloop()


def collect_data_ui(delete_old=True, session=2, proband="defaultUser"):
    user_path = "Collections/" + proband
    raw_path = user_path + "/raw"
    create_directories(proband=proband, delete_old=delete_old, raw_path=raw_path, raw_sep=user_path + "/raw_separate",
                       raw_con=user_path + "/raw_continues")

    Button(master=data_collect_window, text="Collect separate data",
           command=lambda: collect_separate_training_data(save_label=hand_disinfection,
                                                          display_label=hand_disinfection_display,
                                                          raw_path=user_path + "/raw_separate",
                                                          session=session,
                                                          training_time=5,
                                                          proband=proband)).pack(pady=8)
    Button(master=data_collect_window, text="Collect continues data",
           command=lambda: collect_continuous_trainings_data(save_label=hand_disinfection, session=session,
                                                             display_label=hand_disinfection_display,
                                                             raw_path=user_path + "/raw_separate",
                                                             training_time=5, proband=proband)).pack(pady=8)
    data_collect_window.mainloop()



if __name__ == '__main__':
    main()

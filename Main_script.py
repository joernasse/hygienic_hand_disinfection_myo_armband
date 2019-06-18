from __future__ import print_function

from tkinter import Tk, Button

from Classification import train_classifier, predict
from Myo_communication import collect_training_data
from Process_data import placeholder
from Save_Load import load_raw_csv, load_csv

window = Tk()
window.geometry("300x250")
window.title("EMG Recognition")

# ordentliche prediction Ergebnisse
# PREDICT_TIME: float = 0.5

label_display = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6', 'Rest']

hand_disinfection_1 = ['Step 1', 'Step 1.1', 'Step 2', 'Step 2.1', 'Step 3', 'Step 3.1', 'Step 4',
                       'Step 4.1', 'Step 5',
                       'Step 5.1', 'Step 6', 'Step 6.1', 'Rest']
hand_disinfection_light = ['Step 1', 'Step 4', 'Rest']

hand_desinfection_description = ['Step 1 - palm and wrist',
                                 'Step 2 - right palm on left back',
                                 'Step 2.1 - left palm on right back',
                                 'Step 3 - spread and interlocked fingers palm',
                                 'Step 4 - ',
                                 ]


def main():
    Button(master=window, text="Collect data",
           command=lambda: collect_training_data(hand_disinfection_light)).pack(pady=8)
    Button(master=window, text="Train classifier", command=lambda: train_classifier()).pack(pady=8)
    Button(master=window, text="Predict live Gesture", command=predict).pack(pady=8)
    Button(master=window, text="Process data", command=placeholder).pack(pady=8)
    Button(master=window, text="Close", command=window.destroy).pack(pady=8)

    window.mainloop()


if __name__ == '__main__':
    main()

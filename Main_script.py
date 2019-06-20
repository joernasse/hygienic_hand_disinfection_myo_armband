from __future__ import print_function

from tkinter import Tk, Button

from Classification import train_classifier, predict
from Myo_communication import check_sample_rate, status, collect_separate_training_data
from Process_data import process_raw_data
from Save_Load import load_csv

window = Tk()
window.geometry("300x350")
window.title("EMG Recognition")

# ordentliche prediction Ergebnisse
# PREDICT_TIME: float = 0.5

label_display = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6', 'Rest']

hand_disinfection_1 = ['Step 1', 'Step 1.1', 'Step 2', 'Step 2.1', 'Step 3', 'Step 4', 'Step 4.1', 'Step 5', 'Step 5.1',
                       'Step 6', 'Step 6.1', 'Rest']
hand_disinfection_light = ['Step 1', 'Step 4', 'Rest']

hand_disinfection_description = ['Step 1 - palm and wrist',
                                 'Step 2 - right palm on left back',
                                 'Step 2.1 - left palm on right back',
                                 'Step 3 - spread and interlocked fingers palm',
                                 'Step 4 - Interlock fingers',
                                 'Step 5 - Circular rubbing of the right thumb in the closed left palm of the hand',
                                 'Step 5.1 - Circular rubbing of the left thumb in the closed right palm of the hand ',
                                 'Step 6 - Circular movement of closed fingertips of the right hand on the left palm of the hand',
                                 'Step 6.1 - Circular movement of closed fingertips of the left hand on the right palm of the hand',
                                 'Rest']


def main():
    # status = 1
    Button(master=window, text="Collect data",
           command=lambda: collect_separate_training_data(hand_disinfection_light, delete_old=True)).pack(pady=8)
    Button(master=window, text="Train classifier", command=lambda: train_classifier()).pack(pady=8)
    Button(master=window, text="Check sample rate", command=lambda: check_sample_rate(100)).pack(pady=8)
    Button(master=window, text="Predict live Gesture", command=predict).pack(pady=8)
    Button(master=window, text="Process data", command=process_raw_data).pack(pady=8)
    Button(master=window, text="Load feature file", command=load_csv).pack(pady=8)
    Button(master=window, text="Close", command=window.destroy).pack(pady=8)

    window.mainloop()


if __name__ == '__main__':
    main()

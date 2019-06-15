from __future__ import print_function

from tkinter import Tk, Button

from Classification import train_classifier, predict
from Myo_communication import collect_training_data

window = Tk()
window.geometry("300x150")
window.title("EMG Recognition")
collect_data_button = Button(master=window, text="Collect data",
                             command=lambda: collect_training_data(hand_disinfection_light))
train_classifier_button = Button(master=window, text="Train classifier", command=lambda: train_classifier())
predict_live_button = Button(master=window, text="Predict live Gesture", command=predict)
exit_button = Button(master=window, text="Close", command=window.quit())

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
    collect_data_button.pack()
    predict_live_button.pack()
    train_classifier_button.pack()
    exit_button.pack()
    window.mainloop()


if __name__ == '__main__':
    main()

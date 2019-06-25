from __future__ import print_function

from tkinter import Button, Tk

from GUI import MainWindow

# ordentliche prediction Ergebnisse
# PREDICT_TIME: float = 0.5


def main():
    root = Tk()
    app = MainWindow(root)
    root.wm_title("EMG Recognition")
    root.geometry("500x500")
    root.mainloop()


if __name__ == '__main__':
    main()

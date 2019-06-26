from __future__ import print_function


# ordentliche prediction Ergebnisse
# PREDICT_TIME: float = 0.5

from GUI import MainWindow, Tk


def main():
    main_window = Tk()
    app = MainWindow(main_window)
    main_window.wm_title("EMG Recognition")
    main_window.mainloop()

if __name__ == '__main__':
    main()

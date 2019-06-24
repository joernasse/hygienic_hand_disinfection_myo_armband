from tkinter import Tk, Label, Entry, END

main_window = Tk()
main_window.geometry("400x350")
main_window.title("EMG Recognition")

data_collect_window = Tk()
data_collect_window.geometry("300x200")
data_collect_window.title("Data collection")
data_collect_window.withdraw()

Label(data_collect_window, text="Durchgänge").grid(row=1, pady=4)
Label(data_collect_window, text="Aufnahmedauer pro Geste").grid(row=2, pady=4)
# e1.grid(row=1, column=1)

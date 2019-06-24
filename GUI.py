from tkinter import Tk, Label, Entry, HORIZONTAL, Button, VERTICAL, StringVar
from tkinter.ttk import Separator

main_window = Tk()
# main_window.geometry("400x350")
main_window.title("EMG Recognition")
Label(main_window, text="Data Collection").grid(row=0, pady=4)
Label(main_window, text="Proband Name:").grid(row=2, pady=4)
p_val = StringVar(main_window, value="defaultUser")
proband = Entry(main_window, textvariable=p_val)
proband.grid(row=2, column=1)
Separator(main_window, orient=HORIZONTAL).grid(row=3, column=0, sticky="ew", columnspan=5, padx=4)
Separator(main_window, orient=VERTICAL).grid(row=0, column=3, sticky='ns', rowspan=3, padx=1, pady=4)

Label(main_window, text="Process data").grid(row=3, pady=4)
Separator(main_window, orient=HORIZONTAL).grid(row=9, column=0, sticky="ew", columnspan=5, padx=4)

close_val = StringVar(main_window, value="close")
close = Button(master=main_window, textvariable=close_val, command=main_window.destroy).grid(row=10, column=1, pady=8,
                                                                                             padx=4)

# data collect window

data_collect_window = Tk()
# data_collect_window.geometry()
data_collect_window.title("Data collection")
data_collect_window.withdraw()
Label(data_collect_window, text="Durchgänge").grid(row=1, pady=4, padx=2)
Label(data_collect_window, text="Zeit pro Geste").grid(row=2, pady=4, padx=2)
Separator(data_collect_window, orient=HORIZONTAL).grid(row=5, column=0, sticky="ew", columnspan=5)
Button(master=data_collect_window, text="Close", command=data_collect_window.destroy).grid(row=6, column=1, pady=4,
                                                                                           padx=4)
s_val = StringVar(data_collect_window, value="10")
r_val = StringVar(data_collect_window, value="5")

sessions = Entry(data_collect_window, textvariable=s_val, width=3)
sessions.grid(row=1, column=1)
record_time = Entry(data_collect_window, textvariable=r_val, width=3)
record_time.grid(row=2, column=1)

# status window
status_window = Tk()
Label(status_window, text="Status").grid(row=0, pady=4, padx=2)
status_val = StringVar()
status_val.set('Init Status')
Label(status_window, textvariable=status_val).grid(row=1, pady=4, padx=2)
status_window.withdraw()

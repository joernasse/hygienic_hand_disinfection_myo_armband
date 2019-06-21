from tkinter import Tk, PhotoImage, Label

main_window = Tk()
main_window.geometry("400x350")
main_window.title("EMG Recognition")

data_collect_window = Tk()
data_collect_window.geometry("300x200")
data_collect_window.title("Data collection")

# logo = PhotoImage(file="red.png")
# red_box = Label(main_window, image=logo).pack(side="right")
# red_box = Label(data_collect_window, image=logo).pack(side="right")
# # red_box.grid_forget()
# # red_box.pack_forget()
# # red_box.place_forget()


# def show_hide_alert():

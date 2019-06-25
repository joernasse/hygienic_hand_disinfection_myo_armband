import os
import time


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def countdown(introduction_screen, t=5, ):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        introduction_screen.set_descr_val(timeformat)
        time.sleep(1)
        t -= 1

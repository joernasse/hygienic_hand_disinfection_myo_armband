import os
from datetime import time


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def countdown(t=5):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat)
        time.sleep(1)
        t -= 1

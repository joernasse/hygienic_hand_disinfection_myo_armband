import os
import time
from array import array

import numpy


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def countdown(introduction_screen, t=5, ):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        introduction_screen.set_status_text("Pause! " + timeformat)
        time.sleep(1)
        t -= 1


def list_list_to_matrix(list_list):
    array_list=[]
    for i in range(len(list_list)):
        tmp=numpy.asarray(list_list[i])
        array_list.append(numpy.asarray(list_list[i]))
    m=numpy.asmatrix(array_list)
    return numpy.asmatrix(numpy.asarray(item)for item in list_list[:-1])


    matrix1=numpy.asmatrix(tmp)
    tmp2=numpy.array(tmp1)
    tmp3=tmp2.shape(50,8)
    for item in list_list[:-1]:
        tmp.extend(item)


    print("")


def wait(time_in_sec):
    dif = 0
    start = time.time()
    while dif <= time_in_sec:
        end = time.time()
        dif = end - start



def divide_chunks(l, n):
    tmp = []
    for i in range(0, len(l), n):
        tmp.append(l[i:i + n])
    return tmp

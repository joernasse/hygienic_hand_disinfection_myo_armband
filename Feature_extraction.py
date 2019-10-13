#!/usr/bin/env python
"""

"""

from UliEngineering.SignalProcessing.Utils import zero_crossings
import Constant
import numpy as np

__author__ = "Joern Asse"
__copyright__ = ""
__credits__ = ["Joern Asse"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Joern Asse"
__email__ = "joernasse@yahoo.de"
__status__ = "Production"

def rms(x):
    """
    Root mean square
    :param x: array
            Array for which the feature is calculated
    :return: float
            Root mean square for input array
    """
    sum = 0
    n = len(x)
    for i in x:
        sum += i * i
    return np.sqrt(1 / n * sum)


def mav(x):
    """
    Mean Absolute Value
    :param x:array
            Array for which the feature is calculated
    :return:float
        Mean Absolute Value for input array
    """
    sum = 0
    n = len(x)
    for i in x:
        sum += np.abs(i)
    return 1 / n * sum


def zc(x):
    """
    Zero Crossing
    :param x:array
            Array for which the feature is calculated
    :return:
    """
    n = len(x)
    sum = 0
    for i in range(n - 1):
        if (x[i] * x[i + 1] <= Constant.threshold) and (np.abs(x[i] - x[i + 1]) >= Constant.threshold):
            sum += 1
    return sum


def iemg(x):
    """
    Integrated absolute value
    :param x:array
            Array for which the feature is calculated
    :return:
    """
    sum = 0
    for i in x:
        sum += np.abs(i)
    return sum


def ssi(x):
    """
    Simple Square Integral
    :param x:array
            Array for which the feature is calculated
    :return:
    """
    sum = 0
    for i in x:
        sum += np.power(np.abs(i), 2)
    return sum


def wl(x):
    """
    Waveform length
    :param x:array
            Array for which the feature is calculated
    :return:
    """
    sum = 0
    for i in range(0, len(x) - 1):
        sum += abs(x[i + 1] - x[i])
    return sum


def cc(x):
    """
    Cepstral coeffcients
    :param x:array
            Array for which the feature is calculated
    :return:
    """
    try:
        fft = np.fft.fft(x)
        ab = np.abs(fft)
        l = np.log(ab)
        c = np.fft.ifft(l)
        for i in range(len(c)):
            y = +c[i]
        return y
    except RuntimeWarning:
        print(RuntimeWarning)


def aac(x):
    """
    Average amplitude change
    :param x:
            data for which the Average amplitude change should be calculated
    :return: float
            flo
    """
    n = len(x)
    sum = 0
    for i in range(0, n - 1):
        sum += x[i + 1] - x[i]
    return sum / float(n)


def rehman(data):
    """
    Feature set from Rehman
    Paper: Multiday EMG-Based Classification of Hand Motions with Deep Learning Techniques
    :param data:array
           Data from which features are to be extracted
    :return: list
            Feature vector
    """
    return [mav(data), wl(data), ssc(data), zero_crossings(data).size]


def georgi(data, sensor):
    """
    Feature set from georgi
    Paper: Recognizing Hand and Finger Gestures with IMU based Motion and EMG based Muscle Activity Sensing
    :param data:array
           Data from which features are to be extracted
    :param sensor: string
            The sensor type (EMG or IMU)
    :return: list
            Feature vector
    """
    if sensor == Constant.EMG:
        return [np.std(data)]
    if sensor == Constant.IMU:
        return [np.mean(data), np.std(data)]


def feature_extraction(windows, mode, sensor):
    """
    Handles the feature extraction for different feature sets
    :param windows: list
            The list of windows for which a feature extraction should be performed
    :param mode: string
            The feature extraction mode, describes the feature set which should be used
    :param sensor: string
            Describes the sensor (EMG or IMU, from Constrant.py) Only necessary for Georgi
    :return: list
        Returns a list of feature vectors
    """
    features = []
    for window in windows:
        feature = []
        for data in window[:- 1]:
            if mode == Constant.rehman:
                feature.extend(rehman(data=data))
            elif mode == Constant.georgi:
                feature.extend(georgi(data, sensor))
            elif mode == Constant.robinson:
                feature.extend(robinson(data))
            else:
                feature.extend(mantena(data=data))

        features.append({"fs": feature, "label": int(window[-1])})
    return features


def mantena(data):
    """
    Feature Set from Rajiv_Mantena (GitHub)
    https://github.com/rmantena/Myo_gestureArmBand_experiments
    :param data:array
           Data from which features are to be extracted
    :return: list
            Feature vector
    """
    return [rms(data), iemg(data), ssi(data), np.var(data), wl(data), aac(data)]


def ssc(x):
    """
    # Slope Sign Changes
    # https://pdfs.semanticscholar.org/3d85/7e8fa4bc59b614e6d220f2af644c3e886ba9.pdf
    :param x: array
           Data for which the slope sign changes should be calculated
    :return: float
            result of calculation (slope sign changes)
    """
    try:
        f = 0
        for n in range(1, len(x) - 1):
            res = (x[n] - x[n - 1]) * x[n] - x[n + 1]
            if res >= Constant.threshold:
                f += 1
        return f
    except:
        print("")


# Robinson-PatternClassificationHand-2017.pdf
def robinson(data):
    """
    Feature set from Robinson
    Paper: Pattern Classification of Hand Movements using Time Domain Features of Electromyography
    :param data:
            Data from which features are to be extracted
    :return: list
            Feature vector
    """
    return [rms(data), wl(data), ssc(data)]

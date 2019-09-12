# import math
#
# import python_speech_features
# from pyentrp import entropy as ent
from UliEngineering.SignalProcessing.Utils import zero_crossings
# from sampen import sampen2

import Constant
import numpy as np


# def normalization(channel):
#     channel_norm = []
#     x_max = np.max(channel)
#     for xi in channel:
#         channel_norm.append((MAX_EMG_VALUE / x_max) * xi)
#     return channel_norm


def rms(array):  # root mean square
    sum = 0
    n = len(array)
    for i in array:
        sum += i * i
    return np.sqrt(1 / n * sum)


def mav(array):  # Mean Absolute Value
    sum = 0
    n = len(array)
    for i in array:
        sum += np.abs(i)
    return 1 / n * sum


# def energy(array):  # Energy Ratio
#     sum = 0
#     for i in array:
#         sum += i * i
#     return sum


def var(array):  # Variance
    n = len(array)
    sum = 0
    for i in array:
        sum += np.abs(i)
    return 1 / (n - 1) * sum


# def wamp(array):  # Willison Amplitude
#     n = len(array)
#     sum = 0
#     for i in range(n - 1):
#         if np.abs(array[i] - array[i + 1]) >= threshold:
#             sum += 1
#         else:
#             sum += 0
#     return sum


# Zero Crossing
def zc(x):
    n = len(x)
    sum = 0
    for i in range(n - 1):
        if (x[i] * x[i + 1] <= Constant.threshold) and (np.abs(x[i] - x[i + 1]) >= Constant.threshold):
            sum += 1
    return sum


# def unison_shuffled_copies(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     return a[p], b[p]


# integrated absolute value
# IEMG
def iemg(array):
    sum = 0
    for i in array:
        sum += np.abs(i)
    return sum


# Simple Square Integral
def ssi(array):
    sum = 0
    for i in array:
        sum += np.power(np.abs(i), 2)
    return sum


# def tm3(array):
#     n = len(array)
#     print('n : ', n)
#     sum = 0
#     for i in array:
#         sum += i * i * i
#     return np.power((1 / float(n)) * sum, 1 / float(3))


def wl(array):  # Waveform length
    sum = 0
    for i in range(0, len(array) - 1):
        sum += abs(array[i + 1] - array[i])
    return sum


def cc(x):  # cepstral coeffcients
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


# Average amplitude change
def aac(array):
    n = len(array)
    sum = 0
    for i in range(0, n - 1):
        sum += array[i + 1] - array[i]
    return sum / float(n)


# def to_euler(quat):
#     magnitude = math.sqrt(quat.x ** 2 + quat.y ** 2 + quat.z ** 2 + quat.w ** 2)
#     quat.x = quat.x / magnitude
#     quat.y = quat.y / magnitude
#     quat.z = quat.z / magnitude
#     quat.w = quat.w / magnitude
#
#     # Roll
#     roll = math.atan2(2.0 * (quat.w * quat.x + quat.y * quat.z),
#                       1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y))
#
#     # Pitch
#     pitch = math.asin(max(-1.0, min(1.0, 2.0 * quat.w * quat.y - quat.z * quat.x)))
#
#     # Yaw
#     yaw = math.atan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
#                      1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z))
#     return [pitch, roll, yaw]


# # frequency domain
# def amp_spec(array):  # Amplitude Spectrum
#     freq_array = np.fft.fft(array)
#     n = len(freq_array)
#     sum = 0
#     for a in freq_array:
#         sum += np.abs(a)
#     return sum


# Rehman-EMGHandDeepLearning-2018.pdf
def rehman(data):
    return [mav(data), wl(data), ssc(data), zero_crossings(data).size]


# Georgi-HandFingerGesturesIMUEMG-2015.pdf
def georgi(data, sensor):
    if sensor == Constant.EMG:
        return [np.std(data)]
    if sensor == Constant.IMU:
        return [np.mean(data), np.std(data)]


def feature_extraction(windows, mode, sensor):
    features = []
    for window in windows:
        feature = []
        for data in window[:- 1]:
            # if mode == Constant.phinyomark:
            #     feature.extend(phinyomark(data=data))
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


# Default feature extraction from Rajiv_Mantena (GitHub)
def mantena(data):
    return [rms(data), iemg(data), ssi(data), var(data), wl(data), aac(data)]


# # Paper [5] Feature choice
# def phinyomark(data):
#     # a=sampen2(data,2,0.2)
#     b = ent.sample_entropy(data, 2, 0.2)
#     # mfcc=python_speech_features.mfcc(data,len(data))
#     cc1 = cc(data)
#     return [ent.sample_entropy(data, 2, 0.2)[1], cc(data), rms(data), wl(data)]
#
#     # result = sampen2(window[1])  # SampEn test # ch0, result index 0 Epoch
#     # # 1 is SampEn
#     # # 2 is Std deviation


# mean abs Value Slope
def mavs(ch):
    pass


# Slope Sign Changes
# https://pdfs.semanticscholar.org/3d85/7e8fa4bc59b614e6d220f2af644c3e886ba9.pdf
def ssc(x):
    f = 0
    for n in range(1, len(x) - 1):
        res = (x[n] - x[n - 1]) * x[n] - x[n + 1]
        if res >= Constant.threshold:
            f += 1
    return f


# Robinson-PatternClassificationHand-2017.pdf
def robinson(data):
    return [rms(data), wl(data), ssc(data)]  # C6 RMS,WL,Scc 90,53%
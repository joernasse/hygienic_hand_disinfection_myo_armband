import math
import numpy as np

MAX = 127
threshold = 0.30 * MAX


def feat_trans_def(saving_list, features, options=0):
    if len(saving_list) == 0:
        return [rms(features), iav(features), ssi(features), var(features), wl(features), aac(features)]
    else:
        saving_list.append(rms(features))
        saving_list.append(iav(features))
        saving_list.append(ssi(features))
        saving_list.append(var(features))
        saving_list.append(wl(features))
        saving_list.append(aac(features))
        return saving_list
    # feat_transf.append(tm3(array))


def normalization(channel):
    channel_norm = []
    x_max = np.max(channel)
    for xi in channel:
        channel_norm.append((MAX / x_max) * xi)
    return channel_norm


def rms(array):  # root mean square
    sum = 0
    n = len(array)
    for a in array:
        sum += a * a
    return np.sqrt(1 / n * sum)


def mav(array):  # Mean Absolute Value
    sum = 0
    n = len(array)
    for a in array:
        sum += np.abs(a)
    return 1 / n * sum


def energy(array):  # Energy Ratio
    sum = 0
    for a in array:
        sum += a * a
    return sum


def var(array):  # Variance
    n = len(array)
    sum = 0
    for a in array:
        sum += np.abs(a)
    return 1 / (n - 1) * sum


def wamp(array):  # Willison Amplitude
    n = len(array)
    sum = 0
    for i in range(n - 1):
        if np.abs(array[i] - array[i + 1]) >= threshold:
            sum += 1
        else:
            sum += 0
    return sum


def zc(array):  # Zero Crossing
    n = len(array)
    sum = 0
    for i in range(n - 1):
        x = array[i]
        y = array[i + 1]
        if (x * y <= threshold) and (np.abs(x - y) >= threshold):
            sum += 1
        else:
            sum += 0
    return sum


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# def rms(array):
#     n = len(array)
#     sum = 0
#     for a in array:
#         sum += a * a
#     return np.sqrt((1 / float(n)) * sum)


def iav(array):
    sum = 0
    for a in array:
        sum += np.abs(a)
    return sum


def ssi(array):
    sum = 0
    for a in array:
        sum += a * a
    return sum


def tm3(array):
    n = len(array)
    print('n : ', n)
    sum = 0
    for a in array:
        sum += a * a * a
    return np.power((1 / float(n)) * sum, 1 / float(3))


def wl(array):  # wavlet
    sum = 0
    for a in range(0, len(array) - 1):
        sum += array[a + 1] - array[a]
    return sum


def aac(array):
    n = len(array)
    sum = 0
    for a in range(0, n - 1):
        sum += array[a + 1] - array[a]
    return sum / float(n)


def toEuler(quat):
    magnitude = math.sqrt(quat.x ** 2 + quat.y ** 2 + quat.z ** 2 + quat.w ** 2)
    quat.x = quat.x / magnitude
    quat.y = quat.y / magnitude
    quat.z = quat.z / magnitude
    quat.w = quat.w / magnitude

    # Roll
    roll = math.atan2(2.0 * (quat.w * quat.x + quat.y * quat.z),
                      1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y))

    # Pitch
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * quat.w * quat.y - quat.z * quat.x)))

    # Yaw
    yaw = math.atan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                     1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z))
    return [pitch, roll, yaw]


# frequency domain
def ampSpec(array):  # Amplitude Spectrum
    freq_array = np.fft.fft(array)
    n = len(freq_array)
    sum = 0
    for a in freq_array:
        sum += np.abs(a)
    return sum

# def mmdf(array):  # Modified Median Frequency
#     sum=0
#     for a in array:
#         sum+=ampSpec([a])
#     return 0.5*sum

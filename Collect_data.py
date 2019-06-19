import time
import numpy as np
import collections

from Feature_extraction import feature_transformation, transform_data_collection
from HelperFunctions import countdown, cls
from Save_Load import save_feature_csv
from myo import init, Hub, StreamEmg
import myo as libmyo

TRAINING_TIME: int = 6
PREDICT_TIME: float = 2.5
DATA_POINT_WINDOW_SIZE = 20
EMG_INTERVAL = 0.01
POSITION_INTERVAL = 0.04

RIGHT = "right"
LEFT = "left"
# TRAINING_TIME: int = 1
# PREDICT_TIME: float = 2.5

WINDOW_EMG = 20
DEGREE_OF_OVERLAP = 0.5
OFFSET_EMG = WINDOW_EMG * DEGREE_OF_OVERLAP
SCALING_FACTOR_IMU_DESKTOP = 3.815  # calculated value at desktop PC, problems with Bluetooth connection 3.815821888279855


# WINDOW_IMU = WINDOW_EMG / SCALING_FACTOR_IMU_DESKTOP
# OFFSET_IMU = WINDOW_IMU * DEGREE_OF_OVERLAP

class GestureListener(libmyo.DeviceListener):
    def __init__(self, queue_size=1):
        # super(GestureListener, self).__init__()
        # self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)
        self.ori_data_queue = collections.deque(maxlen=queue_size)

    def on_connected(self, event):
        event.device.stream_emg(StreamEmg.enabled)

    def on_emg(self, event):
        with self.lock:
            if status:
                EMG.append(np.asarray([event.timestamp, event.emg]))

    def on_orientation(self, event):
        with self.lock:
            if status:
                ORI.append([event.timestamp, event.orientation])
                ACC.append([event.timestamp, event.acceleration])
                GYR.append([event.timestamp, event.gyroscope])

    def get_ori_data(self):
        with self.lock:
            return list(self.ori_data_queue)


EMG = []  # emg
ORI = []  # orientation
GYR = []  # gyroscope
ACC = []  # accelerometer

hand_disinfection = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6', 'Rest']

hand_disinfection_1 = ['Step 1', 'Step 1.1', 'Step 2', 'Step 2.1', 'Step 3', 'Step 3.1', 'Step 4',
                       'Step 4.1', 'Step 5',
                       'Step 5.1', 'Step 6', 'Step 6.1', 'Rest']
hand_disinfection_light = ['Step 1', 'Step 4', 'Rest']

init()
hub = Hub()
listener = GestureListener()


def collect_raw_data_of_2_armband(device_left, device_right, data_type='emg', windowing=True):
    device_left.stream_emg(True)
    device_right.stream_emg(True)

    emg_left = []
    ori_left = []
    acc_left = []
    gyro_left = []

    emg_right = []
    ori_right = []
    acc_right = []
    gyro_right = []

    time.sleep(0.5)
    dif = 0
    start = time.time()
    while dif < TRAINING_TIME:
        end = time.time()
        dif = end - start
        if data_type == 'emg':
            emg_left.append(np.asarray(device_left.emg))
            emg_right.append(np.asarray(device_right.emg))
            time.sleep(EMG_INTERVAL)
        elif data_type == 'pos':
            ori_left.append(np.asarray(device_left.orientation))
            acc_left.append(np.asarray(device_left.acceleration))
            gyro_left.append(np.asarray(device_left.gyroscope))

            ori_right.append(np.asarray(device_right.orientation))
            acc_right.append(np.asarray(device_right.acceleration))
            gyro_right.append(np.asarray(device_right.gyroscope))
            time.sleep(POSITION_INTERVAL)

    device_right.stream_emg(False)
    device_left.stream_emg(False)

    if data_type == 'emg':
        if windowing:
            emg_left_windowed = [emg_left[x:x + DATA_POINT_WINDOW_SIZE] for x in
                                 range(0, len(emg_left), DATA_POINT_WINDOW_SIZE)]

            emg_right_windowed = [emg_right[x:x + DATA_POINT_WINDOW_SIZE] for x in
                                  range(0, len(emg_right), DATA_POINT_WINDOW_SIZE)]

            n = len(emg_right_windowed)
            if len(emg_right_windowed[n - 1]) < 20 or len(emg_left_windowed[n - 1]) < 20:
                emg_right_windowed.pop(len(emg_right_windowed) - 1)
                emg_left_windowed.pop(len(emg_left_windowed) - 1)
            return [emg_left_windowed, emg_right_windowed]
        return [emg_left, emg_right]
    elif data_type == 'pos':
        if windowing:
            ori_left_windowed = [ori_left[x:x + DATA_POINT_WINDOW_SIZE] for x in
                                 range(0, len(ori_left), DATA_POINT_WINDOW_SIZE)]
            acc_left_windowed = [acc_left[x:x + DATA_POINT_WINDOW_SIZE] for x in
                                 range(0, len(acc_left), DATA_POINT_WINDOW_SIZE)]
            gyro_left_windowed = [gyro_left[x:x + DATA_POINT_WINDOW_SIZE] for x in
                                  range(0, len(gyro_left), DATA_POINT_WINDOW_SIZE)]

            ori_right_windowed = [ori_right[x:x + DATA_POINT_WINDOW_SIZE] for x in
                                  range(0, len(ori_right), DATA_POINT_WINDOW_SIZE)]
            acc_right_windowed = [acc_right[x:x + DATA_POINT_WINDOW_SIZE] for x in
                                  range(0, len(acc_right), DATA_POINT_WINDOW_SIZE)]
            gyro_right_windowed = [gyro_right[x:x + DATA_POINT_WINDOW_SIZE] for x in
                                   range(0, len(gyro_right), DATA_POINT_WINDOW_SIZE)]

            n = len(ori_left_windowed)
            if len(ori_left_windowed[n - 1]) < 20 or len(acc_left_windowed[n - 1]) < 20 or len(
                    gyro_left_windowed) < 20 or len(ori_right_windowed) < 20 or len(acc_right_windowed) < 20 or len(
                gyro_right_windowed) < 20:
                ori_left_windowed.pop(len(ori_left_windowed) - 1)
                acc_left_windowed.pop(len(acc_left_windowed) - 1)
                gyro_left_windowed.pop(len(gyro_left_windowed) - 1)

                ori_right_windowed.pop(len(ori_right_windowed) - 1)
                acc_right_windowed.pop(len(acc_right_windowed) - 1)
                gyro_right_windowed.pop(len(gyro_right_windowed) - 1)

            left = [ori_left_windowed, acc_left_windowed, gyro_left_windowed]
            right = [ori_right_windowed, acc_right_windowed, gyro_right_windowed]
            return left, right
        left = [ori_left, acc_left, gyro_left]
        right = [ori_right, acc_right, gyro_right]
        return left, right


def collect_training_data():
    time_now = time.localtime()
    timestamp = str(time_now.tm_year) + str(time_now.tm_mon) + str(time_now.tm_mday) \
                + str(time_now.tm_hour) + str(time_now.tm_min) + str(time_now.tm_sec)
    # global hub
    # global status
    # global EMG
    # global ORI
    # global ACC
    # global GYR
    # global hand_disinfection

    time.sleep(1)
    # status = 0
    label_window = []
    raw_data_window = {'EMG': [], 'ORI': [], 'GYR': [], 'ACC': []}
    raw_data = {'EMG': [], 'ORI': [], 'GYR': [], 'ACC': []}
    label_raw = []

    with hub.run_in_background(listener.on_event):
        for emg in range(1, 3):
            for i in range(len(hand_disinfection)):
                for j in range(1, 3):
                    print("\nGesture -- ", hand_disinfection[i], " : Ready?")
                    input("Countdown start after press...")
                    countdown(2)
                    cls()
                    print("Start")
                    tmp_data_window, tmp_data_raw = collect_raw_data(TRAINING_TIME)

                    # window data
                    entries = len(tmp_data_window['EMG'])
                    raw_data_window['EMG'].extend(tmp_data_window['EMG'])
                    raw_data_window['ACC'].extend(tmp_data_window['ACC'])
                    raw_data_window['GYR'].extend(tmp_data_window['GYR'])
                    raw_data_window['ORI'].extend(tmp_data_window['ORI'])
                    label_window.extend(np.full((1, entries), i)[0])

                    # raw data
                    entries = len(tmp_data_raw['EMG'])
                    raw_data['EMG'].extend(tmp_data_raw['EMG'])
                    raw_data['ACC'].extend(tmp_data_raw['ACC'])
                    raw_data['GYR'].extend(tmp_data_raw['GYR'])
                    raw_data['ORI'].extend(tmp_data_raw['ORI'])
                    label_raw.extend(np.full((1, entries), i)[0])

                    # for k in range(len(tmp_data_window)):
                    #     raw_data_window.append(tmp_data_window[k])
                    #     train_y_windowed.append(i)
                    # for k in range(len(tmp_data_raw)):
                    #     raw_data.append(tmp_data_raw[k])
                    #     train_y_raw.append(i)

                    print("Stop")
                    print("Switch hands")

                print("Collected windowed data: ", len(raw_data_window))
                print("Collected raw data: ", len(raw_data))

            print("Saving collected data...")

            transformed_data_collection = transform_data_collection(raw_data_window)
            res = save_feature_csv(transformed_data_collection, label_window,
                           "hand_disinfection_collection_windowed" + timestamp + ".csv")
            res = save_feature_csv(raw_data, label_window, "hand_disinfection_collection_raw" + timestamp + ".csv")
            if res is not None:
                print("Saving succeed")

    save_feature_csv(transformed_data_collection, label_window, "hand_disinfection_collection" + timestamp + ".csv")


def collect_raw_data_old(record_duration):
    # global EMG
    # global ORI
    # global ACC
    # global GYR
    global status
    # global WINDOW_IMU
    # global OFFSET_IMU
    EMG = []
    ORI = []
    ACC = []
    GYR = []
    raw_data = {'EMG': EMG, 'ORI': ORI, 'GYR': GYR, 'ACC': ACC}
    raw_data_window = {'EMG': [], 'ORI': [], 'GYR': [], 'ACC': []}
    dif = 0
    status = 0
    start = time.time()

    while dif <= record_duration:
        status = 1
        end = time.time()
        dif = end - start
    status = 0

    WINDOW_IMU = WINDOW_EMG / (len(EMG) / len(ORI))
    OFFSET_IMU = WINDOW_IMU * DEGREE_OF_OVERLAP

    blocks = int(len(EMG) / abs(WINDOW_EMG - OFFSET_EMG))
    first = 0
    for i in range(blocks):
        last = first + WINDOW_EMG
        raw_data_window['EMG'].append(np.asarray(EMG[first:last]))
        first += int(WINDOW_EMG - OFFSET_EMG)

    blocks = int(len(ORI) / abs(WINDOW_IMU - OFFSET_IMU))
    first = 0
    for i in range(blocks):
        last = int(first + WINDOW_IMU)
        raw_data_window['ORI'].append(np.asarray(ORI[first:last]))
        raw_data_window['GYR'].append(GYR[first:last])
        raw_data_window['ACC'].append(ACC[first:last])
        first += int(WINDOW_IMU - OFFSET_IMU)

    return raw_data_window, raw_data

# old version of data collection
# def collect_training_data_of_2_armband():
#     libmyo.init()
#     hub = libmyo.Hub()
#     listener = libmyo.ApiDeviceListener()
#
#     raw_data_collection_right = []
#     raw_data_collection_left = []
#     label_collection = []
#
#     with hub.run_in_background(listener.on_event):
#         pool = ThreadPool(processes=2)
#         time.sleep(1)
#         for d in listener.devices:
#             if d.arm == RIGHT:
#                 device_r = d
#             elif d.arm == LEFT:
#                 device_l = d
#         time.sleep(0.1)
#
#         for emg in range(1, 2):
#             for i in range(len(hand_disinfection_light)):
#                 cls()
#                 print("\nGesture -- ", hand_disinfection_light[i], " : Ready?")
#                 input("Countdown start after press...")
#                 countdown(2)
#                 print("Start")
#
#                 async_result_emg = pool.apply_async(collect_raw_data_of_2_armband, args=(device_l, device_r, 'emg'))
#                 async_result_position = pool.apply_async(collect_raw_data_of_2_armband,
#                                                          args=(device_l, device_r, 'pos'))
#
#                 return_val_emg = async_result_emg.get()
#                 return_val_position = async_result_position.get()
#
#                 for j in range(len(return_val_emg[0])):
#                     raw_data_collection_left.append(np.asarray(return_val_emg[0][j]))  # 0: left
#                     raw_data_collection_right.append(np.asarray(return_val_emg[1][j]))  # 1: right
#                     label_collection.append(np.asarray(i))
#
#                 for j in range(len(return_val_position[0])):
#                     raw_data_collection_left.append(
#                         np.asarray(return_val_position[0][0][j]))  # [0][0]: left,orientation
#                     raw_data_collection_left.append(
#                         np.asarray(return_val_position[0][1][j]))  # [0][1]: left,acceleration
#                     raw_data_collection_left.append(np.asarray(return_val_position[0][2][j]))  # [0][2]: left, gyroscope
#
#                     raw_data_collection_right.append(
#                         np.asarray(return_val_position[1][0][j]))  # [1][0]: right,orientation
#                     raw_data_collection_right.append(
#                         np.asarray(return_val_position[1][1][j]))  # [1][1]: right,acceleration
#                     raw_data_collection_right.append(
#                         np.asarray(return_val_position[1][2][j]))  # [1][2]: right, gyroscope
#
#                     label_collection.append(np.asarray(i))
#
#                 print("Stop")
#
#         transformed_data_collection_left = transform_data_collection(raw_data_collection_left)
#         transformed_data_collection_right = transform_data_collection(raw_data_collection_right)
#
#         time_now = time.localtime()
#         timestamp = str(time_now.tm_year) + str(time_now.tm_mon) + str(time_now.tm_mday) \
#                     + str(time_now.tm_hour) + str(time_now.tm_min) + str(time_now.tm_sec)
#         # save_csv(transformed_data_collection, train_y, "hand_disinfection_collection" + timestamp + ".csv")

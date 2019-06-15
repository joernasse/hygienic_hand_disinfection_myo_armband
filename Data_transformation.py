from Feature_extraction import feat_trans_def
import numpy as np


def transform_data_collection(raw_data_collection, option=0):
    transformed_data_collection = []
    for data_set in raw_data_collection:
        emg = data_set['EMG']
        ori = data_set['ORI']
        gyr = data_set['GYR']
        acc = data_set['ACC']
        roll, yaw, pitch, x_f_h, x, y, z = [], [], [], [], [], [], []

        # EMG data transformation
        for b in range(0, 8):
            feat_trans_def(x_f_h, emg[:, b])

        # Orientation data transformation
        for c in ori:
            roll.append(c.roll)
            yaw.append(c.yaw)
            pitch.append(c.pitch)
        tmp = [roll, pitch, yaw]
        for t in tmp:
            feat_trans_def(x_f_h, t)

        # Gyroscope data transformation
        for g in gyr:
            x.append(g.x)
            y.append(g.y)
            z.append(g.z)
        tmp = [x, y, z]
        for t in tmp:
            feat_trans_def(x_f_h, t)

        # Acceleration data transformation
        for a in acc:
            x.append(a.x)
            y.append(a.y)
            z.append(a.z)
        tmp = [x, y, z]
        for t in tmp:
            feat_trans_def(x_f_h, t)

        transformed_data_collection.append(np.asarray(x_f_h))

    return np.asarray(transformed_data_collection)


# Unused
def order_raw_data_timestamp(emg, ori, gyr, acc):
    for e in emg:
        tmp_sensor_data = []
        tse = e[0]
        for i in range(len(ori)):
            tso = ori[i][0]
            tsg = gyr[i][0]
            tsa = acc[i][0]
            if ori[i][0] == gyr[i][0] == acc[i][0]:
                if round(abs(tse / tso)):
                    tmp_sensor_data.append([e[1], ori[i][1], gyr[i][1], acc[i][1]])
                else:
                    input("x")  # Debug
            else:
                input("x")  # Debug
    input("x")  # Debug

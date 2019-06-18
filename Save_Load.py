import csv
import os
from tkinter import filedialog

import numpy as np

emg_headline = ["timestamp",
                "ch0", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7",
                "label"]
imu_headline = ["timestamp",
                "x_ori", "y_ori", "z_ori",
                "x_gyr", "y_gyr", "z_gyr",
                "x_acc", "y_acc", "z_acc",
                "label"]
imu_identifier = ["x", "y", "z"]

imu_load_data = {"timestamp": [],
                 "x_ori": [], "y_ori": [], "z_ori": [],
                 "x_gyr": [], "y_gyr": [], "z_gyr": [],
                 "x_acc": [], "y_acc": [], "z_acc": [],
                 "label": []}
emg_load_data = {"timestamp": [],
                 "ch0": [], "ch1": [], "ch2": [], "ch3": [], "ch4": [], "ch5": [], "ch6": [], "ch7": [],
                 "label": []}


def save_raw_csv(data, label, file_emg, file_imu):
    f = open(file_emg, 'w', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(emg_headline)
        for emg in data['EMG']:
            tmp = [emg[0]]
            for i in emg[1]:
                tmp.append(i)
            tmp.append(label)
            writer.writerow(tmp)
    f.close()

    g = open(file_imu, 'w', newline='')
    with g:
        writer = csv.writer(g, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(imu_headline)
        length = len(data['ORI'])
        ori = data['ORI']
        acc = data['ACC']
        gyr = data['GYR']

        for i in range(length):
            tmp = [ori[i][0],
                   ori[i][1].x, ori[i][1].y, ori[i][1].z,
                   gyr[i][1].x, gyr[i][1].y, gyr[i][1].z,
                   acc[i][1].x, acc[i][1].y, acc[i][1].z,
                   label]
            writer.writerow(tmp)
    g.close()
    return


def save_csv(data, labels, file_name):
    f = open(file_name, 'w', newline='')
    with f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for n in range(len(data)):
            value_set = []
            for i in data[n]:
                value_set.append(i)
            value_set.append(labels[n])
            writer.writerow(value_set)
    return f


def load_classifier():
    classifier = filedialog.askopenfile(filetypes=[("Classifier", "*.joblib")])


def load_raw_csv(emg_path, imu_path):
    imu_file, emg_file = open(imu_path), open(emg_path)
    load_data, identifier = [], []

    for file in [emg_file, imu_file]:
        if file.name.__contains__('emg'):
            load_data = emg_load_data
        elif file.name.__contains__('imu'):
            load_data = imu_load_data
        reader = csv.reader(file, delimiter=';')
        first_line = True
        for column in reader:
            if first_line:
                first_line = False
                identifier = column
                continue
            length = len(identifier)
            for i in range(length):
                load_data[identifier[i]].append(float(column[i]))
    return emg_load_data, imu_load_data


def load_csv():
    file = filedialog.askopenfile(filetypes=[("CSV files", "*.csv")])
    data_name = file.name.split("/")

    print("Start -- loading data")
    with open(file.name) as csv_file:
        train_x = []
        train_y = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            tmp = []
            for i in range(len(row) - 1):
                tmp.append(float(row[i]))
            train_x.append(np.asarray(tmp))
            train_y.append(int(row[len(row) - 1]))
        print("Done -- loading data")
        return np.asarray(train_x), np.asarray(train_y), data_name[-1]

import csv
import os
import shutil
from tkinter import filedialog

import numpy as np
import logging as log

from Constant import emg_headline, imu_headline, COLLECTION_DIR


def save_raw_csv(data, label, file_emg, file_imu):
    file_exist = os.path.isfile(file_emg)
    f = open(file_emg, 'a', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not file_exist:
            writer.writerow(emg_headline)
        for emg in data['EMG']:
            tmp = [emg[0]]
            for i in emg[1]:
                tmp.append(i)
            tmp.append(label)
            writer.writerow(tmp)
    f.close()

    file_exist = os.path.isfile(file_imu)
    g = open(file_imu, 'a', newline='')
    with g:
        writer = csv.writer(g, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not file_exist:
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


def save_feature_csv(data, file_name):
    f = open(file_name, 'w', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for entry in data:
            for t in entry:
                tmp = t['fs']
                tmp.extend([t['label']])
                writer.writerow(tmp)
    return f


def load_classifier():
    classifier = filedialog.askopenfile(filetypes=[("Classifier", "*.joblib")])


def load_raw_csv(emg_path, imu_path):
    imu_file, emg_file = open(imu_path), open(emg_path)
    load_data, identifier = [], []
    imu_load_data = {"timestamp": [],
                     "x_ori": [], "y_ori": [], "z_ori": [],
                     "x_gyr": [], "y_gyr": [], "z_gyr": [],
                     "x_acc": [], "y_acc": [], "z_acc": [],
                     "label": []}
    emg_load_data = {"timestamp": [],
                     "ch0": [], "ch1": [], "ch2": [], "ch3": [], "ch4": [], "ch5": [], "ch6": [], "ch7": [],
                     "label": []}

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
    imu_file.close()
    emg_file.close()

    return emg_load_data, imu_load_data


def load_csv():
    file = filedialog.askopenfile(filetypes=[("CSV files", "*.csv")])
    data_name = file.name.split("/")

    print("Start -- loading data")
    with open(file.name) as csv_file:
        train_x = []
        train_y = []
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            tmp = []
            for i in range(len(row) - 1):
                tmp.append(float(row[i]))
            train_x.append(np.asarray(tmp))
            train_y.append(int(row[len(row) - 1]))
        print("Done -- loading data")
        return np.asarray(train_x), np.asarray(train_y), data_name[-1]


def create_directories(proband, delete_old, raw_path, raw_con, raw_sep):
    user_path = COLLECTION_DIR + "/" + proband
    if not os.path.isdir(COLLECTION_DIR):  # Collection dir
        os.mkdir(COLLECTION_DIR)
        log.info("Create directory" + COLLECTION_DIR)
    if not os.path.isdir(user_path):  # User dir
        os.mkdir(user_path)
        log.info("Create directory" + user_path)
    if os.path.isdir(raw_sep) and os.path.isdir(raw_con):  # Raw dir
        if delete_old:
            shutil.rmtree(raw_sep)
            shutil.rmtree(raw_con)
            log.info("Remove directory" + raw_path)
            os.mkdir(raw_sep)
            os.mkdir(raw_path + "_continues")
            log.info("Create directory" + raw_path)
    else:
        # os.mkdir(raw_path)
        os.mkdir(raw_sep)
        os.mkdir(raw_con)
        log.info("Create directory" + raw_path)

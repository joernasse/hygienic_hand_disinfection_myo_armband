import csv
from tkinter import filedialog

import numpy as np

emg_headline = ["ch0", "ch1", "ch2", "ch", "ch4", "ch5", "ch6", "ch7", "label"]
imu_headline = ["ori", "gyr", "acc", "label"]


def save_raw_csv(data, label, file_emg, file_imu):
    f = open(file_emg, 'w', newline='')
    with f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(emg_headline)
        for n in range(len(data['EMG'])):
            column = [data['EMG'][n][0], data['EMG'][n][1], label]
            writer.writerow(column)
    f.close()

    g = open(file_imu, 'w', newline='')
    with g:
        writer = csv.writer(g, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(imu_headline)
        for j in range(len(data['ORI'])):
            column = [data['ORI'][j], data['GYR'][j], data['ACC'][j], label]
            writer.writerow(column)
    g.close()

    # for n in range(len(emg)):
    #     for
    # i in emg[n]:
    # value_set.append(i)
    # value_set.append(labels[n])
    # writer.writerow(value_set)
    return f


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

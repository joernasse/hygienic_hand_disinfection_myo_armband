import csv
import os
import shutil
import sys
import numpy
import logging as log
from Constant import *


def save_raw_data(data, label, file_emg, file_imu):
    """
    TODO:
    :param data:
    :param label:
    :param file_emg:
    :param file_imu:
    :return:
    """
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


def save_features(features, file_name):
    """
    TODO:
    :param features:
    :param file_name:
    :return:
    """
    f = open(file_name, 'w', newline='')
    with f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for entry in features:
            for dict in entry:
                tmp = dict['fs']
                tmp.extend([dict['label']])
                writer.writerow(tmp)
    return f


def load_raw_2(path):
    data = []
    try:
        file = open(path)
    except:
        return []
    reader = csv.reader(file, delimiter=';')
    next(reader, None)
    for column in reader:
        data.append(numpy.asarray([float(x) for x in column]))
    return numpy.asarray(data)


def load_raw_data_for_emg_imu(emg_path, imu_path):
    """

    :param emg_path: string
            The path to the emg.csv to open the file
    :param imu_path: string
            The path to the imu.csv to open the file
    :return:dict{"timestamp": [],
                         "x_ori": [], "y_ori": [], "z_ori": [],
                         "x_gyr": [], "y_gyr": [], "z_gyr": [],
                         "x_acc": [], "y_acc": [], "z_acc": [],
                         "label": []}
           ,dict{"timestamp": [],
                         "ch0": [], "ch1": [], "ch2": [], "ch3": [], "ch4": [], "ch5": [], "ch6": [], "ch7": [],
                         "label": []}
    """
    imu_load_data = {"timestamp": [],
                     "x_ori": [], "y_ori": [], "z_ori": [],
                     "x_gyr": [], "y_gyr": [], "z_gyr": [],
                     "x_acc": [], "y_acc": [], "z_acc": [],
                     "label": []}

    emg_load_data = {"timestamp": [],
                     "ch0": [], "ch1": [], "ch2": [], "ch3": [], "ch4": [], "ch5": [], "ch6": [], "ch7": [],
                     "label": []}
    load_data, identifier = [], []
    try:
        imu_file, emg_file = open(imu_path), open(emg_path)
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
                for i in range(1, len(identifier)):
                    load_data[identifier[i]].append(float(column[i]))

        imu_file.close()
        emg_file.close()
        return emg_load_data, imu_load_data

    except:
        print(sys.exc_info()[0])
        raise


def load_features(config, path, users=USERS):
    """

    :param config: string
            Describes the configuration for the features
    :param users: array
            Describes the array of users for which the features should be load
    :param path: sting
            Describe the path to the directory where the features saved
    :return:
    """
    users_data = []
    for user in users:
        data, label_ = [], []
        try:
            file = open(path + user + "-" + config + ".csv")
        except:
            print("Open file is not possible")
            return []
        reader = csv.reader(file, delimiter=';')
        for column in reader:
            data.append([float(x) for x in column[:-1]])
            label_.append(int(column[-1]))
        users_data.append({'data': data, 'label': label_})
        print("Load raw data for", user, "done")
    return users_data


# def load_feature_csv(file):
#     label, data = [], []
#     reader = csv.reader(file, delimiter=';')
#     for column in reader:
#         tmp = []
#         for x in column[:-1]:
#             if x == "inf":
#                 tmp.append(sys.float_info.max)
#             try:
#                 tmp.append(numpy.float64(x))
#             except:
#                 tmp.append(numpy.abs(complex(x)))
#         data.append(tmp)
#         label.append(int(column[-1]))
#
#     return data, label


def create_directories_for_data_collection(proband, delete_old, raw_path, raw_con, raw_sep):
    """

    :param proband:
    :param delete_old:
    :param raw_path:
    :param raw_con:
    :param raw_sep:
    :return:
    """
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
            os.mkdir(raw_con)
            log.info("Create directory" + raw_path)
    else:
        os.mkdir(raw_sep)
        os.mkdir(raw_con)
        log.info("Create directory" + raw_path)


def load_prediction_summary(path):
    """

    :param path:
    :return:
    """
    summary = []
    file = open(path)
    reader = csv.reader(file, delimiter=';')
    for column in reader:
        summary.append(column)
    return summary

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


def load_raw_for_one_sensor(path):
    """
    Load the Data for one sensor from a given file
    :param path:string
            Path to the raw data (folder)
    :return:array
            Return an array of the loaded data.
    """
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


def load_raw_data_for_both_sensors(emg_path, imu_path):
    """
    Load the data for both sensors (EMG and IMU) from a file
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
    imu_load_data = {"timestamp": [], "x_ori": [], "y_ori": [], "z_ori": [], "x_gyr": [], "y_gyr": [], "z_gyr": [],
                     "x_acc": [], "y_acc": [], "z_acc": [], "label": []}

    emg_load_data = {"timestamp": [], "ch0": [], "ch1": [], "ch2": [], "ch3": [], "ch4": [], "ch5": [], "ch6": [],
                     "ch7": [], "label": []}
    load_data, identifier = [], []
    try:
        imu_file, emg_file = open(imu_path), open(emg_path)
        for file in [emg_file, imu_file]:
            if 'emg' in file.name:
                load_data = emg_load_data
            elif 'imu' in file.name:
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


def load_feature_data(config, path, user_list=USERS):
    """
    Load the preprocessed data (after windowing, preprocessing  and features extraction) from a file
    :param config: string
            Describes the configuration for the features
    :param user_list: array
            Describes the array of users for which the features should be load
    :param path: sting
            Describe the path to the directory where the features saved
    :return:
    """
    users_data = []
    for user in user_list:
        data, label_ = [], []
        try:
            file = open(path + user + "-" + config + ".csv")
        except:
            print(path + user + "-" + config + ".csv not found")
            return []
        reader = csv.reader(file, delimiter=';')
        for column in reader:
            data.append([float(x) for x in column[:-1]])
            label_.append(int(column[-1]))
        users_data.append({'data': data, 'label': label_})
        print("Load raw data for", user, "done")
    return users_data


def create_directories_for_data_collection(user, delete_old, raw_path, raw_con, raw_sep):
    """
    Create al necessary directories to save the collected data (user study, GUI)
    :param user:string
            The user for which the directories will be created
    :param delete_old:boolean
            If True, delete all old directories for the given user
    :param raw_path:string
            Path for the raw data directory
    :param raw_con:string
            Path for the continues collected data
    :param raw_sep: string
            Path for the continues collected data
    :return:
    """
    user_path = COLLECTION_DIR + "/" + user
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
    TODO
    :param path:
    :return:
    """
    summary = []
    file = open(path)
    reader = csv.reader(file, delimiter=';')
    for column in reader:
        summary.append(column)
    return summary

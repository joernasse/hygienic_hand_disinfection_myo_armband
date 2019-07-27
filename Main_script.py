from __future__ import print_function

# ordentliche prediction Ergebnisse
# PREDICT_TIME: float = 0.5
import datetime
import os
import timeit
from multiprocessing import Process
from tkinter import filedialog

import numpy

from Classification import train_classifier_1
from Constant import *

from Process_data import process_raw_data
from Save_Load import load_feature_csv


def main():
    classifier = True
    featureExtraction = False

    # process_number=1
    process_number = multiprocessing.cpu_count()
    processes = []
    split = numpy.array_split(numpy.asarray(USERS), process_number)

    start = timeit.default_timer()

    for j in range(process_number):
        s = j * process_number
        t = (j + 1) * process_number
        if classifier:
            process = Process(name="Classify", target=train_classifier, args=(split[j],))
        elif featureExtraction:
            process = Process(name="Feature extraction", target=feature_extraction, args=(split[j],))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()

    stop = timeit.default_timer()
    print('Time: ', datetime.timedelta(seconds=(stop - start)))
    print("Finish")
    return True


def train_classifier(users):
    for user in users:
        if not os.path.isdir(collections_default_path + user + "/classifier"):
            os.mkdir(collections_default_path + user + "/classifier")
        files = os.listdir(collections_default_path + user + "/features")
        for file in files:
            x, y = load_feature_csv(open(collections_default_path + user + "/features/" + file))
            train_classifier_1(x, y, collections_default_path + user + "/classifier", file)
    return True


def feature_extraction(users):
    for user in users:
        for dataset in level_1:
            for sensor in level_2:
                for window in level_3:
                    for overlap in level_4:
                        for feature in level_5:
                            process_raw_data(user, dataset=dataset, overlap=overlap,
                                             sensor=sensor, window=window, feature=feature)
                            # if classifier:
                            #     Classifier.train_classifier()
    return True


if __name__ == '__main__':
    main()

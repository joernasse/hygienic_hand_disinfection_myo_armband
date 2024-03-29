#!/usr/bin/env python
"""
This script contains all Constants for the EMG_Recognition Project
"""

from scipy import signal
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GAUSS

__author__ = "Joern Asse"
__copyright__ = ""
__credits__ = ["Joern Asse"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Joern Asse"
__email__ = "joernasse@yahoo.de"
__status__ = "Production"

# --------------------------------------------List of labels ----------------------------------------------------------#
label_display_with_rest = ['Step 0',
                           'Step 1', 'St    ep 1.1', 'Step 1.2',
                           'Step 2', 'Step 2.1',
                           'Step 3',
                           'Step 4',
                           'Step 5', 'Step 5.1',
                           'Step 6', 'Step 6.1',
                           "Rest"]

label_display_without_rest = ['Step 0',
                              'Step 1', 'Step 1.1', 'Step 1.2',
                              'Step 2', 'Step 2.1',
                              'Step 3',
                              'Step 4',
                              'Step 5', 'Step 5.1',
                              'Step 6', 'Step 6.1']

labels_without_rest = ['Step0',
                       'Step1', 'Step1_1', 'Step1_2',
                       'Step2', 'Step2_1',
                       'Step3',
                       'Step4',
                       'Step5', 'Step5_1',
                       'Step6', 'Step6_1']

hand_disinfection_description = ['Step 0    - Take disinfectant',
                                 'Step 1    - Palm and wrist',
                                 'Step 1.1  - Rubbing the right wrist',
                                 'Step 1.2  - Rubbing the left wrist',
                                 'Step 2    - Right palm on left back',
                                 'Step 2.1  - Left palm on right back',
                                 'Step 3    - Spread and interlocked fingers palm',
                                 'Step 4    - Interlock fingers',
                                 'Step 5    - Circular rubbing of the right thumb in the closed\n left palm of the hand',
                                 'Step 5.1  - Circular rubbing of the left thumb in the closed\n right palm of the hand ',
                                 'Step 6    - Circular movement of closed fingertips of the\n right hand on the left palm of the hand',
                                 'Step 6.1  - Circular movement of closed fingertips of the\n left hand on the right palm of the hand',
                                 'Rest      - Do nothing']
# --------------------------------------------List of labels ----------------------------------------------------------#

# --------------------------------------------Constants----------------------------------------------------------------#
COLLECTION_DIR = "Collections"
RIGHT = "right"
LEFT = "left"
SEPARATE = "separate"
SEPARATE_PATH = "/raw_separate"
CONTINUES_PATH = "/raw_continues"
CONTINUES = "continues"
MAX_EMG_VALUE = 127
EMG = "EMG"
IMU = "IMU"
CNN_1 = "CNN_1"
CNN_KAGGLE = "CNN_Kaggle"

REST_MEAN = "rest_mean"
MAX_VALUE_CHANNEL = "max_value_channel"
# --------------------------------------------Constants----------------------------------------------------------------#

# Process_data
threshold = 0.30 * MAX_EMG_VALUE
identifier_emg = "timestamp", "ch0", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7"
identifier_imu = "timestamp", "x_ori", "y_ori", "z_ori", "x_gyr", "y_gyr", "z_gyr", "x_acc", "y_acc", "z_acc"
collections_default_path = './Collections/'
test_set_size = 0.2
validation_set_size = 0.1
classes = 12

# Save_Load_CSV
emg_headline = ["timestamp",
                "ch0", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7",
                "label"]
imu_headline = ["timestamp",
                "x_ori", "y_ori", "z_ori",
                "x_gyr", "y_gyr", "z_gyr",
                "x_acc", "y_acc", "z_acc",
                "label"]
imu_identifier = ["x", "y", "z"]

# --------------------------------------------Configuration layout-----------------------------------------------------#
W_100 = 100
W_50 = 50
W_60 = 60
W_10 = 10

O_90 = 0.9
O_50 = 0.5
O_0 = 0
O_75 = 0.75

rehman = "rehman"
georgi = "georgi"
robinson = "robinson"
mantena = "mantena"

filter_ = "filter"
z_norm = "z-normalization"
no_pre_processing = "no_pre_pro"

# ----------------------------------Configuration levels-----------------------------------------------------------#
level_0 = [filter_, z_norm, no_pre_processing]
level_1 = [CONTINUES, SEPARATE, SEPARATE + CONTINUES]
level_2 = [EMG, IMU, EMG + IMU]
level_3 = [W_100, W_50]
level_4 = [O_50, O_0, O_90, O_75]
level_5 = [rehman, georgi, robinson, mantena]
# ----------------------------------Configuration levels-----------------------------------------------------------#

# --------------------------------------------Configuration layout-----------------------------------------------------#

# --------------------------------------------List of users from study-------------------------------------------------#
# Standard Varianten
USERS = ["User001", "User002", "User003", "User004",
         "User005", "User006", "User007", "User008",
         "User009", "User010", "User011", "User012",
         "User013", "User014", "User015"]

USERS_SUB = ["User001",
             # "User002",
             "User003",
             "User004",
             "User005",
             "User006",
             "User007",
             "User008",
             "User009",
             "User010",
             "User011",
             "User012",
             "User013",
             "User014",
             "User015"]
# --------------------------------------------List of users from study-------------------------------------------------#


# ----------------------------------Classic Classifier-----------------------------------------------------------------#
random_forest = RandomForestClassifier(n_jobs=-1,
                                       criterion='gini',
                                       n_estimators=256,
                                       min_samples_split=2,
                                       bootstrap=True,
                                       max_depth=16,
                                       max_features=3, verbose=0)

rf_parameter = {'n_estimators': [256, 1024],
                'criterion': ['gini'],
                'max_depth': [8, 16, 64, 128],
                'min_samples_split': [2, 8, 16],
                'min_samples_leaf': [3, 4, 5],
                "max_features": [2, 3, 4],
                'bootstrap': [True]}
gauss = GAUSS()
svc = SVC(C=2, gamma='scale', kernel='rbf', shrinking=True, cache_size=5000, verbose=False)

one_vs_Rest = OneVsRestClassifier(
    BaggingClassifier(SVC(kernel='rbf', probability=True, class_weight='balanced', verbose=False, gamma='scale'),
                      max_samples=1.0 / 12, n_estimators=12, verbose=False), n_jobs=-1)

knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)

lda = LDA(solver='lsqr', shrinkage='auto', n_components=1)
lda_parameter = {'solver': ['lsqr', 'eigen'],
                 'n_components': [1, 2, 5, 10, 20],
                 'shrinkage': ['auto', 0.1, 0.5, 0.9]}
qda = QDA()
classifiers = [random_forest, lda, qda, gauss, knn, svc]
classifier_names = ["Random_Forest", "LDA", "QDA", "Bayers", "KNN", "SVM"]
# ----------------------------------Classic Classifier-----------------------------------------------------------------#

# ----------------------------------Pre processing with Filter and normalization---------------------------------------#
count_devices = 2
fs_emg = 200
fs_imu = 50
nyq_emg = 0.5 * fs_emg * count_devices
nyq_imu = 0.5 * fs_imu * count_devices

# Rehman Filter
cut_off = 2
cut_emg = cut_off / nyq_emg
rehman_b_emg, rehman_a_emg = signal.butter(N=3, Wn=cut_emg, output='ba', btype='highpass')

# Benalcázar Filter
cut_off = 5
cut_emg = cut_off / nyq_emg
benalcazar_b_emg, benalcazar_a_emg = signal.butter(N=4, Wn=cut_emg, output='ba', btype='lowpass')

# Georgi Normalization
# Used no Filter only Z-normalization

# ----------------------------------Pre processing with Filter and normalization---------------------------------------#

# ---------------------------------- Liver Prototype constants---------------------------------------------------------#
seq_duration = [5, 4, 3, 2]
live_prediction_path = "./Live_Prediction/"
delta = 0.15
delta2 = 0.05

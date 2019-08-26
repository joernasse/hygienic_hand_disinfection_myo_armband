import multiprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GAUSS

label_display = ['Step 0',
                 'Step 1', 'Step 1.1', 'Step 1.2',
                 'Step 2', 'Step 2.1',
                 'Step 3',
                 'Step 4',
                 'Step 5', 'Step 5.1',
                 'Step 6', 'Step 6.1',
                 'Rest']

save_label = ['Step0',
              'Step1', 'Step1_1', 'Step1_2',
              'Step2', 'Step2_1',
              'Step3',
              'Step4',
              'Step5', 'Step5_1',
              'Step6', 'Step6_1',
              'Rest']
sub_label = ['Step1', 'Step1_1', 'Step1_2',
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

# Collect_data
PREDICT_TIME = 2.5
DATA_POINT_WINDOW_SIZE = 20
EMG_INTERVAL = 0.01
POSITION_INTERVAL = 0.04
COLLECTION_DIR = "Collections"
RIGHT = "right"
LEFT = "left"
INDIVIDUAL = "individual"

# Process_data
MAX_EMG_VALUE = 127
threshold = 0.30 * MAX_EMG_VALUE

identifier_emg = "timestamp", "ch0", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7"
identifier_imu = "timestamp", "x_ori", "y_ori", "z_ori", "x_gyr", "y_gyr", "z_gyr", "x_acc", "y_acc", "z_acc"
collections_default_path = 'G:/Masterarbeit/Collections/'
# beast_path = "./Collections/"
# collections_default_path = beast_path
# laptop
# collections_default_path = 'C:/EMG_Recognition/Collections/'
# externe HDD
# collections_default_path = 'E:/Masterarbeit/Collections/'

TEST_SIZE = 0.1
SEPARATE_PATH = "/raw_separate"
CONTINUES_PATH = "/raw_continues"

SEPARATE = "separate"
CONTINUES = "continues"

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

EMG = "EMG"
IMU = "IMU"
W_100 = 100
W_50 = 50
O_90 = 0.9
O_50 = 0.5
O_0 = 0
O_75 = 0.75
F1 = "phinyomark"
F2 = "rehman"
georgi = "georgi"
F4 = "robinson"
Fd = "default"

# user_cross_val_feature_selection = ["separate-IMU-100-0.9-default.csv",
#                                     "separate-EMGIMU-100-0.75-georgi.csv", "separate-EMGIMU-100-0.75-rehman.csv",
#                                     "separate-EMGIMU-100-0.5-robinson.csv", "separate-EMGIMU-50-0.5-rehman.csv"]

# Standard Varianten
USERS = ["User001", "User002", "User003", "User004",
         "User005", "User006", "User007", "User008",
         "User009", "User010", "User011", "User012",
         "User013", "User014", "User015"]

# Without user007
USERS_cross = ["User001", "User002", "User003", "User004",
               "User005", "User006", "User008",
               "User009", "User010", "User011", "User012",
               "User013", "User014", "User015"]

# Configuration rating from user dependent accuracy
best_config_qda = "separate-EMGIMU-100-0.75-default"
best_emg_qda = "separate-EMG-100-0.75-rehman"

best_config_rf = "separate-EMGIMU-100-0.75-georgi"
best_emg_rf = "separate-EMG-100-0.75-robinson"
sec_config_rf = "separate-EMGIMU-50-0.75-georgi"

best_config_lda = "separate-EMGIMU-100-0.75-default"
sec_config_lda = "separate-EMGIMU-100-0.5-default"
best_emg_lda = "separate-EMG-100-0.75-default"

best_config_knn = "separate-EMGIMU-100-0.75-georgi"
best_emg_knn = "separate-EMG-100-0.75-georgi"

best_config_bayers = "separate-EMGIMU-100-0.75-georgi"
best_emg_bayers = "separate-EMG-100-0.5-georgi"

best_config_svm = "separate-EMGIMU-100-0.75-georgi"
best_emg_svm = "separate-EMG-100-0.75-georgi"

most_data = "separatecontinues-IMU-50-0.75-default"

# Configuration levels
level_1 = [CONTINUES, SEPARATE, SEPARATE + CONTINUES]
level_2 = [EMG, IMU, EMG + IMU]
level_3 = [W_100, W_50]
level_4 = [O_50, O_0, O_90, O_75]
level_5 = [F2, georgi, F4, Fd]

# Classic Classifier
random_forest = RandomForestClassifier(n_jobs=-1, criterion='gini', n_estimators=256, min_samples_split=2,
                                       bootstrap=True, max_depth=16, max_features=3, verbose=0)
gauss = GAUSS()
svc = SVC(C=2, gamma='scale', kernel='rbf', shrinking=True)
one_vs_Rest = OneVsRestClassifier(svc, n_jobs=-1)
knn = KNeighborsClassifier(n_jobs=-1)

rf_parameter = {'criterion': ['gini'],
                'max_depth': [16, 64],
                'min_samples_split': [2, 3],
                'min_samples_leaf': [3, 4],
                "max_features": [3, 10],
                'bootstrap': [True, False]}
lda_parameter = {'solver': ['lsqr', 'eigen'],
                 'n_components': [1, 2, 5, 10, 20],
                 'shrinkage': ['auto', 0.1, 0.5, 0.9]}

lda = LDA(solver='lsqr', shrinkage='auto')
qda = QDA()


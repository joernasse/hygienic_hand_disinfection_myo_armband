import multiprocessing

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
collections_default_path = 'D:/Masterarbeit/EMG_Recognition/Collections/'

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
# F2=
Fd = "default"

# Standard Varianten
USERS = ["User001", "User002", "User003", "User004",
         "User005", "User006", "User007", "User008",
         "User009", "User010","User011", "User012",
         "User013", "User014", "User015"]
level_1 = [SEPARATE, CONTINUES, SEPARATE + CONTINUES]
# level_2 = [EMG, IMU, EMG + IMU]
level_2 = [EMG + IMU]
level_3 = [W_100, W_50]
level_4 = [O_50, O_0, O_90, O_75]
level_5 = [Fd]  # F1 geht momentan nicht 24.7.19

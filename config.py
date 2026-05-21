DEBUG     = False
NROWS     = 1008
MARKER_FS = 120
BR_FS     = 18
PSS_FS    = 18
ACC_FS    = 100
IMU_FS    = 120
BVP_FS    = 64
EDA_FS    = 4
ECG_FS    = 250
E4_ACC_FS = 32
HR_FS     = 1
E4_HR_FS  = 1
TEMP_FS   = 1
N_MARKERS = 7

ACC_THOLD = 10
WIN_THOLD = 0.03
MQA_THOLD = 0.7
FS_RESAMPLE = 256

WINDOW_SIZE = 20 # seconds
WINDOW_SHIFT = 1 # seconds
MIN_RESP_RATE = 3 # BPM
MAX_RESP_RATE = 45 # BPM

TIME_COLS = ['Timestamps', 'Event', 'Text', 'Color']

MOCAP_ACCEL_SD = 0.00352

TRAIN_VAL_TEST_SPLIT = [0.6, 0.2, 0.2]
TRAIN_TEST_SPLIT = [0.8, 0.2]

import matplotlib as mpl
mpl.rcParams['figure.titlesize']   = 6
mpl.rcParams['axes.titlesize']   = 6
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize']   = 6
mpl.rcParams['xtick.labelsize']  = 6
mpl.rcParams['ytick.labelsize']  = 6

IMU_COLS =  ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

LOW_HACC_FS_ID = [1, 2, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24,
                  25, 26, 27]
NO_HACC_ID = [3, 4, 5, 6]

# issues with marker data on MR conditions:
MARKER_ISSUES_MR = [12, 14, 17, 18, 26, 30]
MARKER_ISSUES_R = [12, 14, 18]
MARKER_ISSUES_M = [12, 14]
# issues with imu data on MR and L0-3 conditions:
IMU_ISSUES = [15, 17, 21, 23, 26, 28, 30]
IMU_ISSUES_L = [15, 17, 21, 23, 26, 28]

# issues with imu data on MR:
IMU_ISSUES_MR = [17, 26, 30]

DPI = 300
FIG_FMT = 'png'
USER = 'raqchia'

from sys import platform
from os import makedirs
if 'linux' in platform:
    DATA_DIR = '/projects/BLVMob/'
    SEAT_DATA_DIR = DATA_DIR + 'aria_seated/Data'
    WALK_DATA_DIR = DATA_DIR + 'aria-walk/Data'
elif platform == 'darwin':
    DATA_DIR = '../data/'
    SEAT_DATA_DIR = DATA_DIR + 'aria_seated'
elif 'win' in platform:
    DATA_DIR = 'D:/Raymond Chia/UTS/Howe Zhu - Data/1stExperiment_sitting'

if platform == 'darwin':
    SBJ_PROCESSED_DIR = f"../data/imu-rr-seated/"
    M_DIR = sbj_processed_dir
elif 'linux' in platform or platform == 'linux':
    SBJ_PROCESSED_DIR = f"/projects/BLVMob/imu-rr-seated/Data/"
    M_DIR = f'/projects/BLVMob/imu-rr-seated/models/'

makedirs(SBJ_PROCESSED_DIR, exist_ok=True)

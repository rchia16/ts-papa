import ipdb
from itertools import repeat
from functools import partial
import pytz
from pyxdf import load_xdf
from multiprocessing import Pool, cpu_count
from ast import literal_eval
from os.path import join, exists, sep
from datetime import datetime, timedelta, timezone, timedelta

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from datapipeline import load_and_snip, get_windowed_data, get_file_list
from digitalsignalprocessing import vectorized_slide_win as vsw
from digitalsignalprocessing import (
    imu_signal_processing, pressure_signal_processing, ecg_signal_processing,
    get_max_freq
)

from tsfresh.feature_selection import relevance as tsfresh_relevance
from tsfresh.utilities.string_manipulation import get_config_from_string

from config import WINDOW_SIZE, WINDOW_SHIFT, IMU_FS, PSS_FS, ECG_FS
from config import IMU_FS, DATA_DIR, SEAT_DATA_DIR

def marker_data_columns(df):
    keep_cols = {
        "glasses_rotation_x",
        "glasses_rotation_y",
        "glasses_rotation_z",
        "glasses_rotation",
        "glasses_position_x",
        "glasses_position_y",
        "glasses_position_z",
    }
    return [
        col for col in df.columns
        if str(col).lower() in keep_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

def reshape_array(data):
    shape = data.shape
    return data.reshape((shape[0], shape[2], shape[1]))

def get_win_conds(time, lbls, wins):
    ''' get median condition of each window '''
    le = LabelEncoder()
    c_enc = le.fit_transform(lbls)
    c_enc_win = get_windowed_data(time, c_enc, wins)
    c_enc_win = np.median(c_enc_win, axis=-1)
    return le.inverse_transform(c_enc_win.astype(int))

# Perform sliding window operation
def create_windows(time, x, y, window_size=WINDOW_SIZE,
                   window_shift=WINDOW_SHIFT, fs=IMU_FS):
    inds = np.arange(0, len(time))
    wins = vsw(inds, len(inds),
               sub_window_size=window_size*fs,
               stride_size=window_shift*fs)
    x_win = get_windowed_data(time, x, wins)
    if isinstance(x_win, list): x_win = np.array(x_win)
    x_win = reshape_array(x_win)
    y_win = get_windowed_data(time, y, wins)
    if isinstance(y_win, list): y_win = np.array(y_win)

    # Take median of the window as label
    y_win = np.median(y_win, axis=-1)
    return x_win, y_win

# Choose top n more relevant feature parameters from tsfresh library
def get_top_tsfresh_params(x_train_df, y_train_df, lbl_str='br',
                           ntop_features=5):
    x_train_df = x_train_df.fillna(0)
    rel_df = tsfresh_relevance.calculate_relevance_table(
        x_train_df, y_train_df[lbl_str])

    params = rel_df['feature'].iloc[:ntop_features].values
    return params

def get_data_cols(df):
    cols = df.columns.values
    data_cols = cols[5:]
    return data_cols

def get_label_cols(df):
    cols = df.columns.values
    br_str = [f for f in cols if f.lower() == 'br'][0]
    lbl_cols = [br_str, 'condition']
    return lbl_cols

def get_conditions_from_glob(glob_pattern):
    if glob_pattern == '[!M]*':
        conditions = ['R', 'L0', 'L1', 'L2', 'L3']
    elif glob_pattern == 'L*':
        conditions = ['L0', 'L1', 'L2', 'L3']
    else:
        sys.exit("Unmatched glob pattern")
    return conditions

# Returns intra subject relevant features
def get_intra_feature_hist(df_list, lbl_str='br', ntop_features=5):
    df = df_list[0].copy()
    data_cols = get_data_cols(df)
    lbl_cols = get_label_cols(df)

    sbj_param_dict = {}

    for df in df_list:
        df.dropna(inplace=True)
        x = df[data_cols]
        y = df[lbl_cols]
        sbj = int(df['subject'].values[0])
        params = get_top_tsfresh_params(x, y, lbl_str=lbl_str,
                                        ntop_features=ntop_features)
        sbj_param_dict[sbj] = params

    sbj_param_df = pd.DataFrame.from_dict(sbj_param_dict, orient='index')
    cols = sbj_param_df.columns.values
    arr = sbj_param_df[cols].values.flatten()

    hist_df = pd.DataFrame.from_dict(Counter(arr), orient='index')
    return hist_df

# Returns inter subject relevant features
def get_inter_feature_hist(df, lbl_str='br', ntop_features=5, nsbjs=30):
    data_cols = get_data_cols(df)
    lbl_cols = get_label_cols(df)

    # drop 
    df.dropna(inplace=True)

    # Check for overlapping times
    x_time = df['ms'].values

    sbj_param_dict = {}
    x = df[data_cols]
    y = df[lbl_cols]
    params = get_top_tsfresh_params(x, y, lbl_str=lbl_str,
                                    ntop_features=ntop_features)
    sbj_param_dict[0] = params

    sbj_param_df = pd.DataFrame.from_dict(sbj_param_dict, orient='index')
    cols = sbj_param_df.columns.values
    arr = sbj_param_df[cols].values.flatten()

    hist_df = pd.DataFrame.from_dict(Counter(arr), orient='index')
    return hist_df

def get_df_windows(df, func, window_size=15, window_shift=0.2, fs=IMU_FS,
                  cols=None):
    time = df['ms'].values
    inds = np.arange(len(df))
    window_shift *= window_size
    wins = vsw(inds, len(inds), sub_window_size=int(window_size*fs),
               stride_size=int(window_shift*fs))
    x, y = [], []
    x_df_out = pd.DataFrame()
    N = len(wins)
    i_list = [n for n in range(N)]
    args = zip(wins.tolist(), repeat(df, N), i_list, [cols]*N)

    out_data = []
    # for i, win in enumerate(wins):
    #     out_data.append(func(win, df, i_list[i]))

    with Pool(cpu_count()) as p:
        out_data = p.starmap(func, args)

    x, y = [], []
    for out in out_data:
        if out is not None:
            x.append(out[0])
            y.append(out[1])

    x_df_out = pd.concat(x).reset_index(drop=True)
    y_df_out = pd.concat(y).reset_index(drop=True)

    x_df_out.sort_values(by='ms', inplace=True)
    y_df_out.sort_values(by='ms', inplace=True)

    return x_df_out, y_df_out

def make_windows_from_id(x_df, cols):
    def make_wins(df):
        ids = df.id.unique()
        wins = []
        for i in ids:
            mask = df.id == i
            wins.append(df[mask][cols])
        return wins
    x = make_wins(x_df)
    x_win = np.array(x)
    return x_win

def get_parameters_from_feature_string(feature_names):
    kind_to_fc_parameters = {}
    for feature_name in feature_names:
        split_name = feature_name.split("__")
        sensor_var = split_name[0]
        feature_var = split_name[1]
        feature_cfg = get_config_from_string(split_name)
        if feature_cfg is not None: feature_cfg = [feature_cfg]
        tmp = {feature_var: feature_cfg}
        if sensor_var in kind_to_fc_parameters.keys():
            params = kind_to_fc_parameters[sensor_var]
            if feature_var in params.keys():
                feature_param = params[feature_var]
                if isinstance(feature_param, list):
                    params[feature_var] = feature_param + feature_cfg
                else:
                    params[feature_var] = [feature_param] + feature_cfg

                # for f_key, f_val in feature_param.items():
                #     new_param = feature_cfg[f_key]
                #     ipdb.set_trace()
                #     if isinstance(f_val, list):
                #         param_list = f_val + [new_param]
                #     else:
                #         param_list = [f_val] + [new_param]
                #     param_list = np.unique(param_list).tolist()
                #     if len(param_list) > 1:
                #         params[feature_var][f_key] = param_list
                #     else:
                #         params[feature_var][f_key] = param_list[0]
            else:
                params[feature_var] = feature_cfg
            kind_to_fc_parameters[sensor_var] = params
        else:
            kind_to_fc_parameters[sensor_var] = tmp
    return kind_to_fc_parameters

def split_timeseries_train_test_df(data_list, test_size=0.2, **kwargs):
    # In each of the files: get the last 20% as the test portion
    df_list = load_and_snip(data_list, **kwargs)
    train_data_df, test_data_df  = [], []
    func = partial(train_test_split, test_size=test_size,
                   shuffle=False)
    with Pool(cpu_count()) as p:
        tmp = p.map(func, df_list)

    train_data_df, test_data_df = zip(*tmp)

    train_data_df = pd.concat(train_data_df, ignore_index=True)
    test_data_df = pd.concat(test_data_df, ignore_index=True)

    train_data_df.sort_values(by='ms', inplace=True)
    test_data_df.sort_values(by='ms', inplace=True)

    overlap_flag = np.isin(train_data_df.ms, test_data_df.ms).any()==False
    if not overlap_flag: ipdb.set_trace()
    assert overlap_flag, print("overlapping test and train data")
    return train_data_df, test_data_df

def utc_to_local(utc_dt, tz=None):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=tz)

def load_bioharness_files(f_list:list, skiprows=None):
    method = partial(pd.read_csv, skipinitialspace=True,
                     skiprows=skiprows, header=0)
    df_list = []
    for f in f_list:
        df_list.append(method(f))

    df = pd.concat(df_list, ignore_index=True)
    return df

def load_imu_file(imu_file:str):
    hdr_file = imu_file.replace('imudata.gz', 'recording.g3')

    df = pd.read_json(imu_file, lines=True, compression='gzip')
    data_df = pd.DataFrame(df['data'].tolist())
    df = pd.concat([df.drop('data', axis=1), data_df], axis=1)

    # sync
    hdr = pd.read_json(hdr_file, orient='index')
    hdr = hdr.to_dict().pop(0)

    iso_tz = hdr['created']
    tzinfo = pytz.timezone(hdr['timezone'])
    # adjust for UTC
    start_time = datetime.fromisoformat(iso_tz[:-1])
    start_time = utc_to_local(start_time, tz=tzinfo).astimezone(tzinfo)
    if 'S02' in imu_file:
        start_time += timedelta(hours=10)

    na_inds = df.loc[pd.isna(df['accelerometer']), :].index.values
    df.drop(index=na_inds, inplace=True)

    imu_times = df['timestamp'].values
    df['timestamp_interp'] = imu_times
    df['timestamp_interp'] = df['timestamp_interp'].interpolate()
    imu_times = df['timestamp_interp'].values
    imu_datetimes = [start_time + timedelta(seconds=val) \
                     for val in imu_times]
    imu_s = np.array([time.timestamp() for time in imu_datetimes])
    df['seconds'] = imu_s

    return df, hdr

def load_imu_files(f_list:list):
    data, hdr = [], []
    tmp = []
    for f in f_list:
        tmp.append(load_imu_file(f))
    for l in tmp:
        data.append(l[0])
        hdr.append(l[1])
    data_df = pd.concat(data, axis=0)
    return data_df, hdr

def bioharness_datetime_to_seconds(val):
    fmt = "%d/%m/%Y %H:%M:%S.%f" 
    dstr = datetime.strptime(val, fmt)
    seconds = dstr.timestamp()
    return seconds

def event_datetime_to_seconds(val):
    fmt = "%m/%d/%Y %H:%M:%S.%f" 
    dstr = datetime.strptime(val, fmt)
    seconds = dstr.timestamp()
    return seconds

def get_start_end_time(sbj='S01', condition='L0', data_dir=None):
    # Load time stamps
    if data_dir == None:
        data_dir = DATA_DIR
    sbj_path = join(data_dir, sbj)
    xdf_path = join(sbj_path, 'xdf')
    has_marker = False
    # xdf_path = join(sbj_path, 'null')
    if exists(xdf_path) and sbj=='S02':
        xdf_file = join(xdf_path, f'{sbj}_{condition}.xdf')
        data, header = load_xdf(xdf_file)
        i = 0
        for i, d in enumerate(data):
            if d['info']['type'][0] == 'Markers':
                has_marker = True
                break
        if condition != 'P':
            try:
                timestamps = pd.DataFrame(data[i]['time_series'])[0].str\
                        .split(" ", expand=True)
            except:
                timestamps = pd.DataFrame(data[i]['time_series'])[0]
            timestamps = timestamps[0] + ' ' + timestamps[1]
            timestamps = timestamps.map(event_datetime_to_seconds).values
        else:
            timestamps = [datetime(year=2023, month=7, day=24, hour=14,
                                   minute=53, second=0).timestamp(),
                          datetime(year=2023, month=7, day=24, hour=14,
                                   minute=53, second=30).timestamp()
                         ]
    else:
        csv_file = join(sbj_path, f'{condition}_{sbj}.csv')
        data = pd.read_csv(csv_file)
        timestamps = data['Timestamps'].map(event_datetime_to_seconds).values
    
    start_time = timestamps[0]
    end_time = timestamps[-1]
    return start_time, end_time

def sync_df(df, time_tuple):
    times = df['seconds'].values

    max_time = datetime.now().timestamp()
    max_mask = times < max_time
    
    start_time = time_tuple[0]
    end_time   = time_tuple[1]

    mask = (times > start_time) & (times < end_time) & max_mask

    return df[mask].reset_index(drop=True).copy()

# def prepare_multimodal(
def prepare_multimodal(
    imu_df, pss_df, ecg_df=None, marker_df=None, window_size=30, window_shift=15
):
    imu_sec = imu_df['sec']
    pss_sec = pss_df['sec']
    wins = vsw(np.arange(len(imu_sec)), len(imu_sec),
              sub_window_size=int(window_size*IMU_FS),
              stride_size=int(window_shift*IMU_FS))

    # take the first time from imu window and match to pss_sec, fix window from
    # there there were any leaps in imu seconds or pss seconds, skip
    thold = 60 # seconds
    pss_win_size = int(window_size * PSS_FS)
    ecg_win_size = int(window_size * ECG_FS) if ecg_df is not None else None
    ecg_sec = ecg_df['sec'] if ecg_df is not None else None
    marker_win_size = int(window_size * IMU_FS) if marker_df is not None else None
    marker_sec = marker_df['sec'] if marker_df is not None else None
    marker_cols = marker_data_columns(marker_df) if marker_df is not None else []
    if marker_df is not None and (
        len(marker_cols) == 0 or len(marker_df) < marker_win_size
    ):
        marker_df = None
        marker_win_size = None
        marker_sec = None

    imu_wins, pss_wins, ecg_wins, marker_wins = [], [], [], []
    for win in wins:
        if win[-1] == 0: break
        imu_win = imu_df.iloc[win]

        diff = np.abs(pss_sec - imu_win['sec'].iloc[0])
        idx = diff.argmin()
        pss_win = pss_df.iloc[idx:idx+pss_win_size]
        ecg_win = None
        if ecg_df is not None:
            ecg_diff = np.abs(ecg_sec - imu_win['sec'].iloc[0])
            ecg_idx = ecg_diff.argmin()
            ecg_win = ecg_df.iloc[ecg_idx:ecg_idx+ecg_win_size]
        marker_win = None
        if marker_df is not None:
            marker_diff = np.abs(marker_sec - imu_win['sec'].iloc[0])
            marker_idx = marker_diff.argmin()
            marker_win = marker_df.iloc[marker_idx:marker_idx+marker_win_size]

        imu_sec0 = datetime.fromtimestamp(imu_win['sec'].iloc[0])
        pss_sec0 = datetime.fromtimestamp(pss_win['sec'].iloc[0])

        # imu_check = np.any(np.diff(imu_win['sec']) > 60)
        # pss_check = np.any(np.diff(pss_win['sec']) > 60)
        # if imu_check or pss_check:
        #     continue
        # else:
        imu_wins.append(imu_win)
        pss_wins.append(pss_win)
        if ecg_df is not None:
            ecg_wins.append(ecg_win)
        if marker_df is not None:
            marker_wins.append(marker_win)

    if len(imu_wins) == 0 or len(pss_wins) == 0:
        return

    for i, (imu_win, pss_win) in enumerate(zip(imu_wins, pss_wins)):
        imu_pss_diff = imu_win['sec'].iloc[0]-pss_win['sec'].iloc[0]
        assert imu_pss_diff < 3, 'first index difference > 3seconds'

    imu_time = np.array([imu_data.sec.values.copy() for imu_data in imu_wins])

    conds_wins = np.array([pss_win['condition'].iloc[0] for pss_win in
                           pss_wins])

    imu_func = partial(imu_signal_processing, fs=IMU_FS)
    pss_func = partial(pressure_signal_processing, fs=PSS_FS)
    with Pool(cpu_count()//2) as p:
        imu_filt = p.map(imu_func, imu_wins)
        pss_filt = p.map(
            pss_func,
            map(lambda pwin: pwin['pss'].values, pss_wins)
        )

    pss_time = [pwin['sec'].values for pwin in pss_wins]
    ecg_time = [ewin['sec'].values for ewin in ecg_wins] if ecg_df is not None else None
    marker_time = [mwin['sec'].values for mwin in marker_wins] if marker_df is not None else None

    freq_func = partial(get_max_freq, fs=PSS_FS)
    with Pool(cpu_count()//2) as p:
        pss_freqs = p.map(freq_func, pss_filt)
    # pss_freqs = [freq_func(pss) for pss in pss_filt]

    ecg_filt = None
    if ecg_df is not None:
        ecg_func = partial(ecg_signal_processing, fs=ECG_FS)
        with Pool(cpu_count()//2) as p:
            ecg_filt = p.map(
                ecg_func,
                map(lambda ewin: ewin['ecg'].values, ecg_wins)
            )
    marker_filt = None
    if marker_df is not None:
        marker_filt = [
            mwin[marker_cols].apply(pd.to_numeric, errors='coerce').interpolate().fillna(0).values
            for mwin in marker_wins
        ]

    # double check if there are any fractured samples
    if not isinstance(imu_filt, np.ndarray) and \
       not isinstance(pss_filt, np.ndarray) and \
       len(imu_filt)==len(pss_filt):
        imu_win_len = window_size*IMU_FS
        imu_idxs = [i for i, data in enumerate(imu_filt) if
                           len(data)!=imu_win_len]
        pss_win_len = window_size*PSS_FS
        pss_idxs = [i for i, data in enumerate(pss_filt) if
                           len(data)!=pss_win_len]
        ecg_idxs = []
        if ecg_filt is not None:
            ecg_win_len = window_size*ECG_FS
            ecg_idxs = [i for i, data in enumerate(ecg_filt) if
                               len(data)!=ecg_win_len]
        marker_idxs = []
        if marker_filt is not None:
            marker_win_len = window_size*IMU_FS
            marker_idxs = [i for i, data in enumerate(marker_filt) if
                               len(data)!=marker_win_len]

        # filter out any that do not meet time requirement
        idxs_reject = np.unique([imu_idxs+pss_idxs+ecg_idxs+marker_idxs])
        idxs_to_keep = [i for i in range(len(imu_filt)) if i not in
                        idxs_reject]

        imu_filt = [data for i, data in enumerate(imu_filt) if i in idxs_to_keep]
        imu_time = [data for i, data in enumerate(imu_time) if i in idxs_to_keep]
        pss_filt = [data for i, data in enumerate(pss_filt) if i in idxs_to_keep]
        pss_time = [data for i, data in enumerate(pss_time) if i in idxs_to_keep]
        pss_freqs = np.array(
            [data for i, data in enumerate(pss_freqs) if i in idxs_to_keep]
        )
        if ecg_filt is not None:
            ecg_filt = [data for i, data in enumerate(ecg_filt) if i in idxs_to_keep]
            ecg_time = [data for i, data in enumerate(ecg_time) if i in idxs_to_keep]
        if marker_filt is not None:
            marker_filt = [data for i, data in enumerate(marker_filt) if i in idxs_to_keep]
            marker_time = [data for i, data in enumerate(marker_time) if i in idxs_to_keep]
        conds_wins = [data for i, data in enumerate(conds_wins) if i in idxs_to_keep]

    if not isinstance(imu_filt, np.ndarray):
        imu_filt = np.array(imu_filt)
    imu_filt[..., 0] = imu_time

    if not isinstance(pss_filt, np.ndarray):
        pss_filt = np.array(pss_filt)
    pss_freqs = np.expand_dims(np.array(pss_freqs)*60, axis=-1)

    if not isinstance(pss_time, np.ndarray):
        pss_time = np.array(pss_time)
    if ecg_filt is not None and not isinstance(ecg_filt, np.ndarray):
        ecg_filt = np.array(ecg_filt)
    if ecg_time is not None and not isinstance(ecg_time, np.ndarray):
        ecg_time = np.array(ecg_time)
    if marker_filt is not None and not isinstance(marker_filt, np.ndarray):
        marker_filt = np.array(marker_filt)
    if marker_time is not None and not isinstance(marker_time, np.ndarray):
        marker_time = np.array(marker_time)

    if len(conds_wins) != len(pss_freqs):
        ipdb.set_trace()

    out = [imu_filt, pss_filt, pss_freqs, pss_time, conds_wins]
    if ecg_filt is not None:
        out.extend([ecg_filt, ecg_time])
    if marker_filt is not None:
        out.extend([marker_filt, marker_time, marker_cols])
    return tuple(out)

def attach_condition_to_df(fname, df):
    file_condition = fname.split(sep)[-1].split('_')[0]
    if 'condition' not in df.columns.values:
        df.insert(0, 'condition', file_condition)
    return df

def load_and_sync_xsens(subject, condition='M', data_dir=SEAT_DATA_DIR):
    # load imu
    def glob_wrapper(condition, sens):
        if condition == '*' or condition == None:
            glob_str = f'*_{sens}_df*'
        else:
            glob_str = f'*{condition}_{sens}_df*'
        return glob_str

    imu_glob = glob_wrapper(condition, 'imu')
    imu_list = get_file_list(data_dir, imu_glob, sbj=subject)
    imu_dfs = []
    for f in imu_list:
        try:
            imu_dfs.append(pd.read_csv(f))
        except Exception as e:
            print(e)
            print("skipping ", f)
            continue

    axes = ['x', 'y', 'z']
    for df in imu_dfs:
        acc = np.array(df['accelerometer'].map(literal_eval).tolist())
        gyr = np.array(df['gyroscope'].map(literal_eval).tolist())
        for i in range(acc.shape[1]):
            df.insert(i+2, f"acc_{axes[i]}", acc[:, i], True)
            df.insert(i+2, f"gyr_{axes[i]}", gyr[:, i], True)

    imu_df = pd.concat(imu_dfs, axis=0)
    imu_df.reset_index(drop=True, inplace=True)
    imu_df = imu_df[['sec', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y',
                     'gyr_z']]

    # load bioharness
    pss_glob = glob_wrapper(condition, 'pressure')
    pss_list = get_file_list(data_dir, pss_glob, sbj=subject)
    if len(pss_list) == 0:
        pss_list = get_file_list(data_dir, f'*{condition}_pressure_df*', 
                                 sbj=subject)
    
    pss_dfs = [
        attach_condition_to_df(fname, pd.read_csv(fname)) for fname in pss_list
    ]
    pss_df = pd.concat(pss_dfs, axis=0)
    waveform_col = [col for col in pss_df.columns.values if\
                    'breathing' in col.lower()][0]
    pss_df = pss_df[['condition', 'sec', waveform_col]]
    pss_df = pss_df.rename(columns={waveform_col: 'pss'})
    pss_df.sort_values(by='sec', inplace=True)
    pss_df.reset_index(drop=True, inplace=True)

    ecg_glob = glob_wrapper(condition, 'ecg')
    ecg_list = get_file_list(data_dir, ecg_glob, sbj=subject)
    if len(ecg_list) == 0:
        ecg_list = get_file_list(data_dir, f'*{condition}_ecg_df*',
                                 sbj=subject)
    ecg_df = None
    if len(ecg_list) > 0:
        ecg_dfs = [
            attach_condition_to_df(fname, pd.read_csv(fname)) for fname in ecg_list
        ]
        ecg_df = pd.concat(ecg_dfs, axis=0)
        ecg_waveform_cols = [col for col in ecg_df.columns.values
                             if 'ecg' in col.lower()]
        if len(ecg_waveform_cols) > 0:
            ecg_col = ecg_waveform_cols[0]
        else:
            ecg_col = ecg_df.columns.values[-1]
        if not pd.api.types.is_numeric_dtype(ecg_df[ecg_col]):
            raise ValueError(
                f"ECG column '{ecg_col}' is not numeric for subject={subject}, "
                f"condition={condition}"
            )
        ecg_df = ecg_df[['condition', 'sec', ecg_col]]
        ecg_df = ecg_df.rename(columns={ecg_col: 'ecg'})
        ecg_df.sort_values(by='sec', inplace=True)
        ecg_df.reset_index(drop=True, inplace=True)

    marker_glob = glob_wrapper(condition, 'marker')
    marker_list = get_file_list(data_dir, marker_glob, sbj=subject)
    if len(marker_list) == 0:
        marker_list = get_file_list(data_dir, f'*{condition}_marker_df*',
                                    sbj=subject)
    marker_df = None
    if len(marker_list) > 0:
        marker_dfs = [
            attach_condition_to_df(fname, pd.read_csv(fname)) for fname in marker_list
        ]
        marker_df = pd.concat(marker_dfs, axis=0)
        if 'sec' not in marker_df.columns:
            raise ValueError(
                f"Marker dataframe is missing 'sec' for subject={subject}, "
                f"condition={condition}"
            )
        marker_cols = marker_data_columns(marker_df)
        marker_df = marker_df[['condition', 'sec'] + marker_cols]
        marker_df.sort_values(by='sec', inplace=True)
        marker_df.reset_index(drop=True, inplace=True)

    br_glob = glob_wrapper(condition, 'summary')
    br_list = get_file_list(data_dir, br_glob, sbj=subject)
    br_dfs = [
        attach_condition_to_df(fname, pd.read_csv(fname)) for fname in br_list
    ]
    br_df = pd.concat(br_dfs, axis=0)
    br_df = br_df[['condition', 'sec', 'BR']]
    br_df = br_df.sort_values(by='sec')
    br_df.reset_index(drop=True, inplace=True)

    xsens_dict = {'subject': subject,
                  'imu': imu_df,
                  'pss': pss_df,
                  'ecg': ecg_df,
                  'marker': marker_df,
                  'br': br_df}

    return xsens_dict

def sync_with_last_val(win, time):
    return np.argmin(np.abs(win[-1]-time))

def load_dataset(subjects,
                 condition='M',
                 data_dir=SEAT_DATA_DIR,
                 debug=False) -> list:
    load_func = partial(load_and_sync_xsens, condition=condition,
                        data_dir=data_dir)

    # load subjects set
    if debug:
        xsens_dicts = [load_func(sbj) for sbj in subjects]
    else:
        with Pool(cpu_count()//2) as p:
            xsens_dicts = p.map(load_func, subjects)

    xsens_dicts = [x for x in xsens_dicts if x != None]

    return xsens_dicts


def load_loocv_dataset(sbj, sbj_dicts, condition='M', do_aug=False,
                       window_size=20, window_shift=1):
    out_dicts = []
    for m_dict in sbj_dicts:
        if m_dict['subject'] == sbj:
            test_data = m_dict.copy()
            break

    for sbj_dict in sbj_dicts:
        if sbj_dict['subject'] == sbj:
            continue
        tmp = {}

        sbj = xsens_dict['subject']
        imu = xsens_dict['imu'].values
        pss = xsens_dict['pss'].values
        br = xsens_dict['br'].values

        imu_time, imu_data = imu_sect[..., 0], imu_sect[..., 1:]

        pss_conds = pss_sect[..., 0]
        pss_time, pss_data = pss_sect[..., 1], pss_sect[..., 2]

        imu_filt = imu_signal_processing(imu_data)
        pss_filt = pressure_signal_processing(pss_data.reshape(-1, 1))

        # train ICA with PCA compression

        pca = PCA(n_components=n_comp, random_state=seed)
        ica = FastICA(n_comp, whiten='arbitrary-variance', random_state=seed)

        imu_pca = pca.fit_transform(imu_filt)
        imu_ica = ica.fit_transform(imu_filt)

        tmp['subject'] = sbj
        tmp['pss_data'] = pss_filt
        tmp['pss_time'] = pss_time
        tmp['imu_filt'] = imu_filt
        tmp['imu_pca'] = imu_pca
        tmp['imu_ica'] = imu_ica
        tmp['imu_time'] = imu_time

        out_dicts.append(tmp)

def segment_to_patches(data, patch_len, pad_value=0, axis=2):
    # data: bs, timesteps, channels
    patch_shape = (patch_len, data.shape[-1])
    remainder = data.shape[axis] % patch_len
    pad_width = (patch_len - remainder) % patch_len

    # Build pad specification for np.pad
    pad_spec = [(0, 0)] * data.ndim
    pad_spec[axis] = (0, pad_width)

    # Pad
    data_padded = np.pad(data, pad_spec, mode="constant", constant_values=pad_value)

    # Reshape into patches
    new_shape = list(data_padded.shape)
    n_patches = new_shape[axis] // patch_len

    # Example for axis=1 specifically
    if axis == 1:
        data_patched = data_padded.reshape(
            new_shape[0], n_patches, patch_len, *new_shape[2:]
        )
    elif axis == 2:
        data_patched = data_padded.reshape(
            *new_shape[:axis], patch_len, n_patches
        )
    else:
        raise NotImplementedError

    return data_patched, data_padded

# ---------------------------------------------------------------------
# Model discovery helpers (shared with benchmark scripts)
# ---------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
import json
import re
from datetime import datetime
import yaml

METHOD_TOKENS_TO_DROP = {"style", "ttt"}


@dataclass
class RunInfo:
    run_dir: Path
    run_yaml: Path
    run_ts: float
    cfg: Dict
    method: str
    backbone: str
    subject: str


def normalize_method(method: str) -> str:
    if not method:
        return "baseline"
    toks = [t for t in method.split("_") if t and (t not in METHOD_TOKENS_TO_DROP)]
    out = "_".join(toks)
    return out if out else "baseline"


def infer_method(
    cfg_method: str,
    run_dir: Path,
    known_methods: Optional[Iterable[str]] = None,
) -> str:
    """
    Prefer metadata method; if missing/ambiguous, infer from run directory name.
    """
    m = normalize_method(cfg_method or "")
    if known_methods is not None and m in set(known_methods):
        return m

    run_name = run_dir.name.lower()
    if "flow_ssa_cmt" in run_name or "flow_cmt_ssa" in run_name:
        return "flow_ssa_cmt"
    if "flow_cmt" in run_name:
        return "flow_cmt"
    if "flow_ssa" in run_name:
        return "flow_ssa"
    if "ssa_cmt" in run_name or "cmt_ssa" in run_name:
        return "ssa_cmt"
    if "flow" in run_name:
        return "flow"
    if "cmt" in run_name:
        return "cmt"
    if "ssa" in run_name:
        return "ssa"
    return "baseline"


def extract_run_timestamp(run_yaml: Path) -> float:
    """
    Extract timestamp from run directory name if present:
      YYYYMMDD-HHMMSS_...
    Fallback to run.yaml mtime.
    """
    run_name = run_yaml.parent.name
    m = re.match(r"^(\d{8})-(\d{6})", run_name)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y%m%d-%H%M%S")
            return dt.timestamp()
        except Exception:
            pass
    return run_yaml.stat().st_mtime


def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def parse_run_prefix_flags(run_dir: Path) -> Dict[str, int]:
    """
    Parse run_prefix flags from bundle.json when available.
    Returns mapping like {"flow": 1, "ssa": 0, ...}
    """
    bundle = _read_json(run_dir / "bundle.json")
    prefix = str(bundle.get("run_prefix", "") or "").strip()
    if not prefix:
        return {}

    toks = [t for t in prefix.strip("_").split("_") if t]
    flags: Dict[str, int] = {}
    for i in range(len(toks) - 1):
        if toks[i + 1].isdigit():
            try:
                flags[toks[i]] = int(toks[i + 1])
            except Exception:
                continue
    return flags


def resolve_best_ckpt(run_dir: Path, ckpt_prefix: str) -> Optional[Path]:
    candidates = []
    if ckpt_prefix:
        candidates.extend([
            run_dir / f"ckpt_{ckpt_prefix}_best.pt",
            run_dir / f"ckpt_{ckpt_prefix}_last.pt",
        ])
    candidates.extend([
        run_dir / "ckpt_best.pt",
        run_dir / "ckpt_last.pt",
    ])
    for p in candidates:
        if p.exists():
            return p
    return None


def find_run_yamls(root: Path) -> List[Path]:
    return list(root.rglob("run.yaml"))


def find_run_yamls_by_model_layout(root: Path, models: List[str]) -> List[Path]:
    """
    Search run.yaml paths using repository layout conventions:
      - vit runs under:        <root>/<subject>/vit/**/run.yaml
      - times-family runs under:
          <root>/<subject>/times/timesnet/**/run.yaml
          <root>/<subject>/times/primus/**/run.yaml
          <root>/<subject>/times/limu/**/run.yaml
    Falls back to a full recursive search if nothing is found.
    """
    selected = set(models)
    out: List[Path] = []

    if "vit" in selected:
        out.extend(root.glob("*/vit/**/run.yaml"))

    for m in ("timesnet", "primus", "limu", "imu2clip", "limu_bert_x", "unihar", "normwear"):
        if m in selected:
            out.extend(root.glob(f"*/times/{m}/**/run.yaml"))

    uniq = sorted(set(out))
    if uniq:
        return uniq
    return sorted(find_run_yamls(root))


def infer_backbone(
    cfg_backbone: str,
    run_dir: Path,
    known_models: Optional[Iterable[str]] = None,
) -> str:
    """
    Prefer run metadata backbone; if it is missing/ambiguous, infer from path.
    """
    b = str(cfg_backbone or "").strip().lower()
    if known_models is not None and b in set(known_models):
        return b

    parts = {p.lower() for p in run_dir.parts}
    if known_models is not None:
        for cand in set(known_models):
            if cand in parts:
                return cand
    return b


def build_latest_run_index(
    root: Path,
    methods: List[str],
    models: List[str],
) -> Dict[Tuple[str, str, str], RunInfo]:
    latest: Dict[Tuple[str, str, str], RunInfo] = {}

    for run_yaml in find_run_yamls_by_model_layout(root, models):
        run_dir = run_yaml.parent
        cfg = load_yaml(run_yaml) or {}

        method = infer_method(str(cfg.get("method", "")), run_dir, known_methods=methods)
        backbone = infer_backbone(str(cfg.get("backbone", "")), run_dir, known_models=models)
        subject = str(cfg.get("subject", ""))

        if method not in methods:
            continue
        if backbone not in models:
            continue
        if not subject:
            continue

        key = (method, backbone, subject)
        info = RunInfo(
            run_dir=run_dir,
            run_yaml=run_yaml,
            run_ts=extract_run_timestamp(run_yaml),
            cfg=cfg,
            method=method,
            backbone=backbone,
            subject=subject,
        )
        prev = latest.get(key)
        if prev is None or info.run_ts > prev.run_ts:
            latest[key] = info

    return latest

def _filter_subjects_with_data(
    subjects: List[str],
    *,
    excluded_subject_nums: Optional[List[int]] = None,
    data_dir: str,
) -> List[str]:
    excluded = {f"S{int(n):02d}" for n in (excluded_subject_nums or [])}
    filtered = []
    skipped_excluded = []
    skipped_missing = []

    for sbj in subjects:
        if sbj in excluded:
            skipped_excluded.append(sbj)
            continue
        sbj_path = join(data_dir, f"{sbj}.pkl")
        if not exists(sbj_path):
            skipped_missing.append(sbj)
            continue
        filtered.append(sbj)

    if skipped_excluded:
        print(f"[DATA] Skipping excluded subjects: {', '.join(skipped_excluded)}")
    if skipped_missing:
        print(f"[DATA] Skipping subjects with missing data: {', '.join(skipped_missing)}")

    return filtered

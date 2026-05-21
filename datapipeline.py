import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import pandas as pd
# import tensorflow as tf
from os import environ, mkdir, makedirs, listdir, stat, walk
from sys import platform
from os.path import join
from os.path import isdir, splitext, sep
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from functools import partial
from ast import literal_eval

import numpy as np
import glob
import ipdb
import mat73
import re
import json
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.cluster import MiniBatchKMeans

# import cv2

from config import DEBUG, NROWS, N_MARKERS\
        ,TIME_COLS, NO_HACC_ID, DATA_DIR

def datetime_to_sec(time_in, is_iso=False):
    dstr = datetime.today()
    try:
        fmt ="%Y-%m-%d %I.%M.%S.%f %p" 
        dstr = datetime.strptime(time_in, fmt)
    except ValueError:
        if 'Take' in time_in:
            time_section = time_in[5:-4]
            if 'Task' or 'MR' in time_in:
                start_ind = re.search(r'\d{4}', time_in)
                end_ind = re.search(r'M_', time_in)
                time_section = time_in[start_ind.start():end_ind.start()+1]
            dstr = datetime.strptime(time_section, "%Y-%m-%d %I.%M.%S %p")
        elif '_' in time_in:
            fmt = "%Y_%m_%d-%H_%M_%S" 
            dstr = datetime.strptime(time_in, fmt)
        elif '/' in time_in:
            fmt = "%d/%m/%Y %H:%M:%S.%f" 
            dstr = datetime.strptime(time_in, fmt)
        elif 'Z' in time_in:
            fmt = "%Y%m%dT%H%M%SZ"
            dstr = datetime.strptime(time_in, fmt)

    sec = dstr.timestamp()
    # td = timedelta(hours=dstr.hour,
    #                minutes=dstr.minute,
    #                seconds=dstr.second,
    #                microseconds=dstr.microsecond)
    # sec = td.total_seconds()
    return sec

def sec_to_datetime(sec):
    return datetime.fromtimestamp(sec)

def mat_to_sec(time):
    if sum([':'==ch for ch in time]) == 1:
        dstr = datetime.strptime(time, '%m/%d/%Y %H:%M')
    elif time[-1] == 'M':
        dstr = datetime.strptime(time, '%m/%d/%Y  %I:%M:%S %p')
    elif '.' not in time:
        dstr = datetime.strptime(time, '%m/%d/%Y  %H:%M:%S')
    else:
        dstr = datetime.strptime(time, '%m/%d/%Y %H:%M:%S.%f')
    sec = dstr.timestamp()
    # td = timedelta(hours=dstr.hour,
    #                minutes=dstr.minute,
    #                seconds=dstr.second,
    #                microseconds=dstr.microsecond)
    # sec = td.total_seconds()
    return sec

def cond_to_label(cond_str:str):
    my_dict = {'M': 0, 'R': 1, 'L0': 2, 'L1': 3, 'L2': 4, 'L3': 5}
    lbl = my_dict[cond_str]
    return lbl

def get_conditions(fname:str):
    return fname.split(sep)[-1].split("_")[0]

def split_csv_method(fname:str, skip_ratio=0.8, is_train=True, skiprows=None,
                    **kwargs):
    nrows = None
    with open(fname) as f:
        nrows_tot = sum(1 for line in f)
    nrows_tot -= 1

    if skiprows is not None:
        nrows_tot = nrows_tot - skiprows

    if skip_ratio > 0:
        if is_train:
            nrows = int(nrows_tot*skip_ratio)
        else:
            if skiprows is not None:
                skiprows += int(nrows_tot*skip_ratio)
            else:
                skiprows = int(nrows_tot*skip_ratio)
            skiprows = range(1, skiprows+1)

    df = pd.read_csv(fname, skipinitialspace=True, skiprows=skiprows,
                     header=0, nrows=nrows, **kwargs)
    cond = get_conditions(fname)
    df['condition'] = cond
    return df

def read_csv_method(fname:str, **kwargs):
    df = pd.read_csv(fname, skipinitialspace=True, 
                     header=0, **kwargs)
    cond = get_conditions(fname)
    df['condition'] = cond
    return df

def load_files_conditions(f_list:list, skip_ratio=None, do_multiprocess=True,
                          **kwargs):
    if skip_ratio is not None:
        method = partial(split_csv_method, skip_ratio=skip_ratio, **kwargs)
    else:
        method = partial(read_csv_method, **kwargs)

    if do_multiprocess:
        with Pool(processes=cpu_count()) as p:
            df_list = p.map(method, f_list)
    else:
        df_gen = map(method, f_list)
        df_list = [df for df in df_gen]

    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by='sec', inplace=True)
    return df

def load_and_snip(f_list:list, ratios=[0.3, 0.1]):
    method = partial(read_csv_method)
    with Pool(processes=cpu_count()) as p:
        df_list = p.map(method, f_list)

    if len(ratios) == 2:
        for i, df in enumerate(df_list):
            l = len(df)
            skiprows = int(ratios[0]*l)
            nrows = int((1-sum(ratios))*l)+1
            df_list[i] = df.iloc[skiprows:(skiprows+nrows)]
    return df_list

def get_file_list(data_dir, glob_pattern:str, sbj=None, hardware=None):
    if sbj is not None:
        f_glob = join(data_dir, sbj, '**')
    else:
        f_glob = join(data_dir, 'S*', '**')

    if hardware is not None:
        f_glob = join(f_glob, hardware+'/**/')
    else:
        f_glob = join(f_glob, '**')

    if glob_pattern is not None:
        f_glob = join(f_glob, glob_pattern)
    else:
        f_glob = join(f_glob, '*.csv')

    f_list = sorted(glob.glob(f_glob, recursive=True))
    # pop zero size files
    return f_list

def get_windowed_data(time, data, vsw, thold=100):
    out = []
    for i, w_inds in tqdm(enumerate(vsw), total=vsw.shape[0]):
        if w_inds[-1] == 0: break
        t0, t1 = time[w_inds][0], time[w_inds][-1]
        diff = time[w_inds[1:]] - time[w_inds[0:-1]]
        mask = diff>thold
        diff_chk = np.any(mask)
        if diff_chk:
            continue
        out.append(data[w_inds])
    
    return np.array(out)

def get_imu_numpy(x_in):
    try:
        acc_data = np.array(x_in['accelerometer']\
                            .map(literal_eval).tolist())
        gyr_data = np.array(x_in['gyroscope']\
                            .map(literal_eval).tolist())
    except:
        acc_data = np.stack(x_in['accelerometer'].values)
        gyr_data = np.stack(x_in['gyroscope'].values)
    data = np.concatenate((acc_data, gyr_data), axis=1)
    return data

def parallelize_dataframe(df, func):
    num_processes = cpu_count()
    df_split = np.array_split(df, num_processes)
    with Pool(num_processes) as p:
        df = pd.concat(p.map(func, df_split))
    return df

def get_train_test_df(data_list, test_size=0.2, **kwargs):
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

    train_data_df.sort_values(by='sec', inplace=True)
    test_data_df.sort_values(by='sec', inplace=True)

    overlap_flag = np.isin(train_data_df.sec, test_data_df.sec).any()==False
    if not overlap_flag: ipdb.set_trace()
    assert overlap_flag, print("overlapping test and train data")
    return train_data_df, test_data_df

# Load the data
def load_split_data(subject, glob_str:str="*xsens*",
              test_size=0.2, ratios=[0.1, 0.1]):
    '''Loads the data into pandas dataframe
    subject: string. Specify which subject ('S12', 'S13' etc)
    glob_str: str (default = '*xsens*'). Specify what glob string
    condition: string. Specify which condition you're interested in 
    '''
    # use glob pattern to create list of files
    data_list = get_file_list(glob_str, sbj=subject)

    # data_list = []
    # valid = re.compile(r"S[0-9][0-9]")
    # for f in tmp:
    #     check_string = valid.search(f).group()[1:]
    #     if int(check_string) not in IMU_ISSUES_L:
    #         print(int(check_string))
    #         data_list.append(f)

    # Clip each file the start and end
    if data_list is None: ipdb.set_trace()
    train_data_df, test_data_df = get_train_test_df(data_list, ratios=ratios,
                                                    test_size=test_size)

    return train_data_df, test_data_df

# Load the data
def load_data(subject, glob_pattern, condition:str=None,
              ratios=[0.1, 0.1]):
    '''Loads the data into pandas dataframe
    subject: string. Specify which subject ('S12', 'S13' etc)
    condition: string. Specify which condition you're interested in 
    '''
    if condition is not None:
        data_glob = f'{condition}_{glob_pattern}_df'
    else:
        data_glob = f'*{glob_pattern}_df'

    # use glob pattern to create list of files
    data_list = get_file_list(data_glob, sbj=subject)

    # Clip each file the start and end
    df_list = load_and_snip(data_list, ratios=ratios)

    df = pd.concat(df_list)
    df.sort_values(by='sec', inplace=True)
    return df

def load_harness_data(subject, data_str:str, condition:str=None, ratios=[0.1, 0.1]):
    if condition is not None:
        data_glob = f'{condition}_{data_str}_df'
    else:
        data_glob = f'*{data_str}_df'

    # use glob pattern to create list of files
    data_list = get_file_list(data_glob, sbj=subject)

    # Clip each file the start and end
    df_list = load_and_snip(data_list, ratios)

    data_df = pd.concat(df_list, ignore_index=True)
    data_df.sort_values(by='sec', inplace=True)

    return data_df

def shuffle_split(x):
    ss = ShuffleSplit(n_splits=1, random_state=10)
    x_train, x_test = [], []
    for train, test in ss.split(x):
        x_train.append(x[train])
        x_test.append(x[test])
    x_train = np.array(x_train)
    x_test  = np.array(x_test)
    return x_train, x_test

class DataImporter():
    def __init__(self):
        self.imu_fname        = ''
        self.marker_fname     = ''
        self.timeline_fname   = ''
        self.summary_fname    = ''
        self.video_fname      = ''
        if DEBUG:
            self.nrows_to_import  = NROWS
        else:
            self.nrows_to_import  = None

        if platform =='linux' or platform == 'linux2':
            self.sep = "/"
        else:
            self.sep = "\\"

        self.parent_dir = DATA_DIR

    def import_rigid_body_data(self):
        col_keys = ['frame', 'time (seconds)', 'mean marker error',
                    'marker quality', 'rigid body', 'position', 'rotation',
                    'x', 'y', 'z']
        filename = self.marker_fname
        header = pd.read_csv(self.marker_hdr_fname, nrows=1, usecols=list(range(0,22)),
                             header=None)
        header = dict(header.values.reshape((11,2)).tolist())
        if self.nrows_to_import is None:
            df = pd.read_csv(
                filename, header=list(range(0,5))
            )
        else:
            df = pd.read_csv(
                filename, nrows=self.nrows_to_import, header=list(range(0,5))
            )
        shape = df.shape
        if shape[1] > 10:
            diff = shape[1] - 10
            df = df.drop(df.columns[-diff::], axis=1)
        cols = df.columns.values
        new_cols = []
        for i, lstr in enumerate(cols):
            col_val = []
            lstr_list = [ls for ls in lstr]
            if 'Rigid Body Marker' in lstr_list: continue
            for j, str_val in enumerate(lstr):
                if str_val.lower() in col_keys or 'glasses' in str_val.lower():
                    if ' ' in str_val:
                        str_val = str_val.replace(' ', '_')
                    col_val.append(str_val)
            new_cols.append('_'.join(col_val))
        df.columns = new_cols

        return df, header

    def cleanup_marker_data(self, filename):
        chunksize = 10
        file_size_mb = stat(filename).st_size/(1024*1024)
        ff = filename.split(self.sep)[:-1]
        if file_size_mb > 0.5:
            print("processing: ", filename)
            header = pd.read_csv(
                filename, nrows=1, usecols=list(range(0,22)), header=None)
            hdr_name = join(self.sep.join(ff),
                                 filename[:-4] + '_header.csv')
            header.to_csv(hdr_name, index=False)
            df_hdr = pd.read_csv(filename, skiprows=2, header=list(range(0,5)),
                                 nrows=0)
            df = pd.read_csv(filename, skiprows=6, usecols=list(range(38)))
            df.columns = df_hdr.columns[:38]
            amended_df_name = join(
                self.sep.join(ff), filename[:-4] + '_amended.csv')
            df.to_csv(amended_df_name, index=False)
            print("saved: ", amended_df_name)

    def import_marker_file(self, filename):
        if self.nrows_to_import is None:
            df = pd.read_csv(
                filename, header=list(range(0,5))
            )
        else:
            df = pd.read_csv(
                filename, nrows=self.nrows_to_import,
                header=list(range(0,5))
            )
        return df
    
    def import_header_file(self, filename):
        df = pd.read_csv(filename, skiprows=1, nrows=1,
                         usecols=list(range(0,22)), header=None)
        return df

# Import .mat files from markers
    def import_marker_data(self):
        col_keys = ['frame', 'time (seconds)', 'mean marker error',
                    'marker quality', 'marker', 'position', 'rotation',
                    'x', 'y', 'z']
        filename = self.marker_fname
        header = self.import_header_file(self.marker_hdr_fname)
        df = self.import_marker_file(filename)

        shape = df.shape
        if shape[1] > 38:
            diff = shape[1] - 38
            df = df.drop(df.columns[-diff::], axis=1)
        cols = df.columns.values
        new_cols = []
        for i, lstr in enumerate(cols):
            col_val = []
            if type(lstr[0]) is str and "('" in lstr[0]:
                tmp = lstr[0][1:-1].split(',')
                lstr = [ll.replace(" '", '').replace("'", "") for ll in tmp]
            for j, str_val in enumerate(lstr):
                if str_val.lower() in col_keys or 'glasses' in str_val.lower():
                    if ' ' in str_val:
                        str_val = str_val.replace(' ', '_')
                    col_val.append(str_val)
            new_cols.append('_'.join(col_val))
        df.columns = new_cols

        header = dict(header.values.reshape((11,2)).tolist())
        return df, header

    # Import labels from csv
    def import_labels(self, filename):
        if self.nrows_to_import is None:
            df = pd.read_csv(filename, skipinitialspace=True)
        else:
            df = pd.read_csv(filename, nrows=self.nrows_to_import, skipinitialspace=True)

        return df

    def import_mat_data(self, filename):
        try:
            data_dict = mat73.loadmat(filename)
        except TypeError:
            data_dict = loadmat(filename)

        times = data_dict['StoreData']
        df = pd.DataFrame(times, columns=TIME_COLS)
        df = df.applymap(np.squeeze)
        #  a few nested lists, repeat once more
        df = df.applymap(np.squeeze)
        return df

    def import_time_data(self):
        filename = self.timeline_fname
        if '.mat' in filename:
            return self.import_mat_data(filename)
        elif '.csv' in filename:
            return pd.read_csv(filename)

    def import_imu_data(self):
        filename = self.imu_fname
        try:
            df = pd.read_json(filename, lines=True, compression='gzip')
        except EOFError:
            df = pd.read_json(splitext(filename)[0], lines=True)
        data_df = pd.DataFrame(df['data'].tolist())
        df = pd.concat([df.drop('data', axis=1), data_df], axis=1)
        hdr = self.import_imu_header()
        hdr = hdr.to_dict().pop(0)
        return df, hdr
    
    def import_imu_header(self):
        return pd.read_json(self.imu_hdr_fname, orient='index')
    
    def import_video(self):
        return cv2.VideoCapture(self.import_video)

class DataSynchronizer():
    def __init__(self):
        self.start_ind = None
        self.end_ind = None

    # Sync and downsample to match frequences across the datasets
    def sync_df_start(self, df):
        ''' sync dataframe '''
        my_df = df.drop(index=df.index[:self.start_ind],
                axis=0,
                inplace=False)
        return my_df

    def sync_df_end(self, df):
        ''' sync dataframe '''
        diff = self.end_ind - self.start_ind + 1
        my_df = df.drop(index=df.index[diff::],
                axis=0,
                inplace=False)
        return my_df

    def sync_df(self, df):
        ''' sync to mat data '''
        my_df = df.iloc[self.start_ind:self.end_ind+1]
        # my_df = self.sync_df_start(df)
        # my_df = self.sync_df_end(my_df)
        return my_df

    def set_bounds(self, times, t_start, t_end):
        ''' sync to using masking method '''
        # find the index that is closest to t_start
        start_mask0 = times <= t_start
        start_mask1 = times > t_start
        if not start_mask0.any() and t_start < times[0]:
            start0 = times[0]
        else:
            start0 = times[start_mask0][-1]
        start1 = times[start_mask1][0]

        # take lowest
        dt0 = np.abs(t_start-start0)
        dt1 = np.abs(t_start-start1)
        if dt0 < dt1:
            start_val = start0
        else:
            start_val = start1
        start_ind = np.where(times==start_val)[0][0]

        end_mask0 = times <= t_end
        end_mask1 = times > t_end
        end0 = times[end_mask0][-1]
        
        times_end = times[-1]
        if np.isnan(times_end): 
            # times_end = times[-2] + 1000
            times_end = times[-2] + 1
            times[-1] = times_end

        if not end_mask1.any() and t_end >= times_end:
            end1 = times_end
        else:
            end1 = times[end_mask1][0]

        # take dt1
        dt0 = np.abs(t_end-end0)
        dt1 = np.abs(t_end-end1)
        if dt0 < dt1:
            end_val = end0
        else:
            end_val = end1
        end_ind = np.where(times==end_val)[0][0]
        # end_diff = end_ind - start_ind + 1

        self.start_ind = start_ind
        self.end_ind   = end_ind

class TFDataPipeline():
    def __init__(self, window_size=60, batch_size=32):
        self.window_size = window_size
        self.window_shift = self.window_size
        self.batch_size = batch_size
        self.mb_kmeans = MiniBatchKMeans(n_clusters=int(self.batch_size//2))
        self.shuffle_flag = True

    def kcenter(self, x, y):
        '''
        * get some batches
        * perform iterative kcenter_greedy on these batches
        * get their scores and weight these batches (x, y, weights)
        * returns final centers and max distances
        '''
        self.mb_kmeans.partial_fit(x, y)
        x, dist = self.mb_kmeans.transform(x, y)
        # apply threshold
        # return weighted labels for each batch, or downsample the batch_size
        # to some percentage
        pass

    def sub_batch(self, x):
        sub_x = x.batch(self.window_size, drop_remainder=True)
        return sub_x

    def get_dataset(self, x, reduce_mean=False):
        import tensorflow as tf
        ds = tf.data.Dataset.from_tensor_slices((x))\
                .window(self.window_size, shift=self.window_shift,
                        drop_remainder=True)\
                .flat_map(self.sub_batch)\
                .batch(self.batch_size, drop_remainder=True)
        if reduce_mean:
            ds = ds.map(lambda y: tf.reduce_mean(y, axis=1))
        ds = ds.prefetch(1)
        return ds
    
    def zip_datasets(self, x_ds, y_ds):
        ds = tf.data.Dataset.zip((x_ds, y_ds))
        if self.shuffle_flag:
            ds.shuffle(3000, reshuffle_each_iteration=True)
        return ds

# Data Feature Handler Class:
    # Deal with metafile
    # Load given a configuration dict
    # Set the directory and filename
class ProjectFileHandler():
    def __init__(self, config:dict):
        self.config  = config
        self.fset_id = -1
        self.metafile_name = 'metafile.json'
        self.set_home_directory()

    def set_home_directory(self, home_directory=DATA_DIR):
        self.home_directory = home_directory

    def set_parent_directory(self, parent_directory='imu_mwl'):
        self.parent_directory = join(self.home_directory,
                                          parent_directory)
        makedirs(self.parent_directory, exist_ok=True)

    def set_id(self, fset_id:int=-1):
        if fset_id != -1:
            self.fset_id = fset_id
        else:
            ww = walk(self.parent_directory, topdown=True)
            for _, dirs, _ in ww:
                tmp = len(dirs)
                break
            self.fset_id = tmp

    def set_project_directory(self):
        if self.fset_id == -1:
            self.set_id()
            print("Data id not set, auto assigned to: ", self.fset_id)

        self.project_directory = join(self.parent_directory,
                                         str(self.fset_id).zfill(2))
        makedirs(self.project_directory, exist_ok=True)

    def get_metafile_path(self):
        fname = join(self.project_directory, self.metafile_name)
        return fname

    def get_id_from_config(self):
        glob_pattern = join(self.parent_directory, '**', '*.json')
        mfiles = glob.glob(glob_pattern, recursive=True)
        cfg_id = {}
        for mfile in mfiles:
            fset_id_in = mfile.split(sep)[-2]
            with open(mfile, 'r') as f:
                cfg_in = json.load(f)
            cfg_id[fset_id_in] = cfg_in
            if self.config == cfg_in:
                return fset_id_in
        
        print("unable to find matching config id")

    def save_metafile(self):
        fname = self.get_metafile_path()
        with open(fname, 'w') as f:
            json.dump(self.config, f)

import asyncio
import glob
import re
from datetime import datetime, timedelta
from os import listdir, stat
from os.path import exists as path_exists
from os.path import isdir, join as path_join, sep, splitext
from sys import platform

import numpy as np
import pandas as pd
import pytz

try:
    import mat73
except ImportError:
    mat73 = None
from scipy.io import loadmat

from config import DEBUG, NROWS, NO_HACC_ID, SEAT_DATA_DIR, TIME_COLS


DATA_DIR = SEAT_DATA_DIR
READ_MARKER = True


def datetime_to_sec(time_in):
    dstr = datetime.today()
    try:
        dstr = datetime.strptime(time_in, "%Y-%m-%d %I.%M.%S.%f %p")
    except ValueError:
        if "Take" in time_in:
            time_section = time_in[5:-4]
            if "Task" or "MR" in time_in:
                start_ind = re.search(r"\d{4}", time_in)
                end_ind = re.search(r"M_", time_in)
                time_section = time_in[start_ind.start():end_ind.start() + 1]
            dstr = datetime.strptime(time_section, "%Y-%m-%d %I.%M.%S %p")
        elif "_" in time_in:
            dstr = datetime.strptime(time_in, "%Y_%m_%d-%H_%M_%S")
        elif "/" in time_in:
            dstr = datetime.strptime(time_in, "%d/%m/%Y %H:%M:%S.%f")
        elif "Z" in time_in:
            dstr = datetime.strptime(time_in, "%Y%m%dT%H%M%SZ")
    return dstr.timestamp()


def sec_to_datetime(sec):
    return datetime.fromtimestamp(sec)


def mat_to_sec(time):
    if sum([":" == ch for ch in time]) == 1:
        dstr = datetime.strptime(time, "%m/%d/%Y %H:%M")
    elif time[-1] == "M":
        dstr = datetime.strptime(time, "%m/%d/%Y  %I:%M:%S %p")
    elif "." not in time:
        dstr = datetime.strptime(time, "%m/%d/%Y  %H:%M:%S")
    else:
        dstr = datetime.strptime(time, "%m/%d/%Y %H:%M:%S.%f")
    return dstr.timestamp()


def find_column(df, text):
    text = text.lower()
    for col in df.columns:
        if text in str(col).lower():
            return col
    return None


class DataSynchronizer:
    def __init__(self):
        self.start_ind = None
        self.end_ind = None

    def sync_df(self, df):
        return df.iloc[self.start_ind:self.end_ind + 1].reset_index(drop=True)

    def set_bounds(self, times, t_start, t_end):
        times = np.asarray(times, dtype=float).copy()
        if len(times) == 0 or np.isnan(times).all():
            raise ValueError("No valid timestamps available for syncing")
        if np.isnan(times[-1]) and len(times) > 1:
            times[-1] = times[-2] + 1

        start_mask0 = times <= t_start
        start_mask1 = times > t_start
        start0 = times[0] if (not start_mask0.any() and t_start < times[0]) else times[start_mask0][-1]
        start1 = times[start_mask1][0] if start_mask1.any() else times[-1]
        start_val = start0 if np.abs(t_start - start0) < np.abs(t_start - start1) else start1

        end_mask0 = times <= t_end
        end_mask1 = times > t_end
        end0 = times[end_mask0][-1] if end_mask0.any() else times[0]
        end1 = times[-1] if (not end_mask1.any() and t_end >= times[-1]) else times[end_mask1][0]
        end_val = end0 if np.abs(t_end - end0) < np.abs(t_end - end1) else end1

        self.start_ind = np.where(times == start_val)[0][0]
        self.end_ind = np.where(times == end_val)[0][0]


class SubjectData:
    def __init__(self, condition="M", subject="S01"):
        self.condition = condition
        self.subject = subject
        self.subject_id = int(re.search(r"\d+", subject).group()) if subject[0] == "S" else subject
        self.study_start = 0
        self.study_end = 0
        self.parent_dir = DATA_DIR
        self.subject_dir = path_join(self.parent_dir, self.subject)
        self.sep = "/" if platform in {"linux", "linux2"} else "\\"
        self.nrows_to_import = NROWS if DEBUG else None

        self.timeline_fname = ""
        self.marker_fname = ""
        self.marker_hdr_fname = ""
        self.pressure_fname = ""
        self.summary_fname = ""
        self.accel_fname = ""
        self.imu_fname = ""
        self.imu_hdr_fname = ""
        self.ecg_fname = ""

        self.timeline_df = pd.DataFrame()
        self.marker_df = pd.DataFrame()
        self.pressure_df = pd.DataFrame()
        self.summary_df = pd.DataFrame()
        self.accel_df = pd.DataFrame()
        self.imu_df = pd.DataFrame()
        self.ecg_df = pd.DataFrame()
        self.read_marker_data = False

    def get_cond_file(self, files):
        for fname in files:
            base = fname.split(sep)[-1]
            if self.condition in base and self.subject in base:
                return fname
        return ""

    def list_sub_dirs(self, parent_dir, endswith=None):
        reg_str = r"[0-9]+$" if endswith is None else r"[0-9]+{0}$".format(endswith)
        regex = re.compile(reg_str)
        return sorted([
            path_join(parent_dir, d)
            for d in listdir(parent_dir)
            if isdir(path_join(parent_dir, d)) and bool(regex.search(d))
        ])

    def check_times(self, sub_dirs, is_utc=False):
        if is_utc:
            hdrs = [
                pd.read_json(path_join(sub_dir, "recording.g3"), orient="index")
                for sub_dir in sub_dirs
            ]
            times = [hdr.to_dict().pop(0)["created"] for hdr in hdrs]
            times = [datetime.fromisoformat(time[:-1]) for time in times]
            times = [(time.timestamp() + timedelta(hours=11).seconds) for time in times]
        else:
            times = [datetime_to_sec(sub_dir.split(self.sep)[-1]) for sub_dir in sub_dirs]

        sel_dir = sub_dirs[-1]
        for i, time in enumerate(times[:-1]):
            if self.study_start > time and self.study_start < times[i + 1]:
                sel_dir = sub_dirs[i]
        return sel_dir

    def selected_bioharness_dir(self):
        sub_dirs = self.list_sub_dirs(self.subject_dir)
        if len(sub_dirs) == 0:
            raise FileNotFoundError(f"No BioHarness data directories found for {self.subject}")
        return self.check_times(sub_dirs) if len(sub_dirs) > 1 else sub_dirs[0]

    def set_marker_fname(self):
        data_dir = path_join(self.subject_dir, "Motive Logs")
        data_glob = path_join(data_dir, "*_amended.csv") if path_exists(data_dir) else path_join(self.subject_dir, "*Take*_amended.csv")
        data_files = sorted(glob.glob(data_glob))
        if self.subject_id > 16:
            if self.condition in "MR":
                data_files = [fname for fname in data_files if "MR" in fname]
            else:
                data_files = [fname for fname in data_files if "MR" not in fname]
        self.marker_fname = self.check_times(data_files) if len(data_files) > 1 else data_files[-1]
        self.marker_hdr_fname = self.marker_fname.split("_amended")[0] + "_header.csv"

    def set_pressure_fname(self):
        sub_dir = self.selected_bioharness_dir()
        pressure_files = sorted(glob.glob(path_join(sub_dir, "BR*.csv")))
        if not pressure_files:
            pressure_files = sorted(glob.glob(path_join(sub_dir, "*_Breathing.csv")))
        self.pressure_fname = pressure_files[-1]

    def set_summary_fname(self):
        sub_dir = self.selected_bioharness_dir()
        summary_files = sorted(glob.glob(path_join(sub_dir, "Summary*.csv")))
        if not summary_files:
            dt_info = sub_dir.split(sep)[-1]
            summary_files = sorted(glob.glob(path_join(sub_dir, dt_info + "_Summary*.csv")))
        self.summary_fname = summary_files[-1]

    def set_ecg_fname(self):
        sub_dir = self.selected_bioharness_dir()
        ecg_files = [
            fname for fname in sorted(glob.glob(path_join(sub_dir, "*.csv")))
            if "ecg" in fname.split(sep)[-1].lower()
        ]
        self.ecg_fname = ecg_files[-1] if ecg_files else ""

    def set_imu_fname(self):
        sub_dirs = self.list_sub_dirs(self.subject_dir, endswith="Z")
        sub_dir = self.check_times(sub_dirs, is_utc=True) if len(sub_dirs) > 1 else sub_dirs[0]
        self.imu_fname = sorted(glob.glob(path_join(sub_dir, "imu*")))[-1]
        self.imu_hdr_fname = sorted(glob.glob(path_join(sub_dir, "recording.g3")))[-1]

    def set_accel_fname(self):
        sub_dir = self.selected_bioharness_dir()
        accel_files = sorted(glob.glob(path_join(sub_dir, "Accel*.csv")))
        if not accel_files:
            accel_files = sorted(glob.glob(path_join(sub_dir, "*_Accel.csv")))
        accel_files = [fname for fname in accel_files if "g" not in fname.lower().split(sep)[-1]]
        self.accel_fname = accel_files[-1]

    def set_timeline(self):
        times_files = sorted(glob.glob(path_join(self.subject_dir, "*.csv")))
        self.timeline_fname = self.get_cond_file(times_files)
        self.timeline_df = self.import_time_data()

        mat_time = self.timeline_df["Timestamps"].map(mat_to_sec)
        mat_start_ind = self.timeline_df.index[self.timeline_df["Event"] == "Start Test"].tolist()[0]
        self.study_start = mat_time.values[mat_start_ind]
        self.study_end = mat_time.values[-1]

    def set_fnames(self):
        if self.read_marker_data:
            self.set_marker_fname()
        self.set_pressure_fname()
        self.set_summary_fname()
        self.set_ecg_fname()
        if self.subject_id > 11:
            self.set_imu_fname()
        if self.subject_id not in NO_HACC_ID:
            self.set_accel_fname()

    def import_labels(self, filename):
        return pd.read_csv(filename, skipinitialspace=True, nrows=self.nrows_to_import)

    def import_mat_data(self, filename):
        if mat73 is not None:
            try:
                data_dict = mat73.loadmat(filename)
            except TypeError:
                data_dict = loadmat(filename)
        else:
            data_dict = loadmat(filename)
        df = pd.DataFrame(data_dict["StoreData"], columns=TIME_COLS)
        return df.applymap(np.squeeze).applymap(np.squeeze)

    def import_time_data(self):
        if ".mat" in self.timeline_fname:
            return self.import_mat_data(self.timeline_fname)
        return pd.read_csv(self.timeline_fname)

    def import_imu_data(self):
        try:
            df = pd.read_json(self.imu_fname, lines=True, compression="gzip")
        except EOFError:
            df = pd.read_json(splitext(self.imu_fname)[0], lines=True)
        data_df = pd.DataFrame(df["data"].tolist())
        df = pd.concat([df.drop("data", axis=1), data_df], axis=1)
        hdr = pd.read_json(self.imu_hdr_fname, orient="index").to_dict().pop(0)
        return df, hdr

    def import_marker_file(self, filename):
        return pd.read_csv(filename, header=list(range(0, 5)), nrows=self.nrows_to_import)

    def import_header_file(self, filename):
        return pd.read_csv(filename, skiprows=1, nrows=1, usecols=list(range(0, 22)), header=None)

    def import_marker_data(self):
        col_keys = ["frame", "time (seconds)", "mean marker error", "marker quality", "marker", "position", "rotation", "x", "y", "z"]
        header = self.import_header_file(self.marker_hdr_fname)
        df = self.import_marker_file(self.marker_fname)
        if df.shape[1] > 38:
            df = df.drop(df.columns[-(df.shape[1] - 38):], axis=1)
        new_cols = []
        for lstr in df.columns.values:
            col_val = []
            if type(lstr[0]) is str and "('" in lstr[0]:
                tmp = lstr[0][1:-1].split(",")
                lstr = [ll.replace(" '", "").replace("'", "") for ll in tmp]
            for str_val in lstr:
                if str_val.lower() in col_keys or "glasses" in str_val.lower():
                    col_val.append(str_val.replace(" ", "_"))
            new_cols.append("_".join(col_val))
        df.columns = new_cols
        return df, dict(header.values.reshape((11, 2)).tolist())

    def load_dataframes(self):
        self.timeline_df = self.import_time_data()
        self.pressure_df = self.import_labels(self.pressure_fname)
        self.summary_df = self.import_labels(self.summary_fname)
        if self.ecg_fname:
            self.ecg_df = self.import_labels(self.ecg_fname)
        if self.read_marker_data:
            try:
                self.marker_df, self.mkr_hdr = self.import_marker_data()
            except Exception:
                print("error reading marker data on {0} - {1}".format(self.subject_id, self.condition))
        if self.subject_id not in NO_HACC_ID:
            try:
                self.accel_df = self.import_labels(self.accel_fname)
            except Exception:
                print("error reading accel data on {0} - {1}".format(self.subject_id, self.condition))
        if self.subject_id > 11:
            try:
                self.imu_df, self.imu_hdr = self.import_imu_data()
            except Exception:
                print("error reading imu data on {0} - {1}".format(self.subject_id, self.condition))

    def sync_bioharness_df(self, df):
        cols = df.columns
        if "Year" in cols:
            year = int(df["Year"].dropna().values[0])
            month = int(df["Month"].dropna().values[0])
            day = int(df["Day"].dropna().values[0])
            dt_obj = datetime.strptime(f"{year}/{month}/{day}", "%Y/%m/%d")
            times = df["ms"].interpolate().values / 1000 + dt_obj.timestamp()
        else:
            times = df["Time"].map(datetime_to_sec).values
        df = df.copy()
        df["sec"] = times
        data_sync = DataSynchronizer()
        data_sync.set_bounds(times, self.study_start, self.study_end)
        return data_sync.sync_df(df)

    def sync_pressure_df(self):
        self.pressure_df = self.sync_bioharness_df(self.pressure_df)

    def sync_accel_df(self):
        self.accel_df = self.sync_bioharness_df(self.accel_df)

    def sync_summary_df(self):
        self.summary_df = self.sync_bioharness_df(self.summary_df)

    def sync_ecg_df(self):
        if self.ecg_df.empty:
            return
        ecg_col = find_column(self.ecg_df, "ecg")
        if ecg_col is None:
            print("No ECG column found on {0} - {1}".format(self.subject_id, self.condition))
            self.ecg_df = pd.DataFrame(columns=["condition", "sec", "ecg"])
            return
        ecg_df = self.sync_bioharness_df(self.ecg_df)
        ecg_df = ecg_df[["sec", ecg_col]].rename(columns={ecg_col: "ecg"})
        ecg_df.insert(0, "condition", self.condition)
        self.ecg_df = ecg_df

    def sync_marker_df(self):
        time_start = datetime_to_sec(self.mkr_hdr["Capture Start Time"])
        marker_time = self.marker_df["Time_(Seconds)"].values + time_start
        self.marker_df["Time_(Seconds)"] = marker_time
        self.marker_df["sec"] = marker_time
        data_sync = DataSynchronizer()
        data_sync.set_bounds(marker_time, self.study_start, self.study_end)
        self.marker_df = data_sync.sync_df(self.marker_df).fillna(0)

    def sync_imu_df(self):
        na_inds = self.imu_df.loc[pd.isna(self.imu_df["accelerometer"]), :].index.values
        self.imu_df.drop(index=na_inds, inplace=True)
        imu_times = self.imu_df["timestamp"].values
        mask = imu_times > 3 * 60 * 60
        if mask.any():
            bad_args = np.arange(0, len(mask))[mask]
            self.imu_df.drop(index=self.imu_df.iloc[bad_args].index, inplace=True)
            imu_times = self.imu_df["timestamp"].values
        print(np.mean(1 / (imu_times[1:] - imu_times[:-1])))
        self.imu_df["timestamp_interp"] = pd.Series(imu_times).interpolate().values

        iso_tz = self.imu_hdr["created"]
        _ = pytz.timezone(self.imu_hdr["timezone"])
        start_time = datetime.fromisoformat(iso_tz[:-1]) + timedelta(hours=11)
        imu_sec = np.array([
            (start_time + timedelta(seconds=val)).timestamp()
            for val in self.imu_df["timestamp_interp"].values
        ])
        self.imu_df["sec"] = imu_sec
        data_sync = DataSynchronizer()
        data_sync.set_bounds(imu_sec, self.study_start, self.study_end)
        self.imu_df = data_sync.sync_df(self.imu_df)

    def sync_all_df(self):
        if self.study_start == 0 or self.study_start is None:
            self.set_timeline()
        self.sync_pressure_df()
        self.sync_summary_df()
        self.sync_ecg_df()
        if self.subject_id not in NO_HACC_ID:
            try:
                self.sync_accel_df()
            except Exception:
                print("Error syncing accel data on {0} - {1}".format(self.subject_id, self.condition))
                self.accel_df = pd.DataFrame()
        if self.read_marker_data:
            try:
                self.sync_marker_df()
            except Exception:
                print("Error syncing marker data on {0} - {1}".format(self.subject_id, self.condition))
                self.marker_df = pd.DataFrame()
        if self.subject_id > 11:
            try:
                self.sync_imu_df()
            except Exception:
                print("Error syncing imu data on {0} - {1}".format(self.subject_id, self.condition))
                self.imu_df = pd.DataFrame()

    def cleanup_marker_data(self, filename):
        file_size_mb = stat(filename).st_size / (1024 * 1024)
        ff = filename.split(self.sep)[:-1]
        if file_size_mb > 0.5:
            print("processing: ", filename)
            header = pd.read_csv(filename, nrows=1, usecols=list(range(0, 22)), header=None)
            hdr_name = path_join(self.sep.join(ff), filename[:-4] + "_header.csv")
            header.to_csv(hdr_name, index=False)
            df_hdr = pd.read_csv(filename, skiprows=2, header=list(range(0, 5)), nrows=0)
            df = pd.read_csv(filename, skiprows=6, usecols=list(range(38)))
            df.columns = df_hdr.columns[:38]
            amended_df_name = path_join(self.sep.join(ff), filename[:-4] + "_amended.csv")
            df.to_csv(amended_df_name, index=False)
            print("saved: ", amended_df_name)


async def sync(condition="M", subject="S01", read_marker_data=False):
    sbj_data = SubjectData(condition=condition, subject=subject)
    sbj_data.read_marker_data = read_marker_data
    sbj_data.set_timeline()
    print("subject no: {0}\tcondition: {1}\ttime set\t||\tstart: {2}\tend: {3}".format(
        int(subject[-2:]), condition, sec_to_datetime(sbj_data.study_start), sec_to_datetime(sbj_data.study_end)
    ))
    sbj_data.set_fnames()
    sbj_data.load_dataframes()

    if sbj_data.subject_id > 11:
        try:
            iso_tz = sbj_data.imu_hdr["created"]
            start_time = datetime.fromisoformat(iso_tz[:-1]) + timedelta(hours=11)
            print("pre sync\t||\timu_start: {0}\timu shape: {1}".format(start_time, sbj_data.imu_df.shape))
        except Exception as exc:
            print(exc)

    sbj_data.sync_all_df()

    if sbj_data.subject_id > 11 and not sbj_data.imu_df.empty:
        print("post sync\t||\timu_start: {0}\timu shape: {1}".format(
            sec_to_datetime(sbj_data.imu_df["sec"].values[0]), sbj_data.imu_df.shape
        ))
    return sbj_data


def get_subjects():
    return sorted(glob.glob(path_join(DATA_DIR, "S[0-9]*")))


def cleanup(sbjs=None):
    sbj_glob = get_subjects()
    sbjs = [sbj.split(sep)[-1] for sbj in sbj_glob] if sbjs is None else sbjs
    for sbj in sbjs:
        sbj_data = SubjectData(subject=sbj)
        sbj_dir = sbj_data.subject_dir
        data_dir = path_join(sbj_dir, "Motive Logs")
        data_glob = path_join(data_dir, "*.csv") if path_exists(data_dir) else path_join(sbj_dir, "*Take*.csv")
        dlist = sorted(glob.glob(data_glob))
        inds_to_pop = [dlist.index(fname) for fname in dlist if "_amended" in fname or "_header" in fname]
        for idx in sorted(inds_to_pop, reverse=True):
            dlist.pop(idx)
        for fname in dlist:
            sbj_data.cleanup_marker_data(fname)


async def coro_sync(sbjs=None):
    conditions = ["M", "R", "L0", "L1", "L2", "L3"]
    sbj_glob = get_subjects()
    sbjs = [sbj.split(sep)[-1] for sbj in sbj_glob] if sbjs is None else sbjs
    for sbj in sbjs:
        print(sbj)
        for condition in conditions:
            yield await sync(condition=condition, subject=sbj, read_marker_data=READ_MARKER)


async def sync_main(**kwargs):
    index = False
    async for sbj_data in coro_sync(**kwargs):
        sbj_dir = sbj_data.subject_dir
        condition = sbj_data.condition

        if sbj_data.subject_id > 11 and not sbj_data.imu_df.empty:
            sbj_data.imu_df.to_csv(path_join(sbj_dir, "{0}_imu_df.csv".format(condition)), index=index)

        if sbj_data.read_marker_data and not sbj_data.marker_df.empty:
            sbj_data.marker_df.to_csv(path_join(sbj_dir, "{0}_marker_df.csv".format(condition)), index=index)

        sbj_data.pressure_df.to_csv(path_join(sbj_dir, "{0}_pressure_df.csv".format(condition)), index=index)
        sbj_data.summary_df.to_csv(path_join(sbj_dir, "{0}_summary_df.csv".format(condition)), index=index)
        if not sbj_data.accel_df.empty:
            sbj_data.accel_df.to_csv(path_join(sbj_dir, "{0}_accel_df.csv".format(condition)), index=index)
        if not sbj_data.ecg_df.empty:
            sbj_data.ecg_df.to_csv(path_join(sbj_dir, "{0}_ecg_df.csv".format(condition)), index=index)

        if sbj_data.subject_id > 11 and not sbj_data.imu_df.empty:
            xsens_csv = path_join(sbj_dir, "{0}_xsens_df.csv".format(condition))
            data_df = sbj_data.imu_df
            pss_df = sbj_data.pressure_df
            lbl_df = sbj_data.summary_df

            acc_data = np.stack(data_df["accelerometer"].values)
            gyr_data = np.stack(data_df["gyroscope"].values)
            x_time = data_df["sec"].values.reshape(-1, 1)

            br_col = [col for col in pss_df.columns.values if "breathing" in col.lower()][0]
            pss_data = np.interp(x_time, pss_df["sec"].values, pss_df[br_col].values).reshape(-1, 1)

            br_lbl = [col for col in lbl_df.columns.values if "br" in col.lower()][0]
            lbl_data = np.interp(x_time, lbl_df["sec"].values, lbl_df[br_lbl].values).reshape(-1, 1)

            xsens_data = np.concatenate((x_time, lbl_data, pss_data, acc_data, gyr_data), axis=1)
            columns = ["sec", "BR", "PSS", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
            xsens_df = pd.DataFrame(xsens_data, columns=columns)
            xsens_df["condition"] = condition
            xsens_df["subject"] = sbj_data.subject_id
            re_order = ["sec", "BR", "PSS", "condition", "subject", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
            xsens_df[re_order].to_csv(xsens_csv, index=False)
            print("xsens df saved for {0} - {1}".format(sbj_data.subject_id, condition))


def get_file_list(starts_with=None, sbj=None):
    f_glob = path_join(DATA_DIR, sbj, "**") if sbj is not None else path_join(DATA_DIR, "S*", "**")
    f_glob = path_join(f_glob, f"{starts_with}*.csv") if starts_with is not None else path_join(f_glob, "*.csv")
    return sorted(glob.glob(f_glob, recursive=True))


def marker_stuff(marker_fname):
    try:
        df = pd.read_csv(marker_fname)
    except Exception:
        return
    if len(df) < 2:
        return
    if "sec" not in df.columns.values:
        df["sec"] = df["Time_(Seconds)"].values
        df.to_csv(marker_fname, index=False)


if __name__ == "__main__":
    sbjs = ["S" + str(i) for i in range(12, 31)]
    asyncio.run(sync_main(sbjs=sbjs))

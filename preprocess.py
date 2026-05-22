import gc
import argparse
import pickle
import warnings
from os import makedirs
from os.path import join, exists

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import LabelEncoder

from config import IMU_FS, BR_FS, SEAT_DATA_DIR, SBJ_PROCESSED_DIR
from utils import load_dataset, prepare_multimodal, sync_with_last_val
from digitalsignalprocessing import window_filter

seed = 42
np.random.seed(seed)

PSS_FS = BR_FS

imu_issues = [17, 26, 30]
subjects = [f"S{str(i).zfill(2)}" for i in range(12, 31) if i not in imu_issues]

sbj_processed_dir = SBJ_PROCESSED_DIR


def normalize_output_directory(output_directory: str) -> str:
    out = str(output_directory).strip().lower()
    if out == "mr":
        return "mr"
    if out in {"level", "levels"}:
        return "levels"
    raise ValueError(f"Unsupported output directory '{output_directory}'")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrite data or keep as is')
    parser.add_argument('--data_str', type=str,
                        default='imu_filt',
                        choices=['imu_filt', 'imu_ica'])
    parser.add_argument('--condition', type=str, default='[M,R]',
                        choices=['[M,R]', '[!M]*', 'L*'])
    parser.add_argument('--output-directory', type=str, default='mr',
                        choices=['mr', 'level', 'levels'])
    parser.add_argument('--window_size', type=float,
                        default=20,
                        help='Window size for sliding window procedure, set in seconds')
    parser.add_argument('--window_shift', type=float,
                        default=1,
                        help='Window shift for sliding window procedure, set in seconds')
    parser.add_argument('--n_components', type=int,
                        default=1,
                        help='ICA components')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def fit_subject_ica(window_list, n_components=6, **kwargs):
    ica = FastICA(n_components, whiten='arbitrary-variance',
                  random_state=seed, **kwargs)
    subject_data = np.concatenate(window_list, axis=0)
    ica.fit(subject_data)
    return ica


def apply_ica_model(window_list, ica, fs=IMU_FS):
    return [
        window_filter(ica.transform(win_data), 2 * fs, window='triang')
        for win_data in window_list
    ]


def get_subject_level_ica_models(sbj_dir, subject, acc_filt_list, gyr_filt_list,
                                 fs=IMU_FS, n_components=2, output_dir='mr',
                                 **kwargs):
    output_dir = normalize_output_directory(output_dir)
    if output_dir == 'mr':
        acc_ica = fit_subject_ica(acc_filt_list, n_components=n_components, **kwargs)
        gyr_ica = fit_subject_ica(gyr_filt_list, n_components=n_components, **kwargs)
        return acc_ica, gyr_ica

    mr_fname = join(sbj_dir, subject + '.pkl')
    if not exists(mr_fname):
        raise FileNotFoundError(f"Missing subject-level MR ICA fit for {subject}: {mr_fname}")

    with open(mr_fname, 'rb') as f:
        mr_processed = pickle.load(f)

    acc_ica = mr_processed.get('acc_ica_model')
    gyr_ica = mr_processed.get('gyr_ica_model')
    if acc_ica is None or gyr_ica is None:
        raise KeyError(f"Subject-level MR ICA models not found in {mr_fname}")

    return acc_ica, gyr_ica


def signal_processing(sbj_processed_dir: str, sbj_dicts: list, fs=IMU_FS,
                      window_size=20, window_shift=1, n_components=2,
                      overwrite=False, output_dir='mr'):
    output_dir = normalize_output_directory(output_dir)
    sbj_processed_list = []

    for sbj_dict in sbj_dicts:
        subject = sbj_dict['subject']
        sbj_dir = join(sbj_processed_dir, output_dir)
        sbj_fname = join(sbj_dir, subject + '.pkl')

        makedirs(sbj_dir, exist_ok=True)

        if exists(sbj_fname) and not overwrite:
            with open(sbj_fname, 'rb') as f:
                sbj_processed = pickle.load(f)
        else:
            sbj_processed = {}
            imu_df = sbj_dict['imu']
            pss_df = sbj_dict['pss']
            ecg_df = sbj_dict.get('ecg')
            br_df = sbj_dict['br']

            data = prepare_multimodal(
                imu_df,
                pss_df,
                ecg_df=ecg_df,
                window_size=window_size,
                window_shift=window_shift,
            )
            if data is None:
                continue

            if len(data) == 5:
                imu_filt, pss_filt, pss_freqs, pss_time, conds_wins = data
                ecg_filt, ecg_time = None, None
            else:
                imu_filt, pss_filt, pss_freqs, pss_time, \
                        conds_wins, ecg_filt, ecg_time = data

            imu_filt_list = [arr[..., 1:] for arr in imu_filt]
            acc_filt_list = [arr[..., :3] for arr in imu_filt_list]
            gyr_filt_list = [arr[..., 3:] for arr in imu_filt_list]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                acc_ica_model, gyr_ica_model = get_subject_level_ica_models(
                    join(sbj_processed_dir, 'mr'),
                    subject,
                    acc_filt_list,
                    gyr_filt_list,
                    fs=fs,
                    n_components=n_components,
                    output_dir=output_dir,
                    whiten_solver='eigh',
                    max_iter=1000,
                )
                acc_ica_list = apply_ica_model(acc_filt_list, acc_ica_model, fs=fs)
                gyr_ica_list = apply_ica_model(gyr_filt_list, gyr_ica_model, fs=fs)
                imu_ica_list = [
                    np.concatenate((acc, gyr), axis=1)
                    for acc, gyr in zip(acc_ica_list, gyr_ica_list)
                ]

            imu_time = imu_filt[..., 0]
            br_time = br_df['sec'].values
            br_idxs = [sync_with_last_val(t, br_time) for t in imu_time]
            br = br_df['BR'].values[br_idxs]

            sbj_processed['subject'] = subject
            sbj_processed['imu_filt'] = imu_filt[..., 1:]
            sbj_processed['imu_ica'] = np.array(imu_ica_list)
            sbj_processed['acc_ica_model'] = acc_ica_model
            sbj_processed['gyr_ica_model'] = gyr_ica_model
            sbj_processed['ica_fit_conditions'] = (
                np.array(['M', 'R']) if output_dir == 'levels' else np.unique(conds_wins)
            )
            sbj_processed['ica_scope'] = 'subject'
            sbj_processed['pss_filt'] = pss_filt
            sbj_processed['pss_freqs'] = pss_freqs
            sbj_processed['ecg_filt'] = ecg_filt
            sbj_processed['br'] = br
            sbj_processed['conds'] = conds_wins
            sbj_processed['imu_time'] = imu_time
            sbj_processed['pss_time'] = pss_time
            sbj_processed['ecg_time'] = ecg_time
            sbj_processed['br_time'] = br_time

            with open(sbj_fname, 'wb') as f:
                pickle.dump(sbj_processed, f)

        sbj_processed_list.append(sbj_processed)

    return sbj_processed_list


def main(args):
    output_dir = normalize_output_directory(args.output_directory)
    print('Saving data to ', sbj_processed_dir)

    sbj_dicts = load_dataset(subjects, condition=args.condition,
                             data_dir=SEAT_DATA_DIR, debug=args.debug)

    sbj_processed_list = signal_processing(
        sbj_processed_dir,
        sbj_dicts,
        fs=IMU_FS,
        window_size=args.window_size,
        window_shift=args.window_shift,
        n_components=args.n_components,
        overwrite=args.overwrite,
        output_dir=output_dir,
    )

    c_list = [sbj_df['conds'] for sbj_df in sbj_processed_list]
    le = LabelEncoder()
    le.fit(np.concatenate(c_list, axis=0))

    out_dir = join(sbj_processed_dir, output_dir)
    makedirs(out_dir, exist_ok=True)
    le_fname = join(out_dir, 'label_encoder.pkl')
    with open(le_fname, 'wb') as f:
        pickle.dump(le, f)


if __name__ == '__main__':
    gc.enable()
    args = arg_parser()

    if args.debug:
        subjects = subjects[:3]

    _ = PSS_FS
    main(args)

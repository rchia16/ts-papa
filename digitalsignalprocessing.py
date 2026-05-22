import ipdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import butter, sosfilt, decimate, stft, iirnotch, filtfilt
from scipy.signal import convolve, find_peaks
from scipy.signal.windows import hann, triang, exponential
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.fft import fft, fftfreq
from pywt import cwt

from collections import defaultdict

# import cv2
from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from ssqueezepy import cwt as sqz_cwt

from config import WIN_THOLD, WINDOW_SIZE, WINDOW_SHIFT, MQA_THOLD, ACC_THOLD
from config import MARKER_FS, ACC_FS, IMU_FS, FS_RESAMPLE, BR_FS
from config import MIN_RESP_RATE, MAX_RESP_RATE

def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, analog=False, btype='low', output='sos')
    return sos

def butter_lowpass_filter(data, lowcut, fs, order=5, axis=0):
    sos = butter_lowpass(lowcut, fs, order=order)
    y = sosfilt(sos, data, axis=axis)
    return y

def butter_highpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, analog=False, btype='high', output='sos')
    return sos

def butter_highpass_filter(data, highcut, fs, order=5, axis=0):
    sos = butter_highpass(highcut, fs, order=order)
    y = sosfilt(sos, data, axis=axis)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # b, a = butter(order, [low, high], btype='band')
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data, axis=axis)
    return y

def l2norm(data, **kwargs):
    return np.linalg.norm(data, **kwargs)

def movingaverage(data, window_size, axis=0, **kwargs):
    data_in = data.copy()
    return uniform_filter1d(data_in, size=window_size, mode='constant',
                            axis=axis,
                            **kwargs)

def movingmedian(data, window_size, axes=0, **kwargs):
    data_in = data.copy()
    return median_filter(data, window_size, axes=axes, **kwargs)

def run_stft(data, fs, kernel_size=512, stride_size=5):
    f, t, z = stft(data, fs=fs, nperseg=kernel_size,
                        noverlap=kernel_size-stride_size, padded=True)
    return f, t, z

def run_fft(data, fs):
    N = len(data)
    T = 1/fs
    x = np.linspace(0.0, N*T, N, endpoint=False)
    yf = 2.0/N * np.abs(fft(data)[0:N//2])
    xf = fftfreq(N,T)[:N//2]
    return xf, yf

def do_pad_fft(sig, fs=IMU_FS):
    pad_len = npads_frequency_resolution(len(sig), fs=fs)
    data_pad = np.pad(sig.squeeze(), (0, pad_len), 'constant', constant_values=0)
    data_xf, data_yf = run_fft(data_pad, fs)
    return data_xf, data_yf

def torch_fft_targets(patches):
    import torch
    """
    patches: [bs, nvars, patch_num, patch_len]  (time-domain patches)

    returns:
      amp_target:   [bs, nvars, patch_len, patch_num] (FFT amplitudes)
      phase_target: [bs, nvars, patch_len, patch_num] (FFT phases)
    """
    # FFT along last axis (time dimension of patch)
    fft_out = torch.fft.fft(patches, dim=-1)   # complex tensor

    # amplitude (magnitude spectrum)
    amp_target = fft_out.abs()

    # phase (angle spectrum)
    phase_target = torch.angle(fft_out)

    # permute to match masked_pretrain_head_* output convention
    # [bs, nch, patch_len, patch_num]
    amp_target = amp_target.permute(0, 1, 3, 2)
    phase_target = phase_target.permute(0, 1, 3, 2)

    return amp_target, phase_target

def run_cwt(data, **kwargs):
    # default is 'gmw', scales='log:minimal'
    cwt_out = sqz_cwt(data, **kwargs)
    return cwt_out[0]

def first_order_diff(data, fs):
    ''' backward pass '''
    dt = 1/fs
    diff_arr = np.zeros_like(data)
    for i, val in data[1::]:
        diff = (val - data[i-1])/dt
        diff_arr[i] = diff
    diff_arr[0] = diff_arr[1]
    return diff_arr

def second_order_diff(data, fs):
    ''' second order central difference taylor series approx '''
    dt = 1/fs
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    data_len = len(data)
    diff_arr = np.zeros_like(data)

    for i, val in enumerate(data):
        if i == 0: continue
        elif i == data_len-1: break

        x0 = data[i-1, :]
        x1 = data[i, :]
        x2 = data[i+1, :]

        diff = (x2 - 2*x1 + x0)/(dt**2)
        
        diff_arr[i, :] = diff

    diff_arr[0, :] = diff_arr[1, :]
    diff_arr[-1, :] = diff_arr[-2, :]
    return diff_arr

def window_filter(data, npts, window='triang'):
    if window=='triang':
        win = triang(npts)
    elif window=='hann':
        win = hann(npts)
    elif window=='exponential':
        win = exponential(npts)
    if data.ndim > 1:
        win = np.repeat(win, data.shape[1])\
                .reshape(npts, data.shape[1])
    filtered = convolve(data, win, mode='same')/sum(win)
    return filtered

def n_samples_over_thold(data, thold):
    mask = np.abs(data) > thold
    if mask.any(): return mask.sum(axis=0).max()
    return 0

def n_samples_under_thold(data, thold):
    mask = np.abs(data) < thold
    if mask.any(): return mask.sum(axis=0).max()
    return 0

def cubic_interp(data, fs, fs_new):
    t_total = len(data)//fs
    x = np.linspace(0, t_total, len(data))
    y = data
    cs = CubicSpline(x, y)
    factor = fs_new/fs
    xs = np.linspace(0, t_total, int(len(data)*factor))
    ys = cs(xs)

    return ys

def linear_interp(data, fs, fs_new):
    ''' data is 1d vector '''
    t_total = len(data)//fs
    x = np.linspace(0, t_total, len(data))
    y = data
    factor = fs_new/fs
    xs = np.linspace(0, t_total, int(len(data)*factor))
    ys = np.interp(xs, x, y)

    return ys

def npads_frequency_resolution(data_len, fr=0.02, fs=256):
    nbins = fs//fr
    npads = nbins*2 - data_len
    if npads<0: npads = 0
    return int(npads)

def get_peaks(x, distance=320, height=0.01, **kwargs):
    return find_peaks(x, distance=distance, height=height, **kwargs)

def get_noisy_sine(t=1000, n_ch=7, f=0.75, fs=60):
    t = np.linspace(0, t, t*fs)

    base_wave = np.tile(
        np.sin(2*np.pi*t*(0.005/fs)) + 1.6*np.sin(2*np.pi*t*(f/fs)),
        (3, 1)).T
    noise = np.random.normal(-0.8, 0.8, size=base_wave.shape)

    noisy_sine = np.tile(base_wave + noise, (n_ch, 1))

    return noisy_sine

def attenuate_edges(signal, nsamples):
    start = nsamples
    end = int(len(signal)-nsamples)
    ramp = (1-np.cos(np.pi*(np.arange(start)/start)))/2
    edge_attenuator = np.ones(len(signal))
    edge_attenuator[0:start] = ramp
    edge_attenuator[end:len(signal)] = np.flip(ramp)
    if len(signal.shape) > 1:
        e_arr = np.array([edge_attenuator])
        return(signal*e_arr.T)
    return(signal*edge_attenuator)

def marker_quality_processing(marker_quality):
    new_arr = []
    for i in range(marker_quality.shape[1]):
        new_arr.append(
            linear_interp(
                marker_quality[:,i], MARKER_FS, FS_RESAMPLE
            )
        )
    return np.array(new_arr).T

def std_scaler(data, **kwargs):
    mu = np.mean(data, **kwargs)
    sd = np.std(data, **kwargs)
    try:
        std = (data-mu)/sd
    except ValueError:
        std = (data-mu.reshape(-1,1)/sd.reshape(-1,1))
    return np.nan_to_num(std, nan=0)

def acc_signal_processing(data, fs:int=100):
    ''' Run harness accel through the following steps:
          * Second order taylor estimation
          * ICA transform
          * Cubic interpolation (~256Hz)
          * Standard scaling
          * Moving average with max resp rate size
          * Fourth order bandpass filter between min and max resp rates
          * Window filter at 2 seconds '''
    thold = 2
    triang_window = 2

    accel = cubic_interp(data, fs, FS_RESAMPLE)

    # ma = accel - movingaverage(accel, 3)

    data_sd = std_scaler(accel, axis=0)
    # data_sd = attenuate_edges(data_sd, 20)

    # mask = np.abs(data_sd) > thold
    # data_sd[mask] = thold*np.sign(data_sd[mask])

    # movmean
    # window_size=int((60/MAX_RESP_RATE)*FS_RESAMPLE)
    # ma = movingaverage(data_sd, window_size)

    ''' This step introduces those large steps we see in the windows,
    perhaps due to high order (4):
        * dropping still see effects but less 
        * increasing to larger order (6) reduces step occurrence '''
    bp = butter_bandpass_filter(data_sd,
                                MIN_RESP_RATE/60,
                                MAX_RESP_RATE/60, fs=FS_RESAMPLE, order=4)
    
    # Hard threshold for 1-SD
    # bp_mask = np.abs(bp) < 1
    # bp[bp_mask] = 0

    smth = np.zeros_like(bp)
    smth = window_filter(bp, triang_window*FS_RESAMPLE,
                         window='triang')

    return smth

def imu_signal_processing(data, fs:int=IMU_FS):
    sd = StandardScaler().fit_transform(data)
    bp = butter_bandpass_filter(sd,
                                3/60,
                                70/60, fs=fs, order=2)
    ma = movingaverage(bp, 8, axis=0)
    return ma

def roddiger_sp(data=None, fs:int=IMU_FS, is_marker:bool=False):
    ''' Run markers through the following steps:
          * Second order taylor estimation
          * 3 sample mov mean subtraction
          * 2s mov mena
          * Cubic interpolation (~256Hz)
          * Standard scaling
          * Fourth order bandpass filter between 0.1 and 0.5Hz
          * Window filter at 2 seconds '''
    triang_window = 2
    thold = 2

    # get accelaration profile
    if is_marker:
        mkr_shape = data.shape
        accel = second_order_diff(data, fs)
        accel = data
    else:
        accel = data.astype(float)

    # movmean
    ma = accel - movingaverage(accel, 3, axis=0)
    if is_marker and len(accel.shape) > 2:
        for i in range(accel.shape[1]):
            for j in range(3):
                ma[:,i,j] = movingaverage(ma[:,i,j], 2*fs, axis=0)
    else:
        ma = movingaverage(ma, 2*fs, axis=0)

    # cubic interp
    accel = cubic_interp(accel, fs, FS_RESAMPLE)

    ma = cubic_interp(ma, fs, FS_RESAMPLE)
    data_sd = std_scaler(ma, axis=0)
    mask = np.abs(data_sd) > thold
    data_sd[mask] = thold*np.sign(data_sd[mask])

    bp = butter_bandpass_filter(data_sd, 0.1, 0.5, fs=FS_RESAMPLE, order=4)
    smth = np.zeros_like(bp)
    if len(accel.shape) > 2:
        for i in range(accel.shape[1]):
            smth[:,i,:] = window_filter(bp[:,i,:].squeeze(),
                                        triang_window*FS_RESAMPLE,
                                        window='triang')
    else:
        smth = window_filter(bp, triang_window*FS_RESAMPLE,
                             window='triang')

    return accel, smth

def hernandez_sp(data=None, fs:int=IMU_FS, is_marker:bool=False):
    ''' Run markers through the following steps:
          * Second order taylor estimation
          * Cubic interpolation (~256Hz)
          * Max Resp Rate mov mean
          * Fourth order bandpass filter between 0.13 and 0.75Hz
          * Standard scaling w/ hard 2SD thold per window '''
    thold = 2
    # max_br = MAX_RESP_RATE #bpm
    max_br = 45 #bpm

    if is_marker:
        accel = second_order_diff(data, fs)
    else:
        accel = data.astype(float)

    # cubic interp
    accel = cubic_interp(accel, fs, FS_RESAMPLE)

    data_sd = std_scaler(accel, axis=0)
    mask = np.abs(data_sd) > thold
    data_sd[mask] = thold*np.sign(data_sd[mask])

    # movmean
    ma = movingaverage(data_sd, int(FS_RESAMPLE*60/max_br), axis=0)

    bp = butter_bandpass_filter(ma, MIN_RESP_RATE/60, max_br/60, fs=FS_RESAMPLE, order=4)

    return accel, bp

def pressure_signal_processing(pressure_data, fs=BR_FS):
    ''' Run pressure signal through the following steps:
          * Moving average with 8 samples
          * Standard scaler
          * Cubic interpolations to ~256 Hz
          * Second order bandpass filter between min and max resp rates'''
    # Normalize
    flag = False
    if pressure_data.ndim == 1:
        pressure_data = pressure_data.reshape(-1, 1)
        flag = True
    data_sd = StandardScaler().fit_transform(pressure_data)

    data_ma = movingaverage(data_sd, 16)

    # bandpass filter the lbls
    bp_data = butter_bandpass_filter(data_ma, 4/60, 70/60, fs=fs, order=2)

    if flag:
        bp_data = np.squeeze(bp_data)

    return bp_data

def ecg_signal_processing(ecg_data, fs=250):
    '''Run ECG signal through standard preprocessing steps:
          * Standard scaling
          * 50 Hz notch filter for line-noise suppression
          * Bandpass filter at 0.5-100 Hz (4th order Butterworth)'''
    flag = False
    if ecg_data.ndim == 1:
        ecg_data = ecg_data.reshape(-1, 1)
        flag = True

    data_sd = StandardScaler().fit_transform(ecg_data.astype(float))
    b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=fs)
    ecg_notch = filtfilt(b_notch, a_notch, data_sd, axis=0)

    nyq = 0.5 * fs
    highcut = min(100.0, 0.95 * nyq)
    bp_data = butter_bandpass_filter(ecg_notch, 0.5, highcut, fs=fs, order=4)

    if flag:
        bp_data = np.squeeze(bp_data)

    return bp_data

def vectorized_slide_win(array, max_time, sub_window_size=3,
                         stride_size=3):
    ''' Sliding window with right-side zero padding '''
    # https://towardsdatascience.com/
    # fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    # array: vec to slide
    # max_time: len of array
    # sub win size: window length
    # stride_size: step stride len

    # Create a rightmost vector as [0, V, 2V, ...].
    sub_windows = (
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time, step=stride_size), 0).T
    )
    pad_len = int(np.max(sub_windows) - max_time + 1)

    # pad right side data array with zero
    arr = np.pad(array,(0, pad_len), 'constant',
                 constant_values=0)

    return arr[sub_windows]

def generate_noisy_sine_windows(sig=None,
                                window_size=WINDOW_SIZE*FS_RESAMPLE,
                                window_shift=WINDOW_SHIFT*FS_RESAMPLE,
                                **kwargs):
    ''' Separates some noisy sine waves into sliding windows.
    Applies artefact rejection for windows and matches times'''
    if sig is None:
        sig = get_noisy_sine(**kwargs)

    if method == 'roddiger':
        noise, noise_smth = roddiger_sp(data=sig)
    elif method == 'hernandez':
        noise, noise_smth = hernandez_sp(data=sig)

    inds = np.arange(0, len(sig))

    vec_win = vectorized_slide_win(
        inds, len(inds),
        sub_window_size=window_size,
        stride_size=window_shift)

    for i, vec_inds in enumerate(vec_win):
        noise_out = noise_smth[vec_inds, :]
        # If we go over the full window, throw away last bits of data
        if vec_inds[-1] == 0:
            break

        if method == 'hernandez':
            thold = 2
            data_sd = std_scaler(noise_out, axis=0)
            mask = np.abs(data_sd) > thold
            data_sd[mask] = thold*np.sign(data_sd[mask])
            noise_out = data_sd

        yield noise_out

def check_tholds(accel, qual, window_size=WINDOW_SIZE*FS_RESAMPLE):
    hernandez_thold = 2
    mag_ns = n_samples_over_thold(accel, ACC_THOLD)
    mag_thold = mag_ns > WIN_THOLD*window_size
    qual_thold = n_samples_under_thold(qual, MQA_THOLD) > \
            WIN_THOLD*window_size

    if method != 'hernandez':
        if mag_thold or qual_thold:
            return False
    elif qual_thold:
        return False

    return True

def reject_artefact(data_win, thold=2, percent=3):
    data = data_win.flatten()
    N = len(data)
    chk = np.sum(data > thold, axis=0) > N*(percent/100)
    if np.any(chk): return True
    else: return False

def infer_frequency(time, thold=60):
    diff = time[1:] - time[:-1]
    mask = np.abs(diff) < thold
    fs = 1/np.mean(diff[mask])
    return fs


import pandas as pd
from scipy.signal import welch
from scipy.stats import linregress 
"""
See: https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_extraction/feature_calculators.py
'acc_x__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2',
'acc_x__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2',
'acc_x__fft_coefficient__attr_"real"__coeff_10',
'acc_x__quantile__q_0.2', 'acc_x__quantile__q_0.3',
'acc_y__fft_coefficient__attr_"real"__coeff_10',
'acc_z__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"max"',
'acc_z__c3__lag_1', 'acc_z__c3__lag_2', 'acc_z__c3__lag_3',
'acc_z__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
'acc_z__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
'acc_z__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
'acc_z__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2',
'acc_z__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4',
'acc_z__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
'acc_z__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
'acc_z__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2',
'acc_z__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
'acc_z__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
'acc_z__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
'acc_z__maximum', 'acc_z__skewness',
'acc_z__spkt_welch_density__coeff_5',
'gyro_x__fft_coefficient__attr_"real"__coeff_70',
'gyro_y__fft_coefficient__attr_"abs"__coeff_13',
'gyro_y__large_standard_deviation__r_0.15000000000000002',
'gyro_z__benford_correlation',
'gyro_z__fft_coefficient__attr_"abs"__coeff_13',
'gyro_z__fft_coefficient__attr_"imag"__coeff_10'],
"""

def change_quantiles(x, f_agg='var', isabs=False, qh=0.4, ql=0.2):
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

    Think about selecting a corridor on the
    y-Axis and only calculating the mean of the absolute change of the time series inside this corridor.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param ql: the lower quantile of the corridor
    :type ql: float
    :param qh: the higher quantile of the corridor
    :type qh: float
    :param isabs: should the absolute differences be taken?
    :type isabs: bool
    :param f_agg: the aggregator function that is applied to the differences in the bin
    :type f_agg: str, name of a numpy function (e.g. mean, var, std, median)

    :return: the value of this feature
    :return type: float
    """
    index = f'change_quantiles__f_agg_{f_agg}__isabs_{isabs}__qh_{qh}__ql_{ql}'
    if ql >= qh:
        return index, 0

    div = np.diff(x)
    if isabs:
        div = np.abs(div)
    # All values that originate from the corridor between the quantiles ql and qh will have the category 0,
    # other will be np.NaN
    try:
        bin_cat = pd.qcut(x, [ql, qh], labels=False)
        bin_cat_0 = bin_cat == 0
    except ValueError:  # Occurs when ql are qh effectively equal, e.g. x is not long enough or is too categorical
        return index, 0
    # We only count changes that start and end inside the corridor
    ind = (bin_cat_0 & _roll(bin_cat_0, 1))[1:]
    if np.sum(ind) == 0:
        return index, 0
    else:
        ind_inside_corridor = np.where(ind == 1)
        aggregator = getattr(np, f_agg)
        return  index, aggregator(div[ind_inside_corridor])

def fft_coefficient(x, agg='real', coeff=10):
    """
    Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
    fourier transformation algorithm

    .. math::
        A_k =  \\sum_{m=0}^{n-1} a_m \\exp \\left \\{ -2 \\pi i \\frac{m k}{n} \\right \\}, \\qquad k = 0,
        \\ldots , n-1.

    The resulting coefficients will be complex, this feature calculator can return the real part (attr=="real"),
    the imaginary part (attr=="imag), the absolute value (attr=""abs) and the angle in degrees (attr=="angle).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param agg:  ["real", "imag", "abs", "angle"]
    :type param: str
    :param coeff: with x int and x >= 0
    :type coeff: int
    :return: the different feature values
    :return type: pandas.Series
    """
    index = f'fft_coefficient__f_agg_{agg}__coeff_{coeff}'

    fft = np.fft.rfft(x)

    def complex_agg(x, agg):
        if agg == "real":
            return x.real
        elif agg == "imag":
            return x.imag
        elif agg == "abs":
            return np.abs(x)
        elif agg == "angle":
            return np.angle(x, deg=True)

    res = complex_agg(fft[coeff], agg) if coeff < len(fft) else np.NaN
    index = 'attr_"{}"__coeff_{}'.format(agg, coeff)
    return index, res

def quantile(x, q):
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param q: the quantile to calculate
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    index = f'quantile__q_{q}'
    if len(x) == 0:
        return index, np.NaN
    return index, np.quantile(x, q)

def _roll(a, shift):
    """
    Roll 1D array elements. Improves the performance of numpy.roll() by reducing the overhead introduced from the
    flexibility of the numpy.roll() method such as the support for rolling over multiple dimensions.

    Elements that roll beyond the last position are re-introduced at the beginning. Similarly, elements that roll
    back beyond the first position are re-introduced at the end (with negative shift).

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=2)
    >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=-2)
    >>> array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=12)
    >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    Benchmark
    ---------
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> %timeit _roll(x, shift=2)
    >>> 1.89 µs ± 341 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> %timeit np.roll(x, shift=2)
    >>> 11.4 µs ± 776 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    :param a: the input array
    :type a: array_like
    :param shift: the number of places by which elements are shifted
    :type shift: int

    :return: shifted array with the same shape as a
    :return type: ndarray
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])

def _aggregate_on_chunks(x, f_agg, chunk_len):
    """
    Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len

    :param x: the time series to calculate the aggregation of
    :type x: numpy.ndarray
    :param f_agg: The name of the aggregation function that should be an attribute of the pandas.Series
    :type f_agg: str
    :param chunk_len: The size of the chunks where to aggregate the time series
    :type chunk_len: int
    :return: A list of the aggregation function over the chunks
    :return type: list
    """
    return [
        getattr(x[i * chunk_len : (i + 1) * chunk_len], f_agg)()
        for i in range(int(np.ceil(len(x) / chunk_len)))
    ]

def agg_linear_trend(x, attr='stderr', chunk_len=50, f_agg='max'):
    """
    Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
    the sequence from 0 up to the number of chunks minus one.

    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

    The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
    "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

    The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

    Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param attr: 'pvalue', 'rvalue', 'intercept', 'slope', 'stderr'
    :type attr: str
    :param chunk_len: chunk size for time series
    :type chunk_len: int
    :param chunk_len: 'pvalue', 'rvalue', 'intercept', 'slope', 'stderr'
    :type f_agg: aggregate function 'max', 'min', 'mean', 'median'
    :type f_agg: str
    :return: the different feature values
    :return type: pandas.Series
    """
    # todo: we could use the index of the DataFrame here

    calculated_agg = defaultdict(dict)
    res_data = []
    res_index = []

    if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
        if chunk_len >= len(x):
            calculated_agg[f_agg][chunk_len] = np.NaN
        else:
            aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
            lin_reg_result = linregress(
                range(len(aggregate_result)), aggregate_result
            )
            calculated_agg[f_agg][chunk_len] = lin_reg_result

    if chunk_len >= len(x):
        res_data = np.NaN
    else:
        res_data = getattr(calculated_agg[f_agg][chunk_len], attr)

    res_index = 'attr_"{}"__chunk_len_{}__f_agg_"{}"'\
            .format(attr, chunk_len, f_agg)

    return res_index, res_data

def c3(x, lag):
    """
    Uses c3 statistics to measure non linearity in the time series

    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag} \\cdot x_{i + lag} \\cdot x_{i}

    which is

    .. math::

        \\mathbb{E}[L^2(X) \\cdot L(X) \\cdot X]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a measure of
    non linearity in the time series.

    .. rubric:: References

    |  [1] Schreiber, T. and Schmitz, A. (1997).
    |  Discrimination power of measures for nonlinearity in a time series
    |  PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """

    index = f"c3__lag_{lag}"
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    n = x.size
    if 2 * lag >= n:
        return index, 0
    else:
        c3 = np.mean(
            (_roll(x, 2 * -lag) * _roll(x, -lag) * x)[0 : (n - 2 * lag)]
        )
        return index, c3

def skewness(x):
    """
    Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return 'skewness', pd.Series.skew(x)

def spkt_welch_density(x, param):
    """
    This feature calculator estimates the cross power spectral density of the time series x at different frequencies.
    To do so, the time series is first shifted from the time domain to the frequency domain.

    The feature calculators returns the power spectrum of the different frequencies.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x} with x int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series

    USAGE:
        spkt_welch_density(x, {'coeff': 5})
    """
    if not isinstance(param, list):
        param = [param]

    freq, pxx = welch(x, nperseg=min(len(x), 256))
    coeff = [config["coeff"] for config in param]
    indices = ["spkt_welch_density__coeff_{}".format(i) for i in coeff]

    if len(pxx) <= np.max(
        coeff
    ):  # There are fewer data points in the time series than requested coefficients

        # filter coefficients that are not contained in pxx
        reduced_coeff = [coefficient for coefficient in coeff if len(pxx) > coefficient]
        not_calculated_coefficients = [
            coefficient for coefficient in coeff if coefficient not in reduced_coeff
        ]

        # Fill up the rest of the requested coefficients with np.NaNs
        fill_coeffs = list(pxx[reduced_coeff])\
                + [np.NaN] * len(not_calculated_coefficients)
        return indices[0], fill_coeffs[0]
    else:
        return indices[0], pxx[coeff][0]

def benford_correlation(x):
    """
     Useful for anomaly detection applications [1][2]. Returns the correlation from first digit distribution when
     compared to the Newcomb-Benford's Law distribution [3][4].

     .. math::

         P(d)=\\log_{10}\\left(1+\\frac{1}{d}\\right)

     where :math:`P(d)` is the Newcomb-Benford distribution for :math:`d` that is the leading digit of the number
     {1, 2, 3, 4, 5, 6, 7, 8, 9}.

     .. rubric:: References

     |  [1] A Statistical Derivation of the Significant-Digit Law, Theodore P. Hill, Statistical Science, 1995
     |  [2] The significant-digit phenomenon, Theodore P. Hill, The American Mathematical Monthly, 1995
     |  [3] The law of anomalous numbers, Frank Benford, Proceedings of the American philosophical society, 1938
     |  [4] Note on the frequency of use of the different digits in natural numbers, Simon Newcomb, American Journal of
     |  mathematics, 1881

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)

    # retrieve first digit from data
    x = np.array(
        [int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(x))]
    )

    # benford distribution
    benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])

    data_distribution = np.array([(x == n).mean() for n in range(1, 10)])

    # np.corrcoef outputs the normalized covariance (correlation) between benford_distribution and data_distribution.
    # In this case returns a 2x2 matrix, the  [0, 1] and [1, 1] are the values between the two arrays
    benford_correlation = np.corrcoef(
        benford_distribution, data_distribution)[0, 1]
    return 'benford_correlation', benford_correlation

def large_standard_deviation(x, r:float=0.15):
    """
    Does time series have *large* standard deviation?

    Boolean variable denoting if the standard dev of x is higher than 'r' times the range = difference between max and
    min of x. Hence it checks if

    .. math::

        std(x) > r * (max(X)-min(X))

    According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    index = f'large_standard_deviation__r_{r}'
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return index, np.std(x) > (r * (np.max(x) - np.min(x)))

def get_max_freq(win, fs=120):
    xf, yf = do_pad_fft(win, fs=fs)
    max_freq = xf[yf.argmax()]
    return max_freq

'''
def get_video_features(img, use_shi=False, max_corners=100,
                       exclude_border=5):
    # cv2.imshow('roi',roi)
    feature_params = dict(maxCorners = max_corners,
                          qualityLevel = 0.2,
                          minDistance = 50,
                          blockSize = 3)
    if use_shi:
        corner_response = cv2.goodFeaturesToTrack(img, **feature_params)
        # corner_response = corner_shi_tomasi(img)
    else:
        # corner_response = corner_harris(img)
        corner_response = cv2.goodFeaturesToTrack(img, useHarrisDetector=True,
                                                  **feature_params)
    # coordinates = peak_local_max(corner_response,
    #                              num_peaks=num_peaks,
    #                              exclude_border=exclude_border)
    # x = coordinates[:, 1]
    # y = coordinates[:, 0]

    # x = corner_response[:, :, 0].astype(int)
    # y = corner_response[:, :, 1].astype(int)
    return corner_response
'''

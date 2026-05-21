import numpy as np
from random import shuffle
import torch


def DataTransform(sample, config, device):

    if len(sample.shape) < 3:
        sample = torch.unsqueeze(sample, 0)
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio,
                       device=device)
    strong_aug = jitter(
        permutation(
            sample, max_segments=config.augmentation.max_seg, device=device
        ),
        config.augmentation.jitter_ratio, device=device
    )

    return weak_aug, strong_aug

def dom_shuffle(x, rate=4, dim=-1):
    # https://arxiv.org/html/2405.16456v1#S3
    # x: (batch_size, channel,  timestep)
    x_f = torch.fft.rfft(x, dim=dim)
    magnitude = abs(x_f)
    topk_indices = torch.argsort(
        magnitude, dim=dim, descending=True)[..., 1:int(rate+1)]
    #minor_indices = torch.argsort(magnitude, dim=1, descending=True)[:, 10:]

    # Shuffle top frequency bins per (batch, channel)
    B, C, K = topk_indices.shape
    perm = torch.stack([torch.randperm(K, device=x.device) for _ in range(B*C)])
    perm = perm.view(B, C, K)

    shuffled_indices = torch.gather(topk_indices, 2, perm)

    # Apply the shuffle in the frequency domain
    x_f_shuf = x_f.clone()
    gather_mask = topk_indices

    # Expand masks to complex-valued indices
    for b in range(B):
        for c in range(C):
            x_f_shuf[b, c, gather_mask[b, c]] = x_f[b, c, shuffled_indices[b, c]]
    x = torch.fft.irfft(x_f_shuf, dim=dim)
    return x

def jitter(x, sigma=0.8, device=None):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1, device=None):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        # ai.append(torch.mul(xi, factor[:, :])[:, np.newaxis, :])
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    # return torch.cat((ai), axis=1)
    return np.concatenate((ai), axis=1)

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1,
                                                replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            # list-wise shuffle rather than np permutation
            shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return ret

def minmax(x):
    x_min, _ = torch.min(x, dim=-1, keepdims=True)
    x_max, _ = torch.max(x, dim=-1, keepdims=True)
    x_range = x_max - x_min
    return (x - x_min)/x_range

def random_phase_augment(x, max_phase=np.pi, dim=-1):
    """
    Random phase perturbation in the frequency domain.

    x:  (B, C, T) real-valued time series
    Returns: (B, C, T) real-valued time series with same shape
    """
    # ensure tensor
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    # rFFT along time dimension
    X = torch.fft.rfft(x, dim=dim)  # (B, C, F_rfft)

    # random phase in [-max_phase, max_phase]
    phase = (torch.rand_like(X.real) * 2.0 - 1.0) * max_phase  # real tensor
    phase_shift = torch.exp(1j * phase)  # complex rotation

    X_shifted = X * phase_shift

    # inverse rFFT back to time domain
    T = x.shape[dim]
    x_shifted = torch.fft.irfft(X_shifted, n=T, dim=dim)

    return x_shifted


def random_freq_shift(x, fs, max_shift_hz=0.5, dim=-1):
    """
    Small frequency shift by rolling the spectrum a few bins in rFFT domain.

    x:  (B, C, T) real-valued time series
    fs: sampling frequency (e.g. IMU_FS = 120)
    max_shift_hz: maximum absolute frequency shift (Hz)
    Returns: (B, C, T) real-valued time series with same shape
    """
    # ensure tensor
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    B, C, T = x.shape
    if T < 2:
        return x

    # rFFT along time dimension
    X = torch.fft.rfft(x, dim=dim)  # (B, C, F_rfft)
    F = X.shape[-1]

    # frequency resolution
    df = fs / float(T)
    max_bins = int(max_shift_hz / df)

    # if shift < 1 bin, do nothing
    if max_bins < 1:
        return x

    X_shifted = X.clone()

    # per (batch, channel) random integer shift in [-max_bins, max_bins]
    for b in range(B):
        for c in range(C):
            k = int(torch.randint(-max_bins, max_bins + 1, (1,)))
            if k != 0:
                X_shifted[b, c] = torch.roll(X[b, c], shifts=k, dims=-1)

    # back to time domain
    x_shifted = torch.fft.irfft(X_shifted, n=T, dim=dim)

    return x_shifted


def spectral_augment(
    x,
    fs,
    max_phase=np.pi / 2,
    max_shift_hz=0.5,
    dim=-1,
    p_phase=0.5,
    p_shift=0.5,
):
    """
    Convenience wrapper: optionally apply random phase + small frequency shift.

    x:  (B, C, T)
    fs: sampling frequency (e.g. IMU_FS = 120)
    Returns: augmented x with same shape and dtype.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    # random phase
    if torch.rand(1).item() < p_phase:
        x = random_phase_augment(x, max_phase=max_phase, dim=dim)

    # small frequency shift
    if torch.rand(1).item() < p_shift:
        x = random_freq_shift(x, fs=fs, max_shift_hz=max_shift_hz, dim=dim)

    return x

'''
def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1,
                                                replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            # list-wise shuffle rather than np permutation
            shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)
'''

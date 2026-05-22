#!/usr/bin/env python3
"""
MA-RD-PAPA: Marker-aware respiratory-dynamics PAPA for IMU mental workload.

This script implements the marker/OptiTrack version of the model discussed in
this thread.  It is deliberately a wrapper around the existing PAPA/PAPA-dyn
training path.  The pressure-STFT/RR backbone remains the physiological anchor:

    IMU -> reconstructed pressure STFT + RR head -> respiratory dynamics

Marker/OptiTrack is NOT used as a deployable input and is NOT treated as a
mental-workload classifier.  It is used as a privileged motion/posture teacher:

    source marker windows -> motion/posture/artefact features
    IMU + reconstructed respiratory dynamics -> inferred motion/posture state
    inferred motion state -> reliability / preference gate for respiratory views

At test/deployment time the main prediction uses IMU only:

    IMU -> reconstructed respiration + IMU activity + inferred marker-like
           motion/posture proxy -> expert reliability gate -> MWL sequence

What this script adds over PAPA-dyn
-----------------------------------
1. Loads optional marker windows exposed by dataloader.make_dataset(...,
   include_marker=True).  Missing marker is allowed by default: subjects without
   marker are retained, but marker-teacher rows are masked.
2. Extracts fixed marker motion/posture features from marker_filt windows:
   translation/rotation energy, velocity, jerk, drift, stationarity, spectral
   entropy, dominant motion frequency, and validity.
3. Trains an IMU/RR/activity -> marker-motion proxy on source subjects only.
   The held-out target marker is never used for the main prediction.
4. Builds simple respiratory/activity experts and a source-inner-LOSO reliability
   gate.  The gate sees respiratory quality, IMU activity, inferred marker
   motion, and marker-proxy uncertainty.
5. Optionally enables marker-proxy-conditioned experts and marker-oracle audits.
   Oracle rows are diagnostic only and should not be treated as deployable.
6. CUDA-heavy paths are available for the marker proxy, expert classifiers, and
   preference gate via torch ridge/logistic-regression backends, so the expensive
   source-inner-LOSO audit can use GPU compute rather than only scikit-learn CPU.

Typical command
---------------
python vit_pressure_crossmodal_marker_rd_papa.py \
  --data-str imu_filt \
  --data-group mr \
  --epochs 20 \
  --batch-size 64 \
  --embed-labels L0,L2,L3 \
  --eval-marker-rd-papa \
  --out-dir runs/marker_rd_papa/$(date -u +%Y%m%dT%H%M%SZ)

Outputs per held-out subject
----------------------------
<out>/<subject>/marker_rd_papa/
  marker_rd_predictions.csv
  marker_rd_preference_trace.csv
  marker_rd_subject_preference_summary.csv
  marker_rd_motion_summary.csv
  marker_rd_feature_meta.json
  marker_rd_marker_oracle_predictions.csv     # diagnostic only, when possible

Top-level summary files emitted by the core runner
--------------------------------------------------
  summary.csv
  marker_rd_papa_summary.csv
  marker_rd_papa_preference_summary.csv
  marker_rd_papa_motion_summary.csv
  marker_rd_papa_oracle_summary.csv
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import warnings
import sys
import types
from collections import OrderedDict

# Debug/raw-data optional dependencies are imported by project utilities even
# during argument parsing.  Provide no-op stubs so processed-data experiments do
# not fail in lightweight environments that lack these debug/raw import packages.
if "ipdb" not in sys.modules:
    sys.modules["ipdb"] = types.SimpleNamespace(set_trace=lambda *args, **kwargs: None)
if "pyxdf" not in sys.modules:
    sys.modules["pyxdf"] = types.SimpleNamespace(load_xdf=lambda *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError("pyxdf is required for raw XDF loading")))
if "mat73" not in sys.modules:
    sys.modules["mat73"] = types.SimpleNamespace(loadmat=lambda *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError("mat73 is required for MATLAB v7.3 loading")))
if "pywt" not in sys.modules:
    sys.modules["pywt"] = types.SimpleNamespace(cwt=lambda *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError("pywt is required for wavelet processing")))
if "tsfresh" not in sys.modules:
    tsfresh_mod = types.ModuleType("tsfresh")
    fs_mod = types.ModuleType("tsfresh.feature_selection")
    rel_mod = types.ModuleType("tsfresh.feature_selection.relevance")
    rel_mod.calculate_relevance_table = lambda *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError("tsfresh is required for raw-feature relevance"))
    util_mod = types.ModuleType("tsfresh.utilities")
    sm_mod = types.ModuleType("tsfresh.utilities.string_manipulation")
    sm_mod.get_config_from_string = lambda *args, **kwargs: None
    sys.modules["tsfresh"] = tsfresh_mod
    sys.modules["tsfresh.feature_selection"] = fs_mod
    sys.modules["tsfresh.feature_selection.relevance"] = rel_mod
    sys.modules["tsfresh.utilities"] = util_mod
    sys.modules["tsfresh.utilities.string_manipulation"] = sm_mod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import vit_pressure_crossmodal_papa_dyn as dyn
from config import BR_FS, IMU_FS, MARKER_FS, SBJ_PROCESSED_DIR
from dataloader import load_data, make_dataset

papa = dyn.papa
core = dyn.core


# -----------------------------------------------------------------------------
# Labels and robust data parsing
# -----------------------------------------------------------------------------
VALID_MWL_LABELS = ("M", "R", "L0", "L1", "L2", "L3")
LABEL_TO_ID = {"M": 0, "R": 1, "L0": 2, "L1": 3, "L2": 4, "L3": 5}


def parse_mwl_labels(x: str | Sequence[str]) -> List[str]:
    if isinstance(x, (list, tuple, np.ndarray)):
        labels = [str(v).strip().upper() for v in x if str(v).strip()]
    else:
        labels = [v.strip().upper() for v in str(x).split(",") if v.strip()]
    bad = [v for v in labels if v not in VALID_MWL_LABELS]
    if bad:
        raise ValueError(f"Unsupported labels {bad}; expected one of {VALID_MWL_LABELS}")
    if not labels:
        raise ValueError("No labels requested. Use --embed-labels L0,L2,L3 etc.")
    return labels


def label_id(label: str) -> int:
    return LABEL_TO_ID[str(label).strip().upper()]


def _is_level_label(label: str) -> bool:
    return str(label).strip().upper() in {"L0", "L1", "L2", "L3"}


def grouped_labels(labels: Sequence[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {"mr": [], "levels": []}
    for lbl in labels:
        grouped["levels" if _is_level_label(lbl) else "mr"].append(str(lbl).strip().upper())
    return {k: v for k, v in grouped.items() if v}


def _subject_from_dict(d: Dict[str, Any], fallback: str = "") -> str:
    for key in ("subject", "subject_id", "sbj", "sid"):
        if key in d and str(d[key]).strip():
            return str(d[key]).strip()
    return str(fallback).strip()


def _filter_subject_dict_by_labels(d: Dict[str, Any], keep_values: set[str]) -> Optional[Dict[str, Any]]:
    conds = np.asarray(d.get("conds"))
    if conds.size == 0:
        return d
    mask = np.asarray([str(c).strip().upper() in keep_values for c in conds], dtype=bool)
    if not mask.any():
        return None
    out = dict(d)
    n = len(conds)
    for k, v in d.items():
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.shape[:1] == (n,):
            out[k] = arr[mask]
    return out


def _as_tensor(x: Any, *, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if x is None:
        return None
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _looks_like_per_window_scalar(x: Any, batch_size: int) -> bool:
    try:
        arr = np.asarray(x)
    except Exception:
        return False
    if arr.shape[:1] != (batch_size,):
        return False
    return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)


def _split_extras_for_marker(extras: Sequence[Any], batch_size: int) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Return (tlx, ecg, marker) from extras after x/pressure/cond/br.

    Our local dataset emits only marker as an extra, but this parser tolerates
    future dataloader variants that append TLX and ECG before marker.
    """
    tlx = None
    ecg = None
    marker = None
    for item in extras:
        if item is None:
            continue
        if _looks_like_per_window_scalar(item, batch_size) and tlx is None:
            tlx = item
        elif marker is None:
            # Prefer treating the first non-scalar extra as marker in this script.
            marker = item
        elif ecg is None:
            ecg = item
    return tlx, ecg, marker


def unpack_batch_optional_marker(
    batch: Iterable[Any] | Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (imu, pressure, cond, br, tlx, marker)."""
    if isinstance(batch, dict):
        imu = batch.get("imu", batch.get("x", batch.get("past_values")))
        pressure = batch.get("pressure", batch.get("pss", batch.get("y")))
        cond = batch.get("cond", batch.get("condition", batch.get("conds")))
        br = batch.get("br", batch.get("rr", batch.get("breathing_rate")))
        tlx = batch.get("tlx", batch.get("nasa_tlx"))
        marker = batch.get("marker", batch.get("marker_filt", batch.get("mocap")))
    else:
        items = list(batch)
        if len(items) < 4:
            raise ValueError(f"Expected at least 4 batch items, got {len(items)}")
        imu, pressure, cond, br = items[:4]
        batch_size = int(np.asarray(cond).shape[0]) if not torch.is_tensor(cond) else int(cond.shape[0])
        tlx, _ecg, marker = _split_extras_for_marker(items[4:], batch_size=batch_size)

    imu_t = _as_tensor(imu, device=device, dtype=torch.float32)
    pressure_t = _as_tensor(pressure, device=device, dtype=torch.float32)
    if pressure_t is not None and pressure_t.ndim == 3 and pressure_t.size(-1) == 1:
        pressure_t = pressure_t.squeeze(-1)
    cond_t = _as_tensor(cond, device=device)
    br_t = _as_tensor(br, device=device, dtype=torch.float32)
    tlx_t = _as_tensor(tlx, device=device, dtype=torch.float32) if tlx is not None else None
    marker_t = _as_tensor(marker, device=device, dtype=torch.float32) if marker is not None else None
    if imu_t is None or pressure_t is None or cond_t is None or br_t is None:
        raise ValueError("Could not unpack imu/pressure/condition/br from batch.")
    return imu_t, pressure_t, cond_t, br_t, tlx_t, marker_t




def _mard_loader_kwargs(args) -> Dict[str, Any]:
    """Dataloader settings that keep GPU fed during feature extraction."""
    nw = max(0, int(getattr(args, "mard_num_workers", 0)))
    kwargs: Dict[str, Any] = {
        "num_workers": nw,
        "pin_memory": bool(getattr(args, "mard_pin_memory", True)) and torch.cuda.is_available(),
    }
    if nw > 0:
        kwargs["persistent_workers"] = bool(getattr(args, "mard_persistent_workers", True))
        kwargs["prefetch_factor"] = max(1, int(getattr(args, "mard_prefetch_factor", 2)))
    return kwargs

class WindowDatasetWithMarker(Dataset):
    """Local dataset with a fixed tuple order: x, pressure, cond, br, marker."""

    def __init__(self, x: np.ndarray, pressure: np.ndarray, cond: np.ndarray, br: np.ndarray, marker: Optional[np.ndarray]):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        if self.x.ndim == 3 and self.x.shape[1] < self.x.shape[2]:
            self.x = self.x.permute(0, 2, 1)
        self.pressure = torch.as_tensor(pressure, dtype=torch.float32)
        self.cond = torch.as_tensor(cond, dtype=torch.long)
        self.br = torch.as_tensor(br, dtype=torch.float32)
        n = int(self.x.shape[0])
        if marker is None:
            self.marker = torch.empty((n, 0), dtype=torch.float32)
        else:
            arr = np.asarray(marker)
            if arr.shape[:1] != (n,):
                raise ValueError(f"marker first dimension {arr.shape[:1]} does not match n={n}")
            self.marker = torch.as_tensor(arr, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.pressure[idx], self.cond[idx], self.br[idx], self.marker[idx]


def _call_make_dataset_marker(data_list: List[Dict[str, Any]], data_group: str, args) -> Tuple[Any, bool]:
    """Call make_dataset for one subject/group; fall back when marker is missing."""
    base_kwargs = dict(
        label_encoder_dir=args.data_dir,
        data_group=data_group,
        include_tlx=False,
        include_ecg=False,
        tlx_csv_path=getattr(args, "tlx_csv_path", None),
    )
    try:
        out = make_dataset(data_list, args.data_str, include_marker=True, **base_kwargs)
        return out, True
    except (TypeError, ValueError, KeyError) as exc:
        if not bool(getattr(args, "mard_allow_missing_marker", True)):
            raise
        msg = str(exc)
        if "include_marker" not in msg and "marker" not in msg.lower():
            # Older loader may reject include_marker.  Fall through to no-marker.
            pass
        out = make_dataset(data_list, args.data_str, include_marker=False, **base_kwargs)
        return out, False


def _parse_make_dataset_marker_output(out: Any, marker_included: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if isinstance(out, dict):
        x = out.get("x", out.get("imu", out.get("past_values")))
        pressure = out.get("pressure", out.get("pss", out.get("y")))
        br = out.get("br", out.get("rr"))
        cond = out.get("cond", out.get("conds", out.get("condition")))
        marker = out.get("marker", out.get("marker_filt", None)) if marker_included else None
    else:
        items = list(out)
        if len(items) < 4:
            raise ValueError(f"make_dataset returned {len(items)} items; expected >=4")
        x, pressure, br, cond = items[:4]
        marker = items[4] if marker_included and len(items) > 4 else None
    if x is None or pressure is None or br is None or cond is None:
        raise ValueError("Could not parse x/pressure/br/cond from make_dataset output")
    return np.asarray(x), np.asarray(pressure), np.asarray(br), np.asarray(cond), None if marker is None else np.asarray(marker)


def _pad_or_trim_to_shape(arr: np.ndarray, shape_tail: Tuple[int, ...], fill: float = np.nan) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    out = np.full((arr.shape[0],) + tuple(shape_tail), fill, dtype=np.float32)
    slices = [slice(None)]
    for got, want in zip(arr.shape[1:], shape_tail):
        slices.append(slice(0, min(int(got), int(want))))
    out[tuple(slices)] = arr[tuple(slices)]
    return out


def _concat_marker_parts(parts: List[Optional[np.ndarray]], ns: List[int]) -> Optional[np.ndarray]:
    good = [np.asarray(p) for p in parts if p is not None]
    if not good:
        return None
    # Prefer preserving the time/channel layout.  Pad/truncate if needed.
    max_ndim = max(p.ndim for p in good)
    good_norm = []
    for p in good:
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim < max_ndim:
            # Treat flattened marker as a one-channel sequence.
            arr = arr.reshape(arr.shape[0], -1, 1)
        good_norm.append(arr)
    shape_tail = tuple(max(p.shape[i] for p in good_norm) for i in range(1, max_ndim))
    out_parts = []
    gi = 0
    for p, n in zip(parts, ns):
        if p is None:
            out_parts.append(np.full((int(n),) + shape_tail, np.nan, dtype=np.float32))
        else:
            arr = np.asarray(p, dtype=np.float32)
            if arr.ndim < max_ndim:
                arr = arr.reshape(arr.shape[0], -1, 1)
            out_parts.append(_pad_or_trim_to_shape(arr, shape_tail, fill=np.nan))
            gi += 1
    return np.concatenate(out_parts, axis=0)


def build_marker_loaders_with_subject_ids(subject: str, subjects: List[str], args) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    keep_labels = parse_mwl_labels(getattr(args, "embed_labels", []))
    keep_ids = np.asarray([label_id(lbl) for lbl in keep_labels], dtype=int)
    groups = grouped_labels(keep_labels)

    def append_subject(xs, ps, brs, conds, markers, ids, raw: Dict[str, Any], group: str, sid: str) -> None:
        out, marker_included = _call_make_dataset_marker([raw], group, args)
        x, pressure, br, cond, marker = _parse_make_dataset_marker_output(out, marker_included)
        cond = np.asarray(cond, dtype=int).reshape(-1)
        if group == "levels" and cond.size and int(np.nanmax(cond)) <= 3:
            cond = cond + 2
        mask = np.isin(cond, keep_ids)
        if not mask.any():
            return
        xs.append(x[mask])
        ps.append(pressure[mask])
        brs.append(br[mask])
        conds.append(cond[mask])
        markers.append(None if marker is None else marker[mask])
        ids.extend([_subject_from_dict(raw, sid)] * int(mask.sum()))

    def build_split(split_subjects: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        xs: List[np.ndarray] = []
        ps: List[np.ndarray] = []
        brs: List[np.ndarray] = []
        conds: List[np.ndarray] = []
        markers: List[Optional[np.ndarray]] = []
        ids: List[str] = []
        for group, labels in groups.items():
            keep_values = {str(lbl).strip().upper() for lbl in labels}
            for sid in split_subjects:
                raw = load_data(sid, data_dir=args.data_dir, data_group=group)
                filt = _filter_subject_dict_by_labels(raw, keep_values)
                if filt is None:
                    continue
                append_subject(xs, ps, brs, conds, markers, ids, filt, group, sid)
        if not xs:
            raise RuntimeError(f"No requested labels {keep_labels} found for split subjects {split_subjects}")
        ns = [len(a) for a in xs]
        marker_arr = _concat_marker_parts(markers, ns)
        return (
            np.concatenate(xs, axis=0),
            np.concatenate(ps, axis=0),
            np.concatenate(brs, axis=0),
            np.concatenate(conds, axis=0),
            marker_arr,
            np.asarray(ids, dtype=object),
        )

    train_subjects = [s for s in subjects if s != subject]
    x_train, y_train, br_train, cond_train, marker_train, train_ids = build_split(train_subjects)
    x_test, y_test, br_test, cond_test, marker_test, test_ids = build_split([subject])

    train_ds = WindowDatasetWithMarker(x_train, y_train, cond_train, br_train, marker_train)
    test_ds = WindowDatasetWithMarker(x_test, y_test, cond_test, br_test, marker_test)
    loader_kwargs = _mard_loader_kwargs(args)
    train_loader = DataLoader(train_ds, batch_size=int(args.embed_batch_size), shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=int(args.embed_batch_size), shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, test_loader, train_ids, test_ids


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
ACTIVITY_FEATURE_NAMES = [
    "acc_norm_mean", "acc_norm_std", "acc_norm_iqr", "acc_norm_rms", "acc_norm_diff_rms", "acc_norm_range",
    "gyro_norm_mean", "gyro_norm_std", "gyro_norm_iqr", "gyro_norm_rms", "gyro_norm_diff_rms", "gyro_norm_range",
    "imu_motion_energy", "imu_stationarity",
    "acc_dom_freq", "acc_entropy", "acc_peakness", "acc_bandwidth", "gyro_dom_freq", "gyro_entropy", "gyro_peakness", "gyro_bandwidth",
    "acc_low_frac", "acc_high_frac", "gyro_low_frac", "gyro_high_frac",
]

SEQ_FEATURE_NAMES = [
    "mean", "std", "iqr", "rms", "abs_mean", "range", "vel_mean", "vel_std", "vel_rms", "jerk_rms", "drift", "dom_freq", "entropy", "peakness", "bandwidth", "low_frac", "high_frac",
]
MARKER_FEATURE_NAMES = (
    ["marker_valid_frac"]
    + [f"marker_pos_{n}" for n in SEQ_FEATURE_NAMES]
    + [f"marker_rot_{n}" for n in SEQ_FEATURE_NAMES]
    + [f"marker_global_{n}" for n in SEQ_FEATURE_NAMES]
    + ["marker_motion_energy", "marker_stationarity", "marker_posture_drift", "marker_rot_trans_coupling", "marker_burstiness"]
)

INTERACTION_FEATURE_NAMES = [
    "rrdiff_x_imu_motion", "resp_entropy_x_imu_entropy", "resp_peakness_x_imu_stationarity", "resp_bandwidth_x_gyro_entropy",
    "token_rrstd_x_imu_motion", "rrdiff_over_imu_motion", "resp_energy_x_acc_energy", "highband_x_gyro_peak",
]

MOTION_INTERACTION_FEATURE_NAMES = [
    "rrdiff_x_marker_motion", "resp_entropy_x_marker_entropy", "resp_peakness_x_marker_stationarity", "resp_bandwidth_x_marker_drift",
    "token_rrstd_x_marker_motion", "rrdiff_over_marker_motion", "imu_motion_x_marker_motion", "imu_stationarity_x_marker_stationarity",
]


def _ensure_btc(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected (B,T,C) or (B,C,T), got {tuple(x.shape)}")
    # Channel axis is usually the smaller one.
    if x.shape[1] < x.shape[2] and x.shape[1] <= 32:
        x = x.transpose(1, 2)
    return x


def _iqr_torch(x: torch.Tensor, dim: int) -> torch.Tensor:
    q75 = torch.quantile(x, 0.75, dim=dim)
    q25 = torch.quantile(x, 0.25, dim=dim)
    return q75 - q25


def _norm_stat_features(x: torch.Tensor) -> List[torch.Tensor]:
    mean = x.mean(dim=1)
    std = x.std(dim=1, unbiased=False)
    iqr = _iqr_torch(x, dim=1) if x.size(1) > 1 else torch.zeros_like(mean)
    rms = torch.sqrt(x.pow(2).mean(dim=1).clamp_min(0.0))
    if x.size(1) > 1:
        diff = torch.diff(x, dim=1)
        diff_rms = torch.sqrt(diff.pow(2).mean(dim=1).clamp_min(0.0))
    else:
        diff_rms = torch.zeros_like(mean)
    ran = x.max(dim=1).values - x.min(dim=1).values
    return [mean, std, iqr, rms, diff_rms, ran]


def _spectral_features_1d(x: torch.Tensor, fs: float, min_hz: float = 0.05, max_hz: float = 8.0) -> List[torch.Tensor]:
    if x.size(1) < 8:
        z = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        return [z, z, z, z, z, z]
    x = x - x.mean(dim=1, keepdim=True)
    spec = torch.fft.rfft(x, dim=1).abs().pow(2)
    freqs = torch.fft.rfftfreq(x.size(1), d=1.0 / float(fs)).to(x.device)
    mask = (freqs >= float(min_hz)) & (freqs <= min(float(max_hz), float(fs) / 2.0))
    if int(mask.sum().item()) < 3:
        mask = torch.ones_like(freqs, dtype=torch.bool)
    f = freqs[mask]
    s = spec[:, mask].clamp_min(1e-8)
    p = s / s.sum(dim=1, keepdim=True).clamp_min(1e-8)
    centroid = (p * f.view(1, -1)).sum(dim=1)
    entropy = -(p * p.log()).sum(dim=1) / math.log(max(2, p.size(1)))
    peakness = p.max(dim=1).values
    bandwidth = torch.sqrt((p * (f.view(1, -1) - centroid.view(-1, 1)).pow(2)).sum(dim=1))
    med = torch.median(f)
    low_frac = p[:, f <= med].sum(dim=1)
    high_frac = p[:, f >= med].sum(dim=1)
    return [centroid, entropy, peakness, bandwidth, low_frac, high_frac]


@torch.no_grad()
def imu_activity_features(imu: torch.Tensor, fs: float = float(IMU_FS)) -> torch.Tensor:
    imu = _ensure_btc(imu.float())
    B, _T, C = imu.shape
    if C >= 6:
        acc = imu[:, :, :3]
        gyr = imu[:, :, 3:6]
    elif C >= 2:
        mid = max(1, C // 2)
        acc = imu[:, :, :mid]
        gyr = imu[:, :, mid:]
    else:
        acc = imu
        gyr = imu.new_zeros(B, imu.size(1), 1)
    acc_norm = torch.linalg.norm(acc, dim=2)
    gyr_norm = torch.linalg.norm(gyr, dim=2)
    acc_stats = _norm_stat_features(acc_norm)
    gyr_stats = _norm_stat_features(gyr_norm)
    motion_energy = acc_stats[3] + gyr_stats[3]
    stationarity = 1.0 / (1.0 + acc_stats[1] + gyr_stats[1] + acc_stats[4] + gyr_stats[4])
    acc_spec = _spectral_features_1d(acc_norm, fs=fs)
    gyr_spec = _spectral_features_1d(gyr_norm, fs=fs)
    feats = acc_stats + gyr_stats + [motion_energy, stationarity] + acc_spec[:4] + gyr_spec[:4] + acc_spec[4:] + gyr_spec[4:]
    return torch.stack(feats, dim=1)


def _seq_features_from_norm(x: torch.Tensor, fs: float) -> List[torch.Tensor]:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    stats = _norm_stat_features(x)
    mean, std, iqr, rms, abs_mean_unused, ran = stats[0], stats[1], stats[2], stats[3], x.abs().mean(dim=1), stats[5]
    if x.size(1) > 1:
        vel = torch.diff(x, dim=1) * float(fs)
        vel_mean = vel.abs().mean(dim=1)
        vel_std = vel.std(dim=1, unbiased=False)
        vel_rms = torch.sqrt(vel.pow(2).mean(dim=1).clamp_min(0.0))
        drift = (x[:, -1] - x[:, 0]).abs()
    else:
        z = torch.zeros_like(mean)
        vel = x.new_zeros(x.size(0), 1)
        vel_mean = vel_std = vel_rms = drift = z
    if vel.size(1) > 1:
        jerk = torch.diff(vel, dim=1) * float(fs)
        jerk_rms = torch.sqrt(jerk.pow(2).mean(dim=1).clamp_min(0.0))
    else:
        jerk_rms = torch.zeros_like(mean)
    spec = _spectral_features_1d(x, fs=fs, min_hz=0.02, max_hz=8.0)
    return [mean, std, iqr, rms, abs_mean_unused, ran, vel_mean, vel_std, vel_rms, jerk_rms, drift] + spec


@torch.no_grad()
def marker_motion_features(marker: Optional[torch.Tensor], fs: float = float(MARKER_FS)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed-size marker motion/posture summary and valid mask.

    Handles marker_filt windows shaped (B,T,C), (B,C,T), or flattened (B,F).
    The expected exported order is rotation columns followed by position columns,
    but the extractor also works generically when the channel count differs.
    """
    if marker is None or marker.numel() == 0:
        B = 0 if marker is None else int(marker.shape[0])
        dev = torch.device("cpu") if marker is None else marker.device
        return torch.zeros(B, len(MARKER_FEATURE_NAMES), device=dev), torch.zeros(B, dtype=torch.bool, device=dev)
    B = int(marker.shape[0])
    if marker.ndim == 2:
        if marker.shape[1] == 0:
            return torch.zeros(B, len(MARKER_FEATURE_NAMES), device=marker.device), torch.zeros(B, dtype=torch.bool, device=marker.device)
        x = marker.reshape(B, marker.shape[1], 1).float()
    elif marker.ndim == 3:
        x = _ensure_btc(marker.float())
    else:
        x = marker.reshape(B, -1, 1).float()

    finite = torch.isfinite(x)
    valid_frac = finite.float().mean(dim=(1, 2))
    clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    C = int(clean.shape[2])
    if C >= 7:
        rot = clean[:, :, :4]
        pos = clean[:, :, -3:]
    elif C >= 4:
        split = max(1, C - 3)
        rot = clean[:, :, :split]
        pos = clean[:, :, split:]
    elif C >= 2:
        split = max(1, C // 2)
        rot = clean[:, :, :split]
        pos = clean[:, :, split:]
    else:
        pos = clean
        rot = clean.new_zeros(B, clean.size(1), 1)

    pos_norm = torch.linalg.norm(pos, dim=2)
    rot_norm = torch.linalg.norm(rot, dim=2)
    global_norm = torch.linalg.norm(clean, dim=2)
    pos_feats = _seq_features_from_norm(pos_norm, fs=fs)
    rot_feats = _seq_features_from_norm(rot_norm, fs=fs)
    glob_feats = _seq_features_from_norm(global_norm, fs=fs)

    pos_vel_rms = pos_feats[8]
    rot_vel_rms = rot_feats[8]
    pos_drift = pos_feats[10]
    rot_drift = rot_feats[10]
    motion_energy = pos_vel_rms + rot_vel_rms
    stationarity = 1.0 / (1.0 + motion_energy + pos_drift + rot_drift)
    posture_drift = pos_drift + rot_drift
    coupling = pos_vel_rms / (1.0 + rot_vel_rms)
    burstiness = glob_feats[7] / (1.0 + glob_feats[6])

    feats = torch.stack([valid_frac] + pos_feats + rot_feats + glob_feats + [motion_energy, stationarity, posture_drift, coupling, burstiness], dim=1)
    valid = (valid_frac > 0.50) & (motion_energy >= 0.0)
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), valid


def interaction_features(resp_static: np.ndarray, activity_static: np.ndarray) -> np.ndarray:
    r = np.asarray(resp_static, dtype=np.float32)
    a = np.asarray(activity_static, dtype=np.float32)
    n = r.shape[0]
    if n == 0:
        return np.zeros((0, len(INTERACTION_FEATURE_NAMES)), dtype=np.float32)
    rr_disagree = r[:, 2] if r.shape[1] > 2 else np.zeros(n)
    band_energy = r[:, 3] if r.shape[1] > 3 else np.zeros(n)
    resp_entropy = r[:, 4] if r.shape[1] > 4 else np.zeros(n)
    resp_peakness = r[:, 5] if r.shape[1] > 5 else np.zeros(n)
    bandwidth = r[:, 6] if r.shape[1] > 6 else np.zeros(n)
    high_frac = r[:, 15] if r.shape[1] > 15 else np.zeros(n)
    token_rr_std = r[:, 16] if r.shape[1] > 16 else np.zeros(n)
    acc_energy = a[:, 3] if a.shape[1] > 3 else np.zeros(n)
    gyro_energy = a[:, 9] if a.shape[1] > 9 else np.zeros(n)
    motion_energy = a[:, 12] if a.shape[1] > 12 else acc_energy + gyro_energy
    stationarity = a[:, 13] if a.shape[1] > 13 else np.ones(n)
    acc_entropy = a[:, 15] if a.shape[1] > 15 else np.zeros(n)
    gyro_entropy = a[:, 19] if a.shape[1] > 19 else np.zeros(n)
    gyro_peak = a[:, 20] if a.shape[1] > 20 else np.zeros(n)
    motion_entropy = 0.5 * (acc_entropy + gyro_entropy)
    feats = np.stack([
        rr_disagree * motion_energy,
        resp_entropy * motion_entropy,
        resp_peakness * stationarity,
        bandwidth * gyro_entropy,
        token_rr_std * motion_energy,
        rr_disagree / (1.0 + motion_energy),
        band_energy * acc_energy,
        high_frac * gyro_peak,
    ], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def motion_interaction_features(resp_static: np.ndarray, activity_static: np.ndarray, marker_proxy: np.ndarray) -> np.ndarray:
    r = np.asarray(resp_static, dtype=np.float32)
    a = np.asarray(activity_static, dtype=np.float32)
    m = np.asarray(marker_proxy, dtype=np.float32)
    n = r.shape[0]
    if n == 0:
        return np.zeros((0, len(MOTION_INTERACTION_FEATURE_NAMES)), dtype=np.float32)
    rr_disagree = r[:, 2] if r.shape[1] > 2 else np.zeros(n)
    resp_entropy = r[:, 4] if r.shape[1] > 4 else np.zeros(n)
    resp_peakness = r[:, 5] if r.shape[1] > 5 else np.zeros(n)
    bandwidth = r[:, 6] if r.shape[1] > 6 else np.zeros(n)
    token_rr_std = r[:, 16] if r.shape[1] > 16 else np.zeros(n)
    imu_motion = a[:, 12] if a.shape[1] > 12 else np.zeros(n)
    imu_stationarity = a[:, 13] if a.shape[1] > 13 else np.ones(n)
    marker_entropy = m[:, 1 + 2 * len(SEQ_FEATURE_NAMES) + 12] if m.shape[1] > 1 + 2 * len(SEQ_FEATURE_NAMES) + 12 else np.zeros(n)
    marker_motion = m[:, -5] if m.shape[1] >= 5 else np.zeros(n)
    marker_stationarity = m[:, -4] if m.shape[1] >= 4 else np.ones(n)
    marker_drift = m[:, -3] if m.shape[1] >= 3 else np.zeros(n)
    feats = np.stack([
        rr_disagree * marker_motion,
        resp_entropy * marker_entropy,
        resp_peakness * marker_stationarity,
        bandwidth * marker_drift,
        token_rr_std * marker_motion,
        rr_disagree / (1.0 + marker_motion),
        imu_motion * marker_motion,
        imu_stationarity * marker_stationarity,
    ], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@torch.no_grad()
def collect_marker_rd_features(model, loader, device: str, args, sample_subjects: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    model.eval()
    resp, act, mark, mark_valid, y, rr, subjects, states = [], [], [], [], [], [], [], []
    offset = 0
    sample_subjects_arr = None if sample_subjects is None else np.asarray(sample_subjects, dtype=object)
    for batch in loader:
        imu, _pressure, cond, _br, _tlx, marker = unpack_batch_optional_marker(batch, device)
        pred_logmag, rr_pred, hidden = model(imu)
        rfeat = dyn.respiratory_stft_features(
            pred_logmag=pred_logmag,
            rr_pred=rr_pred,
            fs=float(args.resp_dyn_fs),
            min_hz=float(args.resp_dyn_min_hz),
            max_hz=float(args.resp_dyn_max_hz),
        )
        afeat = imu_activity_features(imu, fs=float(args.imu_fs))
        mfeat, mvalid = marker_motion_features(marker, fs=float(args.marker_fs))
        n = int(cond.numel())
        resp.append(rfeat.detach().cpu().numpy())
        act.append(afeat.detach().cpu().numpy())
        mark.append(mfeat.detach().cpu().numpy())
        mark_valid.append(mvalid.detach().cpu().numpy().astype(bool))
        y.append(cond.detach().cpu().numpy().astype(int).reshape(-1))
        rr.append(rr_pred.detach().cpu().numpy().reshape(-1))
        if sample_subjects_arr is not None:
            subjects.append(sample_subjects_arr[offset: offset + n])
            offset += n
        if hasattr(model, "respiration_state_from_outputs"):
            try:
                st = model.respiration_state_from_outputs(pred_logmag, rr_pred, hidden, adapt=False)
                states.append(st.detach().cpu().numpy())
            except Exception as exc:
                print(f"[MARKER_RD] Could not collect PAPA state: {exc}")
                states = []
    if not resp:
        raise RuntimeError("No batches available for marker/RD feature collection.")
    out: Dict[str, np.ndarray] = {
        "x_resp_static": np.nan_to_num(np.concatenate(resp, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "x_activity_static": np.nan_to_num(np.concatenate(act, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "x_marker": np.nan_to_num(np.concatenate(mark, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "marker_valid": np.concatenate(mark_valid, axis=0).astype(bool),
        "y": np.concatenate(y, axis=0).astype(int),
        "rr_pred": np.concatenate(rr, axis=0).astype(np.float32),
    }
    if subjects:
        out["subject_ids"] = np.concatenate(subjects, axis=0)
    if states:
        out["papa_state"] = np.nan_to_num(np.concatenate(states, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


# -----------------------------------------------------------------------------
# Feature transforms and marker proxy
# -----------------------------------------------------------------------------
def _robust_scale_np(x: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    iqr = np.nanquantile(x, 0.75, axis=0) - np.nanquantile(x, 0.25, axis=0)
    scale = np.maximum(mad, iqr / 1.349)
    return np.maximum(scale, float(floor)).astype(np.float32)


def subject_robust_z(x: np.ndarray, subject_ids: Optional[np.ndarray], floor: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if subject_ids is None:
        med = np.nanmedian(x, axis=0)
        scale = _robust_scale_np(x, floor=floor)
        return np.nan_to_num((x - med) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ids = np.asarray(subject_ids, dtype=object)
    out = np.zeros_like(x, dtype=np.float32)
    for sid in pd.unique(ids):
        mask = ids == sid
        med = np.nanmedian(x[mask], axis=0)
        scale = _robust_scale_np(x[mask], floor=floor)
        out[mask] = (x[mask] - med) / scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def make_segments(subject_ids: Optional[np.ndarray], y: Optional[np.ndarray] = None, by_label: bool = False) -> Optional[np.ndarray]:
    if subject_ids is None:
        if y is None or not by_label:
            return None
        subject_ids = np.full(len(y), "__all__", dtype=object)
    ids = np.asarray(subject_ids, dtype=object)
    if by_label and y is not None:
        yy = np.asarray(y, dtype=int)
        return np.asarray([f"{sid}:{int(lbl)}" for sid, lbl in zip(ids, yy)], dtype=object)
    return ids


def concat_features(*parts: Optional[np.ndarray]) -> np.ndarray:
    xs = [np.asarray(p, dtype=np.float32) for p in parts if p is not None]
    if not xs:
        raise ValueError("concat_features received no arrays")
    return np.nan_to_num(np.concatenate(xs, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _finite_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    finite = np.isfinite(arr).any(axis=1)
    nonzero = np.nan_to_num(np.abs(arr), nan=0.0, posinf=0.0, neginf=0.0).sum(axis=1) > 1e-8
    return finite & nonzero



# -----------------------------------------------------------------------------
# CUDA helpers for marker proxy, expert classifiers, and preference gate
# -----------------------------------------------------------------------------
def _cuda_device_from_args(args) -> torch.device:
    requested = str(getattr(args, "mard_cuda_device", "auto"))
    if requested and requested.lower() not in {"auto", "none"}:
        try:
            dev = torch.device(requested)
            if dev.type == "cuda" and torch.cuda.is_available():
                return dev
            if dev.type == "cpu":
                return dev
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_standardize_fit(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(float(eps))
    return (x - mu) / sd, mu, sd


def _torch_standardize_apply(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (x - mu) / sd.clamp_min(1e-6)


def _fit_ridge_predict_torch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    alpha: float,
    args=None,
) -> np.ndarray:
    """Multi-output ridge regression on GPU/torch with sklearn-compatible behavior."""
    dev = _cuda_device_from_args(args)
    dtype = torch.float64 if bool(getattr(args, "mard_torch_ridge_float64", False)) else torch.float32
    x = torch.as_tensor(np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0), dtype=dtype, device=dev)
    y = torch.as_tensor(np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0), dtype=dtype, device=dev)
    xp = torch.as_tensor(np.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0), dtype=dtype, device=dev)
    if y.ndim == 1:
        y = y[:, None]
    xz, mu, sd = _torch_standardize_fit(x)
    xpz = _torch_standardize_apply(xp, mu, sd)
    ones = torch.ones((xz.size(0), 1), dtype=dtype, device=dev)
    X = torch.cat([xz, ones], dim=1)
    Xp = torch.cat([xpz, torch.ones((xpz.size(0), 1), dtype=dtype, device=dev)], dim=1)
    eye = torch.eye(X.size(1), dtype=dtype, device=dev)
    eye[-1, -1] = 0.0  # do not penalize intercept
    A = X.T @ X + float(alpha) * eye
    B = X.T @ y
    try:
        W = torch.linalg.solve(A, B)
    except RuntimeError:
        W = torch.linalg.pinv(A) @ B
    pred = Xp @ W
    return pred.detach().cpu().float().numpy().astype(np.float32)


class TorchLogRegClassifier:
    """Small sklearn-like multinomial logistic regression trained in torch."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 200,
        lr: float = 1e-2,
        batch_size: int = 4096,
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        seed: int = 42,
        class_weight_balanced: bool = True,
    ):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.seed = int(seed)
        self.class_weight_balanced = bool(class_weight_balanced)
        self.classes_: np.ndarray = np.array([], dtype=int)
        self.mu_: Optional[torch.Tensor] = None
        self.sd_: Optional[torch.Tensor] = None
        self.model_: Optional[nn.Linear] = None
        self.constant_: Optional[int] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchLogRegClassifier":
        x_np = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_np = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.array(sorted(np.unique(y_np).tolist()), dtype=int)
        if len(self.classes_) < 2 or x_np.shape[0] < max(3, len(self.classes_)):
            vals, counts = np.unique(y_np, return_counts=True)
            self.constant_ = int(vals[np.argmax(counts)]) if vals.size else 0
            self.classes_ = np.array([self.constant_], dtype=int)
            return self
        dev = torch.device(self.device) if self.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        x_t = torch.as_tensor(x_np, dtype=torch.float32, device=dev)
        y_idx_np = np.searchsorted(self.classes_, y_np).astype(np.int64)
        y_t = torch.as_tensor(y_idx_np, dtype=torch.long, device=dev)
        x_z, self.mu_, self.sd_ = _torch_standardize_fit(x_t)
        n, d = x_z.shape
        k = len(self.classes_)
        self.model_ = nn.Linear(d, k).to(dev)
        nn.init.zeros_(self.model_.bias)
        opt = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=(self.weight_decay if self.weight_decay > 0 else 1.0 / max(self.C, 1e-6) * 1e-4),
        )
        class_weight = None
        if self.class_weight_balanced:
            counts = torch.bincount(y_t, minlength=k).float().clamp_min(1.0)
            class_weight = (counts.sum() / (k * counts)).to(dev)
        bs = max(1, min(int(self.batch_size), n))
        for _ in range(int(self.max_iter)):
            perm = torch.randperm(n, device=dev)
            for st in range(0, n, bs):
                idx = perm[st : st + bs]
                opt.zero_grad(set_to_none=True)
                logits = self.model_(x_z[idx])
                loss = F.cross_entropy(logits, y_t[idx], weight=class_weight)
                loss.backward()
                opt.step()
        self.constant_ = None
        return self

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_np = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.model_ is None or self.constant_ is not None:
            return np.ones((x_np.shape[0], 1), dtype=np.float32)
        dev = next(self.model_.parameters()).device
        out: List[np.ndarray] = []
        bs = max(1, int(self.batch_size))
        for st in range(0, x_np.shape[0], bs):
            xb = torch.as_tensor(x_np[st : st + bs], dtype=torch.float32, device=dev)
            xb = _torch_standardize_apply(xb, self.mu_.to(dev), self.sd_.to(dev))
            prob = torch.softmax(self.model_(xb), dim=1)
            out.append(prob.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        p = self.predict_proba(x)
        return self.classes_[np.argmax(p, axis=1)].astype(int)

def _fit_ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray, alpha: float, args=None) -> np.ndarray:
    backend = str(getattr(args, "mard_ridge_backend", "torch" if torch.cuda.is_available() else "sklearn")).lower()
    if backend in {"torch", "cuda", "gpu"}:
        try:
            return _fit_ridge_predict_torch(x_train, y_train, x_pred, alpha, args=args)
        except Exception as exc:
            if not bool(getattr(args, "mard_strict_cuda", False)):
                warnings.warn(f"Torch ridge backend failed ({exc}); falling back to sklearn Ridge on CPU.")
            else:
                raise
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    model.fit(np.nan_to_num(x_train, nan=0.0), np.nan_to_num(y_train, nan=0.0))
    return model.predict(np.nan_to_num(x_pred, nan=0.0)).astype(np.float32)


def fit_marker_proxy(
    x_ar_train: np.ndarray,
    marker_train: np.ndarray,
    marker_valid_train: np.ndarray,
    train_subject_ids: np.ndarray,
    x_ar_test: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train source-only AR -> marker-motion proxy and return train/test predictions.

    Source predictions are source-LOSO inferred by default to avoid giving the
    gate observed marker values for pseudo-held-out source subjects.  Set
    --mard-marker-source-mode observed for an optimistic diagnostic.
    """
    x_ar_train = np.nan_to_num(np.asarray(x_ar_train, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x_ar_test = np.nan_to_num(np.asarray(x_ar_test, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_m = np.nan_to_num(np.asarray(marker_train, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.asarray(marker_valid_train, dtype=bool).reshape(-1) & _finite_rows(y_m)

    # Negative controls from the ablation grid.  These are applied only to the
    # source marker teacher target before fitting the proxy.
    mode = str(getattr(args, "marker_mode", "quality_gate"))
    if mode == "shuffled_teacher_control" and int(valid.sum()) > 1:
        rng = np.random.default_rng(int(getattr(args, "seed", 42)))
        y_m[valid] = y_m[valid][rng.permutation(int(valid.sum()))]
    elif mode == "time_shift_control" and int(valid.sum()) > 1:
        shift = max(1, int(round(0.25 * int(valid.sum()))))
        y_m[valid] = np.roll(y_m[valid], shift=shift, axis=0)
    elif mode == "time_only_control":
        t = np.linspace(-1.0, 1.0, y_m.shape[0], dtype=np.float32).reshape(-1, 1)
        basis = np.concatenate([t, t ** 2, np.sin(np.pi * t), np.cos(np.pi * t)], axis=1)
        reps = int(np.ceil(y_m.shape[1] / basis.shape[1])) if y_m.ndim == 2 and y_m.shape[1] else 1
        y_m = np.tile(basis, (1, reps))[:, : y_m.shape[1]].astype(np.float32)

    min_rows = int(getattr(args, "mard_min_marker_rows", 50))
    min_subjects = int(getattr(args, "mard_marker_min_subjects", 3))
    source_mode = str(getattr(args, "mard_marker_source_mode", "inferred")).lower()
    d = int(y_m.shape[1]) if y_m.ndim == 2 else len(MARKER_FEATURE_NAMES)
    zeros_tr = np.zeros((x_ar_train.shape[0], d), dtype=np.float32)
    zeros_te = np.zeros((x_ar_test.shape[0], d), dtype=np.float32)
    high_unc_tr = np.ones((x_ar_train.shape[0], 1), dtype=np.float32)
    high_unc_te = np.ones((x_ar_test.shape[0], 1), dtype=np.float32)
    sids = np.asarray(train_subject_ids, dtype=object)
    valid_subjects = [sid for sid in pd.unique(sids[valid])]
    if int(valid.sum()) < min_rows or len(valid_subjects) < min_subjects:
        return zeros_tr, high_unc_tr, zeros_te, high_unc_te, {
            "marker_proxy_available": False,
            "marker_proxy_valid_rows": int(valid.sum()),
            "marker_proxy_valid_subjects": int(len(valid_subjects)),
            "marker_proxy_dim": d,
        }

    alpha = float(getattr(args, "mard_marker_proxy_alpha", 10.0))
    pred_test = _fit_ridge_predict(x_ar_train[valid], y_m[valid], x_ar_test, alpha, args=args)
    pred_train_full = _fit_ridge_predict(x_ar_train[valid], y_m[valid], x_ar_train, alpha, args=args)
    pred_train = pred_train_full.copy()
    if source_mode == "observed":
        pred_train[valid] = y_m[valid]
    else:
        for sid in pd.unique(sids):
            row_mask = sids == sid
            val_mask = row_mask & valid
            tr_mask = (~row_mask) & valid
            if int(val_mask.sum()) == 0:
                continue
            if int(tr_mask.sum()) >= min_rows and len(pd.unique(sids[tr_mask])) >= max(2, min_subjects - 1):
                pred_train[val_mask] = _fit_ridge_predict(x_ar_train[tr_mask], y_m[tr_mask], x_ar_train[val_mask], alpha, args=args)
            else:
                pred_train[val_mask] = pred_train_full[val_mask]

    residual = np.sqrt(np.mean((pred_train[valid] - y_m[valid]) ** 2, axis=1))
    resid_scale = float(np.nanmedian(residual) + np.nanstd(residual)) if residual.size else 1.0
    resid_scale = max(resid_scale, 1e-6)
    scaler = StandardScaler().fit(x_ar_train[valid])
    z_src = scaler.transform(x_ar_train[valid])
    z_mu = z_src.mean(axis=0, keepdims=True)

    def uncertainty(x: np.ndarray) -> np.ndarray:
        z = scaler.transform(np.nan_to_num(x, nan=0.0))
        dist = np.sqrt(np.mean((z - z_mu) ** 2, axis=1, keepdims=True))
        return np.clip(resid_scale * (1.0 + dist), 0.0, 100.0).astype(np.float32)

    unc_train = uncertainty(x_ar_train)
    unc_test = uncertainty(x_ar_test)
    meta = {
        "marker_proxy_available": True,
        "marker_proxy_valid_rows": int(valid.sum()),
        "marker_proxy_valid_subjects": int(len(valid_subjects)),
        "marker_proxy_dim": int(d),
        "marker_proxy_source_mode": source_mode,
        "marker_proxy_resid_median": float(np.nanmedian(residual)),
        "marker_proxy_resid_scale": float(resid_scale),
    }
    return pred_train.astype(np.float32), unc_train, pred_test.astype(np.float32), unc_test, meta


def build_marker_rd_feature_blocks(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray], args) -> Tuple[OrderedDict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, np.ndarray], Dict[str, Any]]:
    y_train = train["y"].astype(int)
    train_ids = train.get("subject_ids")
    test_ids = test.get("subject_ids")
    scale_floor = float(args.resp_dyn_scale_floor)

    ladder = dyn.build_ladder_features(
        x_train_static=train["x_resp_static"],
        y_train=y_train,
        x_test_static=test["x_resp_static"],
        args=args,
        train_subject_ids=train_ids,
        test_subject_ids=test_ids,
        x_train_state=train.get("papa_state"),
        x_test_state=test.get("papa_state"),
    )
    resp_variant = str(args.mard_resp_variant)
    if resp_variant not in ladder:
        print(f"[MARKER_RD] Requested resp variant {resp_variant!r} not available; using dyn_hybrid or first available.")
        resp_variant = "dyn_hybrid" if "dyn_hybrid" in ladder else sorted(ladder.keys())[0]
    x_resp_tr, x_resp_te = ladder[resp_variant]

    act_tr_z = subject_robust_z(train["x_activity_static"], train_ids, floor=scale_floor)
    act_te_z = subject_robust_z(test["x_activity_static"], test_ids, floor=scale_floor)
    x_act_tr = dyn.add_rolling_dynamics(
        act_tr_z,
        win=int(args.resp_dyn_roll_win),
        segments=make_segments(train_ids, y_train, by_label=bool(args.resp_dyn_source_segment_by_label)),
        centered=bool(args.resp_dyn_centered_roll),
        boundary_jump_z=float(args.resp_dyn_boundary_jump_z),
        scale_floor=scale_floor,
    )
    x_act_te = dyn.add_rolling_dynamics(
        act_te_z,
        win=int(args.resp_dyn_roll_win),
        segments=make_segments(test_ids),
        centered=bool(args.resp_dyn_centered_roll),
        boundary_jump_z=float(args.resp_dyn_boundary_jump_z),
        scale_floor=scale_floor,
    )
    x_int_tr = interaction_features(train["x_resp_static"], train["x_activity_static"])
    x_int_te = interaction_features(test["x_resp_static"], test["x_activity_static"])
    x_ar_tr = concat_features(x_resp_tr, x_act_tr, x_int_tr)
    x_ar_te = concat_features(x_resp_te, x_act_te, x_int_te)

    m_proxy_tr, m_unc_tr, m_proxy_te, m_unc_te, marker_meta = fit_marker_proxy(
        x_ar_train=x_ar_tr,
        marker_train=train["x_marker"],
        marker_valid_train=train["marker_valid"],
        train_subject_ids=np.asarray(train_ids if train_ids is not None else np.full(len(y_train), "__source__", dtype=object), dtype=object),
        x_ar_test=x_ar_te,
        args=args,
    )
    x_mint_tr = motion_interaction_features(train["x_resp_static"], train["x_activity_static"], m_proxy_tr)
    x_mint_te = motion_interaction_features(test["x_resp_static"], test["x_activity_static"], m_proxy_te)

    mode = str(getattr(args, "marker_mode", "quality_gate"))
    blocks: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
    blocks["resp_dyn"] = (x_resp_tr, x_resp_te)

    use_activity_blocks = mode not in {"none", "marker_only_oracle"}
    if use_activity_blocks:
        blocks["activity_dyn"] = (x_act_tr, x_act_te)
        blocks["resp_activity"] = (concat_features(x_resp_tr, x_act_tr, x_int_tr), concat_features(x_resp_te, x_act_te, x_int_te))

    use_marker_proxy_blocks = bool(getattr(args, "mard_use_marker_proxy_blocks", False)) or mode in {
        "teacher_motion_state", "posture_profile_expert", "shuffled_teacher_control", "time_shift_control", "time_only_control"
    }
    if use_marker_proxy_blocks:
        blocks["marker_proxy"] = (concat_features(m_proxy_tr, m_unc_tr), concat_features(m_proxy_te, m_unc_te))
        blocks["motion_aware_resp"] = (concat_features(x_resp_tr, m_proxy_tr, m_unc_tr, x_mint_tr), concat_features(x_resp_te, m_proxy_te, m_unc_te, x_mint_te))
        blocks["motion_aware_resp_activity"] = (concat_features(x_ar_tr, m_proxy_tr, m_unc_tr, x_mint_tr), concat_features(x_ar_te, m_proxy_te, m_unc_te, x_mint_te))
    if "papa_state" in train and "papa_state" in test:
        blocks["papa_state"] = (train["papa_state"], test["papa_state"])

    def quality(resp_static: np.ndarray, act_static: np.ndarray, marker_proxy: np.ndarray, marker_unc: np.ndarray) -> np.ndarray:
        n = resp_static.shape[0]
        motion_energy = act_static[:, 12] if act_static.shape[1] > 12 else np.zeros(n)
        stationarity = act_static[:, 13] if act_static.shape[1] > 13 else np.ones(n)
        acc_entropy = act_static[:, 15] if act_static.shape[1] > 15 else np.zeros(n)
        gyro_entropy = act_static[:, 19] if act_static.shape[1] > 19 else np.zeros(n)
        marker_valid_frac = marker_proxy[:, 0] if marker_proxy.shape[1] > 0 else np.zeros(n)
        marker_motion = marker_proxy[:, -5] if marker_proxy.shape[1] >= 5 else np.zeros(n)
        marker_stationarity = marker_proxy[:, -4] if marker_proxy.shape[1] >= 4 else np.ones(n)
        marker_drift = marker_proxy[:, -3] if marker_proxy.shape[1] >= 3 else np.zeros(n)
        marker_entropy = marker_proxy[:, 1 + 2 * len(SEQ_FEATURE_NAMES) + 12] if marker_proxy.shape[1] > 1 + 2 * len(SEQ_FEATURE_NAMES) + 12 else np.zeros(n)
        marker_burst = marker_proxy[:, -1] if marker_proxy.shape[1] >= 1 else np.zeros(n)
        cols = [
            resp_static[:, 2] if resp_static.shape[1] > 2 else np.zeros(n),
            resp_static[:, 4] if resp_static.shape[1] > 4 else np.zeros(n),
            resp_static[:, 5] if resp_static.shape[1] > 5 else np.zeros(n),
            resp_static[:, 6] if resp_static.shape[1] > 6 else np.zeros(n),
            resp_static[:, 12] if resp_static.shape[1] > 12 else np.zeros(n),
            motion_energy,
            stationarity,
            0.5 * (acc_entropy + gyro_entropy),
            marker_unc.reshape(-1),
            marker_valid_frac,
            marker_motion,
            marker_stationarity,
            marker_drift,
            marker_entropy,
            marker_burst,
        ]
        return np.stack(cols, axis=1).astype(np.float32)

    q_train = quality(train["x_resp_static"], train["x_activity_static"], m_proxy_tr, m_unc_tr)
    q_test = quality(test["x_resp_static"], test["x_activity_static"], m_proxy_te, m_unc_te)
    q = {
        "train": q_train,
        "test": q_test,
        "marker_proxy_train": m_proxy_tr,
        "marker_proxy_test": m_proxy_te,
        "marker_unc_train": m_unc_tr,
        "marker_unc_test": m_unc_te,
    }
    meta = {
        "resp_variant": resp_variant,
        "marker_mode": mode,
        "use_activity_blocks": bool(use_activity_blocks),
        "use_marker_proxy_blocks": bool(use_marker_proxy_blocks),
        "blocks": {k: {"train_dim": int(v[0].shape[1]), "test_dim": int(v[1].shape[1])} for k, v in blocks.items()},
        **marker_meta,
    }
    return blocks, q, meta


# -----------------------------------------------------------------------------
# Safe classifiers and preference gate
# -----------------------------------------------------------------------------
class SafeProbClassifier:
    def __init__(self, kind: str = "logreg", C: float = 1.0, max_iter: int = 1500, args=None):
        self.kind = str(kind).lower()
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.args = args
        self.model: Optional[Any] = None
        self.classes_: np.ndarray = np.array([], dtype=int)
        self.constant_: Optional[int] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SafeProbClassifier":
        x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.array(sorted(np.unique(y).tolist()), dtype=int)
        if len(self.classes_) < 2 or x.shape[0] < max(3, len(self.classes_)):
            vals, counts = np.unique(y, return_counts=True)
            self.constant_ = int(vals[np.argmax(counts)]) if vals.size else 0
            self.classes_ = np.array([self.constant_], dtype=int)
            self.model = None
            return self
        try:
            if self.kind == "lda":
                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                self.model = make_pipeline(StandardScaler(), clf)
                self.model.fit(x, y)
                self.classes_ = np.asarray(self.model[-1].classes_, dtype=int)
            elif self.kind == "logreg":
                clf = LogisticRegression(max_iter=self.max_iter, C=self.C, class_weight="balanced", multi_class="auto")
                self.model = make_pipeline(StandardScaler(), clf)
                self.model.fit(x, y)
                self.classes_ = np.asarray(self.model[-1].classes_, dtype=int)
            elif self.kind in {"torch_logreg", "cuda_logreg", "gpu_logreg"}:
                self.model = TorchLogRegClassifier(
                    C=self.C,
                    max_iter=int(getattr(self.args, "mard_torch_clf_epochs", self.max_iter)),
                    lr=float(getattr(self.args, "mard_torch_clf_lr", 1e-2)),
                    batch_size=int(getattr(self.args, "mard_torch_clf_batch_size", 4096)),
                    weight_decay=float(getattr(self.args, "mard_torch_clf_weight_decay", 0.0)),
                    device=str(_cuda_device_from_args(self.args)),
                    seed=int(getattr(self.args, "seed", 42)),
                )
                self.model.fit(x, y)
                self.classes_ = self.model.classes_
            else:
                raise ValueError(f"Unknown classifier kind {self.kind!r}")
            self.constant_ = None
        except Exception as exc:
            if bool(getattr(self.args, "mard_strict_cuda", False)):
                raise
            warnings.warn(f"Expert classifier failed ({exc}); using majority class fallback.")
            vals, counts = np.unique(y, return_counts=True)
            self.constant_ = int(vals[np.argmax(counts)]) if vals.size else 0
            self.classes_ = np.array([self.constant_], dtype=int)
            self.model = None
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.model is None:
            return np.ones((x.shape[0], 1), dtype=np.float32)
        return self.model.predict_proba(x).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        p = self.predict_proba(x)
        return self.classes_[np.argmax(p, axis=1)].astype(int)


def align_proba(proba: np.ndarray, classes: np.ndarray, global_classes: np.ndarray) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(global_classes)), dtype=np.float32)
    cls_to_i = {int(c): i for i, c in enumerate(global_classes)}
    for j, c in enumerate(classes.astype(int)):
        if int(c) in cls_to_i:
            out[:, cls_to_i[int(c)]] = proba[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum.reshape(-1) <= 1e-8
    if np.any(missing):
        out[missing, :] = 1.0 / max(1, len(global_classes))
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum.clip(min=1e-8)


def proba_meta_features(proba_aligned: np.ndarray, quality: np.ndarray, expert_index: int, n_experts: int) -> np.ndarray:
    p = np.clip(proba_aligned, 1e-12, 1.0)
    conf = p.max(axis=1)
    top2 = np.sort(p, axis=1)[:, -2:] if p.shape[1] >= 2 else np.concatenate([np.zeros((p.shape[0], 1)), p], axis=1)
    margin = top2[:, -1] - top2[:, -2]
    entropy = -(p * np.log(p)).sum(axis=1) / math.log(max(2, p.shape[1]))
    pred_idx = np.argmax(p, axis=1)
    pred_onehot = np.zeros_like(p, dtype=np.float32)
    pred_onehot[np.arange(p.shape[0]), pred_idx] = 1.0
    expert_onehot = np.zeros((p.shape[0], n_experts), dtype=np.float32)
    expert_onehot[:, expert_index] = 1.0
    return np.concatenate([conf[:, None], margin[:, None], entropy[:, None], p.astype(np.float32), pred_onehot.astype(np.float32), quality.astype(np.float32), expert_onehot], axis=1).astype(np.float32)


def softmax_np(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - np.nanmax(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=axis, keepdims=True).clip(min=1e-12)).astype(np.float32)


def train_preference_gate(blocks_train: OrderedDict[str, Tuple[np.ndarray, np.ndarray]], y_train: np.ndarray, subject_ids: np.ndarray, quality_train: np.ndarray, global_classes: np.ndarray, args) -> Tuple[Optional[BaseEstimator], Dict[str, Any]]:
    rows: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sids = np.asarray(subject_ids, dtype=object)
    expert_names = list(blocks_train.keys())
    n_experts = len(expert_names)
    for pseudo in list(pd.unique(sids)):
        val_mask = sids == pseudo
        tr_mask = ~val_mask
        if int(val_mask.sum()) < int(args.mard_min_pseudo_windows) or len(np.unique(y_train[tr_mask])) < 2:
            continue
        for ei, name in enumerate(expert_names):
            x_all = blocks_train[name][0]
            clf = SafeProbClassifier(kind=args.mard_expert_classifier, C=args.mard_logreg_c, max_iter=args.mard_logreg_max_iter, args=args)
            clf.fit(x_all[tr_mask], y_train[tr_mask])
            p = clf.predict_proba(x_all[val_mask])
            p_aligned = align_proba(p, clf.classes_, global_classes)
            pred = global_classes[np.argmax(p_aligned, axis=1)]
            correct = (pred.astype(int) == y_train[val_mask].astype(int)).astype(int)
            rows.append(proba_meta_features(p_aligned, quality_train[val_mask], ei, n_experts))
            labels.append(correct)
    if not rows:
        return None, {"gate_available": False, "gate_rows": 0, "gate_pos_rate": float("nan")}
    X = np.concatenate(rows, axis=0)
    y = np.concatenate(labels, axis=0).astype(int)
    if len(np.unique(y)) < 2 or len(y) < int(args.mard_min_gate_rows):
        return None, {"gate_available": False, "gate_rows": int(len(y)), "gate_pos_rate": float(np.mean(y)) if len(y) else float("nan")}
    gate_backend = str(getattr(args, "mard_gate_backend", "torch" if torch.cuda.is_available() else "sklearn")).lower()
    if gate_backend in {"torch", "cuda", "gpu", "torch_logreg"}:
        try:
            gate = TorchLogRegClassifier(
                C=float(args.mard_gate_c),
                max_iter=int(getattr(args, "mard_torch_gate_epochs", args.mard_gate_max_iter)),
                lr=float(getattr(args, "mard_torch_gate_lr", 1e-2)),
                batch_size=int(getattr(args, "mard_torch_gate_batch_size", 8192)),
                weight_decay=float(getattr(args, "mard_torch_gate_weight_decay", 0.0)),
                device=str(_cuda_device_from_args(args)),
                seed=int(getattr(args, "seed", 42)),
                class_weight_balanced=True,
            )
            gate.fit(X, y)
        except Exception as exc:
            if bool(getattr(args, "mard_strict_cuda", False)):
                raise
            warnings.warn(f"Torch preference gate failed ({exc}); falling back to sklearn LogisticRegression on CPU.")
            gate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=int(args.mard_gate_max_iter), C=float(args.mard_gate_c), class_weight="balanced"))
            gate.fit(X, y)
    else:
        gate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=int(args.mard_gate_max_iter), C=float(args.mard_gate_c), class_weight="balanced"))
        gate.fit(X, y)
    return gate, {"gate_available": True, "gate_rows": int(len(y)), "gate_pos_rate": float(np.mean(y)), "gate_backend": gate_backend}


def fit_outer_experts(blocks: OrderedDict[str, Tuple[np.ndarray, np.ndarray]], y_train: np.ndarray, global_classes: np.ndarray, quality_test: np.ndarray, gate: Optional[BaseEstimator], args) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    expert_names = list(blocks.keys())
    n_experts = len(expert_names)
    proba_list = []
    score_list = []
    pred_by_expert: Dict[str, np.ndarray] = {}
    conf_by_expert: Dict[str, np.ndarray] = {}
    for ei, name in enumerate(expert_names):
        xtr, xte = blocks[name]
        clf = SafeProbClassifier(kind=args.mard_expert_classifier, C=args.mard_logreg_c, max_iter=args.mard_logreg_max_iter, args=args)
        clf.fit(xtr, y_train)
        p = clf.predict_proba(xte)
        p_aligned = align_proba(p, clf.classes_, global_classes)
        meta = proba_meta_features(p_aligned, quality_test, ei, n_experts)
        if gate is not None and bool(getattr(args, "mard_use_preference_gate", True)):
            try:
                score = gate.predict_proba(meta)[:, 1]
            except Exception:
                score = p_aligned.max(axis=1)
        else:
            score = p_aligned.max(axis=1)
        proba_list.append(p_aligned)
        score_list.append(np.clip(score, 1e-6, 1.0).astype(np.float32))
        pred_by_expert[name] = global_classes[np.argmax(p_aligned, axis=1)].astype(int)
        conf_by_expert[name] = p_aligned.max(axis=1).astype(np.float32)
    scores = np.stack(score_list, axis=1)
    weights = softmax_np(np.log(scores.clip(min=1e-6)), axis=1, temperature=float(args.mard_gate_temperature))
    final = np.zeros_like(proba_list[0])
    for ei, p in enumerate(proba_list):
        final += weights[:, [ei]] * p
    return final.astype(np.float32), weights.astype(np.float32), pred_by_expert, conf_by_expert


def safe_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes_train: np.ndarray, prefix: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes_true = np.array(sorted(np.unique(y_true).astype(int).tolist()), dtype=int)
    classes_union = np.array(sorted(set(classes_train.astype(int).tolist()) | set(classes_true.tolist())), dtype=int)
    absent_pred_rate = float(np.mean(~np.isin(y_pred, classes_true))) if len(y_pred) else float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal = balanced_accuracy_score(y_true, y_pred) if len(classes_true) > 1 else float("nan")
    return {
        f"{prefix}_acc": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_bal_acc": float(bal),
        f"{prefix}_f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=classes_union, zero_division=0)),
        f"{prefix}_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=classes_union, zero_division=0)),
        f"{prefix}_present_f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=classes_true, zero_division=0)),
        f"{prefix}_absent_pred_rate": absent_pred_rate,
        f"{prefix}_n_test": int(len(y_true)),
        f"{prefix}_n_classes_train": int(len(classes_train)),
        f"{prefix}_n_classes_test": int(len(classes_true)),
        f"{prefix}_n_classes_union": int(len(classes_union)),
    }


# -----------------------------------------------------------------------------
# Main marker-aware hook
# -----------------------------------------------------------------------------
def marker_rd_papa_hook(model, sbj: str, subjects: List[str], _train_loader, _test_loader, device: str, args, sbj_dir: Path):
    if not bool(getattr(args, "eval_marker_rd_papa", True)):
        return []
    train_loader, test_loader, train_ids, test_ids = build_marker_loaders_with_subject_ids(sbj, subjects, args)
    train_pack = collect_marker_rd_features(model, train_loader, device, args, sample_subjects=train_ids)
    test_pack = collect_marker_rd_features(model, test_loader, device, args, sample_subjects=test_ids)
    y_train = train_pack["y"].astype(int)
    y_test = test_pack["y"].astype(int)
    classes_train = np.array(sorted(np.unique(y_train).tolist()), dtype=int)
    if len(classes_train) < 1 or len(y_test) == 0:
        return []
    blocks, quality, feature_meta = build_marker_rd_feature_blocks(train_pack, test_pack, args)
    global_classes = classes_train
    gate, gate_meta = train_preference_gate(blocks, y_train, np.asarray(train_pack.get("subject_ids", np.full(len(y_train), "__all__", dtype=object)), dtype=object), quality["train"], global_classes, args)
    final_proba, weights, pred_by_expert, conf_by_expert = fit_outer_experts(blocks, y_train, global_classes, quality["test"], gate, args)

    if bool(getattr(args, "mard_hmm_smooth", False)) and final_proba.shape[1] > 1:
        idx = dyn.viterbi_smooth(np.log(np.clip(final_proba, 1e-12, 1.0)), stay_prob=dyn.effective_hmm_stay(final_proba, args))
        y_pred = global_classes[idx].astype(int)
        effective_stay = float(dyn.effective_hmm_stay(final_proba, args))
    else:
        y_pred = global_classes[np.argmax(final_proba, axis=1)].astype(int)
        effective_stay = float("nan")
    final_conf = final_proba.max(axis=1).astype(np.float32)
    metrics = safe_classification_metrics(y_test, y_pred, classes_train, prefix="marker_rd")

    out = sbj_dir / "marker_rd_papa"
    out.mkdir(parents=True, exist_ok=True)
    trace: Dict[str, Any] = {
        "subject_id": test_pack.get("subject_ids", np.asarray([sbj] * len(y_test), dtype=object)).astype(str),
        "window_idx": np.arange(len(y_test), dtype=int),
        "y_true": y_test.astype(int),
        "y_pred": y_pred.astype(int),
        "final_conf": final_conf,
        "rr_head_stft_abs_diff": test_pack["x_resp_static"][:, 2] if test_pack["x_resp_static"].shape[1] > 2 else np.nan,
        "resp_band_entropy": test_pack["x_resp_static"][:, 4] if test_pack["x_resp_static"].shape[1] > 4 else np.nan,
        "imu_motion_energy": test_pack["x_activity_static"][:, 12] if test_pack["x_activity_static"].shape[1] > 12 else np.nan,
        "imu_stationarity": test_pack["x_activity_static"][:, 13] if test_pack["x_activity_static"].shape[1] > 13 else np.nan,
        "marker_proxy_uncertainty": quality["marker_unc_test"].reshape(-1),
        "marker_proxy_motion_energy": quality["marker_proxy_test"][:, -5] if quality["marker_proxy_test"].shape[1] >= 5 else np.nan,
        "marker_proxy_stationarity": quality["marker_proxy_test"][:, -4] if quality["marker_proxy_test"].shape[1] >= 4 else np.nan,
        "marker_available_target": test_pack["marker_valid"].astype(int),
    }
    for ei, name in enumerate(blocks.keys()):
        trace[f"weight_{name}"] = weights[:, ei]
        trace[f"pred_{name}"] = pred_by_expert[name]
        trace[f"conf_{name}"] = conf_by_expert[name]
    pd.DataFrame(trace).to_csv(out / "marker_rd_preference_trace.csv", index=False)
    pd.DataFrame({"y_true": y_test.astype(int), "y_pred": y_pred.astype(int), "final_conf": final_conf}).to_csv(out / "marker_rd_predictions.csv", index=False)

    pref_row: Dict[str, Any] = {
        "__summary_name__": "marker_rd_papa_preference_summary",
        "subject": sbj,
        "tag": "marker_rd_papa",
        "gate_available": bool(gate_meta.get("gate_available", False)),
        "gate_rows": int(gate_meta.get("gate_rows", 0)),
        "gate_pos_rate": float(gate_meta.get("gate_pos_rate", np.nan)),
        "n_experts": int(len(blocks)),
    }
    motion = test_pack["x_activity_static"][:, 12] if test_pack["x_activity_static"].shape[1] > 12 else np.zeros(len(y_test))
    high_mask = motion >= np.nanmedian(motion)
    low_mask = ~high_mask
    for ei, name in enumerate(blocks.keys()):
        pref_row[f"mean_weight_{name}"] = float(np.nanmean(weights[:, ei]))
        pref_row[f"low_motion_weight_{name}"] = float(np.nanmean(weights[low_mask, ei])) if low_mask.any() else float("nan")
        pref_row[f"high_motion_weight_{name}"] = float(np.nanmean(weights[high_mask, ei])) if high_mask.any() else float("nan")
    pd.DataFrame([pref_row]).drop(columns=["__summary_name__"], errors="ignore").to_csv(out / "marker_rd_subject_preference_summary.csv", index=False)

    motion_row: Dict[str, Any] = {
        "__summary_name__": "marker_rd_papa_motion_summary",
        "subject": sbj,
        "tag": "marker_rd_papa",
        "target_marker_valid_rate": float(np.mean(test_pack["marker_valid"])),
        "source_marker_valid_rate": float(np.mean(train_pack["marker_valid"])),
        "marker_proxy_available": bool(feature_meta.get("marker_proxy_available", False)),
        "marker_proxy_valid_rows": int(feature_meta.get("marker_proxy_valid_rows", 0)),
        "marker_proxy_valid_subjects": int(feature_meta.get("marker_proxy_valid_subjects", 0)),
        "marker_proxy_unc_mean": float(np.nanmean(quality["marker_unc_test"])),
        "marker_proxy_unc_std": float(np.nanstd(quality["marker_unc_test"])),
        "marker_proxy_motion_mean": float(np.nanmean(quality["marker_proxy_test"][:, -5])) if quality["marker_proxy_test"].shape[1] >= 5 else float("nan"),
        "marker_proxy_stationarity_mean": float(np.nanmean(quality["marker_proxy_test"][:, -4])) if quality["marker_proxy_test"].shape[1] >= 4 else float("nan"),
    }
    pd.DataFrame([motion_row]).drop(columns=["__summary_name__"], errors="ignore").to_csv(out / "marker_rd_motion_summary.csv", index=False)

    feature_meta.update({
        "subject": sbj,
        "classes": classes_train.tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "resp_static_feature_names": dyn.RESP_DYN_STATIC_FEATURE_NAMES,
        "activity_feature_names": ACTIVITY_FEATURE_NAMES,
        "marker_feature_names": MARKER_FEATURE_NAMES,
        "interaction_feature_names": INTERACTION_FEATURE_NAMES,
        "motion_interaction_feature_names": MOTION_INTERACTION_FEATURE_NAMES,
        "quality_columns": [
            "rr_head_stft_abs_diff", "resp_band_entropy", "resp_band_peakness", "resp_band_bandwidth_bpm", "rr_spectral_uncertainty",
            "imu_motion_energy", "imu_stationarity", "imu_motion_entropy", "marker_proxy_uncertainty", "marker_valid_frac",
            "marker_motion_energy", "marker_stationarity", "marker_posture_drift", "marker_motion_entropy", "marker_burstiness",
        ],
        "gate_meta": gate_meta,
        "main_prediction_uses_target_marker": False,
        "marker_role": "source marker supervises an IMU/RR-derived motion/posture proxy and reliability gate; target marker is ignored outside oracle audit",
        "mard_cuda_device": str(getattr(args, "mard_cuda_device", "auto")),
        "mard_ridge_backend": str(getattr(args, "mard_ridge_backend", "torch")),
        "mard_gate_backend": str(getattr(args, "mard_gate_backend", "torch")),
        "mard_expert_classifier": str(getattr(args, "mard_expert_classifier", "torch_logreg")),
        "mard_tf32_enabled": bool(getattr(args, "mard_enable_tf32", True)),
    })
    with open(out / "marker_rd_feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    row: Dict[str, Any] = {
        "__summary_name__": "marker_rd_papa_summary",
        "subject": sbj,
        "tag": "marker_rd_papa",
        "marker_rd_resp_variant": str(feature_meta.get("resp_variant", args.mard_resp_variant)),
        "marker_rd_n_train": int(len(y_train)),
        "marker_rd_n_test": int(len(y_test)),
        "marker_rd_n_classes_train": int(len(classes_train)),
        "marker_rd_n_classes_test": int(len(np.unique(y_test))),
        "marker_rd_mean_conf": float(np.nanmean(final_conf)),
        "marker_rd_hmm_smooth": bool(getattr(args, "mard_hmm_smooth", False)),
        "marker_rd_effective_hmm_stay": effective_stay,
        "marker_rd_use_marker_proxy_blocks": bool(getattr(args, "mard_use_marker_proxy_blocks", False)),
        "marker_rd_uses_target_marker": False,
        **metrics,
    }

    oracle_rows = []
    if bool(getattr(args, "mard_oracle_audit", True)) and int(np.sum(train_pack["marker_valid"])) >= int(args.mard_min_marker_rows) and int(np.sum(test_pack["marker_valid"])) >= int(args.mard_min_marker_target_oracle):
        try:
            clf = SafeProbClassifier(kind=args.mard_expert_classifier, C=args.mard_logreg_c, max_iter=args.mard_logreg_max_iter, args=args)
            tr_mask = train_pack["marker_valid"]
            te_mask = test_pack["marker_valid"]
            clf.fit(train_pack["x_marker"][tr_mask], y_train[tr_mask])
            p = clf.predict_proba(test_pack["x_marker"][te_mask])
            p_aligned = align_proba(p, clf.classes_, classes_train)
            y_oracle = classes_train[np.argmax(p_aligned, axis=1)].astype(int)
            om = safe_classification_metrics(y_test[te_mask], y_oracle, classes_train, prefix="marker_oracle")
            oracle_row = {
                "__summary_name__": "marker_rd_papa_oracle_summary",
                "subject": sbj,
                "tag": "marker_oracle_audit",
                "marker_oracle_n_train": int(tr_mask.sum()),
                "marker_oracle_n_test": int(te_mask.sum()),
                "marker_oracle_uses_target_marker": True,
                **om,
            }
            oracle_rows.append(oracle_row)
            pd.DataFrame({"y_true": y_test[te_mask], "y_pred": y_oracle, "oracle_conf": p_aligned.max(axis=1)}).to_csv(out / "marker_rd_marker_oracle_predictions.csv", index=False)
        except Exception as exc:
            print(f"[MARKER_RD] Marker oracle audit failed for {sbj}: {exc}")

    print(
        f"MARKER_RD {sbj}: acc={row['marker_rd_acc']:.4f} bal={row['marker_rd_bal_acc']:.4f} "
        f"macro={row['marker_rd_f1_macro']:.4f} conf={row['marker_rd_mean_conf']:.3f} "
        f"gate={'yes' if gate_meta.get('gate_available', False) else 'no'} "
        f"marker_proxy={'yes' if feature_meta.get('marker_proxy_available', False) else 'no'}"
    )
    return [row, pref_row, motion_row, *oracle_rows]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def add_marker_rd_args(parser) -> None:
    parser.add_argument("--eval-marker-rd-papa", dest="eval_marker_rd_papa", action="store_true", default=True)
    parser.add_argument("--no-eval-marker-rd-papa", dest="eval_marker_rd_papa", action="store_false")

    # Names used by run_marker_rd_papa_ablation_grid.sh.  These are aliases for
    # the mard_* arguments below so older and newer run scripts both work.
    parser.add_argument("--include-marker", dest="include_marker", action="store_true", default=True)
    parser.add_argument("--no-include-marker", dest="include_marker", action="store_false")
    parser.add_argument("--marker-source-only", action="store_true", default=True, help="Compatibility flag; main prediction never uses target marker.")
    parser.add_argument("--marker-mode", default="quality_gate", choices=[
        "none", "imu_activity_gate", "teacher_motion_state", "quality_gate", "posture_profile_gate", "posture_profile_expert",
        "oracle_audit", "marker_only_oracle", "shuffled_teacher_control", "time_shift_control", "time_only_control",
        "low_motion_oracle", "high_motion_stress_test",
    ])
    parser.add_argument("--marker-resp-variant", dest="mard_resp_variant", default="dyn_hybrid", help="Alias for --mard-resp-variant.")
    parser.add_argument("--marker-expert-classifier", dest="mard_expert_classifier", default="torch_logreg", choices=["logreg", "lda", "torch_logreg"])
    parser.add_argument("--marker-gate-temperature", dest="mard_gate_temperature", type=float, default=0.75)
    parser.add_argument("--marker-teacher-alpha", dest="mard_marker_proxy_alpha", type=float, default=10.0)
    parser.add_argument("--marker-profile-alpha", type=float, default=10.0, help="Reserved for future posture-profile regression; accepted for grid compatibility.")
    parser.add_argument("--marker-quality-q", type=float, default=0.75, help="Accepted for grid compatibility; quality thresholds are emitted in summaries.")
    parser.add_argument("--marker-low-motion-q", type=float, default=0.25)
    parser.add_argument("--marker-high-motion-q", type=float, default=0.75)
    parser.add_argument("--marker-min-valid", dest="mard_min_marker_rows", type=int, default=50)
    parser.add_argument("--marker-motion-conditioned-hmm", dest="mard_hmm_smooth", action="store_true", default=False)

    # Native MA-RD-PAPA names.
    parser.add_argument("--mard-resp-variant", dest="mard_resp_variant", help=argparse.SUPPRESS)
    parser.add_argument("--mard-expert-classifier", dest="mard_expert_classifier", choices=["logreg", "lda", "torch_logreg"], help=argparse.SUPPRESS)
    parser.add_argument("--mard-logreg-c", type=float, default=1.0)
    parser.add_argument("--mard-logreg-max-iter", type=int, default=1500)
    parser.add_argument("--mard-marker-proxy-alpha", dest="mard_marker_proxy_alpha", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mard-marker-source-mode", default="inferred", choices=["inferred", "observed"])
    parser.add_argument("--mard-marker-min-subjects", type=int, default=3)
    parser.add_argument("--mard-min-marker-rows", dest="mard_min_marker_rows", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mard-min-marker-target-oracle", type=int, default=20)
    parser.add_argument("--mard-allow-missing-marker", dest="mard_allow_missing_marker", action="store_true", default=True)
    parser.add_argument("--mard-require-marker", dest="mard_allow_missing_marker", action="store_false")
    parser.add_argument("--mard-use-marker-proxy-blocks", action="store_true", default=False, help="Add proxy-conditioned marker blocks as classifier features. Default keeps marker as gate/quality only.")
    parser.add_argument("--mard-use-preference-gate", dest="mard_use_preference_gate", action="store_true", default=True)
    parser.add_argument("--mard-no-preference-gate", dest="mard_use_preference_gate", action="store_false")
    parser.add_argument("--mard-min-pseudo-windows", type=int, default=5)
    parser.add_argument("--mard-min-gate-rows", type=int, default=50)
    parser.add_argument("--mard-gate-c", type=float, default=1.0)
    parser.add_argument("--mard-gate-max-iter", type=int, default=1500)
    parser.add_argument("--mard-gate-temperature", dest="mard_gate_temperature", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mard-hmm-smooth", dest="mard_hmm_smooth", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mard-oracle-audit", dest="mard_oracle_audit", action="store_true", default=True)
    parser.add_argument("--mard-no-oracle-audit", dest="mard_oracle_audit", action="store_false")

    # CUDA-heavy post-hoc model components.  The backbone already trains on the
    # selected device via the shared core runner; these switches move the marker
    # proxy regression, expert classifiers, and preference gate onto torch/CUDA.
    parser.add_argument("--mard-ridge-backend", default="torch", choices=["torch", "sklearn"], help="Backend for AR->marker proxy ridge regressions.")
    parser.add_argument("--mard-gate-backend", default="torch", choices=["torch", "sklearn"], help="Backend for the preference gate.")
    parser.add_argument("--mard-cuda-device", default="auto", help="Device for torch post-hoc components: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--mard-strict-cuda", action="store_true", help="Raise instead of falling back to sklearn/CPU when a torch backend fails.")
    parser.add_argument("--mard-torch-ridge-float64", action="store_true", help="Use float64 for torch ridge solves; slower but sometimes more stable.")
    parser.add_argument("--mard-torch-clf-epochs", type=int, default=200)
    parser.add_argument("--mard-torch-clf-lr", type=float, default=1e-2)
    parser.add_argument("--mard-torch-clf-batch-size", type=int, default=4096)
    parser.add_argument("--mard-torch-clf-weight-decay", type=float, default=0.0)
    parser.add_argument("--mard-torch-gate-epochs", type=int, default=200)
    parser.add_argument("--mard-torch-gate-lr", type=float, default=1e-2)
    parser.add_argument("--mard-torch-gate-batch-size", type=int, default=8192)
    parser.add_argument("--mard-torch-gate-weight-decay", type=float, default=0.0)
    parser.add_argument("--mard-num-workers", type=int, default=4, help="DataLoader workers for marker downstream feature collection.")
    parser.add_argument("--mard-prefetch-factor", type=int, default=2)
    parser.add_argument("--mard-pin-memory", dest="mard_pin_memory", action="store_true", default=True)
    parser.add_argument("--mard-no-pin-memory", dest="mard_pin_memory", action="store_false")
    parser.add_argument("--mard-persistent-workers", dest="mard_persistent_workers", action="store_true", default=True)
    parser.add_argument("--mard-no-persistent-workers", dest="mard_persistent_workers", action="store_false")
    parser.add_argument("--mard-enable-tf32", dest="mard_enable_tf32", action="store_true", default=True, help="Enable TF32 matmul/cuDNN for CUDA post-hoc torch components.")
    parser.add_argument("--mard-disable-tf32", dest="mard_enable_tf32", action="store_false", help="Disable TF32 matmul/cuDNN for stricter reproducibility.")

    parser.add_argument("--imu-fs", type=float, default=float(IMU_FS))
    parser.add_argument("--marker-fs", type=float, default=float(MARKER_FS))


def add_papa_dyn_compatible_args(parser) -> None:
    # Frozen embedding evaluation.
    parser.add_argument("--eval-frozen-embeddings", action="store_true")
    parser.add_argument("--eval-frozen-tlx", action="store_true")
    parser.add_argument("--tlx-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--embed-data-group", default=None, choices=["mr", "level", "levels", "mr_levels"])
    parser.add_argument("--embed-labels", default="L0,L2,L3")
    parser.add_argument("--embed-classifier", default="linear", choices=["lda", "logreg", "linear"])
    parser.add_argument("--embed-pooling", default="rich", choices=["mean", "max", "cls_last", "mean_std", "mean_std_max", "rich"])
    parser.add_argument("--embed-stft-profile", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--linear-probe-epochs", type=int, default=30)
    parser.add_argument("--linear-probe-lr", type=float, default=1e-3)
    parser.add_argument("--linear-probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--linear-probe-batch-size", type=int, default=64)

    # Static PAPA branch.
    parser.add_argument("--eval-papa", action="store_true")
    parser.add_argument("--papa-state-dim", type=int, default=48)
    parser.add_argument("--papa-adapter-init-scale", type=float, default=0.05)
    parser.add_argument("--papa-no-bottleneck", action="store_true")
    parser.add_argument("--papa-no-adapter", action="store_true")
    parser.add_argument("--papa-tta", default="none", choices=["none", "tent", "nrc", "cotta", "papa"])
    parser.add_argument("--papa-epochs", type=int, default=3)
    parser.add_argument("--papa-lr", type=float, default=5e-4)
    parser.add_argument("--papa-weight-decay", type=float, default=0.0)
    parser.add_argument("--papa-temperature", type=float, default=1.0)
    parser.add_argument("--papa-nrc-k", type=int, default=5)
    parser.add_argument("--lambda-papa-align", type=float, default=0.20)
    parser.add_argument("--lambda-papa-proto", type=float, default=0.20)
    parser.add_argument("--lambda-papa-entropy", type=float, default=0.05)
    parser.add_argument("--lambda-papa-diversity", type=float, default=0.05)
    parser.add_argument("--lambda-papa-nrc", type=float, default=0.10)
    parser.add_argument("--lambda-papa-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-resp-rr", type=float, default=0.05)
    parser.add_argument("--lambda-resp-recon", type=float, default=0.01)

    # Respiratory dynamics branch.
    parser.add_argument("--eval-resp-dyn", action="store_true")
    parser.add_argument("--resp-dyn-ladder", default="all")
    parser.add_argument("--resp-dyn-classifier", default="lda", choices=["lda", "logreg"])
    parser.add_argument("--resp-dyn-logreg-c", type=float, default=1.0)
    parser.add_argument("--resp-dyn-logreg-max-iter", type=int, default=1000)
    parser.add_argument("--resp-dyn-fs", type=float, default=float(BR_FS))
    parser.add_argument("--resp-dyn-min-hz", type=float, default=0.05)
    parser.add_argument("--resp-dyn-max-hz", type=float, default=0.75)
    parser.add_argument("--resp-dyn-roll-win", type=int, default=7)
    parser.add_argument("--resp-dyn-baseline-label", default="L0")
    parser.add_argument("--resp-dyn-target-baseline-q", type=float, default=0.20)
    parser.add_argument("--resp-dyn-source-baseline-q", type=float, default=0.20)
    parser.add_argument("--resp-dyn-source-baseline-mode", choices=["subject", "global"], default="subject")
    parser.add_argument("--resp-dyn-scale-floor", type=float, default=1e-3)
    parser.add_argument("--resp-dyn-boundary-jump-z", type=float, default=8.0)
    parser.add_argument("--resp-dyn-hmm-stay", type=float, default=0.75)
    parser.add_argument("--resp-dyn-hmm-min-stay", type=float, default=0.50)
    parser.add_argument("--resp-dyn-hmm-adaptive", dest="resp_dyn_hmm_adaptive", action="store_true", default=True)
    parser.add_argument("--resp-dyn-no-hmm-adaptive", dest="resp_dyn_hmm_adaptive", action="store_false")
    parser.add_argument("--resp-dyn-collect-state", action="store_true", default=True)
    parser.add_argument("--no-resp-dyn-collect-state", dest="resp_dyn_collect_state", action="store_false")
    parser.add_argument("--resp-dyn-centered-roll", action="store_true", default=True)
    parser.add_argument("--no-resp-dyn-centered-roll", dest="resp_dyn_centered_roll", action="store_false")
    parser.add_argument("--resp-dyn-source-segment-by-label", action="store_true", default=True)
    parser.add_argument("--no-resp-dyn-source-segment-by-label", dest="resp_dyn_source_segment_by_label", action="store_false")
    parser.add_argument("--resp-dyn-also-frozen", action="store_true")


def finalize_args_marker_rd(args) -> None:
    dyn.finalize_args_dyn(args)
    # Defensive defaults for aliased argparse destinations.
    if getattr(args, "mard_resp_variant", None) is None:
        args.mard_resp_variant = "dyn_hybrid"
    if getattr(args, "mard_expert_classifier", None) is None:
        args.mard_expert_classifier = "logreg"
    if getattr(args, "mard_marker_proxy_alpha", None) is None:
        args.mard_marker_proxy_alpha = 10.0
    if getattr(args, "mard_min_marker_rows", None) is None:
        args.mard_min_marker_rows = 50
    if getattr(args, "mard_gate_temperature", None) is None:
        args.mard_gate_temperature = 0.75

    # CRA-CUDA style speed controls for Ampere+ GPUs. This only affects torch/CUDA
    # operations; sklearn fallbacks are unchanged.
    if torch.cuda.is_available():
        enable_tf32 = bool(getattr(args, "mard_enable_tf32", True))
        try:
            torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32

    # If the launcher passes --device cuda:N and leaves --mard-cuda-device auto,
    # bind post-hoc torch ridge/logreg/gate work to that same GPU.
    if str(getattr(args, "mard_cuda_device", "auto")).lower() == "auto":
        args.mard_cuda_device = str(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    mode = str(getattr(args, "marker_mode", "quality_gate"))
    if mode in {"teacher_motion_state", "posture_profile_expert", "shuffled_teacher_control", "time_shift_control", "time_only_control"}:
        args.mard_use_marker_proxy_blocks = True
    if mode in {"oracle_audit", "marker_only_oracle", "low_motion_oracle", "high_motion_stress_test"}:
        args.mard_oracle_audit = True
    if mode == "none":
        args.mard_use_marker_proxy_blocks = False

    if bool(getattr(args, "eval_marker_rd_papa", True)):
        args.eval_frozen_embeddings = bool(args.eval_frozen_embeddings)


def main() -> None:
    parser = core.build_base_parser(
        dyn.SUBJECTS,
        str(Path(SBJ_PROCESSED_DIR) / "vit_pressure_crossmodal_marker_rd_papa"),
    )
    add_papa_dyn_compatible_args(parser)
    add_marker_rd_args(parser)
    args = parser.parse_args()
    core.run_loocv_experiment(
        args,
        post_eval_hooks=[
            dyn.frozen_embedding_hook,
            papa.papa_hook,
            dyn.resp_dyn_hook,
            marker_rd_papa_hook,
        ],
        config_mutator=finalize_args_marker_rd,
    )


if __name__ == "__main__":
    main()

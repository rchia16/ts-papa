#!/usr/bin/env python3
"""
CRA-PAPA: Cardio-Respiratory-Activity preference modelling for IMU MWL.

This script is intentionally a small wrapper around the existing PAPA/PAPA-dyn
training code.  It does NOT claim that head IMU can reconstruct ECG.  Instead,
ECG is treated as a privileged, training-time physiological view that can shape
and audit a test-time IMU-only workload model.

Pipeline per LOSO fold
----------------------
1. Reuse the existing IMU -> reconstructed pressure STFT + RR PAPA model.
2. Collect window-level learned respiratory dynamics from the reconstructed
   pressure STFT/RR head, using the same feature extractor as PAPA-dyn.
3. Collect explicit IMU activity features from the raw IMU window.
4. Collect ECG summary features when the dataloader exposes ECG.  Missing ECG is
   allowed and is masked, not imputed as a hard label.
5. Train a source-pool ECG->MWL oracle expert and an IMU/respiration/activity
   -> ECG-feature proxy.  The oracle is saved only as an audit.  The proxy is an
   uncertainty-aware auxiliary feature available at test time.
6. Train multiple simple experts on source subjects:
      - respiratory dynamics
      - IMU activity dynamics
      - respiration x activity interactions
      - ECG-proxy/autonomic latent
      - fused multimodal latent
      - optional learned PAPA respiration state
7. Inside the LOSO source pool, run an inner subject-held-out reliability audit
   to learn a preference gate.  The gate learns which expert tends to be correct
   under different quality/activity/cardio-proxy conditions.
8. Apply the learned gate to the true held-out subject using IMU only.  ECG from
   the held-out subject is ignored for the main prediction, even if present; it
   is used only for optional oracle auditing.

Missing data/classes
--------------------
- Missing ECG: all ECG-derived losses/experts are skipped or backed off to zero
  proxy features with high uncertainty.
- Missing labels/classes for a subject: subjects/classes are filtered per fold;
  classifiers are trained on the available source classes.  Metrics report the
  train/test/union class counts so impossible target classes are visible rather
  than crashing the run.
- One-class source blocks or tiny class counts: the corresponding expert falls
  back to a constant majority-class predictor instead of failing.

Typical command
---------------
python vit_pressure_crossmodal_cra_papa.py \
  --data-str imu_filt \
  --data-group mr \
  --epochs 20 \
  --batch-size 64 \
  --embed-labels L0,L2,L3 \
  --eval-cra-papa \
  --out-dir runs/cra_papa/$(date -u +%Y%m%dT%H%M%SZ)

Outputs per held-out subject
----------------------------
<out>/<subject>/cra_papa/
  cra_predictions.csv
  cra_preference_trace.csv
  cra_subject_preference_summary.csv
  cra_feature_meta.json
  cra_ecg_oracle_predictions.csv       # only when source+target ECG are usable

Top-level summary files are also emitted by the shared core runner:
  summary.csv
  cra_papa_summary.csv
  cra_papa_preference_summary.csv
  cra_papa_ecg_oracle_summary.csv
"""
from __future__ import annotations

import inspect
import json
import math
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import vit_pressure_crossmodal_papa_dyn as dyn
from config import ECG_FS, IMU_FS, SBJ_PROCESSED_DIR
from dataloader import load_data, make_dataset

# Import through PAPA-dyn so we reuse the existing monkey-patched PAPA model and
# core training path.
papa = dyn.papa
core = dyn.core


# -----------------------------------------------------------------------------
# Robust batch handling
# -----------------------------------------------------------------------------
def _as_tensor(x: Any, *, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _looks_like_per_window_scalar(x: Any, batch_size: int) -> bool:
    """Heuristic for distinguishing TLX-like vectors from ECG arrays."""
    if x is None:
        return False
    try:
        arr = np.asarray(x)
    except Exception:
        return False
    if arr.shape[:1] != (batch_size,):
        return False
    return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)


def _split_tlx_ecg_from_extras(extras: Sequence[Any], batch_size: int) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Parse optional tensors after (imu, pressure, cond, br).

    The updated project loaders may expose ECG in a few forms.  This parser is
    intentionally tolerant:
      len==5 old code: fifth item is usually TLX.
      len==5 new code: fifth item may be ECG; if it is not scalar-like, treat it
                       as ECG.
      len>=6: choose the first scalar-like item as TLX and the first non-scalar
              item as ECG.
    """
    tlx = None
    ecg = None
    for item in extras:
        if item is None:
            continue
        if _looks_like_per_window_scalar(item, batch_size) and tlx is None:
            tlx = item
        elif ecg is None:
            ecg = item
        elif tlx is None:
            # Last resort: keep the first unassigned extra as TLX.  This is only
            # used for bookkeeping; training ignores TLX unless the user asks for
            # TLX probes elsewhere.
            tlx = item
    return tlx, ecg


def unpack_batch_optional_ecg(batch: Iterable[Any] | Dict[str, Any], device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Return (imu, pressure, cond, br, tlx, ecg) from old or new dataloader batches.

    This is used by CRA-PAPA feature collection.  A separate monkey patch below
    lets the original core/PAPA code keep using a five-value unpacker while safely
    ignoring ECG if it appears in batches.
    """
    if isinstance(batch, dict):
        imu = batch.get("imu", batch.get("x", batch.get("past_values")))
        pressure = batch.get("pressure", batch.get("pss", batch.get("y")))
        cond = batch.get("cond", batch.get("condition", batch.get("conds")))
        br = batch.get("br", batch.get("rr", batch.get("breathing_rate")))
        tlx = batch.get("tlx", batch.get("nasa_tlx"))
        ecg = batch.get("ecg", batch.get("ecg_feat", batch.get("hr", batch.get("ibi_feat"))))
    else:
        items = list(batch)
        if len(items) < 4:
            raise ValueError(f"Expected at least 4 batch items, got {len(items)}")
        imu, pressure, cond, br = items[:4]
        batch_size = int(np.asarray(cond).shape[0]) if not torch.is_tensor(cond) else int(cond.shape[0])
        tlx, ecg = _split_tlx_ecg_from_extras(items[4:], batch_size=batch_size)

    imu_t = _as_tensor(imu, device=device, dtype=torch.float32)
    pressure_t = _as_tensor(pressure, device=device, dtype=torch.float32)
    if pressure_t is not None and pressure_t.ndim == 3 and pressure_t.size(-1) == 1:
        pressure_t = pressure_t.squeeze(-1)
    cond_t = _as_tensor(cond, device=device)
    br_t = _as_tensor(br, device=device, dtype=torch.float32)
    tlx_t = _as_tensor(tlx, device=device, dtype=torch.float32) if tlx is not None else None
    ecg_t = _as_tensor(ecg, device=device, dtype=torch.float32) if ecg is not None else None

    if imu_t is None or pressure_t is None or cond_t is None or br_t is None:
        raise ValueError("Could not unpack imu/pressure/condition/br from batch.")
    return imu_t, pressure_t, cond_t, br_t, tlx_t, ecg_t


# Let the inherited core/PAPA training code tolerate an extra ECG item in batches.
def _core_unpack_ignore_ecg(batch: Iterable[Any] | Dict[str, Any], device: str):
    imu, pressure, cond, br, tlx, _ecg = unpack_batch_optional_ecg(batch, device)
    return imu, pressure, cond, br, tlx


core.unpack_batch = _core_unpack_ignore_ecg


# -----------------------------------------------------------------------------
# Label/load helpers
# -----------------------------------------------------------------------------
VALID_MWL_LABELS = ("M", "R", "L0", "L1", "L2", "L3")


def parse_mwl_labels(x: str | Sequence[str]) -> List[str]:
    if isinstance(x, (list, tuple, np.ndarray)):
        labels = [str(v).strip().upper() for v in x if str(v).strip()]
    else:
        labels = [v.strip().upper() for v in str(x).split(",") if v.strip()]
    bad = [v for v in labels if v not in VALID_MWL_LABELS]
    if bad:
        raise ValueError(f"Unsupported labels {bad}; expected one of {VALID_MWL_LABELS}")
    if not labels:
        raise ValueError("No labels requested. Use --embed-labels L0,L1,L2,L3 etc.")
    return labels


def label_id(label: str) -> int:
    return {"M": 0, "R": 1, "L0": 2, "L1": 3, "L2": 4, "L3": 5}[str(label).strip().upper()]


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


def _call_make_dataset(data_list: List[Dict[str, Any]], data_group: str, args):
    """Call project make_dataset while tolerating old/new include_ecg signatures."""
    include_tlx = bool(getattr(args, "include_tlx", False) or getattr(args, "eval_frozen_tlx", False))
    base_kwargs = dict(
        label_encoder_dir=args.data_dir,
        data_group=data_group,
        include_tlx=include_tlx,
        tlx_csv_path=getattr(args, "tlx_csv_path", None),
    )
    attempts = []
    # Newer loaders may use one of these names; old loaders will reject them.
    attempts.append({**base_kwargs, "include_ecg": True})
    attempts.append({**base_kwargs, "return_ecg": True})
    attempts.append({**base_kwargs, "include_ecg": True, "return_ecg": True})
    attempts.append(base_kwargs)

    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return make_dataset(data_list, args.data_str, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def _parse_make_dataset_output(out: Any, n_expected: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse make_dataset output into x, pressure, br, cond, tlx, ecg.

    The existing loader returns x, pressure, br, cond and optionally TLX.  The
    updated loader may append ECG in addition to or instead of TLX.  This parser
    keeps the first four fields fixed and infers the rest by shape.
    """
    if isinstance(out, dict):
        x = out.get("x", out.get("imu", out.get("past_values")))
        pressure = out.get("pressure", out.get("pss", out.get("y")))
        br = out.get("br", out.get("rr"))
        cond = out.get("cond", out.get("conds", out.get("condition")))
        tlx = out.get("tlx", out.get("nasa_tlx"))
        ecg = out.get("ecg", out.get("ecg_feat", out.get("hr", out.get("ibi_feat"))))
    else:
        items = list(out)
        if len(items) < 4:
            raise ValueError(f"make_dataset returned {len(items)} items; expected >=4")
        x, pressure, br, cond = items[:4]
        batch_size = int(np.asarray(cond).shape[0])
        tlx, ecg = _split_tlx_ecg_from_extras(items[4:], batch_size=batch_size)

    if x is None or pressure is None or br is None or cond is None:
        raise ValueError("Could not parse x/pressure/br/cond from make_dataset output")
    n = len(np.asarray(cond))
    if n_expected is not None and n != int(n_expected):
        raise ValueError(f"Parsed dataset has {n} rows; expected {n_expected}")
    return np.asarray(x), np.asarray(pressure), np.asarray(br), np.asarray(cond), (None if tlx is None else np.asarray(tlx)), (None if ecg is None else np.asarray(ecg))


class WindowDatasetWithECG(Dataset):
    """Small local dataset so CRA-PAPA controls the tuple order."""

    def __init__(self, x: np.ndarray, pressure: np.ndarray, cond: np.ndarray, br: np.ndarray, tlx: Optional[np.ndarray], ecg: Optional[np.ndarray]):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        # Match the existing project convention: IMU should be (B,T,C).  If the
        # channel axis appears to be in the middle, transpose it.
        if self.x.ndim == 3 and self.x.shape[1] < self.x.shape[2]:
            self.x = self.x.permute(0, 2, 1)
        self.pressure = torch.as_tensor(pressure, dtype=torch.float32)
        self.cond = torch.as_tensor(cond, dtype=torch.long)
        self.br = torch.as_tensor(br, dtype=torch.float32)
        n = self.x.shape[0]
        if tlx is None:
            self.tlx = torch.full((n,), float("nan"), dtype=torch.float32)
        else:
            self.tlx = torch.as_tensor(tlx, dtype=torch.float32).reshape(n, -1).squeeze(-1)
        if ecg is None:
            self.ecg = torch.empty((n, 0), dtype=torch.float32)
        else:
            arr = np.asarray(ecg)
            if arr.shape[:1] != (n,):
                raise ValueError(f"ECG array first dimension {arr.shape[:1]} does not match n={n}")
            self.ecg = torch.as_tensor(arr, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.pressure[idx], self.cond[idx], self.br[idx], self.tlx[idx], self.ecg[idx]


def build_cra_loaders_with_subject_ids(subject: str, subjects: List[str], args) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Build downstream loaders and one subject id per window.

    Unlike the original PAPA-dyn builder, this function keeps optional ECG arrays
    and never assumes every subject has every requested class.  Subjects with no
    matching windows for a group are skipped; a target subject only raises if it
    has no matching windows at all.
    """
    keep_labels = parse_mwl_labels(getattr(args, "embed_labels", []))
    keep_ids = np.asarray([label_id(lbl) for lbl in keep_labels], dtype=int)
    groups = grouped_labels(keep_labels)

    def append_subject(xs, ps, brs, conds, tlxs, ecgs, ids, raw: Dict[str, Any], group: str, sid: str) -> None:
        out = _call_make_dataset([raw], group, args)
        x, pressure, br, cond, tlx, ecg = _parse_make_dataset_output(out)
        cond = np.asarray(cond, dtype=int).reshape(-1)
        # The levels label encoder usually maps L0-L3 to 0-3.  Convert to the
        # canonical six-class id used throughout the existing scripts.
        if group == "levels" and cond.size and int(np.nanmax(cond)) <= 3:
            cond = cond + 2
        mask = np.isin(cond, keep_ids)
        if not mask.any():
            return
        xs.append(x[mask])
        ps.append(pressure[mask])
        brs.append(br[mask])
        conds.append(cond[mask])
        tlxs.append(None if tlx is None else np.asarray(tlx[mask], dtype=np.float32).reshape(int(mask.sum()), -1).squeeze(-1))
        if ecg is None:
            ecgs.append(None)
        else:
            ecg_arr = np.asarray(ecg[mask], dtype=np.float32)
            # Flatten ECG/raw-HR/feature tensors to a fixed per-window vector so
            # subjects with equivalent ECG content but different singleton axes
            # can still be concatenated.
            ecgs.append(ecg_arr.reshape(ecg_arr.shape[0], -1))
        ids.extend([_subject_from_dict(raw, sid)] * int(mask.sum()))

    def build_split(split_subjects: List[str]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        Optional[np.ndarray], Optional[np.ndarray], np.ndarray
    ]:
        xs: List[np.ndarray] = []
        ps: List[np.ndarray] = []
        brs: List[np.ndarray] = []
        conds: List[np.ndarray] = []
        tlxs: List[Optional[np.ndarray]] = []
        ecgs: List[Optional[np.ndarray]] = []
        ids: List[str] = []

        for group, labels in groups.items():
            keep_values = {str(lbl).strip().upper() for lbl in labels}
            for sid in split_subjects:
                raw = load_data(
                    sid, data_dir=args.data_dir, data_group=group
                )
                filt = _filter_subject_dict_by_labels(raw, keep_values)
                if filt is None:
                    continue
                append_subject(
                    xs, ps, brs, conds, tlxs, ecgs, ids, filt, group, 
                    sid)

        if not xs:
            raise RuntimeError(
                f"No requested labels {keep_labels} found for split "\
                f"subjects {split_subjects}"
            )

        def concat_optional(
            parts: List[Optional[np.ndarray]]
        ) -> Optional[np.ndarray]:
            good = [p for p in parts if p is not None]
            if not good:
                return None
            n_parts = [len(a) for a in xs]
            # TLX is usually 1-D.  ECG may be 2-D and, depending on preprocessing,
            # may have slightly different feature lengths.  Flatten ECG-like arrays
            # and right-pad to the maximum feature length rather than failing.
            if any(np.asarray(p).ndim > 1 for p in good):
                max_d = max(int(np.asarray(p).reshape(np.asarray(p).shape[0], -1).shape[1]) for p in good)
                filled_2d = []
                for p, n_part in zip(parts, n_parts):
                    if p is None:
                        filled_2d.append(np.full((n_part, max_d), np.nan, dtype=np.float32))
                    else:
                        arr = np.asarray(p, dtype=np.float32).reshape(np.asarray(p).shape[0], -1)
                        if arr.shape[1] < max_d:
                            pad = np.full((arr.shape[0], max_d - arr.shape[1]), np.nan, dtype=np.float32)
                            arr = np.concatenate([arr, pad], axis=1)
                        elif arr.shape[1] > max_d:
                            arr = arr[:, :max_d]
                        filled_2d.append(arr)
                return np.concatenate(filled_2d, axis=0)

            filled_1d = []
            for p, n_part in zip(parts, n_parts):
                if p is None:
                    filled_1d.append(np.full((n_part,), np.nan, dtype=np.float32))
                else:
                    filled_1d.append(np.asarray(p, dtype=np.float32).reshape(-1))
            return np.concatenate(filled_1d, axis=0)

        return (
            np.concatenate(xs, axis=0),
            np.concatenate(ps, axis=0),
            np.concatenate(brs, axis=0),
            np.concatenate(conds, axis=0),
            concat_optional(tlxs),
            concat_optional(ecgs),
            np.asarray(ids, dtype=object),
        )

    train_subjects = [s for s in subjects if s != subject]
    xtr, ptr, brtr, ytr, tlxtr, ecgtr, idtr = build_split(train_subjects)
    xte, pte, brte, yte, tlxte, ecgte, idte = build_split([subject])

    train_ds = WindowDatasetWithECG(xtr, ptr, ytr, brtr, tlxtr, ecgtr)
    test_ds = WindowDatasetWithECG(xte, pte, yte, brte, tlxte, ecgte)

    batch_size = int(getattr(args, "embed_batch_size", getattr(args, "batch_size", 128)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, test_loader, idtr, idte


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
ACTIVITY_FEATURE_NAMES = [
    "acc_norm_mean", "acc_norm_std", "acc_norm_iqr", "acc_norm_energy", "acc_jerk_energy", "acc_burstiness",
    "gyro_norm_mean", "gyro_norm_std", "gyro_norm_iqr", "gyro_norm_energy", "gyro_jerk_energy", "gyro_burstiness",
    "motion_energy", "stationarity", "acc_dom_freq", "acc_spec_entropy", "acc_spec_peakness", "acc_spec_bandwidth",
    "gyro_dom_freq", "gyro_spec_entropy", "gyro_spec_peakness", "gyro_spec_bandwidth", "acc_low_frac", "acc_high_frac",
    "gyro_low_frac", "gyro_high_frac",
]

ECG_FEATURE_NAMES = [
    "ecg_mean", "ecg_std", "ecg_iqr", "ecg_rms", "ecg_abs_mean", "ecg_diff_std", "ecg_diff_abs_mean", "ecg_line_length",
    "ecg_zero_cross_rate", "ecg_hr_band_log_energy", "ecg_hr_band_entropy", "ecg_hr_band_peakness", "ecg_hr_centroid_bpm",
    "ecg_hr_bandwidth_bpm", "ecg_low_hr_frac", "ecg_high_hr_frac",
]

INTERACTION_FEATURE_NAMES = [
    "rr_disagree_x_motion_energy", "resp_entropy_x_motion_entropy", "resp_peakness_x_stationarity", "bandwidth_x_gyro_entropy",
    "token_rr_std_x_motion_energy", "rr_shift_per_motion", "band_energy_x_acc_energy", "high_resp_frac_x_gyro_peakness",
]


def _ensure_btc(imu: torch.Tensor) -> torch.Tensor:
    if imu.ndim != 3:
        raise ValueError(f"Expected IMU tensor (B,T,C) or (B,C,T), got {tuple(imu.shape)}")
    # Usually T >> C.  If second dim is channels, transpose to (B,T,C).
    if imu.size(1) <= 16 and imu.size(2) > imu.size(1):
        imu = imu.transpose(1, 2)
    return imu


def _iqr_torch(x: torch.Tensor, dim: int) -> torch.Tensor:
    q75 = torch.quantile(x, 0.75, dim=dim)
    q25 = torch.quantile(x, 0.25, dim=dim)
    return q75 - q25


def _norm_stat_features(norm: torch.Tensor) -> List[torch.Tensor]:
    mean = norm.mean(dim=1)
    std = norm.std(dim=1, unbiased=False)
    iqr = _iqr_torch(norm, dim=1)
    energy = torch.log1p((norm.pow(2)).mean(dim=1))
    if norm.size(1) > 1:
        jerk_energy = torch.log1p(torch.diff(norm, dim=1).pow(2).mean(dim=1))
    else:
        jerk_energy = torch.zeros_like(mean)
    burstiness = norm.amax(dim=1) / norm.mean(dim=1).abs().clamp_min(1e-6)
    return [mean, std, iqr, energy, jerk_energy, burstiness]


def _spectral_features_1d(x: torch.Tensor, fs: float, min_hz: float = 0.05, max_hz: float = 6.0) -> List[torch.Tensor]:
    B, T = x.shape
    if T < 8:
        z = x.new_zeros(B)
        return [z, z, z, z, z, z]
    x = x - x.mean(dim=1, keepdim=True)
    spec = torch.fft.rfft(x, dim=1).abs().pow(2)
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(fs)).to(x.device)
    mask = (freqs >= min_hz) & (freqs <= min(max_hz, fs / 2.0))
    if int(mask.sum().item()) < 3:
        mask = torch.ones_like(freqs, dtype=torch.bool)
    f = freqs[mask]
    s = spec[:, mask].clamp_min(1e-8)
    p = s / s.sum(dim=1, keepdim=True).clamp_min(1e-8)
    centroid = (p * f.view(1, -1)).sum(dim=1)
    entropy = -(p * p.log()).sum(dim=1) / math.log(max(2, p.size(1)))
    peakness = p.max(dim=1).values
    bandwidth = torch.sqrt((p * (f.view(1, -1) - centroid.view(-1, 1)).pow(2)).sum(dim=1))
    low_mask = f <= torch.quantile(f, 0.33)
    high_mask = f >= torch.quantile(f, 0.67)
    low_frac = p[:, low_mask].sum(dim=1)
    high_frac = p[:, high_mask].sum(dim=1)
    return [centroid, entropy, peakness, bandwidth, low_frac, high_frac]


@torch.no_grad()
def imu_activity_features(imu: torch.Tensor, fs: float = float(IMU_FS)) -> torch.Tensor:
    """Compact activity state from the same IMU window used by the model."""
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


@torch.no_grad()
def ecg_summary_features(ecg: Optional[torch.Tensor], fs: float = float(ECG_FS), max_passthrough: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fixed-size ECG physiology summary.

    The feature extractor handles raw ECG windows, ECG-derived vectors, HR/IBI
    features, or missing ECG.  It returns (features, valid_mask).  The model never
    reconstructs ECG waveforms; these features are only privileged training-time
    physiology and optional oracle diagnostics.
    """
    if ecg is None or ecg.numel() == 0:
        # Batch size cannot be inferred from a missing tensor here; callers pass
        # an empty tensor with shape (B,0) when ECG is missing.
        B = 0 if ecg is None else int(ecg.shape[0])
        return torch.zeros(B, len(ECG_FEATURE_NAMES), device=(ecg.device if ecg is not None else "cpu")), torch.zeros(B, dtype=torch.bool, device=(ecg.device if ecg is not None else "cpu"))

    B = int(ecg.shape[0])
    flat = ecg.reshape(B, -1).float()
    finite = torch.isfinite(flat)
    finite_frac = finite.float().mean(dim=1)
    clean = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    mean = clean.mean(dim=1)
    std = clean.std(dim=1, unbiased=False)
    iqr = _iqr_torch(clean, dim=1) if clean.size(1) > 1 else torch.zeros_like(mean)
    rms = torch.sqrt(clean.pow(2).mean(dim=1).clamp_min(0.0))
    abs_mean = clean.abs().mean(dim=1)
    if clean.size(1) > 1:
        diff = torch.diff(clean, dim=1)
        diff_std = diff.std(dim=1, unbiased=False)
        diff_abs_mean = diff.abs().mean(dim=1)
        line_length = diff.abs().sum(dim=1) / max(1, clean.size(1) - 1)
        zc = ((clean[:, 1:] >= 0) != (clean[:, :-1] >= 0)).float().mean(dim=1)
    else:
        diff_std = torch.zeros_like(mean)
        diff_abs_mean = torch.zeros_like(mean)
        line_length = torch.zeros_like(mean)
        zc = torch.zeros_like(mean)

    if clean.size(1) >= 16:
        x = clean - clean.mean(dim=1, keepdim=True)
        spec = torch.fft.rfft(x, dim=1).abs().pow(2)
        freqs = torch.fft.rfftfreq(clean.size(1), d=1.0 / float(fs)).to(clean.device)
        # Broad cardiac band.  Works for raw ECG; for low-dimensional ECG feature
        # vectors, this block is skipped above.
        mask = (freqs >= 0.50) & (freqs <= min(3.50, fs / 2.0))
        if int(mask.sum().item()) < 3:
            mask = torch.ones_like(freqs, dtype=torch.bool)
        f = freqs[mask]
        s = spec[:, mask].clamp_min(1e-8)
        p = s / s.sum(dim=1, keepdim=True).clamp_min(1e-8)
        centroid_bpm = (p * f.view(1, -1)).sum(dim=1) * 60.0
        entropy = -(p * p.log()).sum(dim=1) / math.log(max(2, p.size(1)))
        peakness = p.max(dim=1).values
        bandwidth_bpm = torch.sqrt((p * (f.view(1, -1) - centroid_bpm.view(-1, 1) / 60.0).pow(2)).sum(dim=1)) * 60.0
        low_mask = f <= 1.00
        high_mask = f >= 2.00
        if not bool(low_mask.any()):
            low_mask = f <= torch.median(f)
        if not bool(high_mask.any()):
            high_mask = f >= torch.median(f)
        low_frac = p[:, low_mask].sum(dim=1)
        high_frac = p[:, high_mask].sum(dim=1)
        log_energy = torch.log1p(s.sum(dim=1))
    else:
        log_energy = torch.zeros_like(mean)
        entropy = torch.zeros_like(mean)
        peakness = torch.zeros_like(mean)
        centroid_bpm = torch.zeros_like(mean)
        bandwidth_bpm = torch.zeros_like(mean)
        low_frac = torch.zeros_like(mean)
        high_frac = torch.zeros_like(mean)

    valid = (finite_frac > 0.50) & ((abs_mean > 1e-8) | (std > 1e-8))
    feats = torch.stack(
        [
            mean, std, iqr, rms, abs_mean, diff_std, diff_abs_mean, line_length,
            zc, log_energy, entropy, peakness, centroid_bpm, bandwidth_bpm,
            low_frac, high_frac,
        ],
        dim=1,
    )
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), valid


def interaction_features(resp_static: np.ndarray, activity_static: np.ndarray) -> np.ndarray:
    """Small hand-built block of physiologically meaningful interactions."""
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

    feats = np.stack(
        [
            rr_disagree * motion_energy,
            resp_entropy * motion_entropy,
            resp_peakness * stationarity,
            bandwidth * gyro_entropy,
            token_rr_std * motion_energy,
            rr_disagree / (1.0 + motion_energy),
            band_energy * acc_energy,
            high_frac * gyro_peak,
        ],
        axis=1,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@torch.no_grad()
def collect_cra_static_features(model, loader, device: str, args, sample_subjects: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    model.eval()
    resp, act, ecg_feats, ecg_valids, y, rr, subjects, states = [], [], [], [], [], [], [], []
    offset = 0
    sample_subjects_arr = None if sample_subjects is None else np.asarray(sample_subjects, dtype=object)

    for batch in loader:
        imu, _pressure, cond, _br, _tlx, ecg = unpack_batch_optional_ecg(batch, device)
        pred_logmag, rr_pred, hidden = model(imu)

        rfeat = dyn.respiratory_stft_features(
            pred_logmag=pred_logmag,
            rr_pred=rr_pred,
            fs=float(args.resp_dyn_fs),
            min_hz=float(args.resp_dyn_min_hz),
            max_hz=float(args.resp_dyn_max_hz),
        )
        afeat = imu_activity_features(imu, fs=float(args.imu_fs))
        efeat, evalid = ecg_summary_features(ecg, fs=float(args.ecg_fs))

        n = int(cond.numel())
        resp.append(rfeat.detach().cpu().numpy())
        act.append(afeat.detach().cpu().numpy())
        ecg_feats.append(efeat.detach().cpu().numpy())
        ecg_valids.append(evalid.detach().cpu().numpy().astype(bool))
        y.append(cond.detach().cpu().numpy().astype(int).reshape(-1))
        rr.append(rr_pred.detach().cpu().numpy().reshape(-1))

        if sample_subjects_arr is not None:
            subjects.append(sample_subjects_arr[offset : offset + n])
            offset += n

        if hasattr(model, "respiration_state_from_outputs"):
            try:
                st = model.respiration_state_from_outputs(pred_logmag, rr_pred, hidden, adapt=False)
                states.append(st.detach().cpu().numpy())
            except Exception as exc:
                print(f"[CRA] Could not collect PAPA state: {exc}")
                states = []

    if not resp:
        raise RuntimeError("No batches available for CRA feature collection.")

    out: Dict[str, np.ndarray] = {
        "x_resp_static": np.nan_to_num(np.concatenate(resp, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "x_activity_static": np.nan_to_num(np.concatenate(act, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "x_ecg": np.nan_to_num(np.concatenate(ecg_feats, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "ecg_valid": np.concatenate(ecg_valids, axis=0).astype(bool),
        "y": np.concatenate(y, axis=0).astype(int),
        "rr_pred": np.concatenate(rr, axis=0).astype(np.float32),
    }
    if subjects:
        out["subject_ids"] = np.concatenate(subjects, axis=0)
    if states:
        out["papa_state"] = np.nan_to_num(np.concatenate(states, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


# -----------------------------------------------------------------------------
# Feature block construction
# -----------------------------------------------------------------------------
def _robust_scale_np(x: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(x, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(x - med.reshape(1, -1)), axis=0)
    sd = np.nanstd(x, axis=0)
    scale = np.where(mad > floor, mad, sd)
    return np.nan_to_num(scale, nan=floor, posinf=floor, neginf=floor).clip(min=floor).astype(np.float32)


def subject_robust_z(x: np.ndarray, subject_ids: Optional[np.ndarray], floor: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    if subject_ids is None:
        subject_ids = np.full(len(x), "__all__", dtype=object)
    for sid in pd.unique(subject_ids):
        mask = np.asarray(subject_ids, dtype=object) == sid
        xs = x[mask]
        med = np.nanmedian(xs, axis=0)
        sc = _robust_scale_np(xs, floor=floor)
        out[mask] = (xs - med.reshape(1, -1)) / sc.reshape(1, -1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def concat_features(*parts: Optional[np.ndarray]) -> np.ndarray:
    xs = [np.asarray(p, dtype=np.float32) for p in parts if p is not None and np.asarray(p).size > 0]
    if not xs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.nan_to_num(np.concatenate(xs, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def make_segments(subject_ids: Optional[np.ndarray], y: Optional[np.ndarray] = None, by_label: bool = False) -> Optional[np.ndarray]:
    if subject_ids is None:
        return None
    sid = np.asarray(subject_ids, dtype=object)
    if by_label and y is not None:
        return np.asarray([f"{s}:{int(c)}" for s, c in zip(sid, y)], dtype=object)
    return sid


def fit_ecg_proxy(x_ar_train: np.ndarray, ecg_train: np.ndarray, valid_train: np.ndarray, x_ar_test: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Learn AR -> ECG-feature proxy on source ECG-valid windows.

    Returns proxy_train, proxy_unc_train, proxy_test, proxy_unc_test, metadata.
    Uncertainty is a simple combination of source residual scale and distance to
    the valid source AR cloud.
    """
    n_train = x_ar_train.shape[0]
    n_test = x_ar_test.shape[0]
    d_ecg = ecg_train.shape[1]
    valid = np.asarray(valid_train, dtype=bool)
    min_valid = int(getattr(args, "cra_min_ecg_valid", 20))

    if int(valid.sum()) < min_valid or len(np.unique(valid)) < 2 and int(valid.sum()) == 0:
        zeros_train = np.zeros((n_train, d_ecg), dtype=np.float32)
        zeros_test = np.zeros((n_test, d_ecg), dtype=np.float32)
        unc_train = np.ones((n_train, 1), dtype=np.float32)
        unc_test = np.ones((n_test, 1), dtype=np.float32)
        return zeros_train, unc_train, zeros_test, unc_test, {"ecg_proxy_available": False, "ecg_valid_source": int(valid.sum())}

    model = make_pipeline(StandardScaler(), Ridge(alpha=float(args.cra_ecg_proxy_alpha)))
    model.fit(x_ar_train[valid], ecg_train[valid])
    ptr = model.predict(x_ar_train).astype(np.float32)
    pte = model.predict(x_ar_test).astype(np.float32)

    resid = np.sqrt(np.mean((ptr[valid] - ecg_train[valid]) ** 2, axis=1)) if int(valid.sum()) else np.asarray([1.0])
    resid_scale = float(np.nanmedian(resid) + np.nanstd(resid)) if resid.size else 1.0
    resid_scale = max(resid_scale, 1e-6)

    scaler = StandardScaler().fit(x_ar_train[valid])
    z_valid = scaler.transform(x_ar_train[valid])
    mu = z_valid.mean(axis=0, keepdims=True)

    def dist_unc(x: np.ndarray) -> np.ndarray:
        z = scaler.transform(x)
        dist = np.sqrt(np.mean((z - mu) ** 2, axis=1, keepdims=True))
        # Bound the uncertainty to keep the gate numerically stable.
        return np.clip(resid_scale * (1.0 + dist), 0.0, 100.0).astype(np.float32)

    unc_train = dist_unc(x_ar_train)
    unc_test = dist_unc(x_ar_test)
    meta = {
        "ecg_proxy_available": True,
        "ecg_valid_source": int(valid.sum()),
        "ecg_proxy_resid_median": float(np.nanmedian(resid)),
        "ecg_proxy_resid_scale": float(resid_scale),
    }
    return ptr, unc_train, pte, unc_test, meta


def build_cra_feature_blocks(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray], args) -> Tuple[OrderedDict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, np.ndarray], Dict[str, Any]]:
    y_train = train["y"].astype(int)
    train_ids = train.get("subject_ids")
    test_ids = test.get("subject_ids")
    scale_floor = float(args.resp_dyn_scale_floor)

    # Respiratory dynamics are exactly the learned reconstruction pathway used in
    # PAPA-dyn: IMU -> pressure STFT/RR -> morphology/dynamics/baseline shifts.
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
    resp_variant = str(args.cra_resp_variant)
    if resp_variant not in ladder:
        print(f"[CRA] Requested resp variant {resp_variant!r} not available; using dyn_hybrid.")
        resp_variant = "dyn_hybrid" if "dyn_hybrid" in ladder else sorted(ladder.keys())[0]
    x_resp_tr, x_resp_te = ladder[resp_variant]

    # Activity features use their own subject-wise robust baseline and rolling
    # dynamics.  This keeps motion/activity as a separate explanatory block.
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

    ecg_proxy_tr, ecg_unc_tr, ecg_proxy_te, ecg_unc_te, ecg_meta = fit_ecg_proxy(
        x_ar_tr,
        train["x_ecg"],
        train["ecg_valid"],
        x_ar_te,
        args,
    )

    blocks: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
    blocks["resp_dyn"] = (x_resp_tr, x_resp_te)
    blocks["activity_dyn"] = (x_act_tr, x_act_te)
    blocks["resp_activity"] = (concat_features(x_resp_tr, x_act_tr, x_int_tr), concat_features(x_resp_te, x_act_te, x_int_te))
    blocks["ecg_proxy"] = (concat_features(ecg_proxy_tr, ecg_unc_tr), concat_features(ecg_proxy_te, ecg_unc_te))
    blocks["fused"] = (concat_features(x_resp_tr, x_act_tr, x_int_tr, ecg_proxy_tr, ecg_unc_tr), concat_features(x_resp_te, x_act_te, x_int_te, ecg_proxy_te, ecg_unc_te))
    if "papa_state" in train and "papa_state" in test:
        blocks["papa_state"] = (train["papa_state"], test["papa_state"])

    # Quality context for the gate.  These values are never labels; they describe
    # when a feature block may be reliable or confounded.
    def quality(resp_static: np.ndarray, act_static: np.ndarray, ecg_unc: np.ndarray, ecg_valid: np.ndarray) -> np.ndarray:
        motion_energy = act_static[:, 12] if act_static.shape[1] > 12 else np.zeros(len(act_static))
        stationarity = act_static[:, 13] if act_static.shape[1] > 13 else np.ones(len(act_static))
        acc_entropy = act_static[:, 15] if act_static.shape[1] > 15 else np.zeros(len(act_static))
        gyro_entropy = act_static[:, 19] if act_static.shape[1] > 19 else np.zeros(len(act_static))
        cols = [
            resp_static[:, 2] if resp_static.shape[1] > 2 else np.zeros(len(resp_static)),
            resp_static[:, 4] if resp_static.shape[1] > 4 else np.zeros(len(resp_static)),
            resp_static[:, 5] if resp_static.shape[1] > 5 else np.zeros(len(resp_static)),
            resp_static[:, 6] if resp_static.shape[1] > 6 else np.zeros(len(resp_static)),
            resp_static[:, 12] if resp_static.shape[1] > 12 else np.zeros(len(resp_static)),
            motion_energy,
            stationarity,
            0.5 * (acc_entropy + gyro_entropy),
            ecg_unc.reshape(-1),
            ecg_valid.astype(np.float32),
        ]
        return np.stack(cols, axis=1).astype(np.float32)

    q_train = quality(train["x_resp_static"], train["x_activity_static"], ecg_unc_tr, train["ecg_valid"])
    q_test = quality(test["x_resp_static"], test["x_activity_static"], ecg_unc_te, test["ecg_valid"])

    q = {"train": q_train, "test": q_test, "ecg_proxy_train": ecg_proxy_tr, "ecg_proxy_test": ecg_proxy_te, "ecg_unc_train": ecg_unc_tr, "ecg_unc_test": ecg_unc_te}
    meta = {
        "resp_variant": resp_variant,
        "blocks": {k: {"train_dim": int(v[0].shape[1]), "test_dim": int(v[1].shape[1])} for k, v in blocks.items()},
        **ecg_meta,
    }
    return blocks, q, meta


# -----------------------------------------------------------------------------
# Safe classifiers, preference gate, and metrics
# -----------------------------------------------------------------------------
class SafeProbClassifier:
    """Predict-proba wrapper that backs off gracefully when classes are missing."""

    def __init__(self, kind: str = "logreg", C: float = 1.0, max_iter: int = 1000):
        self.kind = str(kind).lower()
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.classes_: np.ndarray = np.asarray([], dtype=int)
        self.constant_: Optional[int] = None
        self.model: Optional[BaseEstimator] = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.array(sorted(np.unique(y).tolist()), dtype=int)
        if len(self.classes_) <= 1:
            self.constant_ = int(self.classes_[0]) if len(self.classes_) else 0
            self.model = None
            return self

        try:
            if self.kind == "lda":
                # LDA can be fragile with tiny classes/high dimensions, so fall
                # back to logistic if any class has fewer than two windows.
                counts = np.bincount(y - y.min())
                if counts.min() < 2 or x.shape[0] <= len(self.classes_):
                    raise ValueError("Too few samples for LDA")
                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            else:
                clf = LogisticRegression(max_iter=self.max_iter, C=self.C, class_weight="balanced", multi_class="auto")
            self.model = make_pipeline(StandardScaler(), clf)
            self.model.fit(x, y)
            self.constant_ = None
        except Exception as exc:
            warnings.warn(f"Expert classifier failed ({exc}); using majority class fallback.")
            vals, counts = np.unique(y, return_counts=True)
            self.constant_ = int(vals[np.argmax(counts)])
            self.model = None
            self.classes_ = np.array([self.constant_], dtype=int)
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
    onehot = np.zeros((p.shape[0], n_experts), dtype=np.float32)
    onehot[:, expert_index] = 1.0
    return np.concatenate([conf[:, None], margin[:, None], entropy[:, None], quality.astype(np.float32), onehot], axis=1).astype(np.float32)


def softmax_np(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - np.nanmax(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=axis, keepdims=True).clip(min=1e-12)).astype(np.float32)


def train_preference_gate(blocks_train: OrderedDict[str, Tuple[np.ndarray, np.ndarray]], y_train: np.ndarray, subject_ids: np.ndarray, quality_train: np.ndarray, global_classes: np.ndarray, args) -> Tuple[Optional[BaseEstimator], Dict[str, Any]]:
    """
    Inner-source LOSO reliability model.

    For every pseudo-held-out source subject, each expert is trained on the other
    source subjects and evaluated on the pseudo subject.  The gate then learns
    which expert is correct from confidence + quality + expert id features.
    """
    rows: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sids = np.asarray(subject_ids, dtype=object)
    unique_sids = list(pd.unique(sids))
    expert_names = list(blocks_train.keys())
    n_experts = len(expert_names)

    for pseudo in unique_sids:
        val_mask = sids == pseudo
        tr_mask = ~val_mask
        if int(val_mask.sum()) < int(args.cra_min_pseudo_windows) or len(np.unique(y_train[tr_mask])) < 2:
            continue
        for ei, name in enumerate(expert_names):
            x_all = blocks_train[name][0]
            clf = SafeProbClassifier(kind=args.cra_expert_classifier, C=args.cra_logreg_c, max_iter=args.cra_logreg_max_iter)
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
    if len(np.unique(y)) < 2 or len(y) < int(args.cra_min_gate_rows):
        return None, {"gate_available": False, "gate_rows": int(len(y)), "gate_pos_rate": float(np.mean(y)) if len(y) else float("nan")}

    gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=int(args.cra_gate_max_iter), C=float(args.cra_gate_c), class_weight="balanced"),
    )
    gate.fit(X, y)
    return gate, {"gate_available": True, "gate_rows": int(len(y)), "gate_pos_rate": float(np.mean(y))}


def fit_outer_experts(blocks: OrderedDict[str, Tuple[np.ndarray, np.ndarray]], y_train: np.ndarray, global_classes: np.ndarray, quality_test: np.ndarray, gate: Optional[BaseEstimator], args) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    expert_names = list(blocks.keys())
    n_experts = len(expert_names)
    proba_list = []
    score_list = []
    pred_by_expert: Dict[str, np.ndarray] = {}
    conf_by_expert: Dict[str, np.ndarray] = {}

    for ei, name in enumerate(expert_names):
        xtr, xte = blocks[name]
        clf = SafeProbClassifier(kind=args.cra_expert_classifier, C=args.cra_logreg_c, max_iter=args.cra_logreg_max_iter)
        clf.fit(xtr, y_train)
        p = clf.predict_proba(xte)
        p_aligned = align_proba(p, clf.classes_, global_classes)
        meta = proba_meta_features(p_aligned, quality_test, ei, n_experts)
        if gate is not None:
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

    scores = np.stack(score_list, axis=1)  # (N,E)
    weights = softmax_np(np.log(scores.clip(min=1e-6)), axis=1, temperature=float(args.cra_gate_temperature))
    final = np.zeros_like(proba_list[0])
    for ei, p in enumerate(proba_list):
        final += weights[:, [ei]] * p
    return final.astype(np.float32), weights.astype(np.float32), pred_by_expert, conf_by_expert


def safe_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes_train: np.ndarray, prefix: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes_union = np.array(sorted(set(classes_train.astype(int).tolist()) | set(np.unique(y_true).astype(int).tolist())), dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        f"{prefix}_acc": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_bal_acc": float(bal),
        f"{prefix}_f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=classes_union, zero_division=0)),
        f"{prefix}_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=classes_union, zero_division=0)),
        f"{prefix}_n_test": int(len(y_true)),
        f"{prefix}_n_classes_train": int(len(classes_train)),
        f"{prefix}_n_classes_test": int(len(np.unique(y_true))),
        f"{prefix}_n_classes_union": int(len(classes_union)),
    }


# -----------------------------------------------------------------------------
# Main CRA-PAPA hook
# -----------------------------------------------------------------------------
def cra_papa_hook(model, sbj: str, subjects: List[str], _train_loader, _test_loader, device: str, args, sbj_dir: Path):
    if not bool(getattr(args, "eval_cra_papa", True)):
        return []

    train_loader, test_loader, train_ids, test_ids = build_cra_loaders_with_subject_ids(sbj, subjects, args)
    train_pack = collect_cra_static_features(model, train_loader, device, args, sample_subjects=train_ids)
    test_pack = collect_cra_static_features(model, test_loader, device, args, sample_subjects=test_ids)

    y_train = train_pack["y"].astype(int)
    y_test = test_pack["y"].astype(int)
    classes_train = np.array(sorted(np.unique(y_train).tolist()), dtype=int)
    classes_global = np.array(sorted(set(classes_train.tolist()) | set(np.unique(y_test).astype(int).tolist())), dtype=int)

    out = sbj_dir / "cra_papa"
    out.mkdir(parents=True, exist_ok=True)

    if len(classes_train) < 2:
        print(f"[CRA] Skipping {sbj}: source pool has <2 classes ({classes_train.tolist()})")
        return []

    blocks, quality, feature_meta = build_cra_feature_blocks(train_pack, test_pack, args)
    gate, gate_meta = train_preference_gate(blocks, y_train, train_pack.get("subject_ids", train_ids), quality["train"], classes_global, args)
    final_proba, weights, pred_by_expert, conf_by_expert = fit_outer_experts(blocks, y_train, classes_global, quality["test"], gate, args)
    y_pred = classes_global[np.argmax(final_proba, axis=1)].astype(int)
    final_conf = final_proba.max(axis=1).astype(np.float32)

    metrics = safe_classification_metrics(y_test, y_pred, classes_train, prefix="cra")
    row = {
        "__summary_name__": "cra_papa_summary",
        "subject": sbj,
        "tag": "cra_papa",
        "cra_n_train": int(len(y_train)),
        "cra_n_test": int(len(y_test)),
        "cra_n_source_subjects": int(len(np.unique(train_pack.get("subject_ids", train_ids)))),
        "cra_train_classes": json.dumps(classes_train.astype(int).tolist()),
        "cra_test_classes": json.dumps(np.unique(y_test).astype(int).tolist()),
        "cra_global_classes": json.dumps(classes_global.astype(int).tolist()),
        "cra_mean_conf": float(np.nanmean(final_conf)),
        "cra_ecg_valid_source": int(train_pack["ecg_valid"].sum()),
        "cra_ecg_valid_target": int(test_pack["ecg_valid"].sum()),
        **metrics,
        **gate_meta,
        **{k: v for k, v in feature_meta.items() if isinstance(v, (int, float, str, bool))},
    }

    # Per-window trace with the subject-specific preference weights.
    trace: Dict[str, Any] = {
        "subject_id": test_pack.get("subject_ids", np.asarray([sbj] * len(y_test), dtype=object)).astype(str),
        "window_idx": np.arange(len(y_test), dtype=int),
        "y_true": y_test.astype(int),
        "y_pred": y_pred.astype(int),
        "final_conf": final_conf,
        "rr_head_stft_abs_diff": test_pack["x_resp_static"][:, 2] if test_pack["x_resp_static"].shape[1] > 2 else np.nan,
        "resp_band_entropy": test_pack["x_resp_static"][:, 4] if test_pack["x_resp_static"].shape[1] > 4 else np.nan,
        "motion_energy": test_pack["x_activity_static"][:, 12] if test_pack["x_activity_static"].shape[1] > 12 else np.nan,
        "stationarity": test_pack["x_activity_static"][:, 13] if test_pack["x_activity_static"].shape[1] > 13 else np.nan,
        "ecg_proxy_uncertainty": quality["ecg_unc_test"].reshape(-1),
        "ecg_available_target": test_pack["ecg_valid"].astype(int),
    }
    for ei, name in enumerate(blocks.keys()):
        trace[f"weight_{name}"] = weights[:, ei]
        trace[f"pred_{name}"] = pred_by_expert[name]
        trace[f"conf_{name}"] = conf_by_expert[name]
    pd.DataFrame(trace).to_csv(out / "cra_preference_trace.csv", index=False)
    pd.DataFrame({"y_true": y_test.astype(int), "y_pred": y_pred.astype(int), "final_conf": final_conf}).to_csv(out / "cra_predictions.csv", index=False)

    # Subject preference summary overall and split by low/high motion.
    pref_row: Dict[str, Any] = {
        "__summary_name__": "cra_papa_preference_summary",
        "subject": sbj,
        "tag": "cra_papa_preference",
        "n_test": int(len(y_test)),
    }
    motion = np.asarray(trace["motion_energy"], dtype=np.float32)
    if np.all(np.isfinite(motion)) and len(motion) > 0:
        low_thr = float(np.nanquantile(motion, 0.33))
        high_thr = float(np.nanquantile(motion, 0.67))
        low_mask = motion <= low_thr
        high_mask = motion >= high_thr
    else:
        low_mask = np.zeros(len(y_test), dtype=bool)
        high_mask = np.zeros(len(y_test), dtype=bool)
    for ei, name in enumerate(blocks.keys()):
        pref_row[f"mean_weight_{name}"] = float(np.nanmean(weights[:, ei]))
        pref_row[f"low_motion_weight_{name}"] = float(np.nanmean(weights[low_mask, ei])) if low_mask.any() else float("nan")
        pref_row[f"high_motion_weight_{name}"] = float(np.nanmean(weights[high_mask, ei])) if high_mask.any() else float("nan")
    pd.DataFrame([pref_row]).drop(columns=["__summary_name__"], errors="ignore").to_csv(out / "cra_subject_preference_summary.csv", index=False)

    # Optional ECG oracle audit: this is not used in the main test-time model.
    oracle_rows = []
    if int(train_pack["ecg_valid"].sum()) >= int(args.cra_min_ecg_valid) and int(test_pack["ecg_valid"].sum()) >= max(2, int(args.cra_min_ecg_target_oracle)):
        vtr = train_pack["ecg_valid"]
        vte = test_pack["ecg_valid"]
        oracle = SafeProbClassifier(kind=args.cra_expert_classifier, C=args.cra_logreg_c, max_iter=args.cra_logreg_max_iter)
        oracle.fit(train_pack["x_ecg"][vtr], y_train[vtr])
        p = align_proba(oracle.predict_proba(test_pack["x_ecg"][vte]), oracle.classes_, classes_global)
        yp = classes_global[np.argmax(p, axis=1)].astype(int)
        oracle_metrics = safe_classification_metrics(y_test[vte], yp, np.unique(y_train[vtr]), prefix="cra_ecg_oracle")
        oracle_row = {
            "__summary_name__": "cra_papa_ecg_oracle_summary",
            "subject": sbj,
            "tag": "ecg_oracle_audit",
            **oracle_metrics,
        }
        oracle_rows.append(oracle_row)
        pd.DataFrame({"y_true": y_test[vte].astype(int), "y_pred": yp.astype(int), "oracle_conf": p.max(axis=1)}).to_csv(out / "cra_ecg_oracle_predictions.csv", index=False)

    meta = {
        "subject": sbj,
        "classes_train": classes_train.astype(int).tolist(),
        "classes_test": np.unique(y_test).astype(int).tolist(),
        "classes_global": classes_global.astype(int).tolist(),
        "resp_static_feature_names": dyn.RESP_DYN_STATIC_FEATURE_NAMES,
        "activity_feature_names": ACTIVITY_FEATURE_NAMES,
        "ecg_feature_names": ECG_FEATURE_NAMES,
        "interaction_feature_names": INTERACTION_FEATURE_NAMES,
        "quality_columns": [
            "rr_head_stft_abs_diff", "resp_band_entropy", "resp_band_peakness", "resp_band_bandwidth_bpm", "rr_spectral_uncertainty",
            "motion_energy", "stationarity", "motion_entropy", "ecg_proxy_uncertainty", "ecg_valid",
        ],
        "feature_meta": feature_meta,
        "gate_meta": gate_meta,
        "main_prediction_uses_target_ecg": False,
    }
    with open(out / "cra_feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"CRA_PAPA {sbj}: acc={row['cra_acc']:.4f} bal={row['cra_bal_acc']:.4f} "
        f"macro={row['cra_f1_macro']:.4f} conf={row['cra_mean_conf']:.3f} "
        f"gate={'yes' if gate is not None else 'fallback'} ecg_src={row['cra_ecg_valid_source']} ecg_tgt={row['cra_ecg_valid_target']}"
    )

    return [row, pref_row, *oracle_rows]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def add_common_downstream_args(parser) -> None:
    # Frozen embedding/PAPA args retained for comparison and because PAPA's
    # finalize_args expects these attributes to exist.
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

    # PAPA-dyn feature controls reused by CRA-PAPA.
    parser.add_argument("--resp-dyn-fs", type=float, default=18.0)
    parser.add_argument("--resp-dyn-min-hz", type=float, default=0.05)
    parser.add_argument("--resp-dyn-max-hz", type=float, default=0.75)
    parser.add_argument("--resp-dyn-roll-win", type=int, default=7)
    parser.add_argument("--resp-dyn-baseline-label", default="L0")
    parser.add_argument("--resp-dyn-target-baseline-q", type=float, default=0.20)
    parser.add_argument("--resp-dyn-source-baseline-q", type=float, default=0.20)
    parser.add_argument("--resp-dyn-source-baseline-mode", choices=["subject", "global"], default="subject")
    parser.add_argument("--resp-dyn-scale-floor", type=float, default=1e-3)
    parser.add_argument("--resp-dyn-boundary-jump-z", type=float, default=8.0)
    parser.add_argument("--resp-dyn-centered-roll", action="store_true", default=True)
    parser.add_argument("--no-resp-dyn-centered-roll", dest="resp_dyn_centered_roll", action="store_false")
    parser.add_argument("--resp-dyn-source-segment-by-label", action="store_true", default=True)
    parser.add_argument("--no-resp-dyn-source-segment-by-label", dest="resp_dyn_source_segment_by_label", action="store_false")
    # These exist only so dyn.resp_dyn_hook can be optionally included without a
    # missing-attribute failure.
    parser.add_argument("--eval-resp-dyn", action="store_true")
    parser.add_argument("--resp-dyn-ladder", default="all")
    parser.add_argument("--resp-dyn-classifier", default="lda", choices=["lda", "logreg"])
    parser.add_argument("--resp-dyn-logreg-c", type=float, default=1.0)
    parser.add_argument("--resp-dyn-logreg-max-iter", type=int, default=1000)
    parser.add_argument("--resp-dyn-hmm-stay", type=float, default=0.75)
    parser.add_argument("--resp-dyn-hmm-min-stay", type=float, default=0.50)
    parser.add_argument("--resp-dyn-hmm-adaptive", dest="resp_dyn_hmm_adaptive", action="store_true", default=True)
    parser.add_argument("--resp-dyn-no-hmm-adaptive", dest="resp_dyn_hmm_adaptive", action="store_false")
    parser.add_argument("--resp-dyn-collect-state", action="store_true", default=True)
    parser.add_argument("--no-resp-dyn-collect-state", dest="resp_dyn_collect_state", action="store_false")
    parser.add_argument("--resp-dyn-also-frozen", action="store_true")


def add_cra_args(parser) -> None:
    parser.add_argument("--eval-cra-papa", dest="eval_cra_papa", action="store_true", default=True)
    parser.add_argument("--no-eval-cra-papa", dest="eval_cra_papa", action="store_false")
    parser.add_argument("--cra-resp-variant", default="dyn_hybrid", help="PAPA-dyn ladder variant used as the respiratory expert input.")
    parser.add_argument("--cra-expert-classifier", default="logreg", choices=["logreg", "lda"])
    parser.add_argument("--cra-logreg-c", type=float, default=1.0)
    parser.add_argument("--cra-logreg-max-iter", type=int, default=1500)
    parser.add_argument("--cra-ecg-proxy-alpha", type=float, default=10.0)
    parser.add_argument("--cra-min-ecg-valid", type=int, default=20)
    parser.add_argument("--cra-min-ecg-target-oracle", type=int, default=5)
    parser.add_argument("--cra-min-pseudo-windows", type=int, default=5)
    parser.add_argument("--cra-min-gate-rows", type=int, default=50)
    parser.add_argument("--cra-gate-c", type=float, default=1.0)
    parser.add_argument("--cra-gate-max-iter", type=int, default=1500)
    parser.add_argument("--cra-gate-temperature", type=float, default=0.75)
    parser.add_argument("--imu-fs", type=float, default=float(IMU_FS))
    parser.add_argument("--ecg-fs", type=float, default=float(ECG_FS))


def finalize_args_cra(args) -> None:
    # Reuse existing PAPA model configuration exactly.  This sets the global class
    # defaults before the core runner instantiates the PAPA model.
    papa.finalize_args(args)
    args.embed_labels = parse_mwl_labels(args.embed_labels)
    if args.embed_data_group is None:
        # Match existing downstream grouping behavior.
        s = set(args.embed_labels)
        has_mr = bool(s & {"M", "R"})
        has_levels = bool(s & {"L0", "L1", "L2", "L3"})
        args.embed_data_group = "mr_levels" if has_mr and has_levels else ("mr" if has_mr else "levels")
    if bool(args.eval_cra_papa):
        # CRA-PAPA uses downstream condition loaders, but the main pretraining
        # remains exactly the existing pressure/RR reconstruction objective.
        args.include_tlx = bool(args.include_tlx or args.eval_frozen_tlx)


def main() -> None:
    parser = core.build_base_parser(
        dyn.SUBJECTS,
        str(Path(SBJ_PROCESSED_DIR) / "vit_pressure_crossmodal_cra_papa"),
    )
    add_common_downstream_args(parser)
    add_cra_args(parser)
    args = parser.parse_args()

    core.run_loocv_experiment(
        args,
        post_eval_hooks=[
            dyn.frozen_embedding_hook,
            papa.papa_hook,
            dyn.resp_dyn_hook,
            cra_papa_hook,
        ],
        config_mutator=finalize_args_cra,
    )


if __name__ == "__main__":
    main()

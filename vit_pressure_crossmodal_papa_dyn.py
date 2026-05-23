#!/usr/bin/env python3
"""
PAPA-dyn: reconstructed respiratory dynamics branch for IMU mental workload.

This is a wrapper around vit_pressure_crossmodal_papa.py.

It adds an evaluation/TTA branch that uses the learned reconstruction pathway:

    IMU -> reconstructed pressure STFT + RR head

to form respiratory dynamics features:

    RR
    STFT-derived RR
    RR disagreement
    respiratory-band energy
    spectral entropy
    peakness
    bandwidth
    rolling mean/std/slope/fast-slow dynamics
    subject baseline-normalized shifts

This is intentionally not another entropy-minimization adapter. The adaptation is:

    estimate target subject respiratory baseline from unlabeled low-drive windows
    compare baseline-normalized reconstructed respiratory dynamics to source classes
    optionally smooth the predicted workload state sequence with a simple HMM prior

Expected files next to this script:
  - vit_pressure_crossmodal_papa.py
  - vit_pressure_crossmodal_stft_rr_core.py
  - vit_pressure_crossmodal_stft_rr_tta_mwl_main.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import vit_pressure_crossmodal_papa as papa
from config import BR_FS, SBJ_PROCESSED_DIR
from dataloader import LoadDataset, load_data, make_dataset
from vit_pressure_crossmodal_stft_rr_mwl_tta_main import (
    SUBJECTS,
    frozen_embedding_hook,
    infer_embed_data_group_from_labels,
    parse_mwl_labels,
)

core = papa.core


# -----------------------------------------------------------------------------
# Label helpers
# -----------------------------------------------------------------------------
def label_id(name: str) -> int:
    name = str(name).strip().upper()
    mapping = {
        "M": 0,
        "R": 1,
        "L0": 2,
        "L1": 3,
        "L2": 4,
        "L3": 5,
    }
    if name not in mapping:
        raise ValueError(f"Unknown label name {name!r}. Expected one of {sorted(mapping)}")
    return mapping[name]



def _is_level_label(label: str) -> bool:
    return str(label).strip().upper() in {"L0", "L1", "L2", "L3"}


def _grouped_embed_labels(labels: List[str]) -> Dict[str, List[str]]:
    grouped = {"mr": [], "levels": []}
    for lbl in labels:
        key = "levels" if _is_level_label(lbl) else "mr"
        grouped[key].append(str(lbl).strip().upper())
    return {k: v for k, v in grouped.items() if v}


def _subject_from_dict(d: Dict, fallback: str = "") -> str:
    for key in ("subject", "subject_id", "sbj", "sid"):
        if key in d and str(d[key]).strip():
            return str(d[key]).strip()
    return str(fallback).strip()


def _filter_subject_dict_for_resp_dyn(
    d: Dict,
    keep_values: set[str],
    *,
    strict: bool = False,
) -> Optional[Dict]:
    """Filter one loaded subject dict to the labels used by the downstream split."""
    conds = np.asarray(d.get("conds"))
    if conds.size == 0:
        return d

    mask = np.asarray([str(c).strip().upper() in keep_values for c in conds], dtype=bool)
    if not mask.any():
        if strict:
            raise RuntimeError(
                f"No labels {sorted(keep_values)} found for subject "
                f"{d.get('subject', '<unknown>')} while building respiratory dynamics split."
            )
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


def _build_resp_dyn_loaders_with_subject_ids(
    subject: str,
    subjects: List[str],
    args,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Build the same downstream loaders as _build_frozen_embedding_loaders, but
    also return one subject id per window. The subject ids let PAPA-dyn compute
    source baselines and rolling dynamics without crossing source-subject
    boundaries.
    """
    keep_labels = parse_mwl_labels(getattr(args, "embed_labels", []))
    grouped = _grouped_embed_labels(keep_labels)
    include_tlx = bool(getattr(args, "eval_frozen_tlx", False))
    keep_ids = np.asarray([label_id(lbl) for lbl in keep_labels], dtype=int)

    def _append_one_subject(
        xs: List[np.ndarray],
        ys: List[np.ndarray],
        brs: List[np.ndarray],
        conds_out: List[np.ndarray],
        tlxs: List[np.ndarray],
        ids: List[str],
        d: Dict,
        group: str,
        fallback_subject: str,
    ) -> None:
        out = make_dataset(
            [d],
            args.data_str,
            label_encoder_dir=args.data_dir,
            data_group=group,
            include_tlx=include_tlx,
            tlx_csv_path=getattr(args, "tlx_csv_path", None),
        )
        if include_tlx:
            x, pressure, br, cond, tlx = out
        else:
            x, pressure, br, cond = out
            tlx = None

        cond = np.asarray(cond, dtype=int).reshape(-1)
        if group == "levels":
            cond = cond + 2

        mask = np.isin(cond, keep_ids)
        if not mask.any():
            return

        xs.append(x[mask])
        ys.append(pressure[mask])
        brs.append(br[mask])
        conds_out.append(cond[mask])
        sid = _subject_from_dict(d, fallback_subject)
        ids.extend([sid] * int(mask.sum()))
        if include_tlx and tlx is not None:
            tlxs.append(tlx[mask])

    def _build_split(split_subjects: List[str], *, strict_target: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        brs: List[np.ndarray] = []
        conds_out: List[np.ndarray] = []
        tlxs: List[np.ndarray] = []
        ids: List[str] = []

        for group, group_labels in grouped.items():
            keep_values = {str(lbl).strip().upper() for lbl in group_labels}
            for sid in split_subjects:
                raw = load_data(sid, data_dir=args.data_dir, data_group=group)
                filt = _filter_subject_dict_for_resp_dyn(
                    raw,
                    keep_values,
                    strict=strict_target,
                )
                if filt is None:
                    continue
                _append_one_subject(xs, ys, brs, conds_out, tlxs, ids, filt, group, sid)

        if not xs:
            raise RuntimeError(
                f"No labels {keep_labels} found while building respiratory dynamics split "
                f"for held-out subject {subject}."
            )

        tlx_arr = np.concatenate(tlxs, axis=0) if include_tlx and tlxs else None
        return (
            np.concatenate(xs, axis=0),
            np.concatenate(ys, axis=0),
            np.concatenate(brs, axis=0),
            np.concatenate(conds_out, axis=0),
            tlx_arr,
            np.asarray(ids, dtype=object),
        )

    train_subjects = [s for s in subjects if s != subject]
    x_train, y_train, br_train, cond_train, tlx_train, train_ids = _build_split(train_subjects, strict_target=False)
    x_test, y_test, br_test, cond_test, tlx_test, test_ids = _build_split([subject], strict_target=True)

    train_ds = (
        LoadDataset(x_train, y_train, cond_train, br_train, tlx_train, aug_ratio=0.0)
        if include_tlx and tlx_train is not None
        else LoadDataset(x_train, y_train, cond_train, br_train, aug_ratio=0.0)
    )
    test_ds = (
        LoadDataset(x_test, y_test, cond_test, br_test, tlx_test, aug_ratio=0.0)
        if include_tlx and tlx_test is not None
        else LoadDataset(x_test, y_test, cond_test, br_test, aug_ratio=0.0)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.embed_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.embed_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, test_loader, train_ids, test_ids


# -----------------------------------------------------------------------------
# Reconstructed respiratory dynamics features
# -----------------------------------------------------------------------------
RESP_DYN_STATIC_FEATURE_NAMES = [
    "rr_head",
    "rr_stft_centroid",
    "rr_head_stft_abs_diff",
    "resp_band_log_energy",
    "resp_band_entropy",
    "resp_band_peakness",
    "resp_band_bandwidth_bpm",
    "recon_stft_global_mean",
    "recon_stft_global_std",
    "recon_stft_global_max",
    "resp_band_flatness",
    "resp_band_peak_margin",
    "rr_spectral_uncertainty",
    "rr_stft_peak",
    "low_band_fraction",
    "high_band_fraction",
    "token_rr_std",
    "token_rr_slope",
    "token_entropy_std",
    "token_energy_std",
]


def respiratory_stft_features(
    pred_logmag: torch.Tensor,
    rr_pred: torch.Tensor,
    fs: float,
    min_hz: float = 0.05,
    max_hz: float = 0.75,
) -> torch.Tensor:
    """
    Extract respiratory morphology features from reconstructed pressure STFT.

    The key change versus the static PAPA branch is that the reconstructed
    pressure spectrum is treated as the TTA object.  Features include RR,
    STFT-derived RR, entropy/peak morphology, and compact uncertainty proxies.

    pred_logmag:
        Tensor shaped approximately (B, T_tokens, F_bins).
    rr_pred:
        Tensor shaped (B,) or (B, 1).

    Returns:
        Tensor shaped (B, len(RESP_DYN_STATIC_FEATURE_NAMES)).
    """
    if pred_logmag.ndim != 3:
        raise ValueError(f"Expected pred_logmag with shape (B,T,F), got {tuple(pred_logmag.shape)}")

    amp = torch.expm1(pred_logmag).clamp_min(0.0)
    spec = amp.mean(dim=1)  # (B, F)

    n_freq = spec.size(-1)
    n_fft = max(2, (n_freq - 1) * 2)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(fs)).to(spec.device)[:n_freq]

    mask = (freqs >= float(min_hz)) & (freqs <= float(max_hz))
    if int(mask.sum().item()) < 3:
        mask = torch.ones_like(freqs, dtype=torch.bool)

    f = freqs[mask]
    s = spec[:, mask].clamp_min(1e-8)
    p = s / s.sum(dim=1, keepdim=True).clamp_min(1e-8)

    centroid_hz = (p * f.view(1, -1)).sum(dim=1)
    rr_stft = centroid_hz * 60.0

    entropy = -(p * p.log()).sum(dim=1) / np.log(max(2, p.size(1)))
    peakness = p.max(dim=1).values
    rr_peak = f[p.argmax(dim=1)] * 60.0

    bandwidth = torch.sqrt(
        (p * (f.view(1, -1) - centroid_hz.view(-1, 1)).pow(2)).sum(dim=1)
    ) * 60.0

    band_energy = torch.log1p(s.sum(dim=1))

    rr_head = rr_pred.view(-1)
    rr_disagree = (rr_head - rr_stft).abs()

    global_mean = amp.mean(dim=(1, 2))
    global_std = amp.std(dim=(1, 2), unbiased=False)
    global_max = amp.amax(dim=(1, 2))

    flatness = torch.exp(torch.log(s).mean(dim=1)) / s.mean(dim=1).clamp_min(1e-8)
    if p.size(1) >= 2:
        top2 = torch.topk(p, k=2, dim=1).values
        peak_margin = top2[:, 0] - top2[:, 1]
    else:
        peak_margin = torch.zeros_like(peakness)

    low_mask = f <= max(float(min_hz), 0.20)
    high_mask = f >= min(float(max_hz), 0.33)
    if not bool(low_mask.any()):
        low_mask = f <= torch.median(f)
    if not bool(high_mask.any()):
        high_mask = f >= torch.median(f)
    low_frac = p[:, low_mask].sum(dim=1)
    high_frac = p[:, high_mask].sum(dim=1)

    # Within-window dynamics across reconstructed-STFT time tokens. These are
    # static per-window features, while add_rolling_dynamics() captures slower
    # changes across neighboring windows.
    st = amp[:, :, mask].clamp_min(1e-8)
    pt = st / st.sum(dim=2, keepdim=True).clamp_min(1e-8)
    centroid_t = (pt * f.view(1, 1, -1)).sum(dim=2) * 60.0
    entropy_t = -(pt * pt.log()).sum(dim=2) / np.log(max(2, pt.size(2)))
    energy_t = torch.log1p(st.sum(dim=2))
    token_rr_std = centroid_t.std(dim=1, unbiased=False)
    token_entropy_std = entropy_t.std(dim=1, unbiased=False)
    token_energy_std = energy_t.std(dim=1, unbiased=False)
    token_rr_slope = centroid_t[:, -1] - centroid_t[:, 0] if centroid_t.size(1) > 1 else torch.zeros_like(token_rr_std)

    rr_scale = 0.5 * (rr_head.abs() + rr_stft.abs()).clamp_min(1.0)
    rr_spectral_uncertainty = entropy * rr_disagree / rr_scale

    return torch.stack(
        [
            rr_head,
            rr_stft,
            rr_disagree,
            band_energy,
            entropy,
            peakness,
            bandwidth,
            global_mean,
            global_std,
            global_max,
            flatness,
            peak_margin,
            rr_spectral_uncertainty,
            rr_peak,
            low_frac,
            high_frac,
            token_rr_std,
            token_rr_slope,
            token_entropy_std,
            token_energy_std,
        ],
        dim=1,
    )



@torch.no_grad()
def collect_resp_dyn_static_features(
    model,
    loader,
    device: str,
    args,
    sample_subjects: Optional[np.ndarray] = None,
    collect_papa_state: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Collect reconstructed respiratory features for a loader.

    Returns a dictionary containing:
        x_static:       (N, D_resp)
        y:              (N,)
        rr_pred:        (N,)
        subject_ids:    (N,) object array when available
        papa_state:     (N, D_state) when requested and supported by the model
    """
    model.eval()

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    rrs: List[np.ndarray] = []
    subject_chunks: List[np.ndarray] = []
    papa_states: List[np.ndarray] = []

    sample_subjects_arr = None
    if sample_subjects is not None:
        sample_subjects_arr = np.asarray(sample_subjects, dtype=object)

    offset = 0
    for batch in loader:
        imu, _pressure, cond, _br, _tlx = core.unpack_batch(batch, device)
        pred_logmag, rr_pred, hidden = model(imu)

        feat = respiratory_stft_features(
            pred_logmag=pred_logmag,
            rr_pred=rr_pred,
            fs=float(args.resp_dyn_fs),
            min_hz=float(args.resp_dyn_min_hz),
            max_hz=float(args.resp_dyn_max_hz),
        )

        n = int(feat.shape[0])
        xs.append(feat.detach().cpu().numpy())
        ys.append(cond.detach().cpu().numpy().astype(int).reshape(-1))
        rrs.append(rr_pred.detach().cpu().numpy().reshape(-1))

        if sample_subjects_arr is not None:
            subject_chunks.append(sample_subjects_arr[offset : offset + n])
            offset += n

        if collect_papa_state and hasattr(model, "respiration_state_from_outputs"):
            try:
                r = model.respiration_state_from_outputs(
                    pred_logmag,
                    rr_pred,
                    hidden,
                    adapt=False,
                )
                papa_states.append(r.detach().cpu().numpy())
            except Exception as exc:
                print(f"[RESP_DYN] Could not collect PAPA state: {exc}")
                collect_papa_state = False
                papa_states = []

    if not xs:
        raise RuntimeError("No batches available for respiratory dynamics collection.")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    rr = np.concatenate(rrs, axis=0)

    out: Dict[str, np.ndarray] = {
        "x_static": np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "y": y.astype(int),
        "rr_pred": rr.astype(np.float32),
    }
    if subject_chunks:
        out["subject_ids"] = np.concatenate(subject_chunks, axis=0)
    if papa_states:
        out["papa_state"] = np.nan_to_num(
            np.concatenate(papa_states, axis=0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
    return out


def _contiguous_slices(keys: Optional[np.ndarray], n: int) -> List[Tuple[int, int]]:
    if keys is None:
        return [(0, n)]
    keys = np.asarray(keys, dtype=object)
    if len(keys) != n:
        raise ValueError(f"segments length {len(keys)} does not match n={n}")
    if n == 0:
        return []
    spans: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, n):
        if keys[i] != keys[i - 1]:
            spans.append((start, i))
            start = i
    spans.append((start, n))
    return spans


def _robust_scale(x: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.shape}")
    med = np.nanmedian(x, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(x - med.reshape(1, -1)), axis=0)
    q25, q75 = np.nanpercentile(x, [25, 75], axis=0)
    iqr = (q75 - q25) / 1.349
    sd = np.nanstd(x, axis=0)
    scale = np.where(mad > floor, mad, np.where(iqr > floor, iqr, sd))
    return np.nan_to_num(scale, nan=float(floor), posinf=float(floor), neginf=float(floor)).clip(min=float(floor)).astype(np.float32)


def _split_span_on_feature_jumps(
    x_seg: np.ndarray,
    start: int,
    jump_z: float,
    floor: float,
) -> List[Tuple[int, int]]:
    if jump_z <= 0 or len(x_seg) < 3:
        return [(start, start + len(x_seg))]
    med = np.nanmedian(x_seg, axis=0)
    scale = _robust_scale(x_seg, floor=floor)
    z = (x_seg - med.reshape(1, -1)) / scale.reshape(1, -1)
    jump = np.sqrt(np.nanmean(np.diff(z, axis=0) ** 2, axis=1))
    cuts = np.where(jump > float(jump_z))[0] + 1
    if cuts.size == 0:
        return [(start, start + len(x_seg))]
    out: List[Tuple[int, int]] = []
    local_start = 0
    for cut in cuts.tolist():
        if cut > local_start:
            out.append((start + local_start, start + cut))
        local_start = cut
    if local_start < len(x_seg):
        out.append((start + local_start, start + len(x_seg)))
    return out


def _rolling_one_segment(x: np.ndarray, win: int, centered: bool) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((0, x.shape[1] * 5), dtype=np.float32)

    df = pd.DataFrame(np.asarray(x, dtype=np.float32))
    roll_mean = df.rolling(win, min_periods=1, center=bool(centered)).mean().to_numpy(np.float32)
    roll_std = (
        df.rolling(win, min_periods=2, center=bool(centered))
        .std()
        .fillna(0.0)
        .to_numpy(np.float32)
    )

    slope = np.zeros_like(x, dtype=np.float32)
    if len(x) > 1:
        slope[1:] = x[1:] - x[:-1]

    fast_span = max(2, win // 2)
    slow_span = max(3, win * 3)
    fast = df.ewm(span=fast_span, adjust=False).mean().to_numpy(np.float32)
    slow = df.ewm(span=slow_span, adjust=False).mean().to_numpy(np.float32)
    fast_slow = fast - slow
    return np.concatenate([x, roll_mean, roll_std, slope, fast_slow], axis=1)


def add_rolling_dynamics(
    x: np.ndarray,
    win: int = 7,
    segments: Optional[np.ndarray] = None,
    centered: bool = True,
    boundary_jump_z: float = 0.0,
    scale_floor: float = 1e-3,
) -> np.ndarray:
    """
    Add temporal dynamics without crossing subject/session boundaries.

    The previous implementation used the whole loader order as one sequence,
    which can create artificial slopes when one source subject or condition is
    concatenated after another.  This version resets rolling statistics at
    provided segment boundaries and can optionally split very large feature
    jumps using an unlabeled robust-z threshold.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D, got {x.shape}")

    win = max(1, int(win))
    out = np.zeros((x.shape[0], x.shape[1] * 5), dtype=np.float32)
    for start, end in _contiguous_slices(segments, len(x)):
        for sub_start, sub_end in _split_span_on_feature_jumps(
            x[start:end],
            start=start,
            jump_z=float(boundary_jump_z),
            floor=float(scale_floor),
        ):
            out[sub_start:sub_end] = _rolling_one_segment(
                x[sub_start:sub_end],
                win=win,
                centered=bool(centered),
            )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _z_for_drive(v: np.ndarray) -> np.ndarray:
    return (v - np.nanmedian(v)) / (_robust_scale(np.asarray(v).reshape(-1, 1), floor=1e-6).reshape(-1)[0] + 1e-6)


def respiratory_drive_score(x_static: np.ndarray) -> np.ndarray:
    """
    Low drive should approximate baseline-like breathing.

    x_static columns:
      0 rr_head
      1 rr_stft
      3 band_energy
      4 entropy
      5 peakness
    """
    rr = x_static[:, 1]
    energy = x_static[:, 3]
    entropy = x_static[:, 4]
    peakness = x_static[:, 5]
    flatness = x_static[:, 10] if x_static.shape[1] > 10 else 0.0

    return (
        _z_for_drive(rr)
        + _z_for_drive(energy)
        + _z_for_drive(entropy)
        + _z_for_drive(flatness)
        - _z_for_drive(peakness)
    )


def _low_drive_mask(x_static: np.ndarray, q: float) -> np.ndarray:
    drive = respiratory_drive_score(x_static)
    cutoff = np.nanquantile(drive, float(q))
    mask = drive <= cutoff
    min_count = max(5, int(0.05 * len(x_static)))
    if int(mask.sum()) < min_count:
        mask = drive <= np.nanmedian(drive)
    return mask


def estimate_unlabeled_baseline(
    x_static: np.ndarray,
    q: float = 0.20,
) -> np.ndarray:
    """Estimate target baseline from unlabeled low respiratory-drive windows."""
    mask = _low_drive_mask(x_static, q=q)
    return np.nanmedian(x_static[mask], axis=0).astype(np.float32)


def estimate_source_baseline(
    x_static: np.ndarray,
    y: np.ndarray,
    baseline_class: int,
    q: float = 0.20,
) -> np.ndarray:
    """Prefer source L0 windows; otherwise fall back to low-drive windows."""
    mask = y.astype(int) == int(baseline_class)
    if int(mask.sum()) >= max(5, int(0.02 * len(y))):
        return np.nanmedian(x_static[mask], axis=0).astype(np.float32)
    return estimate_unlabeled_baseline(x_static, q=q)


def _subject_normalize(
    x_static: np.ndarray,
    y: Optional[np.ndarray],
    subject_ids: Optional[np.ndarray],
    baseline_class: int,
    q: float,
    scale: bool,
    scale_floor: float,
    source: bool,
) -> np.ndarray:
    x_static = np.asarray(x_static, dtype=np.float32)
    out = np.zeros_like(x_static, dtype=np.float32)
    if subject_ids is None:
        subject_ids = np.full(len(x_static), "__all__", dtype=object)
    else:
        subject_ids = np.asarray(subject_ids, dtype=object)

    for sid in pd.unique(subject_ids):
        mask = subject_ids == sid
        xs = x_static[mask]
        ys = y[mask] if y is not None else None
        if source and ys is not None:
            base = estimate_source_baseline(xs, ys, baseline_class=baseline_class, q=q)
        else:
            base = estimate_unlabeled_baseline(xs, q=q)
        centered = xs - base.reshape(1, -1)
        if scale:
            sc = _robust_scale(centered, floor=scale_floor)
            centered = centered / sc.reshape(1, -1)
        out[mask] = centered
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _make_rolling_segments(
    subject_ids: Optional[np.ndarray],
    y: Optional[np.ndarray] = None,
    by_label: bool = False,
) -> Optional[np.ndarray]:
    if subject_ids is None:
        if y is None or not by_label:
            return None
        subject_ids = np.full(len(y), "__all__", dtype=object)
    subject_ids = np.asarray(subject_ids, dtype=object)
    if by_label and y is not None:
        y = np.asarray(y, dtype=int)
        return np.asarray([f"{sid}:{int(lbl)}" for sid, lbl in zip(subject_ids, y)], dtype=object)
    return subject_ids


def _concat_optional(*parts: Optional[np.ndarray]) -> Optional[np.ndarray]:
    xs = [p for p in parts if p is not None]
    if not xs:
        return None
    return np.concatenate(xs, axis=1).astype(np.float32)


def build_ladder_features(
    x_train_static: np.ndarray,
    y_train: np.ndarray,
    x_test_static: np.ndarray,
    args,
    train_subject_ids: Optional[np.ndarray] = None,
    test_subject_ids: Optional[np.ndarray] = None,
    x_train_state: Optional[np.ndarray] = None,
    x_test_state: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Build respiratory-dynamics feature variants.

    New variants added after the first ladder:
      *_z         subject baseline subtraction + robust scale calibration
      state_abs   learned PAPA respiration-state/bottleneck features
      hybrid_*    reconstructed physiology features concatenated with state_abs

    The preliminary run showed absolute STFT/dynamics features were competitive
    while high-stay HMM smoothing hurt macro F1, so absolute variants remain in
    the default ladder and HMM is kept as a diagnostic rather than the only
    sequence option.
    """
    baseline_class = label_id(args.resp_dyn_baseline_label)
    target_q = float(args.resp_dyn_target_baseline_q)
    source_q = float(getattr(args, "resp_dyn_source_baseline_q", target_q))
    scale_floor = float(args.resp_dyn_scale_floor)
    train_norm_subject_ids = None if str(getattr(args, "resp_dyn_source_baseline_mode", "subject")).lower() == "global" else train_subject_ids

    x_train_delta = _subject_normalize(
        x_train_static,
        y_train,
        train_norm_subject_ids,
        baseline_class=baseline_class,
        q=source_q,
        scale=False,
        scale_floor=scale_floor,
        source=True,
    )
    x_test_delta = _subject_normalize(
        x_test_static,
        None,
        test_subject_ids,
        baseline_class=baseline_class,
        q=target_q,
        scale=False,
        scale_floor=scale_floor,
        source=False,
    )
    x_train_z = _subject_normalize(
        x_train_static,
        y_train,
        train_norm_subject_ids,
        baseline_class=baseline_class,
        q=source_q,
        scale=True,
        scale_floor=scale_floor,
        source=True,
    )
    x_test_z = _subject_normalize(
        x_test_static,
        None,
        test_subject_ids,
        baseline_class=baseline_class,
        q=target_q,
        scale=True,
        scale_floor=scale_floor,
        source=False,
    )

    source_segments = _make_rolling_segments(
        train_subject_ids,
        y_train,
        by_label=bool(getattr(args, "resp_dyn_source_segment_by_label", True)),
    )
    target_segments = _make_rolling_segments(test_subject_ids, None, by_label=False)

    rolling_kwargs = dict(
        win=int(args.resp_dyn_roll_win),
        centered=bool(args.resp_dyn_centered_roll),
        boundary_jump_z=float(args.resp_dyn_boundary_jump_z),
        scale_floor=scale_floor,
    )

    dyn_train_abs = add_rolling_dynamics(x_train_static, segments=source_segments, **rolling_kwargs)
    dyn_test_abs = add_rolling_dynamics(x_test_static, segments=target_segments, **rolling_kwargs)

    dyn_train_delta = add_rolling_dynamics(x_train_delta, segments=source_segments, **rolling_kwargs)
    dyn_test_delta = add_rolling_dynamics(x_test_delta, segments=target_segments, **rolling_kwargs)

    dyn_train_z = add_rolling_dynamics(x_train_z, segments=source_segments, **rolling_kwargs)
    dyn_test_z = add_rolling_dynamics(x_test_z, segments=target_segments, **rolling_kwargs)

    # Non-state hybrid variants are motivated by the preliminary run: absolute
    # STFT/dynamics were often more stable, while delta/z calibration rescued
    # some subjects. Concatenating them lets shrinkage LDA decide rather than
    # forcing one normalization globally.
    stft_train_hybrid = _concat_optional(x_train_static, x_train_delta, x_train_z)
    stft_test_hybrid = _concat_optional(x_test_static, x_test_delta, x_test_z)
    dyn_train_hybrid = _concat_optional(dyn_train_abs, dyn_train_delta, dyn_train_z)
    dyn_test_hybrid = _concat_optional(dyn_test_abs, dyn_test_delta, dyn_test_z)

    ladder: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "rr_abs": (x_train_static[:, [0]], x_test_static[:, [0]]),
        "rr_delta": (x_train_delta[:, [0]], x_test_delta[:, [0]]),
        "rr_z": (x_train_z[:, [0]], x_test_z[:, [0]]),
        "stft_abs": (x_train_static, x_test_static),
        "stft_delta": (x_train_delta, x_test_delta),
        "stft_z": (x_train_z, x_test_z),
        "stft_hybrid": (stft_train_hybrid, stft_test_hybrid),
        "dyn_abs": (dyn_train_abs, dyn_test_abs),
        "dyn_abs_hmm": (dyn_train_abs, dyn_test_abs),
        "dyn_delta": (dyn_train_delta, dyn_test_delta),
        "dyn_delta_hmm": (dyn_train_delta, dyn_test_delta),
        "dyn_z": (dyn_train_z, dyn_test_z),
        "dyn_z_hmm": (dyn_train_z, dyn_test_z),
        "dyn_hybrid": (dyn_train_hybrid, dyn_test_hybrid),
        "dyn_hybrid_hmm": (dyn_train_hybrid, dyn_test_hybrid),
    }

    if x_train_state is not None and x_test_state is not None:
        ladder.update(
            {
                "state_abs": (x_train_state, x_test_state),
                "state_abs_hmm": (x_train_state, x_test_state),
                "hybrid_abs": (_concat_optional(x_train_static, x_train_state), _concat_optional(x_test_static, x_test_state)),
                "hybrid_dyn_abs": (_concat_optional(dyn_train_abs, x_train_state), _concat_optional(dyn_test_abs, x_test_state)),
                "hybrid_dyn_abs_hmm": (_concat_optional(dyn_train_abs, x_train_state), _concat_optional(dyn_test_abs, x_test_state)),
                "hybrid_delta": (_concat_optional(dyn_train_delta, x_train_state), _concat_optional(dyn_test_delta, x_test_state)),
                "hybrid_z": (_concat_optional(dyn_train_z, x_train_state), _concat_optional(dyn_test_z, x_test_state)),
                "hybrid_z_hmm": (_concat_optional(dyn_train_z, x_train_state), _concat_optional(dyn_test_z, x_test_state)),
            }
        )

    return {k: v for k, v in ladder.items() if v[0] is not None and v[1] is not None}


# -----------------------------------------------------------------------------
# Classifiers and sequence smoothing
# -----------------------------------------------------------------------------
class TorchLinearProbeClassifier:
    """Sklearn-like multinomial linear probe backed by a single nn.Linear."""

    def __init__(self, max_iter: int = 30, lr: float = 1e-3, batch_size: int = 64, weight_decay: float = 1e-4, device: str = "cpu", seed: int = 42):
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.device = str(device)
        self.seed = int(seed)
        self.classes_: np.ndarray = np.array([], dtype=int)
        self.constant_: Optional[int] = None
        self.model_: Optional[nn.Linear] = None
        self.mu_: Optional[torch.Tensor] = None
        self.sd_: Optional[torch.Tensor] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchLinearProbeClassifier":
        x_np = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_np = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.array(sorted(np.unique(y_np).tolist()), dtype=int)
        if len(self.classes_) < 2 or x_np.shape[0] < max(3, len(self.classes_)):
            vals, counts = np.unique(y_np, return_counts=True)
            self.constant_ = int(vals[np.argmax(counts)]) if vals.size else 0
            self.classes_ = np.array([self.constant_], dtype=int)
            return self
        dev = torch.device(self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        torch.manual_seed(self.seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        x_t = torch.as_tensor(x_np, dtype=torch.float32, device=dev)
        self.mu_ = x_t.mean(dim=0, keepdim=True)
        self.sd_ = x_t.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        x_z = (x_t - self.mu_) / self.sd_
        y_idx = np.searchsorted(self.classes_, y_np).astype(np.int64)
        y_t = torch.as_tensor(y_idx, dtype=torch.long, device=dev)
        self.model_ = nn.Linear(x_z.shape[1], len(self.classes_)).to(dev)
        nn.init.zeros_(self.model_.bias)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        counts = torch.bincount(y_t, minlength=len(self.classes_)).float().clamp_min(1.0)
        class_weight = (counts.sum() / (len(self.classes_) * counts)).to(dev)
        bs = max(1, min(int(self.batch_size), x_z.shape[0]))
        for _ in range(int(self.max_iter)):
            perm = torch.randperm(x_z.shape[0], device=dev)
            for st in range(0, x_z.shape[0], bs):
                idx = perm[st : st + bs]
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.model_(x_z[idx]), y_t[idx], weight=class_weight)
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
        outs = []
        bs = max(1, int(self.batch_size))
        for st in range(0, x_np.shape[0], bs):
            xb = torch.as_tensor(x_np[st : st + bs], dtype=torch.float32, device=dev)
            xb = (xb - self.mu_.to(dev)) / self.sd_.to(dev).clamp_min(1e-6)
            outs.append(torch.softmax(self.model_(xb), dim=1).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        p = self.predict_proba(x)
        return self.classes_[np.argmax(p, axis=1)].astype(int)


def make_resp_dyn_classifier(kind: str, args):
    kind = str(kind).lower()

    if kind == "lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        return make_pipeline(StandardScaler(), clf)

    if kind == "logreg":
        clf = LogisticRegression(
            max_iter=int(args.resp_dyn_logreg_max_iter),
            C=float(args.resp_dyn_logreg_c),
            class_weight="balanced",
            multi_class="auto",
        )
        return make_pipeline(StandardScaler(), clf)

    if kind in {"linear_probe", "torch_linear_probe"}:
        return TorchLinearProbeClassifier(
            max_iter=int(getattr(args, "linear_probe_epochs", 30)),
            lr=float(getattr(args, "linear_probe_lr", 1e-3)),
            batch_size=int(getattr(args, "linear_probe_batch_size", 64)),
            weight_decay=float(getattr(args, "linear_probe_weight_decay", 1e-4)),
            device=str(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")),
            seed=int(getattr(args, "seed", 42)),
        )

    raise ValueError(f"Unknown --resp-dyn-classifier {kind!r}")



def effective_hmm_stay(proba: np.ndarray, args) -> float:
    """
    Avoid over-sticky HMM smoothing when the classifier is uncertain.

    The preliminary run used stay=0.95 and often collapsed macro-F1. This keeps
    the user-specified value as an upper bound, but lowers it for low-confidence
    sequences unless --resp-dyn-no-hmm-adaptive is passed.
    """
    stay = float(args.resp_dyn_hmm_stay)
    if not bool(getattr(args, "resp_dyn_hmm_adaptive", True)):
        return stay

    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[0] == 0 or proba.shape[1] <= 1:
        return stay

    C = proba.shape[1]
    mean_conf = float(np.nanmean(np.nanmax(proba, axis=1)))
    floor = max(float(args.resp_dyn_hmm_min_stay), 1.0 / C + 0.05)
    floor = min(max(floor, 1.0 / C), 0.95)
    stay = min(max(stay, floor), 0.999)
    adaptive = floor + (stay - floor) * mean_conf
    return float(min(stay, max(floor, adaptive)))


def viterbi_smooth(
    log_probs: np.ndarray,
    stay_prob: float = 0.95,
) -> np.ndarray:
    """
    Simple HMM smoothing.

    Returns class indices in probability-column space, not original labels.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T, C = log_probs.shape

    if T == 0:
        return np.array([], dtype=np.int64)

    if C == 1:
        return np.zeros(T, dtype=np.int64)

    stay_prob = float(stay_prob)
    stay_prob = min(max(stay_prob, 1.0 / C), 0.999)

    trans = np.full((C, C), (1.0 - stay_prob) / max(1, C - 1), dtype=np.float64)
    np.fill_diagonal(trans, stay_prob)
    log_trans = np.log(trans + 1e-12)

    dp = np.zeros((T, C), dtype=np.float64)
    ptr = np.zeros((T, C), dtype=np.int64)

    dp[0] = log_probs[0]

    for t in range(1, T):
        scores = dp[t - 1][:, None] + log_trans
        ptr[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + log_probs[t]

    y = np.zeros(T, dtype=np.int64)
    y[-1] = int(dp[-1].argmax())

    for t in range(T - 2, -1, -1):
        y[t] = ptr[t + 1, y[t + 1]]

    return y


def fit_predict_resp_dyn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    variant: str,
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit classifier on source subjects and predict held-out target subject.
    """
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

    clf = make_resp_dyn_classifier(args.resp_dyn_classifier, args)
    clf.fit(x_train, y_train.astype(int))

    if variant.endswith("_hmm"):
        if not hasattr(clf, "predict_proba"):
            pred = clf.predict(x_test).astype(int)
            return pred, np.zeros((len(pred), len(np.unique(y_train))), dtype=np.float32)

        proba = clf.predict_proba(x_test)
        proba = np.clip(proba, 1e-12, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)

        idx = viterbi_smooth(
            np.log(proba),
            stay_prob=effective_hmm_stay(proba, args),
        )
        classes = clf.classes_.astype(int)
        pred = classes[idx]
        return pred.astype(int), proba.astype(np.float32)

    pred = clf.predict(x_test).astype(int)

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x_test).astype(np.float32)
    else:
        proba = np.zeros((len(pred), len(np.unique(y_train))), dtype=np.float32)

    return pred, proba


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    return {
        "resp_dyn_acc": float(accuracy_score(y_true, y_pred)),
        "resp_dyn_bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "resp_dyn_f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)
        ),
        "resp_dyn_f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", labels=classes, zero_division=0)
        ),
    }


# -----------------------------------------------------------------------------
# Respiratory dynamics evaluation hook
# -----------------------------------------------------------------------------
def _default_resp_dyn_ladder(has_state: bool) -> List[str]:
    # Keep absolute morphology first because preliminary results showed it was
    # more stable than pure delta features. Use calibrated/hybrid variants as
    # complementary evidence. HMM variants are limited to gentler/adaptive
    # smoothing candidates, not the old over-sticky dyn_delta_hmm default.
    variants = [
        "rr_abs",
        "rr_delta",
        "rr_z",
        "stft_abs",
        "stft_delta",
        "stft_z",
        "stft_hybrid",
        "dyn_abs",
        "dyn_delta",
        "dyn_z",
        "dyn_hybrid",
        "dyn_abs_hmm",
        "dyn_z_hmm",
        "dyn_hybrid_hmm",
    ]
    if has_state:
        variants.extend(
            [
                "state_abs",
                "hybrid_abs",
                "hybrid_dyn_abs",
                "hybrid_dyn_abs_hmm",
                "hybrid_z",
                "hybrid_z_hmm",
            ]
        )
    return variants


def resp_dyn_hook(
    model,
    sbj: str,
    subjects: List[str],
    _train_loader,
    _test_loader,
    device: str,
    args,
    sbj_dir: Path,
):
    if not bool(getattr(args, "eval_resp_dyn", False)):
        return []

    train_loader, test_loader, train_subject_ids, test_subject_ids = (
        _build_resp_dyn_loaders_with_subject_ids(sbj, subjects, args)
    )

    should_collect_state = bool(getattr(args, "resp_dyn_collect_state", True))
    train_pack = collect_resp_dyn_static_features(
        model,
        train_loader,
        device,
        args,
        sample_subjects=train_subject_ids,
        collect_papa_state=should_collect_state,
    )
    test_pack = collect_resp_dyn_static_features(
        model,
        test_loader,
        device,
        args,
        sample_subjects=test_subject_ids,
        collect_papa_state=should_collect_state,
    )

    x_train_static = train_pack["x_static"]
    y_train = train_pack["y"]
    x_test_static = test_pack["x_static"]
    y_test = test_pack["y"]

    x_train_state = train_pack.get("papa_state")
    x_test_state = test_pack.get("papa_state")
    has_state = x_train_state is not None and x_test_state is not None

    ladder = build_ladder_features(
        x_train_static=x_train_static,
        y_train=y_train,
        x_test_static=x_test_static,
        args=args,
        train_subject_ids=train_pack.get("subject_ids"),
        test_subject_ids=test_pack.get("subject_ids"),
        x_train_state=x_train_state,
        x_test_state=x_test_state,
    )

    requested = [
        v.strip()
        for v in str(args.resp_dyn_ladder).split(",")
        if v.strip()
    ]

    if "all" in requested:
        requested = _default_resp_dyn_ladder(has_state=has_state)

    out = sbj_dir / "resp_dyn"
    out.mkdir(parents=True, exist_ok=True)

    classes = np.array(sorted(np.unique(y_train.astype(int)).tolist()), dtype=int)
    rows = []

    feature_meta = {
        "subject": sbj,
        "classes": classes.tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_source_subjects": int(len(np.unique(train_pack.get("subject_ids", np.array([]))))),
        "static_dim": int(x_train_static.shape[1]),
        "static_feature_names": RESP_DYN_STATIC_FEATURE_NAMES,
        "state_dim": int(x_train_state.shape[1]) if has_state else 0,
        "roll_win": int(args.resp_dyn_roll_win),
        "centered_roll": bool(args.resp_dyn_centered_roll),
        "boundary_jump_z": float(args.resp_dyn_boundary_jump_z),
        "source_segment_by_label": bool(args.resp_dyn_source_segment_by_label),
        "classifier": str(args.resp_dyn_classifier),
        "baseline_label": str(args.resp_dyn_baseline_label),
        "target_baseline_q": float(args.resp_dyn_target_baseline_q),
        "source_baseline_q": float(args.resp_dyn_source_baseline_q),
        "scale_floor": float(args.resp_dyn_scale_floor),
        "resp_dyn_fs": float(args.resp_dyn_fs),
        "resp_dyn_min_hz": float(args.resp_dyn_min_hz),
        "resp_dyn_max_hz": float(args.resp_dyn_max_hz),
        "hmm_stay": float(args.resp_dyn_hmm_stay),
        "hmm_adaptive": bool(getattr(args, "resp_dyn_hmm_adaptive", True)),
        "requested_variants": requested,
        "available_variants": sorted(ladder.keys()),
    }

    with open(out / "resp_dyn_feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    for variant in requested:
        if variant not in ladder:
            print(f"[RESP_DYN] Skipping unknown variant {variant!r}")
            continue

        xtr, xte = ladder[variant]
        y_pred, proba = fit_predict_resp_dyn(
            x_train=xtr,
            y_train=y_train,
            x_test=xte,
            variant=variant,
            args=args,
        )

        metrics = evaluate_predictions(y_test.astype(int), y_pred.astype(int), classes=classes)
        pred_conf = (
            proba.max(axis=1).astype(np.float32)
            if proba.ndim == 2 and proba.shape[0] == len(y_pred) and proba.shape[1] > 0
            else np.full(len(y_pred), np.nan, dtype=np.float32)
        )
        effective_stay = (
            effective_hmm_stay(proba, args)
            if variant.endswith("_hmm") and proba.ndim == 2 and proba.shape[1] > 1
            else np.nan
        )

        row = {
            "__summary_name__": "resp_dyn_summary",
            "subject": sbj,
            "tag": variant,
            "resp_dyn_variant": variant,
            "resp_dyn_classifier": str(args.resp_dyn_classifier),
            "resp_dyn_feature_dim": int(xtr.shape[1]),
            "resp_dyn_n_train": int(len(y_train)),
            "resp_dyn_n_test": int(len(y_test)),
            "resp_dyn_n_classes_train": int(len(classes)),
            "resp_dyn_n_classes_test": int(len(np.unique(y_test))),
            "resp_dyn_hmm_stay": float(args.resp_dyn_hmm_stay) if variant.endswith("_hmm") else np.nan,
            "resp_dyn_effective_hmm_stay": float(effective_stay) if variant.endswith("_hmm") else np.nan,
            "resp_dyn_hmm_adaptive": bool(getattr(args, "resp_dyn_hmm_adaptive", True)) if variant.endswith("_hmm") else False,
            "resp_dyn_fs": float(args.resp_dyn_fs),
            "resp_dyn_target_baseline_q": float(args.resp_dyn_target_baseline_q),
            "resp_dyn_source_baseline_q": float(args.resp_dyn_source_baseline_q),
            "resp_dyn_source_baseline_mode": str(args.resp_dyn_source_baseline_mode),
            "resp_dyn_scale_floor": float(args.resp_dyn_scale_floor),
            "resp_dyn_boundary_jump_z": float(args.resp_dyn_boundary_jump_z),
            "resp_dyn_mean_conf": float(np.nanmean(pred_conf)),
            **metrics,
        }
        rows.append(row)

        pred_data = {
            "subject_id": test_pack.get("subject_ids", np.asarray([sbj] * len(y_test), dtype=object)).astype(str),
            "y_true": y_test.astype(int),
            "y_pred": y_pred.astype(int),
            "pred_conf": pred_conf,
        }
        for i, name in enumerate(RESP_DYN_STATIC_FEATURE_NAMES):
            if i < x_test_static.shape[1]:
                pred_data[name] = x_test_static[:, i]
        pred_df = pd.DataFrame(pred_data)
        pred_df.to_csv(out / f"resp_dyn_predictions_{variant}.csv", index=False)

        print(
            f"RESP_DYN {sbj} {variant}: "
            f"acc={row['resp_dyn_acc']:.4f} "
            f"bal={row['resp_dyn_bal_acc']:.4f} "
            f"macro={row['resp_dyn_f1_macro']:.4f} "
            f"weighted={row['resp_dyn_f1_weighted']:.4f} "
            f"conf={row['resp_dyn_mean_conf']:.3f}"
        )

    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def finalize_args_dyn(args) -> None:
    """
    Reuse PAPA's finalize_args so the base PAPA model and frozen embedding loaders
    are configured exactly as before.
    """
    papa.finalize_args(args)

    if args.eval_resp_dyn:
        args.eval_frozen_embeddings = bool(args.eval_frozen_embeddings or args.resp_dyn_also_frozen)


def main() -> None:
    parser = core.build_base_parser(
        SUBJECTS,
        str(Path(SBJ_PROCESSED_DIR) / "vit_pressure_crossmodal_papa_dyn"),
    )

    # Existing frozen-embedding evaluation.
    parser.add_argument("--eval-frozen-embeddings", action="store_true")
    parser.add_argument("--eval-frozen-tlx", action="store_true")
    parser.add_argument("--tlx-ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--embed-data-group",
        default=None,
        choices=["mr", "level", "levels", "mr_levels"],
    )
    parser.add_argument("--embed-labels", default="L0,L2,L3")
    parser.add_argument(
        "--embed-classifier",
        default="linear_probe",
        choices=["lda", "logreg", "linear_probe"],
    )
    parser.add_argument(
        "--embed-pooling",
        default="rich",
        choices=["mean", "max", "cls_last", "mean_std", "mean_std_max", "rich"],
    )
    parser.add_argument("--embed-stft-profile", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--linear-probe-epochs", type=int, default=30)
    parser.add_argument("--linear-probe-lr", type=float, default=1e-3)
    parser.add_argument("--linear-probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--linear-probe-batch-size", type=int, default=64)

    # Static PAPA branch, retained for comparison.
    parser.add_argument("--eval-papa", action="store_true")
    parser.add_argument("--papa-state-dim", type=int, default=48)
    parser.add_argument("--papa-adapter-init-scale", type=float, default=0.05)
    parser.add_argument("--papa-no-bottleneck", action="store_true")
    parser.add_argument("--papa-no-adapter", action="store_true")
    parser.add_argument(
        "--papa-tta",
        default="none",
        choices=["none", "tent", "nrc", "cotta", "papa"],
    )
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

    # Bottleneck auxiliary losses used by vit_pressure_crossmodal_papa.py.
    parser.add_argument("--lambda-resp-rr", type=float, default=0.05)
    parser.add_argument("--lambda-resp-recon", type=float, default=0.01)

    # New PAPA-dyn branch.
    parser.add_argument(
        "--eval-resp-dyn",
        action="store_true",
        help="Evaluate reconstructed respiratory dynamics ladder.",
    )
    parser.add_argument(
        "--resp-dyn-ladder",
        default="all",
        help=(
            "Comma-separated variants: rr_abs,rr_delta,rr_z,stft_abs,stft_delta,"
            "stft_z,stft_hybrid,dyn_abs,dyn_abs_hmm,dyn_delta,"
            "dyn_delta_hmm,dyn_z,dyn_z_hmm,dyn_hybrid,dyn_hybrid_hmm,"
            "state_abs,state_abs_hmm,hybrid_abs,hybrid_dyn_abs,"
            "hybrid_dyn_abs_hmm,hybrid_delta,hybrid_z,hybrid_z_hmm, or all."
        ),
    )
    parser.add_argument(
        "--resp-dyn-classifier",
        default="linear_probe",
        choices=["lda", "logreg", "linear_probe"],
    )
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
    parser.add_argument(
        "--resp-dyn-also-frozen",
        action="store_true",
        help="Also run the frozen embedding baseline when --eval-resp-dyn is active.",
    )

    args = parser.parse_args()

    core.run_loocv_experiment(
        args,
        post_eval_hooks=[
            frozen_embedding_hook,
            papa.papa_hook,
            resp_dyn_hook,
        ],
        config_mutator=finalize_args_dyn,
    )


if __name__ == "__main__":
    main()

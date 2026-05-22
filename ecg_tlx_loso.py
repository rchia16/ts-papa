#!/usr/bin/env python3
"""
LOSO analysis of ECG-derived RRI/HR shifts vs NASA-TLX.

The script mirrors rr_tlx_loso.py, but uses condition ECG files named like:

    M_ecg_df.csv, R_ecg_df.csv, L0_ecg_df.csv, ..., L3_ecg_df.csv

It infers subject ids from path components such as S12, estimates R peaks in
windowed ECG, derives median RRI and HR, normalizes both to each subject's rest
baseline, and reports basic shrinkage-LDA mental-workload classification results
in leave-one-subject-out folds.

Example:
  python ecg_tlx_loso.py \
    --data-root /projects/BLVMob/aria_seated/Data \
    --tlx-csv /projects/BLVMob/imu-rr-seated/Data/seated_tlx.csv \
    --conditions L0,L1,L2,L3 \
    --out-dir results/ecg_tlx_loso
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, iirnotch
from scipy.stats import shapiro, ttest_rel, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VALID_CONDITIONS = ("M", "R", "L0", "L1", "L2", "L3")
LEVEL_CONDITIONS = ("L0", "L1", "L2", "L3")


@dataclass
class ECGConfig:
    data_root: str
    tlx_csv: str
    conditions: List[str]
    out_dir: str
    ecg_col: str = "ecg"
    fs: float = 250.0
    window_sec: float = 60.0
    shift_sec: float = 10.0
    min_hr_bpm: float = 35.0
    max_hr_bpm: float = 220.0
    rest_condition: str = "R"
    min_valid_fraction: float = 0.8
    min_windows_per_subject_condition: int = 1
    normalize_mode: str = "relative"
    lda_features: List[str] = None

    def __post_init__(self) -> None:
        if self.lda_features is None:
            self.lda_features = ["rri_norm", "hr_norm"]


def canonical_subject_id(x: object) -> str:
    txt = str(x).strip()
    m = re.search(r"S\s*0*(\d+)", txt, flags=re.IGNORECASE)
    if m:
        return f"S{int(m.group(1)):02d}"
    m = re.search(r"\b0*(\d{1,3})\b", txt)
    if m:
        return f"S{int(m.group(1)):02d}"
    return txt


def parse_conditions(s: str) -> List[str]:
    out = [c.strip().upper() for c in s.split(",") if c.strip()]
    bad = [c for c in out if c not in VALID_CONDITIONS]
    if bad:
        raise ValueError(f"Unsupported conditions {bad}. Use any of: {VALID_CONDITIONS}")
    return out


def parse_feature_list(s: str) -> List[str]:
    feats = [x.strip() for x in s.split(",") if x.strip()]
    allowed = {"rri_ms", "hr_bpm", "rri_norm", "hr_norm", "rest_rri_ms", "rest_hr_bpm"}
    bad = [x for x in feats if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported LDA features {bad}. Use any of: {sorted(allowed)}")
    return feats


def infer_subject_from_path(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        m = re.search(r"S\s*0*(\d+)", part, flags=re.IGNORECASE)
        if m:
            return f"S{int(m.group(1)):02d}"
    return None


def discover_ecg_files(data_root: Path) -> Dict[str, Dict[str, Path]]:
    mapping: Dict[str, Dict[str, Path]] = {}
    for path in sorted(data_root.rglob("*_ecg_df.csv")):
        cond = path.name.replace("_ecg_df.csv", "").upper()
        if cond not in VALID_CONDITIONS:
            continue
        sbj = infer_subject_from_path(path)
        if sbj is None:
            sbj = canonical_subject_id(path.parent.name)
        mapping.setdefault(sbj, {})[cond] = path
    return mapping


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def read_ecg_signal(path: Path, ecg_col: str) -> np.ndarray:
    df = clean_columns(pd.read_csv(path))
    lower = {c.lower(): c for c in df.columns}
    if ecg_col.lower() in lower:
        col = lower[ecg_col.lower()]
    else:
        matches = [c for c in df.columns if ecg_col.lower() in c.lower()]
        if matches:
            col = matches[0]
        else:
            ecg_matches = [c for c in df.columns if "ecg" in c.lower()]
            col = ecg_matches[0] if ecg_matches else df.columns[-1]
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return x


def load_tlx_table(path: Path) -> Dict[str, Dict[str, float]]:
    df = clean_columns(pd.read_csv(path))
    lower = {c.lower(): c for c in df.columns}
    subject_col = lower.get("subject") or lower.get("subj") or lower.get("participant") or df.columns[0]
    table: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        sbj = canonical_subject_id(row[subject_col])
        scores: Dict[str, float] = {}
        for cond in LEVEL_CONDITIONS:
            col = cond if cond in df.columns else lower.get(cond.lower())
            if col is None:
                continue
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(val):
                scores[cond] = float(val)
        if scores:
            table[sbj] = scores
    return table


def preprocess_ecg(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.mean() < 0.5:
        return np.asarray([], dtype=float)
    if not finite.all():
        idx = np.arange(len(x))
        x = np.interp(idx, idx[finite], x[finite])
    x = x - np.nanmedian(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.asarray([], dtype=float)
    x = x / sd

    nyq = 0.5 * fs
    if fs > 110:
        b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=fs)
        x = filtfilt(b_notch, a_notch, x)
    high = min(20.0, 0.95 * nyq)
    low = min(5.0, high * 0.5)
    if low <= 0 or high <= low:
        return x
    b, a = butter(3, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def rri_hr_from_window(
    x: np.ndarray,
    fs: float,
    min_hr_bpm: float,
    max_hr_bpm: float,
    min_valid_fraction: float,
) -> Tuple[float, float, int]:
    if x.size < max(8, int(fs * 5)):
        return np.nan, np.nan, 0
    if float(np.isfinite(x).mean()) < min_valid_fraction:
        return np.nan, np.nan, 0
    y = preprocess_ecg(x, fs)
    if y.size < max(8, int(fs * 5)):
        return np.nan, np.nan, 0

    min_distance = max(1, int(round(fs * 60.0 / max_hr_bpm)))
    prominence = max(0.25, 0.35 * float(np.nanstd(y)))
    peaks, _ = find_peaks(y, distance=min_distance, prominence=prominence)
    if peaks.size < 3:
        peaks, _ = find_peaks(-y, distance=min_distance, prominence=prominence)
    if peaks.size < 3:
        return np.nan, np.nan, int(peaks.size)

    rri_ms = np.diff(peaks) / fs * 1000.0
    min_rri = 60000.0 / max_hr_bpm
    max_rri = 60000.0 / min_hr_bpm
    rri_ms = rri_ms[(rri_ms >= min_rri) & (rri_ms <= max_rri)]
    if rri_ms.size < 2:
        return np.nan, np.nan, int(peaks.size)
    med_rri = float(np.nanmedian(rri_ms))
    return med_rri, float(60000.0 / med_rri), int(peaks.size)


def windowed_ecg_features(
    x: np.ndarray,
    fs: float,
    window_sec: float,
    shift_sec: float,
    min_hr_bpm: float,
    max_hr_bpm: float,
    min_valid_fraction: float,
) -> pd.DataFrame:
    win = int(round(window_sec * fs))
    shift = int(round(shift_sec * fs))
    if win <= 0 or shift <= 0:
        raise ValueError("window_sec and shift_sec must be positive")
    if len(x) < win:
        rri, hr, n_peaks = rri_hr_from_window(x, fs, min_hr_bpm, max_hr_bpm, min_valid_fraction)
        rows = [{"window_idx": 0, "rri_ms": rri, "hr_bpm": hr, "n_r_peaks": n_peaks}]
        return pd.DataFrame(rows).dropna(subset=["rri_ms", "hr_bpm"])

    rows = []
    idx = 0
    for start in range(0, len(x) - win + 1, shift):
        seg = x[start : start + win]
        rri, hr, n_peaks = rri_hr_from_window(seg, fs, min_hr_bpm, max_hr_bpm, min_valid_fraction)
        if np.isfinite(rri) and np.isfinite(hr):
            rows.append({"window_idx": idx, "rri_ms": rri, "hr_bpm": hr, "n_r_peaks": n_peaks})
        idx += 1
    return pd.DataFrame(rows)


def normalize_value(value: np.ndarray, rest: float, mode: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if not np.isfinite(rest) or rest <= 0:
        return np.full_like(value, np.nan, dtype=float)
    if mode == "relative":
        return (value - rest) / rest
    if mode == "difference":
        return value - rest
    if mode == "ratio":
        return value / rest
    raise ValueError("normalize_mode must be one of: relative, difference, ratio")


def build_feature_table(cfg: ECGConfig) -> pd.DataFrame:
    tlx = load_tlx_table(Path(cfg.tlx_csv))
    files = discover_ecg_files(Path(cfg.data_root))
    rows: List[dict] = []
    missing_rest: List[str] = []

    for sbj, cond_files in sorted(files.items()):
        rest_path = cond_files.get(cfg.rest_condition)
        if rest_path is None:
            missing_rest.append(sbj)
            continue
        rest_feat = windowed_ecg_features(
            read_ecg_signal(rest_path, cfg.ecg_col),
            cfg.fs,
            cfg.window_sec,
            cfg.shift_sec,
            cfg.min_hr_bpm,
            cfg.max_hr_bpm,
            cfg.min_valid_fraction,
        )
        if rest_feat.empty:
            missing_rest.append(sbj)
            continue
        rest_rri = float(rest_feat["rri_ms"].median())
        rest_hr = float(rest_feat["hr_bpm"].median())

        for cond in cfg.conditions:
            path = cond_files.get(cond)
            if path is None:
                continue
            feat = windowed_ecg_features(
                read_ecg_signal(path, cfg.ecg_col),
                cfg.fs,
                cfg.window_sec,
                cfg.shift_sec,
                cfg.min_hr_bpm,
                cfg.max_hr_bpm,
                cfg.min_valid_fraction,
            )
            if len(feat) < cfg.min_windows_per_subject_condition:
                continue
            feat = feat.copy()
            feat["rri_norm"] = normalize_value(feat["rri_ms"].to_numpy(), rest_rri, cfg.normalize_mode)
            feat["hr_norm"] = normalize_value(feat["hr_bpm"].to_numpy(), rest_hr, cfg.normalize_mode)
            tlx_score = tlx.get(sbj, {}).get(cond, np.nan)
            for _, row in feat.iterrows():
                rows.append(
                    {
                        "subject": sbj,
                        "condition": cond,
                        "window_idx": int(row["window_idx"]),
                        "rri_ms": float(row["rri_ms"]),
                        "hr_bpm": float(row["hr_bpm"]),
                        "rest_rri_ms": rest_rri,
                        "rest_hr_bpm": rest_hr,
                        "rri_norm": float(row["rri_norm"]),
                        "hr_norm": float(row["hr_norm"]),
                        "n_r_peaks": int(row["n_r_peaks"]),
                        "tlx": float(tlx_score) if np.isfinite(tlx_score) else np.nan,
                        "source_file": str(path),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable ECG windows were found. Check --data-root, --ecg-col, --fs, and --window-sec.")
    df = df[np.isfinite(df["rri_ms"]) & np.isfinite(df["hr_bpm"])].reset_index(drop=True)
    if missing_rest:
        print(f"[WARN] Skipped {len(missing_rest)} subjects without usable {cfg.rest_condition} ECG baseline: {missing_rest}")
    return df


def run_loso_lda(df: pd.DataFrame, cfg: ECGConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(df["subject"].unique())
    metrics_rows: List[dict] = []
    pred_rows: List[dict] = []

    for heldout in subjects:
        train = df[df["subject"] != heldout].dropna(subset=cfg.lda_features + ["condition"]).copy()
        test = df[df["subject"] == heldout].dropna(subset=cfg.lda_features + ["condition"]).copy()
        present_train_classes = sorted(train["condition"].unique())
        present_test_classes = sorted(test["condition"].unique())

        fold = {
            "heldout_subject": heldout,
            "lda_features": json.dumps(cfg.lda_features),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_classes": json.dumps(present_train_classes),
            "test_classes": json.dumps(present_test_classes),
        }
        if len(present_train_classes) < 2 or len(test) == 0:
            fold.update({"lda_acc": np.nan, "lda_bal_acc": np.nan, "lda_f1_macro": np.nan})
            metrics_rows.append(fold)
            continue

        clf = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
        x_train = train[cfg.lda_features].to_numpy(dtype=float)
        y_train = train["condition"].to_numpy()
        x_test = test[cfg.lda_features].to_numpy(dtype=float)
        y_test = test["condition"].to_numpy()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        labels = sorted(set(y_train).union(set(y_test)))

        fold.update(
            {
                "lda_acc": float(accuracy_score(y_test, y_pred)),
                "lda_bal_acc": float(balanced_accuracy_score(y_test, y_pred)),
                "lda_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "confusion_labels": json.dumps(labels),
                "confusion_matrix": json.dumps(confusion_matrix(y_test, y_pred, labels=labels).tolist()),
                "classification_report": json.dumps(classification_report(y_test, y_pred, zero_division=0, output_dict=True)),
            }
        )
        metrics_rows.append(fold)

        keep_cols = ["subject", "condition", "window_idx", "rri_ms", "hr_bpm", "rri_norm", "hr_norm", "tlx"]
        tmp = test[keep_cols].copy()
        tmp["heldout_subject"] = heldout
        tmp["pred_condition"] = y_pred
        pred_rows.extend(tmp.to_dict(orient="records"))

    return pd.DataFrame(metrics_rows), pd.DataFrame(pred_rows)


def _subject_condition_summary(df: pd.DataFrame, cfg: ECGConfig) -> pd.DataFrame:
    d = df[df["condition"].isin(cfg.conditions)].copy()
    return (
        d.groupby(["subject", "condition"], as_index=False)
        .agg(
            tlx=("tlx", "first"),
            rri_ms=("rri_ms", "median"),
            hr_bpm=("hr_bpm", "median"),
            rri_norm=("rri_norm", "median"),
            hr_norm=("hr_norm", "median"),
            rest_rri_ms=("rest_rri_ms", "first"),
            rest_hr_bpm=("rest_hr_bpm", "first"),
            n_windows=("hr_bpm", "size"),
        )
    )


def aggregate_plots(df: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, cfg: ECGConfig) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not metrics.empty and "lda_bal_acc" in metrics.columns:
        plt.figure(figsize=(7.0, 4.2))
        m = metrics.sort_values("heldout_subject")
        plt.bar(m["heldout_subject"], m["lda_bal_acc"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("LDA balanced accuracy")
        plt.xlabel("Held-out subject")
        plt.tight_layout()
        plt.savefig(plot_dir / "lda_balanced_accuracy_by_subject.png", dpi=300)
        plt.close()

    summary = _subject_condition_summary(df, cfg)
    summary.to_csv(out_dir / "condition_tlx_ecg_summary.csv", index=False)
    for metric, ylabel, out_name in [
        ("rri_ms", "RRI, median per subject-condition (ms)", "condition_tlx_boxplot_overlay_rri_ms.png"),
        ("hr_bpm", "HR, median per subject-condition (BPM)", "condition_tlx_boxplot_overlay_hr_bpm.png"),
        ("rri_norm", f"RRI normalized to {cfg.rest_condition}", "condition_tlx_boxplot_overlay_rri_norm.png"),
        ("hr_norm", f"HR normalized to {cfg.rest_condition}", "condition_tlx_boxplot_overlay_hr_norm.png"),
    ]:
        plot_condition_tlx_ecg_boxplot(summary, cfg.conditions, metric, ylabel, plot_dir / out_name)


def plot_condition_tlx_ecg_boxplot(
    summary: pd.DataFrame,
    conds: Sequence[str],
    metric_col: str,
    metric_label: str,
    out_path: Path,
) -> None:
    d = summary.dropna(subset=["condition", "tlx", metric_col]).copy()
    if d.empty:
        print(f"[WARN] No rows available for {out_path.name}.")
        return

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    tlx_groups = [d.loc[d["condition"] == c, "tlx"].dropna().to_numpy() for c in conds]
    positions = np.arange(1, len(conds) + 1)
    non_empty = [i for i, vals in enumerate(tlx_groups) if len(vals)]
    if not non_empty:
        plt.close(fig)
        return

    ax1.boxplot(
        [tlx_groups[i] for i in non_empty],
        positions=positions[non_empty],
        widths=0.55,
        patch_artist=False,
    )
    ax1.set_xticks(positions)
    ax1.set_xticklabels(conds)
    ax1.set_xlabel("Condition")
    ax1.set_ylabel("NASA-TLX")

    ax2 = ax1.twinx()
    rng = np.random.default_rng(0)
    for i, cond in enumerate(conds, start=1):
        vals = d.loc[d["condition"] == cond, metric_col].dropna().to_numpy()
        if vals.size == 0:
            continue
        jitter = rng.normal(loc=0.0, scale=0.045, size=vals.size)
        ax2.scatter(np.full(vals.size, i, dtype=float) + jitter, vals, s=30, alpha=0.75)
        ax2.plot([i - 0.22, i + 0.22], [np.nanmedian(vals), np.nanmedian(vals)], linewidth=2)
    ax2.set_ylabel(metric_label)
    ax1.set_title(f"NASA-TLX by condition with overlaid {metric_col}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _bh_fdr(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * float(len(ranked)) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.clip(adj, 0.0, 1.0)
    return q.tolist()


def pairwise_condition_statistics(df: pd.DataFrame, cfg: ECGConfig, out_dir: Path) -> pd.DataFrame:
    summary = _subject_condition_summary(df, cfg).dropna(subset=["subject", "condition"])
    conds = [c for c in cfg.conditions if c in set(summary["condition"])]
    metrics = [
        ("tlx", "NASA-TLX"),
        ("rri_ms", "RRI ms"),
        ("hr_bpm", "Heart rate BPM"),
        ("rri_norm", f"RRI normalized to {cfg.rest_condition}"),
        ("hr_norm", f"HR normalized to {cfg.rest_condition}"),
    ]
    rows: List[dict] = []
    for metric_col, metric_name in metrics:
        wide = summary.pivot(index="subject", columns="condition", values=metric_col)
        for i, cond_a in enumerate(conds):
            for cond_b in conds[i + 1:]:
                if cond_a not in wide.columns or cond_b not in wide.columns:
                    continue
                pair = wide[[cond_a, cond_b]].dropna()
                n = int(len(pair))
                if n == 0:
                    continue
                a = pair[cond_a].to_numpy(dtype=float)
                b = pair[cond_b].to_numpy(dtype=float)
                diff = b - a
                sd_diff = float(np.nanstd(diff, ddof=1)) if n > 1 else np.nan
                shapiro_w, shapiro_p = np.nan, np.nan
                if 3 <= n <= 5000 and np.nanstd(diff) > 0:
                    try:
                        shapiro_w, shapiro_p = shapiro(diff)
                    except Exception:
                        pass
                t_stat, t_p = np.nan, np.nan
                if n >= 2 and np.nanstd(diff) > 0:
                    try:
                        t_stat, t_p = ttest_rel(b, a, nan_policy="omit")
                    except Exception:
                        pass
                wilcoxon_stat, wilcoxon_p = np.nan, np.nan
                nonzero_diff_n = int(np.sum(np.abs(diff) > 1e-12))
                if n >= 2 and nonzero_diff_n > 0:
                    try:
                        wilcoxon_stat, wilcoxon_p = wilcoxon(b, a, zero_method="wilcox", alternative="two-sided", mode="auto")
                    except Exception:
                        pass
                rows.append(
                    {
                        "metric": metric_col,
                        "metric_name": metric_name,
                        "condition_a": cond_a,
                        "condition_b": cond_b,
                        "contrast": f"{cond_b} - {cond_a}",
                        "n_subjects_paired": n,
                        "mean_a": float(np.nanmean(a)),
                        "mean_b": float(np.nanmean(b)),
                        "median_a": float(np.nanmedian(a)),
                        "median_b": float(np.nanmedian(b)),
                        "mean_diff_b_minus_a": float(np.nanmean(diff)),
                        "median_diff_b_minus_a": float(np.nanmedian(diff)),
                        "sd_diff": sd_diff,
                        "cohen_dz": float(np.nanmean(diff) / sd_diff) if np.isfinite(sd_diff) and sd_diff > 0 else np.nan,
                        "shapiro_w_diff": float(shapiro_w) if np.isfinite(shapiro_w) else np.nan,
                        "shapiro_p_diff": float(shapiro_p) if np.isfinite(shapiro_p) else np.nan,
                        "paired_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                        "paired_t_p": float(t_p) if np.isfinite(t_p) else np.nan,
                        "wilcoxon_stat": float(wilcoxon_stat) if np.isfinite(wilcoxon_stat) else np.nan,
                        "wilcoxon_p": float(wilcoxon_p) if np.isfinite(wilcoxon_p) else np.nan,
                        "nonzero_diff_n": nonzero_diff_n,
                    }
                )
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        print("[WARN] No pairwise statistics were calculated.")
        return stats_df
    stats_df["paired_t_q_fdr"] = np.nan
    stats_df["wilcoxon_q_fdr"] = np.nan
    for metric_col in stats_df["metric"].unique():
        mask = stats_df["metric"] == metric_col
        stats_df.loc[mask, "paired_t_q_fdr"] = _bh_fdr(stats_df.loc[mask, "paired_t_p"].to_numpy())
        stats_df.loc[mask, "wilcoxon_q_fdr"] = _bh_fdr(stats_df.loc[mask, "wilcoxon_p"].to_numpy())
    stats_df.to_csv(out_dir / "pairwise_condition_statistics.csv", index=False)
    print(f"[OK] wrote {out_dir / 'pairwise_condition_statistics.csv'}")
    return stats_df


def parse_args() -> ECGConfig:
    p = argparse.ArgumentParser(description="LOSO ECG RRI/HR analysis against NASA-TLX and workload condition.")
    p.add_argument("--data-root", default="/projects/BLVMob/aria_seated/Data")
    p.add_argument("--tlx-csv", default="/projects/BLVMob/imu-rr-seated/Data/seated_tlx.csv")
    p.add_argument("--conditions", default="L0,L2,L3", help="Comma-separated conditions for LDA/test, e.g. L0,L1,L2,L3 or L1,L3")
    p.add_argument("--out-dir", default="ecg_tlx_loso_out")
    p.add_argument("--ecg-col", default="ecg")
    p.add_argument("--fs", type=float, default=250.0, help="ECG sampling rate.")
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--shift-sec", type=float, default=10.0)
    p.add_argument("--min-hr-bpm", type=float, default=35.0)
    p.add_argument("--max-hr-bpm", type=float, default=220.0)
    p.add_argument("--rest-condition", default="R")
    p.add_argument("--min-valid-fraction", type=float, default=0.8)
    p.add_argument("--min-windows-per-subject-condition", type=int, default=1)
    p.add_argument("--normalize-mode", choices=["relative", "difference", "ratio"], default="relative")
    p.add_argument(
        "--lda-features",
        default="rri_norm,hr_norm",
        help="Comma-separated ECG features for LDA. Options: rri_ms,hr_bpm,rri_norm,hr_norm,rest_rri_ms,rest_hr_bpm.",
    )
    args = p.parse_args()
    return ECGConfig(
        data_root=args.data_root,
        tlx_csv=args.tlx_csv,
        conditions=parse_conditions(args.conditions),
        out_dir=args.out_dir,
        ecg_col=args.ecg_col,
        fs=args.fs,
        window_sec=args.window_sec,
        shift_sec=args.shift_sec,
        min_hr_bpm=args.min_hr_bpm,
        max_hr_bpm=args.max_hr_bpm,
        rest_condition=args.rest_condition.upper(),
        min_valid_fraction=args.min_valid_fraction,
        min_windows_per_subject_condition=args.min_windows_per_subject_condition,
        normalize_mode=args.normalize_mode,
        lda_features=parse_feature_list(args.lda_features),
    )


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    features = build_feature_table(cfg)
    features.to_csv(out_dir / "ecg_tlx_features.csv", index=False)
    metrics, preds = run_loso_lda(features, cfg)
    metrics.to_csv(out_dir / "loso_lda_metrics.csv", index=False)
    preds.to_csv(out_dir / "loso_lda_predictions.csv", index=False)
    aggregate_plots(features, metrics, out_dir, cfg)
    pairwise_condition_statistics(features, cfg, out_dir)

    print(f"[OK] wrote {out_dir / 'ecg_tlx_features.csv'}")
    print(f"[OK] wrote {out_dir / 'loso_lda_metrics.csv'}")
    print(f"[OK] wrote {out_dir / 'loso_lda_predictions.csv'}")
    print(f"[OK] wrote {out_dir / 'condition_tlx_ecg_summary.csv'}")
    if not metrics.empty:
        cols = ["heldout_subject", "n_train", "n_test", "lda_acc", "lda_bal_acc", "lda_f1_macro"]
        print("\nLOSO LDA summary:")
        print(metrics[cols].to_string(index=False))
        print("\nMean LDA accuracy:", metrics["lda_acc"].mean(skipna=True))
        print("Mean LDA balanced accuracy:", metrics["lda_bal_acc"].mean(skipna=True))
        print("Mean LDA macro F1:", metrics["lda_f1_macro"].mean(skipna=True))


if __name__ == "__main__":
    main()

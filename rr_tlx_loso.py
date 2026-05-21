#!/usr/bin/env python3
"""
LOSO analysis of respiration-rate shifts vs NASA-TLX.

Input data layout is intentionally flexible. The script searches recursively under
--data-root for files named:

    M_pressure_df.csv, R_pressure_df.csv, L0_pressure_df.csv, ..., L3_pressure_df.csv

and infers the subject id from any path component that looks like S12, S013, etc.
The raw CSV only needs a breathing signal column. By default this is `Breathing`.

Pipeline:
  1. Estimate windowed respiration rate from each condition's breathing waveform.
  2. Compute each subject's rest baseline from R_pressure_df.csv.
  3. Normalize condition RR to rest: rr_norm = (rr_bpm - rest_bpm) / rest_bpm.
  4. For each LOSO fold:
       - train-subject-only regression: rr_norm -> NASA-TLX
       - train LDA on rr_norm -> condition labels
       - test LDA on the held-out subject
  5. Save fold metrics, per-window features, and regression plots.

Example:
  python rr_tlx_loso.py \
    --data-root /projects/BLVMob/aria_seated/Data \
    --tlx-csv /projects/BLVMob/imu-rr-seated/Data/seated_tlx.csv \
    --conditions L0,L1,L2,L3 \
    --out-dir results/rr_tlx_loso
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch
from scipy.stats import pearsonr, spearmanr, shapiro, ttest_rel, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VALID_CONDITIONS = ("M", "R", "L0", "L1", "L2", "L3")
LEVEL_CONDITIONS = ("L0", "L1", "L2", "L3")


@dataclass
class RRConfig:
    data_root: str
    tlx_csv: str
    conditions: List[str]
    out_dir: str
    breathing_col: str = "Breathing"
    fs: float = 18.0
    window_sec: float = 60.0
    shift_sec: float = 10.0
    min_bpm: float = 3.0
    max_bpm: float = 45.0
    rest_condition: str = "R"
    min_valid_fraction: float = 0.8
    min_windows_per_subject_condition: int = 1
    normalize_mode: str = "relative"  # relative: (rr-rest)/rest; difference: rr-rest; ratio: rr/rest
    regression_mode: str = "linear"    # linear, log_rr, log1p_rr_norm


def regression_x(df: pd.DataFrame, cfg: RRConfig) -> Tuple[np.ndarray, str]:
    """
    Build the regression predictor.

    linear:
      x = rr_norm

    log_rr:
      x = log(rr_bpm)
      Useful if absolute breathing rate has a multiplicative/nonlinear relation
      with NASA-TLX.

    log1p_rr_norm:
      x = log(1 + rr_norm)
      For normalize_mode='relative', this equals log(rr_bpm / rest_rr_bpm).
      This is usually the cleanest log transform for rest-normalized RR.
    """
    mode = str(cfg.regression_mode).lower()

    if mode == "linear":
        x = df["rr_norm"].to_numpy(dtype=float)
        label = "RR normalized to rest"

    elif mode == "log_rr":
        x = np.log(df["rr_bpm"].to_numpy(dtype=float))
        label = "log(RR BPM)"

    elif mode == "log1p_rr_norm":
        raw = df["rr_norm"].to_numpy(dtype=float)
        x = np.full_like(raw, np.nan, dtype=float)
        ok = raw > -1.0
        x[ok] = np.log1p(raw[ok])
        label = "log(1 + RR normalized to rest)"

    else:
        raise ValueError(
            "regression_mode must be one of: linear, log_rr, log1p_rr_norm"
        )

    return x.reshape(-1, 1), label


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


def infer_subject_from_path(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        m = re.search(r"S\s*0*(\d+)", part, flags=re.IGNORECASE)
        if m:
            return f"S{int(m.group(1)):02d}"
    return None


def discover_pressure_files(data_root: Path) -> Dict[str, Dict[str, Path]]:
    """Return {subject: {condition: csv_path}}."""
    mapping: Dict[str, Dict[str, Path]] = {}
    for path in sorted(data_root.rglob("*_pressure_df.csv")):
        cond = path.name.replace("_pressure_df.csv", "").upper()
        if cond not in VALID_CONDITIONS:
            continue
        sbj = infer_subject_from_path(path)
        if sbj is None:
            # Fallback: if there is exactly one anonymous directory, use its parent name.
            # This keeps the failure explicit downstream if the parent is not a subject id.
            sbj = canonical_subject_id(path.parent.name)
        mapping.setdefault(sbj, {})[cond] = path
    return mapping


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def read_breathing_signal(path: Path, breathing_col: str) -> np.ndarray:
    df = pd.read_csv(path)
    df = clean_columns(df)
    col_lookup = {c.lower(): c for c in df.columns}
    key = [col for col in col_lookup if breathing_col.lower() in col][0]
    if key not in col_lookup:
        raise KeyError(f"{path}: could not find breathing column '{breathing_col}'. Columns={list(df.columns)}")
    x = pd.to_numeric(df[col_lookup[key]], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return x


def estimate_rr_bpm_window(x: np.ndarray, fs: float, min_bpm: float, max_bpm: float) -> float:
    if x.size < max(8, int(fs * 5)):
        return float("nan")
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        x = x[np.isfinite(x)]
    if x.size < max(8, int(fs * 5)):
        return float("nan")
    x = detrend(x - np.nanmedian(x))
    if np.nanstd(x) <= 1e-12:
        return float("nan")

    nperseg = min(len(x), int(round(fs * 30)))
    if nperseg < 8:
        return float("nan")
    freqs, power = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    bpm = freqs * 60.0
    mask = (bpm >= min_bpm) & (bpm <= max_bpm)
    if not np.any(mask):
        return float("nan")
    return float(bpm[mask][np.argmax(power[mask])])


def windowed_rr_bpm(
    x: np.ndarray,
    fs: float,
    window_sec: float,
    shift_sec: float,
    min_bpm: float,
    max_bpm: float,
    min_valid_fraction: float,
) -> np.ndarray:
    win = int(round(window_sec * fs))
    shift = int(round(shift_sec * fs))
    if win <= 0 or shift <= 0:
        raise ValueError("window_sec and shift_sec must be positive")
    if len(x) < win:
        rr = estimate_rr_bpm_window(x, fs, min_bpm, max_bpm)
        return np.asarray([rr], dtype=float) if np.isfinite(rr) else np.asarray([], dtype=float)

    vals: List[float] = []
    for start in range(0, len(x) - win + 1, shift):
        seg = x[start : start + win]
        valid_fraction = float(np.isfinite(seg).mean()) if seg.size else 0.0
        if valid_fraction < min_valid_fraction:
            continue
        rr = estimate_rr_bpm_window(seg, fs, min_bpm, max_bpm)
        if np.isfinite(rr):
            vals.append(rr)
    return np.asarray(vals, dtype=float)


def load_tlx_table(path: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path)
    df = clean_columns(df)
    lower = {c.lower(): c for c in df.columns}
    subject_col = lower.get("subject") or lower.get("subj") or lower.get("participant") or df.columns[0]
    table: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        sbj = canonical_subject_id(row[subject_col])
        scores: Dict[str, float] = {}
        for cond in LEVEL_CONDITIONS:
            if cond in df.columns:
                val = pd.to_numeric(row[cond], errors="coerce")
            elif cond.lower() in lower:
                val = pd.to_numeric(row[lower[cond.lower()]], errors="coerce")
            else:
                val = np.nan
            if pd.notna(val):
                scores[cond] = float(val)
        if scores:
            table[sbj] = scores
    return table


def normalize_rr(rr: np.ndarray, rest_bpm: float, mode: str) -> np.ndarray:
    rr = np.asarray(rr, dtype=float)
    if not np.isfinite(rest_bpm) or rest_bpm <= 0:
        return np.full_like(rr, np.nan, dtype=float)
    if mode == "relative":
        return (rr - rest_bpm) / rest_bpm
    if mode == "difference":
        return rr - rest_bpm
    if mode == "ratio":
        return rr / rest_bpm
    raise ValueError("normalize_mode must be one of: relative, difference, ratio")


def build_feature_table(cfg: RRConfig) -> pd.DataFrame:
    data_root = Path(cfg.data_root)
    tlx = load_tlx_table(Path(cfg.tlx_csv))
    files = discover_pressure_files(data_root)

    rows: List[dict] = []
    missing_rest: List[str] = []
    for sbj, cond_files in sorted(files.items()):
        rest_path = cond_files.get(cfg.rest_condition)
        if rest_path is None:
            missing_rest.append(sbj)
            continue
        rest_signal = read_breathing_signal(rest_path, cfg.breathing_col)
        rest_rr = windowed_rr_bpm(
            rest_signal,
            fs=cfg.fs,
            window_sec=cfg.window_sec,
            shift_sec=cfg.shift_sec,
            min_bpm=cfg.min_bpm,
            max_bpm=cfg.max_bpm,
            min_valid_fraction=cfg.min_valid_fraction,
        )
        if rest_rr.size == 0:
            missing_rest.append(sbj)
            continue
        rest_bpm = float(np.nanmedian(rest_rr))

        for cond in cfg.conditions:
            path = cond_files.get(cond)
            if path is None:
                continue
            sig = read_breathing_signal(path, cfg.breathing_col)
            rr = windowed_rr_bpm(
                sig,
                fs=cfg.fs,
                window_sec=cfg.window_sec,
                shift_sec=cfg.shift_sec,
                min_bpm=cfg.min_bpm,
                max_bpm=cfg.max_bpm,
                min_valid_fraction=cfg.min_valid_fraction,
            )
            if rr.size < cfg.min_windows_per_subject_condition:
                continue
            rr_norm = normalize_rr(rr, rest_bpm, cfg.normalize_mode)
            tlx_score = tlx.get(sbj, {}).get(cond, np.nan)
            for i, (r, rn) in enumerate(zip(rr, rr_norm)):
                rows.append(
                    {
                        "subject": sbj,
                        "condition": cond,
                        "window_idx": i,
                        "rr_bpm": float(r),
                        "rest_rr_bpm": rest_bpm,
                        "rr_norm": float(rn),
                        "tlx": float(tlx_score) if np.isfinite(tlx_score) else np.nan,
                        "source_file": str(path),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No usable respiration windows were found. Check --data-root, subject folder names, "
            "--breathing-col, --fs, and --window-sec."
        )
    df = df[np.isfinite(df["rr_norm"])].reset_index(drop=True)
    if missing_rest:
        print(f"[WARN] Skipped {len(missing_rest)} subjects without usable {cfg.rest_condition} baseline: {missing_rest}")
    return df


def regression_summary(
    train_df: pd.DataFrame, cfg: RRConfig
) -> Tuple[dict, Optional[LinearRegression]]:
    d = train_df.dropna(subset=["rr_norm", "tlx"])
    if len(d) < 3:
        return {
            "regression_mode": cfg.regression_mode,
            "reg_n": len(d),
            "reg_r": np.nan,
            "reg_p": np.nan,
            "reg_spearman": np.nan,
            "reg_spearman_p": np.nan,
            "reg_slope": np.nan,
            "reg_intercept": np.nan,
        }, None

    x, _ = regression_x(d, cfg)
    y = d["tlx"].to_numpy()

    ok = np.isfinite(x.ravel()) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(y) < 3 or np.unique(x.ravel()).size < 2:
        return {
            "regression_mode": cfg.regression_mode,
            "reg_n": int(len(y)),
            "reg_r": np.nan,
            "reg_p": np.nan,
            "reg_spearman": np.nan,
            "reg_spearman_p": np.nan,
            "reg_slope": np.nan,
            "reg_intercept": np.nan,
        }, None

    y = d["tlx"].to_numpy()
    model = LinearRegression().fit(x, y)
    r, p = pearsonr(x.ravel(), y)
    sr, sp = spearmanr(x.ravel(), y)
    return {
        "regression_mode": cfg.regression_mode,
        "reg_n": int(len(d)),
        "reg_r": float(r),
        "reg_p": float(p),
        "reg_spearman": float(sr),
        "reg_spearman_p": float(sp),
        "reg_slope": float(model.coef_[0]),
        "reg_intercept": float(model.intercept_),
    }, model


def plot_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: Optional[LinearRegression],
    out_path: Path,
    title: str,
    cfg: RRConfig,
) -> None:
    plt.figure(figsize=(6.5, 5.0))
    dtrain = train_df.dropna(subset=["rr_norm", "tlx"])
    dtest = test_df.dropna(subset=["rr_norm", "tlx"])

    x_train_all, xlabel = regression_x(dtrain, cfg)
    dtrain = dtrain.copy()
    dtrain["_reg_x"] = x_train_all.ravel()
    dtrain = dtrain[np.isfinite(dtrain["_reg_x"])]

    if len(dtest):
        x_test_all, _ = regression_x(dtest, cfg)
        dtest = dtest.copy()
        dtest["_reg_x"] = x_test_all.ravel()
        dtest = dtest[np.isfinite(dtest["_reg_x"])]

    for cond, g in dtrain.groupby("condition"):
        plt.scatter(g["_reg_x"], g["tlx"], s=18, alpha=0.55, label=f"train {cond}")

    if len(dtest):
        plt.scatter(dtest["_reg_x"], dtest["tlx"], s=34, marker="x", label="held-out")

    if model is not None and len(dtrain):
        lo = float(dtrain["_reg_x"].min())
        hi = float(dtrain["_reg_x"].max())
        xs = np.linspace(lo, hi, 100).reshape(-1, 1)
        ys = model.predict(xs)
        plt.plot(xs.ravel(), ys, linewidth=2, label="train fit")

    plt.xlabel(xlabel)
    plt.ylabel("NASA-TLX")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_loso(df: pd.DataFrame, cfg: RRConfig, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(df["subject"].unique())
    metrics_rows: List[dict] = []
    pred_rows: List[dict] = []
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for heldout in subjects:
        train = df[df["subject"] != heldout].copy()
        test = df[df["subject"] == heldout].copy()

        # Regression is deliberately fitted only on training subjects in each fold.
        reg_stats, reg_model = regression_summary(train, cfg)
        plot_regression(
            train,
            test,
            reg_model,
            plot_dir / f"regression_{cfg.regression_mode}_train_fit_test_{heldout}.png",
            f"LOSO {heldout}: train RR shift vs TLX ({cfg.regression_mode})",
            cfg,
        )

        # LDA classification on normalized RR only.
        train_cls = train.dropna(subset=["rr_norm", "condition"])
        test_cls = test.dropna(subset=["rr_norm", "condition"])
        present_train_classes = sorted(train_cls["condition"].unique())
        present_test_classes = sorted(test_cls["condition"].unique())

        fold = {
            "heldout_subject": heldout,
            "n_train": int(len(train_cls)),
            "n_test": int(len(test_cls)),
            "train_classes": json.dumps(present_train_classes),
            "test_classes": json.dumps(present_test_classes),
            **reg_stats,
        }

        if len(present_train_classes) < 2 or len(test_cls) == 0:
            fold.update({"lda_acc": np.nan, "lda_bal_acc": np.nan, "lda_f1_macro": np.nan})
            metrics_rows.append(fold)
            continue

        clf = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
        x_train = train_cls[["rr_norm"]].to_numpy()
        y_train = train_cls["condition"].to_numpy()
        x_test = test_cls[["rr_norm"]].to_numpy()
        y_test = test_cls["condition"].to_numpy()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        fold.update(
            {
                "lda_acc": float(accuracy_score(y_test, y_pred)),
                "lda_bal_acc": float(balanced_accuracy_score(y_test, y_pred)),
                "lda_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "confusion_labels": json.dumps(sorted(set(y_train).union(set(y_test)))),
                "confusion_matrix": json.dumps(confusion_matrix(y_test, y_pred, labels=sorted(set(y_train).union(set(y_test)))).tolist()),
                "classification_report": json.dumps(classification_report(y_test, y_pred, zero_division=0, output_dict=True)),
            }
        )
        metrics_rows.append(fold)

        tmp = test_cls[["subject", "condition", "window_idx", "rr_bpm", "rest_rr_bpm", "rr_norm", "tlx"]].copy()
        tmp["heldout_subject"] = heldout
        tmp["pred_condition"] = y_pred
        pred_rows.extend(tmp.to_dict(orient="records"))

    metrics = pd.DataFrame(metrics_rows)
    preds = pd.DataFrame(pred_rows)
    return metrics, preds


def aggregate_plots(
    df: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, cfg: RRConfig
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    d = df.dropna(subset=["rr_norm", "tlx"])
    if len(d) >= 3:
        x, _ = regression_x(d, cfg)
        y = d["tlx"].to_numpy(dtype=float)
        ok = np.isfinite(x.ravel()) & np.isfinite(y)
        if ok.sum() >= 3 and np.unique(x.ravel()[ok]).size >= 2:
            model = LinearRegression().fit(x[ok], y[ok])
            plot_regression(
                d,
                d.iloc[0:0],
                model,
                plot_dir / f"regression_{cfg.regression_mode}_all_subjects.png",
                f"All subjects: RR shift vs NASA-TLX ({cfg.regression_mode})",
                cfg,
            )

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

def plot_condition_tlx_rr_boxplots(
    df:pd.DataFrame, cfg:RRConfig, out_dir:Path
) -> None:
    """
    Boxplot NASA-TLX by condition, overlaid with subject-condition median RR.

    Uses one row per subject-condition so repeated RR windows do not duplicate
    the same TLX score in the boxplot.
    """
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Keep condition/TLX rows even if rr_norm is missing so requested
    # conditions still appear in boxplots (RR overlays are handled per metric).
    d = df.dropna(subset=["subject", "condition", "tlx"]).copy()
    d = d[d["condition"].isin(cfg.conditions)]
    if d.empty:
        print("[WARN] No condition/TLX/RR rows available for condition boxplots.")
        return

    summary = (
        d.groupby(["subject", "condition"], as_index=False)
        .agg(
            tlx=("tlx", "first"),
            rr_bpm=("rr_bpm", "median"),
            rr_norm=("rr_norm", "median"),
            rest_rr_bpm=("rest_rr_bpm", "first"),
            n_windows=("rr_bpm", "size"),
        )
    )
    summary.to_csv(out_dir / "condition_tlx_rr_summary.csv", index=False)

    conds = list(cfg.conditions)
    present_conds = set(summary["condition"])
    if not present_conds:
        print("[WARN] No requested conditions found for condition boxplots.")
        return

    missing_conds = [c for c in conds if c not in present_conds]
    if missing_conds:
        print(f"[WARN] No rows found for requested conditions in boxplots: {missing_conds}")

    def _plot(rr_col: str, out_name: str, rr_label: str) -> None:
        fig, ax1 = plt.subplots(figsize=(8.5, 5.5))

        tlx_groups = [
            summary.loc[summary["condition"] == c, "tlx"].dropna().to_numpy()
            for c in conds
        ]
        positions = np.arange(1, len(conds) + 1)
        non_empty = [i for i, vals in enumerate(tlx_groups) if len(vals)]

        if not non_empty:
            print(f"[WARN] No TLX values available to plot for {out_name}.")
            plt.close(fig)
            return

        bp = ax1.boxplot(
            [tlx_groups[i] for i in non_empty],
            positions=positions[non_empty],
            widths=0.55,
            showmeans=False,
            patch_artist=False,
        )

        ax1.set_xticks(positions)
        ax1.set_xticklabels(conds)
        ax1.set_xlabel("Condition")
        ax1.set_ylabel("NASA-TLX")

        # ---------------------------------------------------------
        # Overlay respiration points
        # ---------------------------------------------------------
        ax2 = ax1.twinx()

        rng = np.random.default_rng(0)

        for i, cond in enumerate(conds, start=1):
            vals = summary.loc[
                summary["condition"] == cond,
                rr_col,
            ].dropna().to_numpy()

            if vals.size == 0:
                continue

            jitter = rng.normal(
                loc=0.0,
                scale=0.045,
                size=vals.size,
            )

            ax2.scatter(
                np.full(vals.size, i, dtype=float) + jitter,
                vals,
                s=30,
                alpha=0.75,
                marker="o",
            )

            ax2.plot(
                [i - 0.22, i + 0.22],
                [np.nanmedian(vals), np.nanmedian(vals)],
                linewidth=2,
            )

        ax2.set_ylabel(rr_label)

        # ---------------------------------------------------------
        # Pairwise significance markers
        # ---------------------------------------------------------
        stats_path = out_dir / "pairwise_condition_statistics.csv"

        if stats_path.exists():
            stats_df = pd.read_csv(stats_path)

            stats_df = stats_df[
                (stats_df["metric"] == rr_col)
            ].copy()

            # Use FDR-corrected Wilcoxon significance
            stats_df = stats_df[
                stats_df["wilcoxon_q_fdr"] < 0.05
            ]

            if not stats_df.empty:

                ymax = max(
                    [np.nanmax(g) for g in tlx_groups if len(g)]
                )

                yrange = max(ymax * 0.15, 5.0)
                current_y = ymax + yrange * 0.25
                step = yrange * 0.18

                cond_to_pos = {
                    cond: pos
                    for cond, pos in zip(conds, positions)
                }

                for _, row in stats_df.iterrows():

                    a = row["condition_a"]
                    b = row["condition_b"]

                    if a not in cond_to_pos or b not in cond_to_pos:
                        continue

                    x1 = cond_to_pos[a]
                    x2 = cond_to_pos[b]

                    q = float(row["wilcoxon_q_fdr"])

                    if q < 0.001:
                        stars = "***"
                    elif q < 0.01:
                        stars = "**"
                    elif q < 0.05:
                        stars = "*"
                    else:
                        continue

                    # bracket
                    ax1.plot(
                        [x1, x1, x2, x2],
                        [
                            current_y,
                            current_y + step * 0.25,
                            current_y + step * 0.25,
                            current_y,
                        ],
                        linewidth=1.5,
                    )

                    # stars
                    ax1.text(
                        (x1 + x2) / 2,
                        current_y + step * 0.30,
                        stars,
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )

                    current_y += step

                ax1.set_ylim(top=current_y + step)

        ax1.set_title(
            "NASA-TLX by condition with overlaid respiration rate"
        )

        fig.tight_layout()
        fig.savefig(plot_dir / out_name, dpi=300)
        plt.close(fig)

    _plot(
        "rr_bpm",
        "condition_tlx_boxplot_overlay_rr_bpm.png",
        "Respiration rate, median per subject-condition (BPM)",
    )
    _plot(
        "rr_norm",
        "condition_tlx_boxplot_overlay_rr_norm.png",
        f"Respiration rate normalized to {cfg.rest_condition}",
    )

def _bh_fdr(pvals: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR correction.
    Returns q-values in the original order.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q.tolist()

    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = float(len(ranked))

    adj = ranked * m / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    q[order] = adj
    return q.tolist()


def pairwise_condition_statistics(
    df: pd.DataFrame,
    cfg: RRConfig,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Pairwise within-subject condition comparisons.

    Operates on one subject-condition summary row, not raw windows, so subjects
    with more valid respiration windows do not dominate the inference.
    """
    d = df.dropna(subset=["subject", "condition", "tlx", "rr_bpm", "rr_norm"]).copy()
    d = d[d["condition"].isin(cfg.conditions)]
    if d.empty:
        print("[WARN] No condition/TLX/RR rows available for pairwise statistics.")
        return pd.DataFrame()

    summary = (
        d.groupby(["subject", "condition"], as_index=False)
        .agg(
            tlx=("tlx", "first"),
            rr_bpm=("rr_bpm", "median"),
            rr_norm=("rr_norm", "median"),
            rest_rr_bpm=("rest_rr_bpm", "first"),
            n_windows=("rr_bpm", "size"),
        )
    )

    conds = [c for c in cfg.conditions if c in set(summary["condition"])]
    metrics = [
        ("tlx", "NASA-TLX"),
        ("rr_bpm", "Respiration rate BPM"),
        ("rr_norm", f"Respiration rate normalized to {cfg.rest_condition}"),
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

                mean_a = float(np.nanmean(a))
                mean_b = float(np.nanmean(b))
                median_a = float(np.nanmedian(a))
                median_b = float(np.nanmedian(b))
                mean_diff = float(np.nanmean(diff))
                median_diff = float(np.nanmedian(diff))
                sd_diff = float(np.nanstd(diff, ddof=1)) if n > 1 else np.nan

                # Paired standardized effect: Cohen's dz = mean paired diff / SD paired diff.
                cohen_dz = (
                    float(mean_diff / sd_diff)
                    if n > 1 and np.isfinite(sd_diff) and sd_diff > 0
                    else np.nan
                )

                # Normality of paired differences is assessed only when meaningful.
                shapiro_w = np.nan
                shapiro_p = np.nan
                if n >= 3 and n <= 5000 and np.nanstd(diff) > 0:
                    try:
                        shapiro_w, shapiro_p = shapiro(diff)
                        shapiro_w = float(shapiro_w)
                        shapiro_p = float(shapiro_p)
                    except Exception:
                        pass

                t_stat = np.nan
                t_p = np.nan
                if n >= 2 and np.nanstd(diff) > 0:
                    try:
                        t_stat, t_p = ttest_rel(b, a, nan_policy="omit")
                        t_stat = float(t_stat)
                        t_p = float(t_p)
                    except Exception:
                        pass

                wilcoxon_stat = np.nan
                wilcoxon_p = np.nan
                nonzero_diff_n = int(np.sum(np.abs(diff) > 1e-12))
                if n >= 2 and nonzero_diff_n > 0:
                    try:
                        wilcoxon_stat, wilcoxon_p = wilcoxon(
                            b,
                            a,
                            zero_method="wilcox",
                            alternative="two-sided",
                            mode="auto",
                        )
                        wilcoxon_stat = float(wilcoxon_stat)
                        wilcoxon_p = float(wilcoxon_p)
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
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "median_a": median_a,
                        "median_b": median_b,
                        "mean_diff_b_minus_a": mean_diff,
                        "median_diff_b_minus_a": median_diff,
                        "sd_diff": sd_diff,
                        "cohen_dz": cohen_dz,
                        "shapiro_w_diff": shapiro_w,
                        "shapiro_p_diff": shapiro_p,
                        "paired_t_stat": t_stat,
                        "paired_t_p": t_p,
                        "wilcoxon_stat": wilcoxon_stat,
                        "wilcoxon_p": wilcoxon_p,
                        "nonzero_diff_n": nonzero_diff_n,
                    }
                )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        print("[WARN] No pairwise statistics were calculated.")
        return stats_df

    # Correct within each metric family because TLX, RR BPM, and RR norm answer
    # related but distinct questions.
    stats_df["paired_t_q_fdr"] = np.nan
    stats_df["wilcoxon_q_fdr"] = np.nan
    for metric_col in stats_df["metric"].unique():
        mask = stats_df["metric"] == metric_col
        stats_df.loc[mask, "paired_t_q_fdr"] = _bh_fdr(stats_df.loc[mask, "paired_t_p"].to_numpy())
        stats_df.loc[mask, "wilcoxon_q_fdr"] = _bh_fdr(stats_df.loc[mask, "wilcoxon_p"].to_numpy())

    stats_df.to_csv(out_dir / "pairwise_condition_statistics.csv", index=False)
    print(f"[OK] wrote {out_dir / 'pairwise_condition_statistics.csv'}")
    return stats_df

def parse_args() -> RRConfig:
    p = argparse.ArgumentParser(description="LOSO RR-normalized-to-rest analysis against NASA-TLX and workload condition.")
    p.add_argument("--data-root", default="/projects/BLVMob/aria_seated/Data")
    p.add_argument("--tlx-csv", default="/projects/BLVMob/imu-rr-seated/Data/seated_tlx.csv")
    p.add_argument("--conditions", default="L0,L2,L3", help="Comma-separated conditions for LDA/test, e.g. L0,L1,L2,L3 or L1,L3")
    p.add_argument("--out-dir", default="rr_tlx_loso_out")
    p.add_argument("--breathing-col", default="Breathing")
    p.add_argument("--fs", type=float, default=18.0, help="Sampling rate of the breathing/pressure signal. Default matches project BR_FS.")
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--shift-sec", type=float, default=10.0)
    p.add_argument("--min-bpm", type=float, default=3.0)
    p.add_argument("--max-bpm", type=float, default=45.0)
    p.add_argument("--rest-condition", default="R")
    p.add_argument("--min-valid-fraction", type=float, default=0.8)
    p.add_argument("--min-windows-per-subject-condition", type=int, default=1)
    p.add_argument("--normalize-mode", choices=["relative", "difference", "ratio"], default="relative")
    p.add_argument(
        "--regression-mode",
        choices=["linear", "log_rr", "log1p_rr_norm"],
        default="linear",
        help=(
            "Regression predictor transform. "
            "linear uses rr_norm; log_rr uses log(rr_bpm); "
            "log1p_rr_norm uses log(1 + rr_norm), equivalent to log(rr/rest) "
            "when --normalize-mode relative."
        ),
    )
    args = p.parse_args()
    return RRConfig(
        data_root=args.data_root,
        tlx_csv=args.tlx_csv,
        conditions=parse_conditions(args.conditions),
        out_dir=args.out_dir,
        breathing_col=args.breathing_col,
        fs=args.fs,
        window_sec=args.window_sec,
        shift_sec=args.shift_sec,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        rest_condition=args.rest_condition.upper(),
        min_valid_fraction=args.min_valid_fraction,
        min_windows_per_subject_condition=args.min_windows_per_subject_condition,
        normalize_mode=args.normalize_mode,
        regression_mode=args.regression_mode,
    )


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    features = build_feature_table(cfg)
    features.to_csv(out_dir / "rr_tlx_features.csv", index=False)

    metrics, preds = run_loso(features, cfg, out_dir)
    metrics.to_csv(out_dir / "loso_lda_regression_metrics.csv", index=False)
    preds.to_csv(out_dir / "loso_lda_predictions.csv", index=False)
    aggregate_plots(features, metrics, out_dir, cfg)
    pairwise_condition_statistics(features, cfg, out_dir)
    plot_condition_tlx_rr_boxplots(features, cfg, out_dir)

    print(f"[OK] wrote {out_dir / 'rr_tlx_features.csv'}")
    print(f"[OK] wrote {out_dir / 'loso_lda_regression_metrics.csv'}")
    print(f"[OK] wrote {out_dir / 'loso_lda_predictions.csv'}")
    print(f"[OK] wrote {out_dir / 'condition_tlx_rr_summary.csv'}")
    if not metrics.empty:
        print("\nLOSO summary:")
        print(
            metrics[
                ["heldout_subject", "regression_mode", "n_train", "n_test",
                 "lda_acc", "lda_bal_acc", "lda_f1_macro", "reg_r", "reg_p"]
            ].to_string(index=False)
        )
        print("\nMean LDA balanced accuracy:", metrics["lda_bal_acc"].mean(skipna=True))


if __name__ == "__main__":
    main()

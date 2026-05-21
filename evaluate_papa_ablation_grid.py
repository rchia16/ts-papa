#!/usr/bin/env python3
"""
Collect PAPA ablation/TTA results into direct-comparison tables.

Expected run layout:

  runs/papa_ablation_grid/<STAMP>/
    full_none/
      summary.csv
      frozen_embedding_summary.csv
      papa_summary.csv
    full_tent/
      ...
    no_bottleneck_papa/
      ...

Outputs:
  <root>/analysis/
    papa_rows_long.csv
    papa_comparison_by_run.csv
    papa_comparison_by_tag.csv
    papa_post_tta_pivot.csv
    papa_pre_vs_post_delta.csv
    frozen_embedding_comparison_by_run.csv
    core_rr_pressure_summary_by_run.csv
    papa_comparison_by_tag.tex
    papa_post_tta_pivot.tex

Example:
  python evaluate_papa_ablation_grid.py --root runs/papa_ablation_grid/20260521T000000Z
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


METRIC_COLS = [
    "papa_acc",
    "papa_f1_macro",
    "papa_f1_weighted",
]

CORE_METRIC_COLS = [
    "loss",
    "stft",
    "rr_loss",
    "spec_mae",
    "spec_rmse",
    "spec_corr",
    "rr_mae",
    "rr_rmse",
    "rr_corr",
    "n_windows",
]

FROZEN_METRIC_COLS = [
    "embed_acc",
    "embed_f1_macro",
    "embed_f1_weighted",
    "embed_n_train",
    "embed_n_test",
    "embed_n_features",
    "embed_n_classes_train",
    "embed_n_classes_test",
    "tlx_available",
    "tlx_n_train",
    "tlx_n_test",
    "tlx_mae",
    "tlx_rmse",
    "tlx_r2",
    "tlx_corr",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Root directory containing ablation run folders.")
    p.add_argument("--out-dir", default=None, help="Output directory. Defaults to <root>/analysis.")
    p.add_argument("--sort-by", default="papa_f1_macro", choices=METRIC_COLS)
    p.add_argument("--higher-is-better", action="store_true", default=True)
    p.add_argument("--tex", action="store_true", help="Also write LaTeX tables.")
    return p.parse_args()


def infer_flags_from_run_name(name: str) -> Dict[str, object]:
    n = name.lower()

    no_bottleneck = "no_bottleneck" in n
    no_adapter = "no_adapter" in n

    method = "none"
    for m in ["tent", "nrc", "cotta", "papa"]:
        if re.search(rf"(^|_){m}($|_)", n):
            method = m
            break

    if re.search(r"(^|_)none($|_)", n):
        method = "none"

    return {
        "run": name,
        "papa_tta": method,
        "use_bottleneck": int(not no_bottleneck),
        "use_adapter": int(not no_adapter),
        "ablation": (
            "full"
            if not no_bottleneck and not no_adapter
            else "no_bottleneck_no_adapter"
            if no_bottleneck and no_adapter
            else "no_bottleneck"
            if no_bottleneck
            else "no_adapter"
        ),
    }


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def add_run_metadata(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    meta = infer_flags_from_run_name(run_dir.name)
    out = df.copy()
    for k, v in meta.items():
        out[k] = v
    out["run_dir"] = str(run_dir)
    return out


def numeric_summary(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return pd.DataFrame()

    agg = (
        df.groupby(group_cols, dropna=False)[available]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten MultiIndex columns.
    flat_cols = []
    for col in agg.columns:
        if isinstance(col, tuple):
            name = "_".join([str(x) for x in col if str(x)])
            flat_cols.append(name)
        else:
            flat_cols.append(str(col))
    agg.columns = flat_cols
    return agg


def add_delta_rows(papa_long: pd.DataFrame) -> pd.DataFrame:
    if papa_long.empty:
        return pd.DataFrame()

    if "tag" not in papa_long.columns:
        return pd.DataFrame()

    keys = ["run", "subject", "ablation", "papa_tta", "use_bottleneck", "use_adapter"]
    rows = []

    for key_vals, g in papa_long.groupby(keys, dropna=False):
        pre = g[g["tag"] == "pre_tta"]
        post = g[g["tag"] == "post_tta"]
        if pre.empty or post.empty:
            continue

        pre_row = pre.iloc[-1]
        post_row = post.iloc[-1]
        row = dict(zip(keys, key_vals))
        row["run_dir"] = post_row.get("run_dir", "")

        for c in METRIC_COLS:
            if c in pre_row and c in post_row:
                row[f"{c}_pre"] = float(pre_row[c])
                row[f"{c}_post"] = float(post_row[c])
                row[f"{c}_delta"] = float(post_row[c]) - float(pre_row[c])

        rows.append(row)

    return pd.DataFrame(rows)


def build_post_tta_pivot(papa_long: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if papa_long.empty:
        return pd.DataFrame()

    if "tag" in papa_long.columns:
        df = papa_long[papa_long["tag"].isin(["post_tta", "pre_tta"])].copy()
        # Prefer post_tta when available, otherwise pre_tta.
        order = {"pre_tta": 0, "post_tta": 1}
        df["_tag_order"] = df["tag"].map(order).fillna(0)
        df = (
            df.sort_values("_tag_order")
            .groupby(["run", "subject"], as_index=False, dropna=False)
            .tail(1)
            .drop(columns=["_tag_order"])
        )
    else:
        df = papa_long.copy()

    summary = numeric_summary(
        df,
        ["ablation", "papa_tta", "use_bottleneck", "use_adapter"],
        METRIC_COLS,
    )

    mean_col = f"{sort_by}_mean"
    if mean_col in summary.columns:
        summary = summary.sort_values(mean_col, ascending=False)

    return summary


def safe_to_latex(df: pd.DataFrame, path: Path, float_format: str = "%.3f") -> None:
    try:
        path.write_text(df.to_latex(index=False, float_format=float_format), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Failed to write LaTeX {path}: {e}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    run_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name != "analysis"])
    if not run_dirs:
        raise RuntimeError(f"No run directories found under {root}")

    papa_rows = []
    frozen_rows = []
    core_rows = []

    for run_dir in run_dirs:
        papa = read_csv_if_exists(run_dir / "papa_summary.csv")
        if papa is not None:
            papa_rows.append(add_run_metadata(papa, run_dir))

        frozen = read_csv_if_exists(run_dir / "frozen_embedding_summary.csv")
        if frozen is not None:
            frozen_rows.append(add_run_metadata(frozen, run_dir))

        core = read_csv_if_exists(run_dir / "summary.csv")
        if core is not None:
            core_rows.append(add_run_metadata(core, run_dir))

    papa_long = pd.concat(papa_rows, ignore_index=True) if papa_rows else pd.DataFrame()
    frozen_long = pd.concat(frozen_rows, ignore_index=True) if frozen_rows else pd.DataFrame()
    core_long = pd.concat(core_rows, ignore_index=True) if core_rows else pd.DataFrame()

    papa_long.to_csv(out_dir / "papa_rows_long.csv", index=False)
    frozen_long.to_csv(out_dir / "frozen_embedding_rows_long.csv", index=False)
    core_long.to_csv(out_dir / "core_rows_long.csv", index=False)

    if not papa_long.empty:
        papa_by_run = numeric_summary(
            papa_long,
            ["run", "ablation", "papa_tta", "use_bottleneck", "use_adapter", "tag"],
            METRIC_COLS,
        )
        papa_by_run.to_csv(out_dir / "papa_comparison_by_run.csv", index=False)

        papa_by_tag = numeric_summary(
            papa_long,
            ["ablation", "papa_tta", "use_bottleneck", "use_adapter", "tag"],
            METRIC_COLS,
        )
        papa_by_tag.to_csv(out_dir / "papa_comparison_by_tag.csv", index=False)

        post_pivot = build_post_tta_pivot(papa_long, args.sort_by)
        post_pivot.to_csv(out_dir / "papa_post_tta_pivot.csv", index=False)

        deltas = add_delta_rows(papa_long)
        deltas.to_csv(out_dir / "papa_pre_vs_post_delta.csv", index=False)

        if not deltas.empty:
            delta_summary = numeric_summary(
                deltas,
                ["ablation", "papa_tta", "use_bottleneck", "use_adapter"],
                [f"{m}_delta" for m in METRIC_COLS],
            )
            delta_summary.to_csv(out_dir / "papa_pre_vs_post_delta_summary.csv", index=False)

        if args.tex:
            safe_to_latex(papa_by_tag, out_dir / "papa_comparison_by_tag.tex")
            safe_to_latex(post_pivot, out_dir / "papa_post_tta_pivot.tex")
            if not deltas.empty:
                safe_to_latex(delta_summary, out_dir / "papa_pre_vs_post_delta_summary.tex")

    if not frozen_long.empty:
        frozen_by_run = numeric_summary(
            frozen_long,
            ["run", "ablation", "papa_tta", "use_bottleneck", "use_adapter", "tag"],
            FROZEN_METRIC_COLS,
        )
        frozen_by_run.to_csv(out_dir / "frozen_embedding_comparison_by_run.csv", index=False)

        frozen_by_tag = numeric_summary(
            frozen_long,
            ["ablation", "papa_tta", "use_bottleneck", "use_adapter", "tag"],
            FROZEN_METRIC_COLS,
        )
        frozen_by_tag.to_csv(out_dir / "frozen_embedding_comparison_by_tag.csv", index=False)

        if args.tex:
            safe_to_latex(frozen_by_tag, out_dir / "frozen_embedding_comparison_by_tag.tex")

    if not core_long.empty:
        core_by_run = numeric_summary(
            core_long,
            ["run", "ablation", "papa_tta", "use_bottleneck", "use_adapter"],
            CORE_METRIC_COLS,
        )
        core_by_run.to_csv(out_dir / "core_rr_pressure_summary_by_run.csv", index=False)

        core_by_tag = numeric_summary(
            core_long,
            ["ablation", "papa_tta", "use_bottleneck", "use_adapter"],
            CORE_METRIC_COLS,
        )
        core_by_tag.to_csv(out_dir / "core_rr_pressure_summary_by_tag.csv", index=False)

        if args.tex:
            safe_to_latex(core_by_tag, out_dir / "core_rr_pressure_summary_by_tag.tex")

    print("\n=== PAPA direct comparison ===")
    if not papa_long.empty:
        display_cols = [
            "ablation",
            "papa_tta",
            "use_bottleneck",
            "use_adapter",
            f"{args.sort_by}_mean",
            f"{args.sort_by}_std",
            f"{args.sort_by}_count",
        ]
        post_pivot = build_post_tta_pivot(papa_long, args.sort_by)
        display_cols = [c for c in display_cols if c in post_pivot.columns]
        print(post_pivot[display_cols].to_string(index=False))
    else:
        print("No papa_summary.csv files found.")

    print(f"\n[WROTE] {out_dir}")


if __name__ == "__main__":
    main()

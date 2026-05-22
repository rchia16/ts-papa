#!/usr/bin/env python3
"""
Evaluate marker-aware respiratory-dynamics PAPA ablation grids.

This evaluator is intentionally tolerant because the marker-aware model branch
is still being designed. It reads the result files produced by both:

  * existing PAPA-dyn respiratory-only runs, and
  * planned marker-aware RD-PAPA runs.

Expected root layout:

  runs/marker_rd_papa_ablation_grid/<STAMP>/
    00_resp_ladder/
      resp_dyn_summary.csv
      summary.csv
    04_marker_teacher_motion_state/
      marker_rd_papa_summary.csv
      marker_preference_summary.csv
      marker_quality_summary.csv
      marker_motion_oracle_summary.csv
      summary.csv

Outputs:

  <out-dir>/all_model_rows.csv
  <out-dir>/metric_summary_by_run.csv
  <out-dir>/paired_delta_vs_baseline.csv
  <out-dir>/subject_best_by_metric.csv
  <out-dir>/negative_control_summary.csv
  <out-dir>/motion_quality_summary.csv
  <out-dir>/evaluation_report.md

Example:

  python evaluate_marker_rd_papa_grid.py \
    --root runs/marker_rd_papa_ablation_grid/20260522T000000Z \
    --baseline-run 00_resp_ladder \
    --baseline-selector best_in_run \
    --metric bal_acc
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

METRICS = [
    "acc",
    "bal_acc",
    "f1_macro",
    "f1_weighted",
    "present_f1_macro",
    "absent_pred_rate",
]

# Candidate column names mapped to normalized names.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "subject": ["subject", "sbj", "heldout", "held_out_subject"],
    "acc": ["acc", "resp_dyn_acc", "marker_rd_acc", "marker_acc", "mard_acc", "cra_papa_acc", "papa_acc", "embed_acc"],
    "bal_acc": ["bal_acc", "balanced_acc", "balanced_accuracy", "resp_dyn_bal_acc", "marker_rd_bal_acc", "marker_bal_acc", "mard_bal_acc", "cra_papa_bal_acc"],
    "f1_macro": ["f1_macro", "macro_f1", "resp_dyn_f1_macro", "marker_rd_f1_macro", "marker_f1_macro", "mard_f1_macro", "cra_papa_f1_macro", "papa_f1_macro", "embed_f1_macro"],
    "f1_weighted": ["f1_weighted", "weighted_f1", "resp_dyn_f1_weighted", "marker_rd_f1_weighted", "marker_f1_weighted", "mard_f1_weighted", "cra_papa_f1_weighted", "papa_f1_weighted", "embed_f1_weighted"],
    "present_f1_macro": ["present_f1_macro", "marker_rd_present_f1_macro", "marker_present_f1_macro", "mard_present_f1_macro", "cra_papa_present_f1_macro"],
    "absent_pred_rate": ["absent_pred_rate", "marker_rd_absent_pred_rate", "marker_absent_pred_rate", "mard_absent_pred_rate", "cra_papa_absent_pred_rate"],
    "resp_variant": ["resp_variant", "resp_dyn_variant", "marker_rd_resp_variant", "marker_resp_variant", "cra_resp_variant"],
    "tag": ["tag", "model_tag", "method", "variant"],
}

SUMMARY_FILES = [
    "resp_dyn_summary.csv",
    "marker_rd_papa_summary.csv",
    "mard_papa_summary.csv",
    "marker_papa_summary.csv",
    "cra_papa_summary.csv",       # accepted for comparison only
    "papa_summary.csv",
    "frozen_embedding_summary.csv",
]

AUX_FILES = [
    # Current MA-RD-PAPA implementation names.
    "marker_rd_papa_preference_summary.csv",
    "marker_rd_papa_motion_summary.csv",
    "marker_rd_papa_oracle_summary.csv",
    # Older/planned aliases kept for backward compatibility.
    "marker_preference_summary.csv",
    "marker_quality_summary.csv",
    "marker_motion_oracle_summary.csv",
    "marker_stratified_summary.csv",
    "cra_papa_preference_summary.csv",
    "cra_papa_profile_summary.csv",
]

NEGATIVE_PATTERNS = [
    "shuffle",
    "shuffled",
    "time_only",
    "time_shift",
    "marker_only_oracle",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Root directory containing ablation run folders.")
    p.add_argument("--out-dir", default=None, help="Output directory. Defaults to <root>/analysis.")
    p.add_argument("--metric", default="bal_acc", choices=METRICS, help="Primary metric for ranking.")
    p.add_argument("--baseline-run", default="00_resp_ladder", help="Run name or regex used as baseline for paired deltas.")
    p.add_argument(
        "--baseline-selector",
        default="best_in_run",
        choices=["best_in_run", "first", "model_tag"],
        help="How to choose a baseline row when a run has multiple rows per subject.",
    )
    p.add_argument("--baseline-model-tag", default=None, help="Required when --baseline-selector=model_tag.")
    p.add_argument("--tex", action="store_true", help="Also write LaTeX versions of compact tables.")
    return p.parse_args()


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] failed reading {path}: {exc}")
        return None


def first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        key = str(name).lower()
        if key in lower:
            return lower[key]
    return None


def normalize_summary(df: pd.DataFrame, run_name: str, source_file: str, run_dir: Path) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["run"] = run_name
    out["run_dir"] = str(run_dir)
    out["source_file"] = source_file

    for norm, aliases in COLUMN_ALIASES.items():
        col = first_existing_column(df, aliases)
        if col is not None:
            out[norm] = df[col]

    if "subject" not in out.columns:
        out["subject"] = np.arange(len(df)).astype(str)
    out["subject"] = out["subject"].astype(str)

    # Create a useful model tag.
    if "tag" not in out.columns:
        if source_file == "resp_dyn_summary.csv" and "resp_variant" in out.columns:
            out["tag"] = "resp_dyn::" + out["resp_variant"].astype(str)
        elif source_file == "frozen_embedding_summary.csv":
            out["tag"] = "frozen_embedding"
        elif source_file == "papa_summary.csv":
            out["tag"] = "papa"
        elif source_file.startswith("marker") or source_file.startswith("mard"):
            out["tag"] = "marker_rd_papa"
        elif source_file.startswith("cra"):
            out["tag"] = "cra_papa"
        else:
            out["tag"] = source_file.replace("_summary.csv", "")

    if "resp_variant" not in out.columns:
        m = out["tag"].astype(str).str.extract(r"resp_dyn::(.+)")[0]
        out["resp_variant"] = m

    for m in METRICS:
        if m not in out.columns:
            out[m] = np.nan
        out[m] = pd.to_numeric(out[m], errors="coerce")

    # Attach run-family hints.
    low = run_name.lower()
    out["is_negative_control"] = any(pat in low for pat in NEGATIVE_PATTERNS)
    out["uses_marker"] = int("marker" in low or source_file.startswith("marker") or source_file.startswith("mard"))
    out["uses_profile"] = int("profile" in low)
    out["uses_quality"] = int("quality" in low)
    out["uses_hmm"] = int("hmm" in low or out["tag"].astype(str).str.contains("hmm", case=False, na=False).any())
    return out.reset_index(drop=True)


def collect_rows(root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "analysis"]):
        for fname in SUMMARY_FILES:
            df = read_csv(run_dir / fname)
            if df is None or df.empty:
                continue
            rows.append(normalize_summary(df, run_dir.name, fname, run_dir))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def collect_aux(root: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, List[pd.DataFrame]] = {fname: [] for fname in AUX_FILES}
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "analysis"]):
        for fname in AUX_FILES:
            df = read_csv(run_dir / fname)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["run"] = run_dir.name
            df["run_dir"] = str(run_dir)
            df["source_file"] = fname
            out[fname].append(df)
    return {k: pd.concat(v, ignore_index=True, sort=False) for k, v in out.items() if v}


def numeric_summary(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in metric_cols if c in df.columns]
    if not cols or df.empty:
        return pd.DataFrame()
    agg = df.groupby(group_cols, dropna=False)[cols].agg(["mean", "std", "median", "count"]).reset_index()
    agg.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in agg.columns]
    return agg


def choose_baseline_rows(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    pattern = re.compile(args.baseline_run)
    base = rows[rows["run"].map(lambda x: bool(pattern.search(str(x))))].copy()
    if base.empty:
        print(f"[WARN] No baseline run matching {args.baseline_run!r}. Paired deltas skipped.")
        return pd.DataFrame()

    metric = args.metric
    selected = []
    for subject, g in base.groupby("subject", dropna=False):
        g = g.copy()
        if args.baseline_selector == "model_tag":
            if not args.baseline_model_tag:
                raise ValueError("--baseline-model-tag is required for --baseline-selector=model_tag")
            h = g[g["tag"].astype(str) == str(args.baseline_model_tag)]
            if h.empty:
                continue
            selected.append(h.iloc[0])
        elif args.baseline_selector == "first":
            selected.append(g.iloc[0])
        else:
            idx = pd.to_numeric(g[metric], errors="coerce").idxmax()
            if pd.notna(idx):
                selected.append(g.loc[idx])
    if not selected:
        return pd.DataFrame()
    out = pd.DataFrame(selected).reset_index(drop=True)
    keep = ["subject", "run", "tag"] + METRICS
    out = out[[c for c in keep if c in out.columns]].copy()
    out = out.rename(columns={"run": "baseline_run", "tag": "baseline_tag", **{m: f"baseline_{m}" for m in METRICS}})
    return out


def paired_deltas(rows: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or baseline.empty:
        return pd.DataFrame()
    comp = rows.merge(baseline, on="subject", how="inner")
    for m in METRICS:
        if m in comp.columns and f"baseline_{m}" in comp.columns:
            comp[f"delta_{m}"] = comp[m] - comp[f"baseline_{m}"]
    return comp


def subject_best(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    if rows.empty or metric not in rows.columns:
        return pd.DataFrame()
    idx = rows.groupby("subject", dropna=False)[metric].idxmax()
    return rows.loc[idx].sort_values([metric, "subject"], ascending=[False, True]).reset_index(drop=True)


def summarize_negative_controls(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    neg = rows[rows["is_negative_control"].astype(bool)].copy()
    if neg.empty:
        return pd.DataFrame()
    return numeric_summary(neg, ["run", "tag"], [metric, "acc", "f1_macro", "absent_pred_rate"])


def maybe_quality_summary(aux: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for fname, df in aux.items():
        if "quality" not in fname and "motion" not in fname and "stratified" not in fname and "oracle" not in fname:
            continue
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        group_cols = [c for c in ["run", "source_file", "motion_bin", "quality_bin", "stratum"] if c in df.columns]
        if not num_cols or not group_cols:
            continue
        agg = df.groupby(group_cols, dropna=False)[num_cols].agg(["mean", "count"]).reset_index()
        agg.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in agg.columns]
        parts.append(agg)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def write_markdown_report(
    out_path: Path,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    best: pd.DataFrame,
    neg: pd.DataFrame,
    metric: str,
) -> None:
    lines: List[str] = []
    lines.append("# Marker-aware RD-PAPA grid evaluation")
    lines.append("")
    lines.append(f"Primary metric: `{metric}`")
    lines.append("")
    lines.append(f"Rows parsed: {len(rows)}")
    lines.append(f"Runs parsed: {rows['run'].nunique() if not rows.empty else 0}")
    lines.append(f"Subjects parsed: {rows['subject'].nunique() if not rows.empty else 0}")
    lines.append("")

    if not summary.empty:
        top_cols = ["run", "tag", f"{metric}_mean", f"{metric}_std", f"{metric}_count"]
        top = summary.sort_values(f"{metric}_mean", ascending=False).head(15)
        lines.append("## Top run/model summaries")
        lines.append("")
        lines.append(top[[c for c in top_cols if c in top.columns]].to_markdown(index=False))
        lines.append("")

    if not deltas.empty:
        dcol = f"delta_{metric}"
        if dcol in deltas.columns:
            dsum = deltas.groupby(["run", "tag"], dropna=False)[dcol].agg(["mean", "median", "std", "count"]).reset_index()
            dsum = dsum.sort_values("mean", ascending=False).head(15)
            lines.append("## Paired deltas vs baseline")
            lines.append("")
            lines.append(dsum.to_markdown(index=False))
            lines.append("")

    if not best.empty:
        lines.append("## Best run/model per subject")
        lines.append("")
        show = best[[c for c in ["subject", "run", "tag", metric, "acc", "f1_macro", "absent_pred_rate"] if c in best.columns]].copy()
        lines.append(show.to_markdown(index=False))
        lines.append("")

    if not neg.empty:
        lines.append("## Negative controls")
        lines.append("")
        lines.append(neg.head(20).to_markdown(index=False))
        lines.append("")
        lines.append("Interpretation rule: shuffled/time-only/marker-only controls should not match or exceed the true marker-teacher runs. If they do, the marker branch is likely learning protocol chronology or motion/task shortcuts rather than respiratory reliability.")
        lines.append("")

    lines.append("## Recommended reading of results")
    lines.append("")
    lines.append("Prefer improvements in balanced accuracy and macro-F1 over raw accuracy. For marker methods, also require lower absent-class prediction rate and improved low-motion/high-motion stratified consistency. A marker-only oracle can be useful as a confound audit, but it should not be treated as deployable MWL evidence.")
    lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(root)
    if rows.empty:
        raise RuntimeError(f"No summary rows found under {root}")
    rows.to_csv(out_dir / "all_model_rows.csv", index=False)

    aux = collect_aux(root)
    for fname, df in aux.items():
        df.to_csv(out_dir / f"all_{fname}", index=False)

    summary = numeric_summary(rows, ["run", "tag"], METRICS)
    summary.to_csv(out_dir / "metric_summary_by_run.csv", index=False)

    baseline = choose_baseline_rows(rows, args)
    baseline.to_csv(out_dir / "baseline_rows.csv", index=False)
    deltas = paired_deltas(rows, baseline)
    if not deltas.empty:
        deltas.to_csv(out_dir / "paired_delta_vs_baseline.csv", index=False)
        delta_summary = numeric_summary(deltas, ["run", "tag"], [f"delta_{m}" for m in METRICS if f"delta_{m}" in deltas.columns])
        delta_summary.to_csv(out_dir / "paired_delta_summary_by_run.csv", index=False)

    best = subject_best(rows, args.metric)
    best.to_csv(out_dir / "subject_best_by_metric.csv", index=False)

    neg = summarize_negative_controls(rows, args.metric)
    if not neg.empty:
        neg.to_csv(out_dir / "negative_control_summary.csv", index=False)

    qsum = maybe_quality_summary(aux)
    if not qsum.empty:
        qsum.to_csv(out_dir / "motion_quality_summary.csv", index=False)

    if args.tex:
        try:
            summary.to_latex(out_dir / "metric_summary_by_run.tex", index=False)
            if not deltas.empty:
                pd.read_csv(out_dir / "paired_delta_summary_by_run.csv").to_latex(out_dir / "paired_delta_summary_by_run.tex", index=False)
        except Exception as exc:
            print(f"[WARN] failed writing LaTeX: {exc}")

    write_markdown_report(
        out_dir / "evaluation_report.md",
        rows=rows,
        summary=summary,
        deltas=deltas,
        best=best,
        neg=neg,
        metric=args.metric,
    )

    print(f"[OK] wrote analysis to {out_dir}")
    print(f"[OK] parsed rows={len(rows)} runs={rows['run'].nunique()} subjects={rows['subject'].nunique()}")


if __name__ == "__main__":
    main()

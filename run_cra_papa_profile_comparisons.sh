#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run_cra_papa_profile_comparisons.sh
#
# Runs four CRA-PAPA profile comparisons against the profile script:
#
#   A. profile default
#      profile blocks ON, legacy ECG-proxy expert blocks OFF
#
#   B. no profile
#      disables ECG-derived subject-profile conditioning
#
#   C. legacy proxy
#      restores the old per-window ECG-proxy/fused expert blocks as an ablation
#      while leaving the new profile blocks on by default
#
#   D. observed source profile diagnostic
#      uses observed source ECG profiles instead of source-LOSO inferred profiles
#      for source expert training; target ECG is still not used for main prediction
#
# Example:
#   bash run_cra_papa_profile_comparisons.sh
#
# Smoke test:
#   SUBJECTS="S12 S13 S14" EPOCHS=2 DEVICES="cuda:0" \
#     bash run_cra_papa_profile_comparisons.sh
#
# Common overrides:
#   SCRIPT=vit_pressure_crossmodal_cra_papa_profile.py
#   DEVICES="cuda:0 cuda:1"
#   EMBED_LABELS="L0,L2,L3"
#   CRA_RESP_VARIANT=dyn_hybrid
#   OUT_ROOT=runs/cra_papa_profile_compare/20260522T000000Z
# -----------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-vit_pressure_crossmodal_cra_papa_profile.py}"
DEVICES=(${DEVICES:-cuda:0 cuda:1})

DATA_STR="${DATA_STR:-imu_filt}"
PRETRAIN_DATA_GROUP="${PRETRAIN_DATA_GROUP:-mr}"
EMBED_LABELS="${EMBED_LABELS:-L0,L2,L3}"
EMBED_DATA_GROUP="${EMBED_DATA_GROUP:-auto}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"

LAMBDA_STFT="${LAMBDA_STFT:-1.0}"
LAMBDA_RR="${LAMBDA_RR:-0.01}"
LAMBDA_CONTRAST="${LAMBDA_CONTRAST:-0.1}"
LAMBDA_RESP_RR="${LAMBDA_RESP_RR:-0.05}"
LAMBDA_RESP_RECON="${LAMBDA_RESP_RECON:-0.01}"
CONTRAST_WARMUP_EPOCHS="${CONTRAST_WARMUP_EPOCHS:-5}"
CONTRAST_RAMP_END_EPOCH="${CONTRAST_RAMP_END_EPOCH:-10}"

CRA_RESP_VARIANT="${CRA_RESP_VARIANT:-dyn_hybrid}"
CRA_EXPERT_CLASSIFIER="${CRA_EXPERT_CLASSIFIER:-logreg}"
CRA_GATE_TEMPERATURE="${CRA_GATE_TEMPERATURE:-0.75}"
CRA_PROFILE_ALPHA="${CRA_PROFILE_ALPHA:-10.0}"
CRA_PROFILE_MIN_SUBJECTS="${CRA_PROFILE_MIN_SUBJECTS:-3}"
CRA_PROFILE_MAX_AR_DIMS="${CRA_PROFILE_MAX_AR_DIMS:-96}"
CRA_ECG_PROXY_ALPHA="${CRA_ECG_PROXY_ALPHA:-10.0}"

RESP_DYN_ROLL_WIN="${RESP_DYN_ROLL_WIN:-7}"
RESP_DYN_TARGET_BASELINE_Q="${RESP_DYN_TARGET_BASELINE_Q:-0.20}"
RESP_DYN_SOURCE_BASELINE_Q="${RESP_DYN_SOURCE_BASELINE_Q:-0.20}"
RESP_DYN_BOUNDARY_JUMP_Z="${RESP_DYN_BOUNDARY_JUMP_Z:-8.0}"
RESP_DYN_SCALE_FLOOR="${RESP_DYN_SCALE_FLOOR:-1e-3}"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-runs/cra_papa_profile_compare/${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

SUBJECT_ARGS=()
if [ -n "${SUBJECTS:-}" ]; then
    # shellcheck disable=SC2206
    SUBJECT_ARGS=(--subjects ${SUBJECTS})
fi

EMBED_GROUP_ARGS=()
if [ "${EMBED_DATA_GROUP}" != "auto" ]; then
    EMBED_GROUP_ARGS=(--embed-data-group "${EMBED_DATA_GROUP}")
fi

COMMON_ARGS=(
    --data-str "${DATA_STR}"
    --data-group "${PRETRAIN_DATA_GROUP}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"

    --lambda-stft "${LAMBDA_STFT}"
    --lambda-rr "${LAMBDA_RR}"
    --lambda-contrast "${LAMBDA_CONTRAST}"
    --lambda-resp-rr "${LAMBDA_RESP_RR}"
    --lambda-resp-recon "${LAMBDA_RESP_RECON}"
    --contrast-warmup-epochs "${CONTRAST_WARMUP_EPOCHS}"
    --contrast-ramp-end-epoch "${CONTRAST_RAMP_END_EPOCH}"

    --eval-cra-papa
    --embed-labels "${EMBED_LABELS}"
    "${EMBED_GROUP_ARGS[@]}"

    --cra-resp-variant "${CRA_RESP_VARIANT}"
    --cra-expert-classifier "${CRA_EXPERT_CLASSIFIER}"
    --cra-gate-temperature "${CRA_GATE_TEMPERATURE}"
    --cra-profile-alpha "${CRA_PROFILE_ALPHA}"
    --cra-profile-min-subjects "${CRA_PROFILE_MIN_SUBJECTS}"
    --cra-profile-max-ar-dims "${CRA_PROFILE_MAX_AR_DIMS}"
    --cra-ecg-proxy-alpha "${CRA_ECG_PROXY_ALPHA}"

    --resp-dyn-roll-win "${RESP_DYN_ROLL_WIN}"
    --resp-dyn-target-baseline-q "${RESP_DYN_TARGET_BASELINE_Q}"
    --resp-dyn-source-baseline-q "${RESP_DYN_SOURCE_BASELINE_Q}"
    --resp-dyn-boundary-jump-z "${RESP_DYN_BOUNDARY_JUMP_Z}"
    --resp-dyn-scale-floor "${RESP_DYN_SCALE_FLOOR}"

    "${SUBJECT_ARGS[@]}"
)

# -----------------------------------------------------------------------------
# Comparison definition
# tag | extra args
# -----------------------------------------------------------------------------
RUN_TAGS=(
    "A_profile_default"
    "B_no_profile"
    "C_legacy_proxy"
    "D_observed_source_profile"
)

RUN_ARGS=(
    ""
    "--cra-no-profile-blocks"
    "--cra-use-ecg-proxy-blocks"
    "--cra-profile-source-mode observed"
)

GPU_PIDS=()
GPU_LABELS=()
for _ in "${DEVICES[@]}"; do
    GPU_PIDS+=("")
    GPU_LABELS+=("")
done

cleanup() {
    for i in "${!GPU_PIDS[@]}"; do
        pid="${GPU_PIDS[$i]}"
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            echo "[CLEANUP] stopping ${GPU_LABELS[$i]} pid=${pid}"
            pkill -TERM -P "${pid}" 2>/dev/null || true
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM

wait_for_free_gpu() {
    while true; do
        for i in "${!GPU_PIDS[@]}"; do
            pid="${GPU_PIDS[$i]}"
            if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
                GPU_PIDS[$i]=""
                GPU_LABELS[$i]=""
                echo "${i}"
                return 0
            fi
        done
        sleep 5
    done
}

echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Log root:    ${LOG_ROOT}"
echo "[INFO] Devices:     ${DEVICES[*]}"
echo "[INFO] Script:      ${SCRIPT}"
echo "[INFO] Labels:      ${EMBED_LABELS}"
echo "[INFO] Resp variant:${CRA_RESP_VARIANT}"

for idx in "${!RUN_TAGS[@]}"; do
    tag="${RUN_TAGS[$idx]}"
    extra="${RUN_ARGS[$idx]}"

    gpu_idx="$(wait_for_free_gpu)"
    device="${DEVICES[$gpu_idx]}"
    out_dir="${OUT_ROOT}/${tag}"
    safe_device="$(echo "${device}" | tr ':/' '__')"
    log_file="${LOG_ROOT}/${tag}__${safe_device}.log"

    mkdir -p "${out_dir}"

    echo "[RUN] tag=${tag} device=${device}"
    echo "[LOG] ${log_file}"

    # shellcheck disable=SC2086
    cmd=(
        "${PYTHON_BIN}" "${SCRIPT}"
        "${COMMON_ARGS[@]}"
        --device "${device}"
        --out-dir "${out_dir}"
        ${extra}
    )

    if [ "${DRY_RUN}" = "1" ]; then
        printf '[DRY_RUN]'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi

    (
        set -x
        "${cmd[@]}"
    ) > "${log_file}" 2>&1 &

    GPU_PIDS[$gpu_idx]=$!
    GPU_LABELS[$gpu_idx]="${tag}"
done

if [ "${DRY_RUN}" != "1" ]; then
    for i in "${!GPU_PIDS[@]}"; do
        pid="${GPU_PIDS[$i]}"
        if [ -n "${pid}" ]; then
            wait "${pid}"
        fi
    done
fi

echo "[DONE] Comparison runs finished."

# -----------------------------------------------------------------------------
# Lightweight aggregation for CRA-PAPA outputs.
# The core runner emits summary files per run directory.  This collector merges
# them and makes compact comparison tables by run tag when the expected columns
# are present.
# -----------------------------------------------------------------------------
if [ "${DRY_RUN}" != "1" ]; then
    OUT_ROOT="${OUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["OUT_ROOT"])
out = root / "analysis"
out.mkdir(parents=True, exist_ok=True)

files = [
    "summary.csv",
    "cra_papa_summary.csv",
    "cra_papa_preference_summary.csv",
    "cra_papa_profile_summary.csv",
    "cra_papa_ecg_oracle_summary.csv",
]

for fname in files:
    rows = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in {"logs", "analysis"}):
        f = run_dir / fname
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"[WARN] could not read {f}: {exc}")
            continue
        df.insert(0, "run", run_dir.name)
        rows.append(df)
    if rows:
        merged = pd.concat(rows, ignore_index=True)
        merged.to_csv(out / fname.replace(".csv", "_by_run.csv"), index=False)

# Main compact comparison.
main_path = out / "cra_papa_summary_by_run.csv"
if main_path.exists():
    df = pd.read_csv(main_path)
    metric_cols = [
        c for c in [
            "cra_acc", "cra_bal_acc", "cra_f1_macro", "cra_present_f1_macro",
            "cra_f1_weighted", "cra_absent_pred_rate", "cra_mean_conf",
            "cra_ecg_valid_source", "cra_ecg_valid_target",
            "profile_source_subjects", "profile_mean_uncertainty_train", "profile_mean_uncertainty_test",
        ]
        if c in df.columns
    ]
    if metric_cols:
        comp = df.groupby("run", dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
        comp.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in comp.columns]
        sort_col = "cra_bal_acc_mean" if "cra_bal_acc_mean" in comp.columns else None
        if sort_col:
            comp = comp.sort_values(sort_col, ascending=False)
        comp.to_csv(out / "cra_papa_comparison_by_run.csv", index=False)
        print("[SUMMARY]", out / "cra_papa_comparison_by_run.csv")

pref_path = out / "cra_papa_preference_summary_by_run.csv"
if pref_path.exists():
    pref = pd.read_csv(pref_path)
    weight_cols = [c for c in pref.columns if c.startswith("weight_mean_")]
    if weight_cols:
        by_run = pref.groupby("run", dropna=False)[weight_cols].mean().reset_index()
        by_run.to_csv(out / "cra_papa_expert_weights_by_run.csv", index=False)
        print("[SUMMARY]", out / "cra_papa_expert_weights_by_run.csv")
PY
fi

echo "[NEXT] Inspect: ${OUT_ROOT}/analysis/cra_papa_comparison_by_run.csv"
echo "[NEXT] Inspect: ${OUT_ROOT}/analysis/cra_papa_expert_weights_by_run.csv"

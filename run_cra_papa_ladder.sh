#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run_cra_papa_ladder.sh
#
# Runs a compact, hypothesis-driven ladder for CRA-PAPA:
#
#   0. PAPA-dyn respiratory-only ladder audit
#   1. CRA-PAPA with selected reconstructed-respiration variants
#   2. Bottleneck vs no-bottleneck checks
#   3. ECG-proxy usefulness checks
#   4. Preference-gate temperature / classifier checks
#   5. Baseline/dynamics robustness checks, for the full tier
#
# Why this is not a giant grid:
#   The goal is to identify whether LOSO performance/preference comes from
#   reconstructed respiratory dynamics, IMU activity dynamics, ECG-proxy latent
#   information, or the learned subject-specific gate.  Most knobs are tested
#   one factor at a time around a strong default.
#
# Example:
#   bash run_cra_papa_ladder.sh
#
# Smoke test:
#   TIER=smoke SUBJECTS="S12 S13 S14" EPOCHS=2 DEVICES="cuda:0" \
#     bash run_cra_papa_ladder.sh
#
# Full run:
#   TIER=full DEVICES="cuda:0 cuda:1" bash run_cra_papa_ladder.sh
#
# Optional env overrides:
#   SCRIPT=vit_pressure_crossmodal_cra_papa.py
#   DATA_STR=imu_filt
#   PRETRAIN_DATA_GROUP=mr
#   EMBED_LABELS=L0,L2,L3
#   EMBED_DATA_GROUP=auto        # auto omits --embed-data-group and lets Python infer
#   DEVICES="cuda:0 cuda:1"
#   SUBJECTS="S12 S13 S14"
#   TIER=smoke|core|full
#   EPOCHS=20 BATCH_SIZE=64 LR=3e-4
#   OUT_ROOT=runs/cra_papa_ladder/<stamp>
# -----------------------------------------------------------------------------

SCRIPT="${SCRIPT:-vit_pressure_crossmodal_cra_papa.py}"
DEVICES=(${DEVICES:-cuda:0 cuda:1})
TIER="${TIER:-core}"

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

RESP_DYN_CLASSIFIER="${RESP_DYN_CLASSIFIER:-lda}"
RESP_DYN_ROLL_WIN="${RESP_DYN_ROLL_WIN:-7}"
RESP_DYN_HMM_STAY="${RESP_DYN_HMM_STAY:-0.75}"
RESP_DYN_HMM_MIN_STAY="${RESP_DYN_HMM_MIN_STAY:-0.50}"
RESP_DYN_TARGET_BASELINE_Q="${RESP_DYN_TARGET_BASELINE_Q:-0.20}"
RESP_DYN_SOURCE_BASELINE_Q="${RESP_DYN_SOURCE_BASELINE_Q:-0.20}"
RESP_DYN_BOUNDARY_JUMP_Z="${RESP_DYN_BOUNDARY_JUMP_Z:-8.0}"
RESP_DYN_SCALE_FLOOR="${RESP_DYN_SCALE_FLOOR:-1e-3}"

# The full PAPA-dyn audit ladder.  CRA-PAPA itself chooses one respiratory
# variant at a time with --cra-resp-variant.
RESP_DYN_LADDER="${RESP_DYN_LADDER:-rr_abs,rr_delta,rr_z,stft_abs,stft_delta,stft_z,stft_hybrid,dyn_abs,dyn_delta,dyn_z,dyn_hybrid,dyn_abs_hmm,dyn_z_hmm,dyn_hybrid_hmm,state_abs,hybrid_abs,hybrid_dyn_abs,hybrid_dyn_abs_hmm,hybrid_z,hybrid_z_hmm}"

CRA_EXPERT_CLASSIFIER="${CRA_EXPERT_CLASSIFIER:-logreg}"
CRA_LOGREG_C="${CRA_LOGREG_C:-1.0}"
CRA_ECG_PROXY_ALPHA="${CRA_ECG_PROXY_ALPHA:-10.0}"
CRA_MIN_ECG_VALID="${CRA_MIN_ECG_VALID:-20}"
CRA_MIN_ECG_TARGET_ORACLE="${CRA_MIN_ECG_TARGET_ORACLE:-5}"
CRA_MIN_PSEUDO_WINDOWS="${CRA_MIN_PSEUDO_WINDOWS:-5}"
CRA_MIN_GATE_ROWS="${CRA_MIN_GATE_ROWS:-50}"
CRA_GATE_C="${CRA_GATE_C:-1.0}"
CRA_GATE_TEMPERATURE="${CRA_GATE_TEMPERATURE:-0.75}"

EVAL_FROZEN="${EVAL_FROZEN:-1}"
EVAL_PAPA_STATIC="${EVAL_PAPA_STATIC:-0}"
ANALYZE="${ANALYZE:-1}"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-runs/cra_papa_ladder/${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

if [ -n "${SUBJECTS:-}" ]; then
    SUBJECT_ARGS=(--subjects ${SUBJECTS})
else
    SUBJECT_ARGS=()
fi

EMBED_DATA_GROUP_ARGS=()
if [ "${EMBED_DATA_GROUP}" != "auto" ] && [ -n "${EMBED_DATA_GROUP}" ]; then
    EMBED_DATA_GROUP_ARGS=(--embed-data-group "${EMBED_DATA_GROUP}")
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
    --contrast-warmup-epochs "${CONTRAST_WARMUP_EPOCHS}"
    --contrast-ramp-end-epoch "${CONTRAST_RAMP_END_EPOCH}"
    --lambda-resp-rr "${LAMBDA_RESP_RR}"
    --lambda-resp-recon "${LAMBDA_RESP_RECON}"

    --embed-labels "${EMBED_LABELS}"
    "${EMBED_DATA_GROUP_ARGS[@]}"

    --resp-dyn-ladder "${RESP_DYN_LADDER}"
    --resp-dyn-classifier "${RESP_DYN_CLASSIFIER}"
    --resp-dyn-roll-win "${RESP_DYN_ROLL_WIN}"
    --resp-dyn-hmm-stay "${RESP_DYN_HMM_STAY}"
    --resp-dyn-hmm-min-stay "${RESP_DYN_HMM_MIN_STAY}"
    --resp-dyn-target-baseline-q "${RESP_DYN_TARGET_BASELINE_Q}"
    --resp-dyn-source-baseline-q "${RESP_DYN_SOURCE_BASELINE_Q}"
    --resp-dyn-scale-floor "${RESP_DYN_SCALE_FLOOR}"
    --resp-dyn-boundary-jump-z "${RESP_DYN_BOUNDARY_JUMP_Z}"

    --cra-expert-classifier "${CRA_EXPERT_CLASSIFIER}"
    --cra-logreg-c "${CRA_LOGREG_C}"
    --cra-ecg-proxy-alpha "${CRA_ECG_PROXY_ALPHA}"
    --cra-min-ecg-valid "${CRA_MIN_ECG_VALID}"
    --cra-min-ecg-target-oracle "${CRA_MIN_ECG_TARGET_ORACLE}"
    --cra-min-pseudo-windows "${CRA_MIN_PSEUDO_WINDOWS}"
    --cra-min-gate-rows "${CRA_MIN_GATE_ROWS}"
    --cra-gate-c "${CRA_GATE_C}"
    --cra-gate-temperature "${CRA_GATE_TEMPERATURE}"

    "${SUBJECT_ARGS[@]}"
)

if [ "${EVAL_FROZEN}" = "1" ]; then
    COMMON_ARGS+=(
        --eval-frozen-embeddings
        --embed-classifier linear
        --embed-pooling rich
        --embed-stft-profile
        --linear-probe-epochs 30
    )
fi

if [ "${EVAL_PAPA_STATIC}" = "1" ]; then
    COMMON_ARGS+=(--eval-papa --papa-tta none)
fi

RUN_TAGS=()
RUN_ARGS=()

add_run() {
    RUN_TAGS+=("$1")
    RUN_ARGS+=("$2")
}

case "${TIER}" in
    smoke)
        add_run "00_cra_dyn_hybrid_smoke" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid"
        ;;

    core)
        # 0. Respiratory-only reference.  This tells you which reconstructed
        # respiratory variant is even worth passing into CRA-PAPA.
        add_run "00_resp_dyn_ladder" \
            "--no-eval-cra-papa --eval-resp-dyn"

        # 1. Main CRA-PAPA respiratory choices.
        add_run "01_cra_dyn_hybrid" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid"
        add_run "02_cra_dyn_z" \
            "--eval-cra-papa --cra-resp-variant dyn_z"
        add_run "03_cra_hybrid_z" \
            "--eval-cra-papa --cra-resp-variant hybrid_z"
        add_run "04_cra_hybrid_dyn_abs" \
            "--eval-cra-papa --cra-resp-variant hybrid_dyn_abs"
        add_run "05_cra_state_abs" \
            "--eval-cra-papa --cra-resp-variant state_abs"

        # 2. Bottleneck check: does the learned respiration-state bottleneck help
        # after we add activity/proxy/preference, or does it discard dynamics?
        add_run "06_cra_no_bottleneck_dyn_hybrid" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --papa-no-bottleneck"

        # 3. ECG proxy checks.  The off run forces the proxy unavailable without
        # requiring a Python flag change.
        add_run "07_cra_ecg_proxy_off" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-min-ecg-valid 100000000"
        add_run "08_cra_ecg_proxy_strong" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-ecg-proxy-alpha 1.0"
        add_run "09_cra_ecg_proxy_conservative" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-ecg-proxy-alpha 100.0"

        # 4. Gate sensitivity.  Lower temperature is sharper preference; higher
        # temperature is more averaged/expert-ensemble-like.
        add_run "10_cra_gate_sharp" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-gate-temperature 0.50"
        add_run "11_cra_gate_soft" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-gate-temperature 1.25"
        ;;

    full)
        # Everything in core.
        add_run "00_resp_dyn_ladder" \
            "--no-eval-cra-papa --eval-resp-dyn"
        add_run "01_cra_dyn_hybrid" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid"
        add_run "02_cra_dyn_z" \
            "--eval-cra-papa --cra-resp-variant dyn_z"
        add_run "03_cra_hybrid_z" \
            "--eval-cra-papa --cra-resp-variant hybrid_z"
        add_run "04_cra_hybrid_dyn_abs" \
            "--eval-cra-papa --cra-resp-variant hybrid_dyn_abs"
        add_run "05_cra_state_abs" \
            "--eval-cra-papa --cra-resp-variant state_abs"
        add_run "06_cra_no_bottleneck_dyn_hybrid" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --papa-no-bottleneck"
        add_run "07_cra_ecg_proxy_off" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-min-ecg-valid 100000000"
        add_run "08_cra_ecg_proxy_strong" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-ecg-proxy-alpha 1.0"
        add_run "09_cra_ecg_proxy_conservative" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-ecg-proxy-alpha 100.0"
        add_run "10_cra_gate_sharp" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-gate-temperature 0.50"
        add_run "11_cra_gate_soft" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-gate-temperature 1.25"

        # Extra full-tier checks.
        add_run "12_cra_experts_lda" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --cra-expert-classifier lda"
        add_run "13_cra_dyn_abs" \
            "--eval-cra-papa --cra-resp-variant dyn_abs"
        add_run "14_cra_stft_hybrid" \
            "--eval-cra-papa --cra-resp-variant stft_hybrid"
        add_run "15_cra_no_bottleneck_hybrid_z" \
            "--eval-cra-papa --cra-resp-variant hybrid_z --papa-no-bottleneck"
        add_run "16_cra_long_roll" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --resp-dyn-roll-win 15"
        add_run "17_cra_short_roll" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --resp-dyn-roll-win 3"
        add_run "18_cra_low_baseline_q" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --resp-dyn-target-baseline-q 0.10 --resp-dyn-source-baseline-q 0.10"
        add_run "19_cra_high_baseline_q" \
            "--eval-cra-papa --cra-resp-variant dyn_hybrid --resp-dyn-target-baseline-q 0.35 --resp-dyn-source-baseline-q 0.35"
        add_run "20_resp_dyn_ladder_no_bottleneck" \
            "--no-eval-cra-papa --eval-resp-dyn --papa-no-bottleneck"
        ;;

    *)
        echo "[ERROR] Unknown TIER='${TIER}'. Use smoke, core, or full." >&2
        exit 2
        ;;
esac

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

run_status=0

printf '[INFO] Output root: %s\n' "${OUT_ROOT}"
printf '[INFO] Log root:    %s\n' "${LOG_ROOT}"
printf '[INFO] Devices:     %s\n' "${DEVICES[*]}"
printf '[INFO] Script:      %s\n' "${SCRIPT}"
printf '[INFO] Tier:        %s\n' "${TIER}"
printf '[INFO] Labels:      %s\n' "${EMBED_LABELS}"
printf '[INFO] Runs:        %s\n' "${#RUN_TAGS[@]}"

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

    # Intentional word-splitting for ${extra}; each RUN_ARGS entry is a plain
    # flag string.  Keep values simple: no spaces inside a single value.
    # shellcheck disable=SC2086
    (
        set -x
        python "${SCRIPT}" \
            "${COMMON_ARGS[@]}" \
            --device "${device}" \
            --out-dir "${out_dir}" \
            ${extra}
    ) > "${log_file}" 2>&1 &

    GPU_PIDS[$gpu_idx]=$!
    GPU_LABELS[$gpu_idx]="${tag}"
done

for i in "${!GPU_PIDS[@]}"; do
    pid="${GPU_PIDS[$i]}"
    label="${GPU_LABELS[$i]}"
    if [ -n "${pid}" ]; then
        if ! wait "${pid}"; then
            echo "[ERROR] Run failed: ${label}. Check ${LOG_ROOT}." >&2
            run_status=1
        fi
    fi
done

if [ "${ANALYZE}" = "1" ]; then
    echo "[ANALYZE] collecting summaries"
    python - "${OUT_ROOT}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
out = root / "analysis"
out.mkdir(parents=True, exist_ok=True)

rows = []
prefs = []
dyn = []
oracle = []
for run_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "logs" and p.name != "analysis"]):
    for name, sink in [
        ("cra_papa_summary.csv", rows),
        ("cra_papa_preference_summary.csv", prefs),
        ("resp_dyn_summary.csv", dyn),
        ("cra_papa_ecg_oracle_summary.csv", oracle),
    ]:
        path = run_dir / name
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["run"] = run_dir.name
                sink.append(df)
            except Exception as e:
                print(f"[WARN] failed to read {path}: {e}")

def write_summary(frames, filename, metrics):
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / filename.replace("_by_run", "_long"), index=False)
    have = [m for m in metrics if m in df.columns]
    if have:
        summary = df.groupby("run", dropna=False)[have].agg(["mean", "std", "count"]).reset_index()
        summary.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in summary.columns]
        sort_col = None
        for candidate in ["cra_bal_acc_mean", "cra_f1_macro_mean", "resp_dyn_bal_acc_mean", "cra_ecg_oracle_bal_acc_mean"]:
            if candidate in summary.columns:
                sort_col = candidate
                break
        if sort_col:
            summary = summary.sort_values(sort_col, ascending=False)
        summary.to_csv(out / filename, index=False)
        print(f"[WROTE] {out / filename}")
    else:
        df.to_csv(out / filename, index=False)
        print(f"[WROTE] {out / filename}")

write_summary(rows, "cra_papa_summary_by_run.csv", ["cra_acc", "cra_bal_acc", "cra_f1_macro", "cra_f1_weighted", "cra_mean_conf"])
write_summary(prefs, "cra_papa_preference_by_run.csv", [c for c in (pd.concat(prefs, ignore_index=True).columns if prefs else []) if c.startswith("pref_") or c.startswith("low_motion_pref_") or c.startswith("high_motion_pref_")])
write_summary(dyn, "resp_dyn_summary_by_run.csv", ["resp_dyn_acc", "resp_dyn_bal_acc", "resp_dyn_f1_macro", "resp_dyn_f1_weighted"])
write_summary(oracle, "cra_ecg_oracle_summary_by_run.csv", ["cra_ecg_oracle_acc", "cra_ecg_oracle_bal_acc", "cra_ecg_oracle_f1_macro", "cra_ecg_oracle_f1_weighted"])
PY
fi

if [ "${run_status}" -ne 0 ]; then
    echo "[DONE_WITH_ERRORS] Some runs failed. Logs are in ${LOG_ROOT}." >&2
    exit "${run_status}"
fi

echo "[DONE] All runs finished."
echo "[NEXT] Main outputs:"
echo "  ${OUT_ROOT}/analysis/cra_papa_summary_by_run.csv"
echo "  ${OUT_ROOT}/analysis/cra_papa_preference_by_run.csv"
echo "  ${OUT_ROOT}/analysis/resp_dyn_summary_by_run.csv"

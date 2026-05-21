#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run_papa_dyn_ladder.sh
#
# Runs the reconstructed respiratory dynamics ladder, including:
#
#   absolute physiology:       rr_abs, stft_abs, dyn_abs
#   baseline calibrated:       rr_delta, rr_z, stft_delta, stft_z, dyn_z
#   abs+calibration hybrids:   stft_hybrid, dyn_hybrid
#   adaptive sequence options: dyn_abs_hmm, dyn_z_hmm, dyn_hybrid_hmm
#   learned-state hybrids:     state_abs, hybrid_dyn_abs, hybrid_z
#
# for:
#   1. full PAPA bottleneck training
#   2. no-bottleneck ablation
#
# Example:
#   bash run_papa_dyn_ladder.sh
#
# Smoke test:
#   SUBJECTS="S12 S13 S14" EPOCHS=2 DEVICES="cuda:0" bash run_papa_dyn_ladder.sh
# -----------------------------------------------------------------------------

SCRIPT="${SCRIPT:-vit_pressure_crossmodal_papa_dyn.py}"

DEVICES=(${DEVICES:-cuda:0 cuda:1})

DATA_STR="${DATA_STR:-imu_filt}"
PRETRAIN_DATA_GROUP="${PRETRAIN_DATA_GROUP:-mr}"
EMBED_LABELS="${EMBED_LABELS:-L0,L2,L3}"

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
RESP_DYN_TARGET_BASELINE_Q="${RESP_DYN_TARGET_BASELINE_Q:-0.20}"
RESP_DYN_SOURCE_BASELINE_Q="${RESP_DYN_SOURCE_BASELINE_Q:-0.20}"
RESP_DYN_HMM_MIN_STAY="${RESP_DYN_HMM_MIN_STAY:-0.50}"
RESP_DYN_BOUNDARY_JUMP_Z="${RESP_DYN_BOUNDARY_JUMP_Z:-8.0}"
RESP_DYN_SCALE_FLOOR="${RESP_DYN_SCALE_FLOOR:-1e-3}"

LADDER="${LADDER:-rr_abs,rr_delta,rr_z,stft_abs,stft_delta,stft_z,stft_hybrid,dyn_abs,dyn_delta,dyn_z,dyn_hybrid,dyn_abs_hmm,dyn_z_hmm,dyn_hybrid_hmm,state_abs,hybrid_abs,hybrid_dyn_abs,hybrid_dyn_abs_hmm,hybrid_z,hybrid_z_hmm}"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-runs/papa_dyn_ladder/${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

if [ -n "${SUBJECTS:-}" ]; then
    SUBJECT_ARGS=(--subjects ${SUBJECTS})
else
    SUBJECT_ARGS=()
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
    --embed-data-group levels

    --eval-resp-dyn
    --resp-dyn-ladder "${LADDER}"
    --resp-dyn-classifier "${RESP_DYN_CLASSIFIER}"
    --resp-dyn-roll-win "${RESP_DYN_ROLL_WIN}"
    --resp-dyn-hmm-stay "${RESP_DYN_HMM_STAY}"
    --resp-dyn-hmm-min-stay "${RESP_DYN_HMM_MIN_STAY}"
    --resp-dyn-target-baseline-q "${RESP_DYN_TARGET_BASELINE_Q}"
    --resp-dyn-source-baseline-q "${RESP_DYN_SOURCE_BASELINE_Q}"
    --resp-dyn-scale-floor "${RESP_DYN_SCALE_FLOOR}"
    --resp-dyn-boundary-jump-z "${RESP_DYN_BOUNDARY_JUMP_Z}"

    --resp-dyn-also-frozen
    --eval-frozen-embeddings
    --embed-classifier linear
    --embed-pooling rich
    --embed-stft-profile
    --linear-probe-epochs 30

    "${SUBJECT_ARGS[@]}"
)

RUN_TAGS=(
    "dyn_full"
    "dyn_no_bottleneck"
)

RUN_ARGS=(
    ""
    "--papa-no-bottleneck"
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
echo "[INFO] Ladder:      ${LADDER}"

for idx in "${!RUN_TAGS[@]}"; do
    tag="${RUN_TAGS[$idx]}"
    extra="${RUN_ARGS[$idx]}"

    gpu_idx="$(wait_for_free_gpu)"
    device="${DEVICES[$gpu_idx]}"

    out_dir="${OUT_ROOT}/${tag}"
    log_file="${LOG_ROOT}/${tag}__$(echo "${device}" | tr ':/' '__').log"

    mkdir -p "${out_dir}"

    echo "[RUN] tag=${tag} device=${device}"
    echo "[LOG] ${log_file}"

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
    if [ -n "${pid}" ]; then
        wait "${pid}"
    fi
done

echo "[DONE] All runs finished."
echo "[NEXT] Look for:"
echo "  ${OUT_ROOT}/dyn_full/resp_dyn_summary.csv"
echo "  ${OUT_ROOT}/dyn_no_bottleneck/resp_dyn_summary.csv"

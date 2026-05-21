#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run_papa_ablation_grid.sh
#
# Runs PAPA ablations across:
#   - bottleneck on/off
#   - subject adapter on/off
#   - TTA method: none, tent, nrc, cotta, papa
#
# Expected script:
#   vit_pressure_crossmodal_papa.py
#
# Example:
#   bash run_papa_ablation_grid.sh
#
# Optional env overrides:
#   DEVICES="cuda:0 cuda:1" bash run_papa_ablation_grid.sh
#   SUBJECTS="S12 S13 S14" bash run_papa_ablation_grid.sh
#   EPOCHS=20 PAPA_EPOCHS=3 bash run_papa_ablation_grid.sh
# -----------------------------------------------------------------------------

DEVICES=(${DEVICES:-cuda:0 cuda:1})
SCRIPT="${SCRIPT:-vit_pressure_crossmodal_papa.py}"

DATA_STR="${DATA_STR:-imu_filt}"
PRETRAIN_DATA_GROUP="${PRETRAIN_DATA_GROUP:-mr}"
EMBED_LABELS="${EMBED_LABELS:-L0,L2,L3}" # prev: L0,L1,L2,L3
EMBED_CLASSIFIER="${EMBED_CLASSIFIER:-linear}"
EMBED_POOLING="${EMBED_POOLING:-rich}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"

LAMBDA_STFT="${LAMBDA_STFT:-1.0}"
LAMBDA_RR="${LAMBDA_RR:-0.01}"
LAMBDA_CONTRAST="${LAMBDA_CONTRAST:-0.1}"
CONTRAST_WARMUP_EPOCHS="${CONTRAST_WARMUP_EPOCHS:-5}"
CONTRAST_RAMP_END_EPOCH="${CONTRAST_RAMP_END_EPOCH:-10}"

PAPA_EPOCHS="${PAPA_EPOCHS:-3}"
PAPA_LR="${PAPA_LR:-5e-4}"
PAPA_STATE_DIM="${PAPA_STATE_DIM:-48}"

LINEAR_PROBE_EPOCHS="${LINEAR_PROBE_EPOCHS:-30}"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/projects/BLVMob/imu-rr-seated/results/papa_ablation_grid/runs/${STAMP}}"
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
    --eval-frozen-embeddings
    --eval-papa
    --embed-labels "${EMBED_LABELS}"
    --embed-classifier "${EMBED_CLASSIFIER}"
    --embed-pooling "${EMBED_POOLING}"
    --embed-stft-profile
    --linear-probe-epochs "${LINEAR_PROBE_EPOCHS}"
    --papa-epochs "${PAPA_EPOCHS}"
    --papa-lr "${PAPA_LR}"
    --papa-state-dim "${PAPA_STATE_DIM}"
    "${SUBJECT_ARGS[@]}"
)

# -----------------------------------------------------------------------------
# Grid definition
#
# tag | extra args
# -----------------------------------------------------------------------------
RUN_TAGS=(
    "full_none"
    "full_tent"
    "full_nrc"
    "full_cotta"
    "full_papa"

    "no_bottleneck_none"
    "no_bottleneck_tent"
    "no_bottleneck_nrc"
    "no_bottleneck_cotta"
    "no_bottleneck_papa"

    "no_adapter_none"

    "no_bottleneck_no_adapter_none"
)

RUN_ARGS=(
    "--papa-tta none"
    "--papa-tta tent"
    "--papa-tta nrc"
    "--papa-tta cotta"
    "--papa-tta papa"

    "--papa-no-bottleneck --papa-tta none"
    "--papa-no-bottleneck --papa-tta tent"
    "--papa-no-bottleneck --papa-tta nrc"
    "--papa-no-bottleneck --papa-tta cotta"
    "--papa-no-bottleneck --papa-tta papa"

    "--papa-no-adapter --papa-tta none"

    "--papa-no-bottleneck --papa-no-adapter --papa-tta none"
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
echo "[NEXT] Summarize with:"
echo "python evaluate_papa_ablation_grid.py --root ${OUT_ROOT}"

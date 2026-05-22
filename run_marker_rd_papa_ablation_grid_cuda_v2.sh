#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run_marker_rd_papa_ablation_grid.sh
#
# Hypothesis-driven ablation grid for the marker-aware respiratory-dynamics
# branch proposed after dropping ECG from the main method.
#
# Central idea:
#   IMU -> reconstructed pressure STFT + RR -> respiratory dynamics -> MWL
#   Marker/OptiTrack is a training-time motion/posture teacher, not a direct
#   test-time MWL input.
#
# The script intentionally separates:
#   1. an existing respiratory-only PAPA-dyn audit, which can run today; and
#   2. planned marker-aware runs, which expect an implementation script exposing
#      the marker flags listed below.
#
# Smoke test:
#   TIER=smoke SUBJECTS="S13 S16 S19" EPOCHS=2 DEVICES="cuda:0" \
#     bash run_marker_rd_papa_ablation_grid.sh
#
# Core run:
#   TIER=core DEVICES="cuda:0 cuda:1" \
#     MARKER_SCRIPT=vit_pressure_crossmodal_marker_rd_papa.py \
#     bash run_marker_rd_papa_ablation_grid.sh
#
# Dry run command preview:
#   DRY_RUN=1 TIER=core bash run_marker_rd_papa_ablation_grid.sh
# -----------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"
RESP_SCRIPT="${RESP_SCRIPT:-vit_pressure_crossmodal_papa_dyn.py}"
MARKER_SCRIPT="${MARKER_SCRIPT:-vit_pressure_crossmodal_marker_rd_papa_cuda_v2.py}"
DEVICES=(${DEVICES:-cuda:0 cuda:1})
TIER="${TIER:-core}"              # smoke|core|full
DRY_RUN="${DRY_RUN:-0}"
SKIP_MARKER_IF_MISSING="${SKIP_MARKER_IF_MISSING:-1}"
# One long-running job is scheduled per listed GPU. Set to 0 to force serial.
PARALLEL="${PARALLEL:-1}"

DATA_STR="${DATA_STR:-imu_filt}"
PRETRAIN_DATA_GROUP="${PRETRAIN_DATA_GROUP:-mr}"
EMBED_LABELS="${EMBED_LABELS:-L0,L2,L3}"
EMBED_DATA_GROUP="${EMBED_DATA_GROUP:-auto}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
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

# Respiratory variants that have been useful in earlier ladders. The marker
# branch should test these as physiological views, not as generic embeddings.
RESP_DYN_LADDER="${RESP_DYN_LADDER:-rr_abs,rr_delta,rr_z,stft_abs,stft_delta,stft_z,stft_hybrid,dyn_abs,dyn_delta,dyn_z,dyn_hybrid,dyn_abs_hmm,dyn_z_hmm,dyn_hybrid_hmm,state_abs,hybrid_abs,hybrid_dyn_abs,hybrid_dyn_abs_hmm,hybrid_z,hybrid_z_hmm}"
PRIMARY_RESP_VARIANT="${PRIMARY_RESP_VARIANT:-dyn_hybrid_hmm}"
SECONDARY_RESP_VARIANT="${SECONDARY_RESP_VARIANT:-stft_hybrid}"
TERTIARY_RESP_VARIANT="${TERTIARY_RESP_VARIANT:-dyn_z_hmm}"

# Marker branch assumptions. These flags are meant for the implementation that
# follows this design discussion. They are deliberately explicit so the grid is
# interpretable rather than a broad hyperparameter sweep.
MARKER_EXPERT_CLASSIFIER="${MARKER_EXPERT_CLASSIFIER:-torch_logreg}"
MARKER_GATE_TEMPERATURE="${MARKER_GATE_TEMPERATURE:-0.75}"
MARKER_TEACHER_ALPHA="${MARKER_TEACHER_ALPHA:-1.0}"
MARKER_PROFILE_ALPHA="${MARKER_PROFILE_ALPHA:-10.0}"
MARKER_QUALITY_Q="${MARKER_QUALITY_Q:-0.75}"
MARKER_LOW_MOTION_Q="${MARKER_LOW_MOTION_Q:-0.25}"
MARKER_HIGH_MOTION_Q="${MARKER_HIGH_MOTION_Q:-0.75}"
MARKER_MIN_VALID="${MARKER_MIN_VALID:-20}"

# CUDA-heavy MA-RD-PAPA post-hoc components, mirroring the CRA CUDA runner.
MARD_RIDGE_BACKEND="${MARD_RIDGE_BACKEND:-torch}"
MARD_GATE_BACKEND="${MARD_GATE_BACKEND:-torch}"
MARD_CUDA_DEVICE="${MARD_CUDA_DEVICE:-auto}"
MARD_STRICT_CUDA="${MARD_STRICT_CUDA:-0}"
MARD_TORCH_CLF_EPOCHS="${MARD_TORCH_CLF_EPOCHS:-200}"
MARD_TORCH_CLF_LR="${MARD_TORCH_CLF_LR:-1e-2}"
MARD_TORCH_CLF_BATCH_SIZE="${MARD_TORCH_CLF_BATCH_SIZE:-4096}"
MARD_TORCH_CLF_WEIGHT_DECAY="${MARD_TORCH_CLF_WEIGHT_DECAY:-0.0}"
MARD_TORCH_GATE_EPOCHS="${MARD_TORCH_GATE_EPOCHS:-200}"
MARD_TORCH_GATE_LR="${MARD_TORCH_GATE_LR:-1e-2}"
MARD_TORCH_GATE_BATCH_SIZE="${MARD_TORCH_GATE_BATCH_SIZE:-8192}"
MARD_TORCH_GATE_WEIGHT_DECAY="${MARD_TORCH_GATE_WEIGHT_DECAY:-0.0}"
MARD_NUM_WORKERS="${MARD_NUM_WORKERS:-2}"
MARD_PREFETCH_FACTOR="${MARD_PREFETCH_FACTOR:-2}"
MARD_PIN_MEMORY="${MARD_PIN_MEMORY:-1}"
MARD_PERSISTENT_WORKERS="${MARD_PERSISTENT_WORKERS:-1}"
MARD_ENABLE_TF32="${MARD_ENABLE_TF32:-1}"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-runs/marker_rd_papa_ablation_grid/${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUT_ROOT}/analysis}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${ANALYSIS_DIR}"

SUBJECT_ARGS=()
if [ -n "${SUBJECTS:-}" ]; then
    # shellcheck disable=SC2206
    SUBJECT_ARGS=(--subjects ${SUBJECTS})
fi

EMBED_GROUP_ARGS=()
if [ "${EMBED_DATA_GROUP}" != "auto" ] && [ -n "${EMBED_DATA_GROUP}" ]; then
    EMBED_GROUP_ARGS=(--embed-data-group "${EMBED_DATA_GROUP}")
fi

COMMON_ARGS=(
    --data-str "${DATA_STR}"
    --data-group "${PRETRAIN_DATA_GROUP}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --embed-batch-size "${EMBED_BATCH_SIZE}"
    --lr "${LR}"
    --lambda-stft "${LAMBDA_STFT}"
    --lambda-rr "${LAMBDA_RR}"
    --lambda-contrast "${LAMBDA_CONTRAST}"
    --lambda-resp-rr "${LAMBDA_RESP_RR}"
    --lambda-resp-recon "${LAMBDA_RESP_RECON}"
    --contrast-warmup-epochs "${CONTRAST_WARMUP_EPOCHS}"
    --contrast-ramp-end-epoch "${CONTRAST_RAMP_END_EPOCH}"
    --embed-labels "${EMBED_LABELS}"
    "${EMBED_GROUP_ARGS[@]}"
    "${SUBJECT_ARGS[@]}"
)

RESP_DYN_ARGS=(
    --resp-dyn-ladder "${RESP_DYN_LADDER}"
    --resp-dyn-classifier "${RESP_DYN_CLASSIFIER}"
    --resp-dyn-roll-win "${RESP_DYN_ROLL_WIN}"
    --resp-dyn-hmm-stay "${RESP_DYN_HMM_STAY}"
    --resp-dyn-hmm-min-stay "${RESP_DYN_HMM_MIN_STAY}"
    --resp-dyn-target-baseline-q "${RESP_DYN_TARGET_BASELINE_Q}"
    --resp-dyn-source-baseline-q "${RESP_DYN_SOURCE_BASELINE_Q}"
    --resp-dyn-scale-floor "${RESP_DYN_SCALE_FLOOR}"
    --resp-dyn-boundary-jump-z "${RESP_DYN_BOUNDARY_JUMP_Z}"
)

MARKER_COMMON_ARGS=(
    --eval-marker-rd-papa
    --include-marker
    --marker-source-only
    --marker-expert-classifier "${MARKER_EXPERT_CLASSIFIER}"
    --marker-gate-temperature "${MARKER_GATE_TEMPERATURE}"
    --marker-teacher-alpha "${MARKER_TEACHER_ALPHA}"
    --marker-profile-alpha "${MARKER_PROFILE_ALPHA}"
    --marker-quality-q "${MARKER_QUALITY_Q}"
    --marker-low-motion-q "${MARKER_LOW_MOTION_Q}"
    --marker-high-motion-q "${MARKER_HIGH_MOTION_Q}"
    --marker-min-valid "${MARKER_MIN_VALID}"
    --mard-ridge-backend "${MARD_RIDGE_BACKEND}"
    --mard-gate-backend "${MARD_GATE_BACKEND}"
    --mard-torch-clf-epochs "${MARD_TORCH_CLF_EPOCHS}"
    --mard-torch-clf-lr "${MARD_TORCH_CLF_LR}"
    --mard-torch-clf-batch-size "${MARD_TORCH_CLF_BATCH_SIZE}"
    --mard-torch-clf-weight-decay "${MARD_TORCH_CLF_WEIGHT_DECAY}"
    --mard-torch-gate-epochs "${MARD_TORCH_GATE_EPOCHS}"
    --mard-torch-gate-lr "${MARD_TORCH_GATE_LR}"
    --mard-torch-gate-batch-size "${MARD_TORCH_GATE_BATCH_SIZE}"
    --mard-torch-gate-weight-decay "${MARD_TORCH_GATE_WEIGHT_DECAY}"
    --mard-num-workers "${MARD_NUM_WORKERS}"
    --mard-prefetch-factor "${MARD_PREFETCH_FACTOR}"
    "${RESP_DYN_ARGS[@]}"
)

if [ "${MARD_PIN_MEMORY}" = "1" ]; then
    MARKER_COMMON_ARGS+=(--mard-pin-memory)
else
    MARKER_COMMON_ARGS+=(--mard-no-pin-memory)
fi
if [ "${MARD_PERSISTENT_WORKERS}" = "1" ]; then
    MARKER_COMMON_ARGS+=(--mard-persistent-workers)
else
    MARKER_COMMON_ARGS+=(--mard-no-persistent-workers)
fi
if [ "${MARD_ENABLE_TF32}" = "1" ]; then
    MARKER_COMMON_ARGS+=(--mard-enable-tf32)
else
    MARKER_COMMON_ARGS+=(--mard-disable-tf32)
fi
if [ "${MARD_STRICT_CUDA}" = "1" ]; then
    MARKER_COMMON_ARGS+=(--mard-strict-cuda)
fi

run_cmd() {
    local tag="$1"
    local script="$2"
    local device="$3"
    shift 3
    local out_dir="${OUT_ROOT}/${tag}"
    local log_file="${LOG_ROOT}/${tag}__${device//[:\/]/_}.log"
    mkdir -p "${out_dir}"
    local extra_device_args=()
    if [ "${script}" = "${MARKER_SCRIPT}" ]; then
        if [ "${MARD_CUDA_DEVICE}" = "auto" ]; then
            extra_device_args=(--mard-cuda-device "${device}")
        else
            extra_device_args=(--mard-cuda-device "${MARD_CUDA_DEVICE}")
        fi
    fi
    local cmd=("${PYTHON_BIN}" "${script}" "${COMMON_ARGS[@]}" --device "${device}" "${extra_device_args[@]}" --out-dir "${out_dir}" "$@")
    echo "[RUN] ${tag} on ${device}"
    echo "+ ${cmd[*]}" | tee "${log_file}"
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi
    "${cmd[@]}" 2>&1 | tee -a "${log_file}"
}

maybe_run_marker() {
    local tag="$1"
    local device="$2"
    shift 2
    if [ ! -f "${MARKER_SCRIPT}" ] && [ "${SKIP_MARKER_IF_MISSING}" = "1" ]; then
        echo "[SKIP] ${tag}: ${MARKER_SCRIPT} not found. Set MARKER_SCRIPT or SKIP_MARKER_IF_MISSING=0 after implementation." | tee -a "${LOG_ROOT}/skipped_marker_runs.log"
        return 0
    fi
    run_cmd "${tag}" "${MARKER_SCRIPT}" "${device}" "$@"
}

echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Devices: ${DEVICES[*]}"
echo "[INFO] Marker script: ${MARKER_SCRIPT}"
echo "[INFO] Marker CUDA: expert=${MARKER_EXPERT_CLASSIFIER} ridge=${MARD_RIDGE_BACKEND} gate=${MARD_GATE_BACKEND} mard_device=${MARD_CUDA_DEVICE} embed_batch=${EMBED_BATCH_SIZE} tf32=${MARD_ENABLE_TF32}"

run_i=0

DEVICE_PIDS=()
DEVICE_TAGS=()
for _dev in "${DEVICES[@]}"; do
    DEVICE_PIDS+=("")
    DEVICE_TAGS+=("")
done

wait_for_device_index() {
    local idx="$1"
    local pid="${DEVICE_PIDS[$idx]:-}"
    if [ -n "${pid}" ]; then
        local tag="${DEVICE_TAGS[$idx]:-unknown}"
        echo "[WAIT] ${DEVICES[$idx]} finishing ${tag} pid=${pid}"
        if ! wait "${pid}"; then
            echo "[FAIL] ${tag} on ${DEVICES[$idx]} failed" >&2
            FAILED_JOBS=1
        fi
        DEVICE_PIDS[$idx]=""
        DEVICE_TAGS[$idx]=""
    fi
}

wait_for_all_devices() {
    local idx
    for idx in "${!DEVICES[@]}"; do
        wait_for_device_index "${idx}"
    done
}

# -----------------------------------------------------------------------------
# Tier definitions
# -----------------------------------------------------------------------------
case "${TIER}" in
  smoke)
    RUN_TAGS=(
      "00_resp_ladder_smoke"
      "01_marker_teacher_gate_smoke"
    )
    ;;
  core)
    RUN_TAGS=(
      "00_resp_ladder"
      "01_rd_no_marker_primary"
      "02_rd_no_marker_secondary"
      "03_imu_activity_gate_no_marker"
      "04_marker_teacher_motion_state"
      "05_marker_quality_gate"
      "06_marker_quality_hmm"
      "07_marker_posture_profile_gate"
      "08_marker_shuffled_teacher_control"
      "09_time_only_control"
    )
    ;;
  full)
    RUN_TAGS=(
      "00_resp_ladder"
      "01_rd_no_marker_primary"
      "02_rd_no_marker_secondary"
      "03_rd_no_marker_tertiary"
      "04_imu_activity_gate_no_marker"
      "05_marker_teacher_motion_state"
      "06_marker_quality_gate"
      "07_marker_quality_hmm"
      "08_marker_posture_profile_gate"
      "09_marker_posture_profile_expert"
      "10_marker_oracle_audit"
      "11_marker_only_oracle_control"
      "12_marker_shuffled_teacher_control"
      "13_marker_time_shift_control"
      "14_time_only_control"
      "15_low_motion_only_oracle"
      "16_high_motion_stress_test"
    )
    ;;
  *)
    echo "Unsupported TIER=${TIER}. Use smoke, core, or full." >&2
    exit 2
    ;;
esac

run_tag_on_device() {
    local tag="$1"
    local device="$2"
    case "${tag}" in
      00_resp_ladder*)
        run_cmd "${tag}" "${RESP_SCRIPT}" "${device}" \
          --eval-resp-dyn \
          "${RESP_DYN_ARGS[@]}" \
          --eval-frozen-embeddings \
          --embed-classifier linear \
          --embed-pooling rich \
          --embed-stft-profile
        ;;
      01_marker_teacher_gate_smoke)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode quality_gate \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      01_rd_no_marker_primary)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode none \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      02_rd_no_marker_secondary)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode none \
          --marker-resp-variant "${SECONDARY_RESP_VARIANT}"
        ;;
      03_rd_no_marker_tertiary)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode none \
          --marker-resp-variant "${TERTIARY_RESP_VARIANT}"
        ;;
      03_imu_activity_gate_no_marker|04_imu_activity_gate_no_marker)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --no-include-marker \
          --marker-mode imu_activity_gate \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      04_marker_teacher_motion_state|05_marker_teacher_motion_state)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode teacher_motion_state \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      05_marker_quality_gate|06_marker_quality_gate)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode quality_gate \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      06_marker_quality_hmm|07_marker_quality_hmm)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode quality_gate \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}" \
          --marker-motion-conditioned-hmm
        ;;
      07_marker_posture_profile_gate|08_marker_posture_profile_gate)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode posture_profile_gate \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      09_marker_posture_profile_expert)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode posture_profile_expert \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      10_marker_oracle_audit)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode oracle_audit \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      11_marker_only_oracle_control)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode marker_only_oracle \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      08_marker_shuffled_teacher_control|12_marker_shuffled_teacher_control)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode shuffled_teacher_control \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      13_marker_time_shift_control)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode time_shift_control \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      09_time_only_control|14_time_only_control)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode time_only_control \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      15_low_motion_only_oracle)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode low_motion_oracle \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      16_high_motion_stress_test)
        maybe_run_marker "${tag}" "${device}" \
          "${MARKER_COMMON_ARGS[@]}" \
          --marker-mode high_motion_stress_test \
          --marker-resp-variant "${PRIMARY_RESP_VARIANT}"
        ;;
      *)
        echo "No command mapped for tag ${tag}" >&2
        exit 2
        ;;
    esac
}

FAILED_JOBS=0
for tag in "${RUN_TAGS[@]}"; do
    device_idx=$((run_i % ${#DEVICES[@]}))
    run_i=$((run_i + 1))
    device="${DEVICES[$device_idx]}"
    if [ "${DRY_RUN}" = "1" ] || [ "${PARALLEL}" = "0" ]; then
        run_tag_on_device "${tag}" "${device}"
    else
        wait_for_device_index "${device_idx}"
        ( run_tag_on_device "${tag}" "${device}" ) &
        DEVICE_PIDS[$device_idx]="$!"
        DEVICE_TAGS[$device_idx]="${tag}"
        echo "[LAUNCH] ${tag} on ${device} pid=${DEVICE_PIDS[$device_idx]}"
    fi
done

if [ "${DRY_RUN}" != "1" ] && [ "${PARALLEL}" != "0" ]; then
    wait_for_all_devices
fi

if [ "${FAILED_JOBS}" != "0" ]; then
    echo "[ERROR] one or more ablation jobs failed; skipping evaluation" >&2
    exit 1
fi

if [ "${DRY_RUN}" != "1" ]; then
    EVAL_SCRIPT="${EVAL_SCRIPT:-evaluate_marker_rd_papa_grid.py}"
    if [ -f "${EVAL_SCRIPT}" ]; then
        "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
          --root "${OUT_ROOT}" \
          --out-dir "${ANALYSIS_DIR}" \
          --baseline-run "00_resp_ladder" \
          --metric bal_acc || true
    else
        echo "[INFO] ${EVAL_SCRIPT} not found in current directory. Run it manually after copying it beside this script."
    fi
fi

echo "[DONE] outputs: ${OUT_ROOT}"

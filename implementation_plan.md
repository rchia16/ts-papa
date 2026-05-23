I want to refactor the `vit_pressure_crossmodal_marker_rd_papa_cuda_v2.py` around **low-dimensional motion/posture reliability states**, while keeping most of the current MA-RD-PAPA structure.

The current script already has the right skeleton:

```text
IMU → reconstructed pressure STFT + RR → respiratory dynamics
marker → privileged motion/posture teacher
IMU/RR/activity → inferred marker-like state
inferred state → reliability / preference gate
```

The issue is that the current implementation still uses a high-dimensional marker proxy. The script extracts many marker motion/posture statistics, then trains `fit_marker_proxy(...)` to predict the full marker feature vector from AR features, and `build_marker_rd_feature_blocks(...)` can include `marker_proxy`, `motion_aware_resp`, and `motion_aware_resp_activity` as expert inputs. That is more like marker-feature reconstruction than low-dimensional quality supervision. 

The next implementation should preserve the existing file structure, but replace:

```text
AR → high-dimensional marker feature proxy
```

with:

```text
AR → low-dimensional motion/posture reliability state
```

where:

```text
AR = reconstructed respiratory dynamics + IMU activity + respiration×activity interactions
```

## Revised model goal

The marker branch should answer:

```text
When should the respiratory evidence be trusted?
```

not:

```text
Can we predict detailed OptiTrack kinematics?
```

So the marker teacher should produce a compact state:

```text
m_t = [
  clean_motion_score,
  motion_contamination_score,
  posture_shift_score,
  rotation_instability_score,
  translation_instability_score,
  marker_validity_score,
  optional motion_state_class
]
```

Then the test-time model predicts:

```text
m̂_t = g(R_t, A_t, R_t × A_t)
```

and uses `m̂_t` only to modulate:

```text
expert reliability
respiratory feature trust
HMM/sequence smoothing
failure reporting
```

By default, it should **not** concatenate `m̂_t` into classifier experts.

## Why this should fix the current failure mode

The current marker proxy target is too heterogeneous:

```text
position level / velocity / acceleration / jerk
rotation level / angular velocity / angular acceleration / angular jerk
spectral entropy
drift
burstiness
stationarity
```

Those features live on different physical scales and respond differently to interpolation noise. The current script computes marker features from marker position/rotation windows, including translation/rotation energy, velocity, jerk, drift, stationarity, spectral entropy, and dominant frequency. 

That is useful for deriving a teacher state, but too broad as a direct regression target.

The low-dimensional formulation avoids trying to predict every marker statistic. Instead, it distills marker into states like:

```text
clean / contaminated
stable / posture shift
translation-dominant / rotation-dominant
low / medium / high motion
```

These are the quantities that the respiratory classifier actually needs.

## Implementation plan

### Stage 1 — keep the existing script, add a new teacher path

Do not delete the current full marker proxy immediately. Keep it as a legacy ablation:

```bash
--mard-marker-target full_proxy
```

Add a new default:

```bash
--mard-marker-target quality_state
```

Recommended new CLI flags:

```bash
--mard-marker-target quality_state|full_proxy
--mard-quality-state-source-mode inferred|observed
--mard-quality-state-dim compact|full
--mard-quality-state-threshold-mode subject_quantile|source_global
--mard-use-quality-state-blocks
--mard-no-quality-state-blocks
--mard-quality-state-hmm
```

Default should be:

```bash
--mard-marker-target quality_state
--mard-no-quality-state-blocks
```

Meaning:

```text
predict quality state,
use it in the gate,
do not use it as a direct expert feature.
```

### Stage 2 — build marker reliability targets

Add a function after `marker_motion_features(...)`:

```python
def marker_quality_state_targets(
    marker_features: np.ndarray,
    marker_valid: np.ndarray,
    subject_ids: np.ndarray | None,
    args,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert high-dimensional marker features into low-dimensional
    motion/posture reliability states.

    Returns:
      y_state:   (N, D_state) continuous targets/probabilities
      valid:     (N,) rows with reliable marker-derived supervision
      metadata:  thresholds, feature names, state names
    """
```

Use the existing `MARKER_FEATURE_NAMES` list to avoid brittle indexing. Build a dictionary:

```python
idx = {name: i for i, name in enumerate(MARKER_FEATURE_NAMES)}
```

Then extract a compact set:

```python
valid_frac      = marker[:, idx["marker_valid_frac"]]
motion_energy   = marker[:, idx["marker_motion_energy"]]
stationarity    = marker[:, idx["marker_stationarity"]]
posture_drift   = marker[:, idx["marker_posture_drift"]]
burstiness      = marker[:, idx["marker_burstiness"]]

pos_vel_rms     = marker[:, idx["marker_pos_vel_rms"]]
pos_jerk_rms    = marker[:, idx["marker_pos_jerk_rms"]]
pos_drift       = marker[:, idx["marker_pos_drift"]]

rot_vel_rms     = marker[:, idx["marker_rot_vel_rms"]]
rot_jerk_rms    = marker[:, idx["marker_rot_jerk_rms"]]
rot_drift       = marker[:, idx["marker_rot_drift"]]
```

The important change is to **separate translation and rotation** before combining them.

Use subject-robust z-scoring:

```python
motion_z = subject_robust_z(motion_energy[:, None], subject_ids)
drift_z  = subject_robust_z(posture_drift[:, None], subject_ids)
rot_z    = subject_robust_z(rot_vel_rms[:, None], subject_ids)
pos_z    = subject_robust_z(pos_vel_rms[:, None], subject_ids)
burst_z  = subject_robust_z(burstiness[:, None], subject_ids)
```

Then derive continuous teacher states:

```python
clean_score =
    sigmoid(-(0.8 * motion_z + 0.8 * drift_z + 0.5 * burst_z))

motion_contam_score =
    sigmoid(0.8 * motion_z + 0.5 * burst_z + 0.5 * (1 - stationarity))

posture_shift_score =
    sigmoid(drift_z)

rotation_instability_score =
    sigmoid(rot_z)

translation_instability_score =
    sigmoid(pos_z)

marker_validity_score =
    valid_frac
```

Final state vector:

```text
[
  clean_score,
  motion_contam_score,
  posture_shift_score,
  rotation_instability_score,
  translation_instability_score,
  marker_validity_score
]
```

Optional discrete state:

```text
0 = clean/stable
1 = translation motion
2 = rotation motion
3 = posture shift
4 = invalid/uncertain
```

This gives you a low-dimensional marker target that is directly about respiratory reliability.

### Stage 3 — replace `fit_marker_proxy` with `fit_marker_quality_state_proxy`

Current function:

```python
fit_marker_proxy(...)
```

learns:

```text
AR → full marker feature vector
```

Replace or wrap it with:

```python
def fit_marker_quality_state_proxy(
    x_ar_train: np.ndarray,
    marker_features_train: np.ndarray,
    marker_valid_train: np.ndarray,
    train_subject_ids: np.ndarray,
    x_ar_test: np.ndarray,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Learn AR -> low-dimensional marker quality state.

    Returns:
      state_train_pred
      state_train_unc
      state_test_pred
      state_test_unc
      metadata
    """
```

Inside:

```python
state_y, state_valid, state_meta = marker_quality_state_targets(
    marker_features_train,
    marker_valid_train,
    train_subject_ids,
    args,
)
```

Then train regressors only on `state_valid`.

For continuous state targets, ridge is fine:

```text
AR → six continuous quality scores
```

But standardize the target before ridge:

```python
y_scaler = StandardScaler().fit(state_y[state_valid])
y_z = y_scaler.transform(state_y[state_valid])
pred_z = ridge.predict(...)
pred = y_scaler.inverse_transform(pred_z)
pred = np.clip(pred, 0.0, 1.0)
```

That is important. It prevents one state dimension from dominating the loss.

For the optional discrete state, use torch logistic regression:

```text
AR → motion_state_class
```

The continuous version should be the default.

### Stage 4 — change `build_marker_rd_feature_blocks(...)`

Currently this function does:

```python
m_proxy_tr, m_unc_tr, m_proxy_te, m_unc_te, marker_meta = fit_marker_proxy(...)
x_mint_tr = motion_interaction_features(..., m_proxy_tr)
...
blocks["marker_proxy"]
blocks["motion_aware_resp"]
blocks["motion_aware_resp_activity"]
```

Refactor to:

```python
m_state_tr, m_state_unc_tr, m_state_te, m_state_unc_te, marker_meta =
    fit_marker_quality_state_proxy(...)
```

Then build quality context:

```python
q_train = quality(
    resp_static=train["x_resp_static"],
    act_static=train["x_activity_static"],
    motion_state=m_state_tr,
    motion_unc=m_state_unc_tr,
)

q_test = quality(
    resp_static=test["x_resp_static"],
    act_static=test["x_activity_static"],
    motion_state=m_state_te,
    motion_unc=m_state_unc_te,
)
```

Default blocks should be only:

```python
blocks["resp_dyn"]
blocks["activity_dyn"]
blocks["resp_activity"]
blocks["papa_state"]  # if available
```

Do **not** include marker state blocks by default.

Only if explicitly requested:

```bash
--mard-use-quality-state-blocks
```

add:

```python
blocks["quality_state"] = concat(m_state, m_state_unc)
blocks["quality_conditioned_resp"] = concat(x_resp, m_state, m_state_unc)
```

This makes the scientific default clean:

```text
marker state influences reliability/gating,
not class-feature geometry.
```

### Stage 5 — redefine quality columns

The gate should see:

```text
respiratory quality:
  RR-head/STFT disagreement
  respiratory spectral entropy
  peakness
  bandwidth
  spectral uncertainty

IMU activity:
  IMU motion energy
  IMU stationarity
  IMU motion entropy

predicted marker-quality state:
  clean_score
  motion_contam_score
  posture_shift_score
  rotation_instability_score
  translation_instability_score
  marker_validity_score
  state_uncertainty
```

So update the `quality(...)` inner function in `build_marker_rd_feature_blocks(...)` to use named dimensions:

```python
clean = motion_state[:, 0]
motion_contam = motion_state[:, 1]
posture_shift = motion_state[:, 2]
rot_instability = motion_state[:, 3]
trans_instability = motion_state[:, 4]
validity = motion_state[:, 5]
```

Then:

```python
q = np.stack([
    rr_disagreement,
    resp_entropy,
    resp_peakness,
    resp_bandwidth,
    resp_uncertainty,
    imu_motion_energy,
    imu_stationarity,
    imu_motion_entropy,
    clean,
    motion_contam,
    posture_shift,
    rot_instability,
    trans_instability,
    validity,
    state_uncertainty,
], axis=1)
```

This is much more interpretable than the current `marker_proxy[:, -5]` style indexing.

### Stage 6 — use quality state for HMM smoothing

Add a mode:

```bash
--mard-quality-state-hmm
```

Current HMM smoothing is binary:

```python
if args.mard_hmm_smooth:
    viterbi_smooth(...)
```

Instead compute a per-window reliability score:

```python
resp_reliability_t =
    clean_score
    * (1 - motion_contam_score)
    * (1 - posture_shift_score)
    * low_uncertainty_weight
```

Then use it to soften emissions before HMM:

```python
p_reliable = resp_reliability_t.reshape(-1, 1)
uniform = np.ones_like(final_proba) / final_proba.shape[1]

emission_proba =
    p_reliable * final_proba
  + (1 - p_reliable) * uniform
```

Then smooth:

```python
idx = viterbi_smooth(np.log(emission_proba), stay_prob=adaptive_stay)
```

Where:

```python
adaptive_stay = base_stay + alpha * (1 - mean(resp_reliability))
```

This means:

```text
when motion/posture reliability is poor,
trust per-window respiratory evidence less
and lean more on temporal continuity.
```

### Stage 7 — output teacher diagnostics

Add files per subject:

```text
marker_quality_state_trace.csv
marker_quality_state_summary.csv
marker_quality_state_proxy_metrics.csv
```

Trace columns:

```text
window_idx
y_true
y_pred
clean_score_hat
motion_contam_score_hat
posture_shift_score_hat
rotation_instability_hat
translation_instability_hat
quality_state_uncertainty
target_marker_available
oracle_clean_score       # only if target marker exists, audit only
oracle_motion_contam     # audit only
```

Summary columns:

```text
mean_clean_score_hat
mean_motion_contam_hat
mean_posture_shift_hat
mean_rotation_instability_hat
mean_translation_instability_hat
mean_state_uncertainty
target_marker_valid_rate
source_marker_valid_rate
```

Proxy audit metrics, when target marker exists:

```text
clean_score_rmse
motion_contam_rmse
posture_shift_rmse
rotation_instability_rmse
translation_instability_rmse
quality_state_mean_rmse
motion_state_bal_acc      # if discrete state used
```

These tell you whether the marker teacher is being learned.

### Stage 8 — update ablations

Replace the current marker ablations with sharper ones.

Core grid:

```text
A. resp_ladder
   existing PAPA-dyn respiratory ladder

B. imu_activity_gate_no_marker
   current best model; no marker

C. marker_quality_state_gate
   low-dimensional marker state predicted from AR;
   used only in gate

D. marker_quality_state_hmm
   low-dimensional marker state used for gate + adaptive HMM smoothing

E. marker_quality_state_blocks
   explicit ablation where state is also concatenated into expert features

F. marker_quality_state_oracle_gate
   diagnostic only: true target marker-derived state used in gate
   tells upper bound of marker-quality idea

G. shuffled_quality_state_control
   shuffle low-dimensional teacher states across source rows

H. time_only_quality_state_control
   replace teacher with time basis
```

Expected interpretation:

```text
C/D should beat B if marker quality states help.
F should beat C/D if marker state is useful but hard to infer.
G/H should not match C/D.
E should not be the main claim; if E wins but C/D does not, marker is acting like a feature shortcut.
```

## Code-level change map

Here is the concrete mapping from current script to new implementation.

### Keep mostly unchanged

```python
build_marker_loaders_with_subject_ids(...)
collect_marker_rd_features(...)
imu_activity_features(...)
marker_motion_features(...)
SafeProbClassifier
TorchLogRegClassifier
train_preference_gate(...)
fit_outer_experts(...)
safe_classification_metrics(...)
marker_rd_papa_hook(...)
```

### Add

```python
marker_feature_index()
marker_quality_state_targets(...)
fit_marker_quality_state_proxy(...)
quality_state_prediction_metrics(...)
apply_quality_state_hmm(...)
```

### Replace or branch

Current:

```python
fit_marker_proxy(...)
```

New:

```python
if args.mard_marker_target == "full_proxy":
    fit_marker_proxy(...)  # legacy
else:
    fit_marker_quality_state_proxy(...)
```

Current:

```python
motion_interaction_features(..., marker_proxy)
```

New default:

```python
# no motion_interaction_features by default
# predicted quality state goes to gate only
```

Optional ablation:

```python
quality_state_interaction_features(..., quality_state)
```

### Modify

```python
build_marker_rd_feature_blocks(...)
```

so default expert blocks are:

```python
blocks = OrderedDict()
blocks["resp_dyn"] = (x_resp_tr, x_resp_te)
blocks["activity_dyn"] = (x_act_tr, x_act_te)
blocks["resp_activity"] = (...)
if "papa_state" in train:
    blocks["papa_state"] = (...)
```

and quality uses:

```python
m_state_tr
m_state_te
m_state_unc_tr
m_state_unc_te
```

## Suggested new function skeleton

```python
QUALITY_STATE_NAMES = [
    "clean_score",
    "motion_contam_score",
    "posture_shift_score",
    "rotation_instability_score",
    "translation_instability_score",
    "marker_validity_score",
]

def marker_quality_state_targets(marker_features, marker_valid, subject_ids, args):
    idx = {name: i for i, name in enumerate(MARKER_FEATURE_NAMES)}
    m = np.asarray(marker_features, dtype=np.float32)
    valid = np.asarray(marker_valid, dtype=bool)

    def col(name, default=0.0):
        if name in idx:
            return m[:, idx[name]]
        return np.full(m.shape[0], default, dtype=np.float32)

    valid_frac = col("marker_valid_frac")
    motion = col("marker_motion_energy")
    stationarity = col("marker_stationarity", 1.0)
    drift = col("marker_posture_drift")
    burst = col("marker_burstiness")

    pos_vel = col("marker_pos_vel_rms")
    pos_jerk = col("marker_pos_jerk_rms")
    pos_drift = col("marker_pos_drift")

    rot_vel = col("marker_rot_vel_rms")
    rot_jerk = col("marker_rot_jerk_rms")
    rot_drift = col("marker_rot_drift")

    X = np.stack([
        motion, 1.0 - stationarity, drift, burst,
        pos_vel, pos_jerk, pos_drift,
        rot_vel, rot_jerk, rot_drift,
    ], axis=1)

    Z = subject_robust_z(X, subject_ids, floor=float(args.resp_dyn_scale_floor))

    motion_z = Z[:, 0]
    nonstat_z = Z[:, 1]
    drift_z = Z[:, 2]
    burst_z = Z[:, 3]
    pos_z = np.maximum.reduce([Z[:, 4], Z[:, 5], Z[:, 6]])
    rot_z = np.maximum.reduce([Z[:, 7], Z[:, 8], Z[:, 9]])

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    motion_contam = sigmoid(0.7 * motion_z + 0.4 * nonstat_z + 0.4 * burst_z)
    posture_shift = sigmoid(0.8 * drift_z + 0.4 * np.maximum(pos_z, rot_z))
    rot_instability = sigmoid(rot_z)
    trans_instability = sigmoid(pos_z)
    clean = 1.0 - np.maximum.reduce([
        motion_contam,
        posture_shift,
        1.0 - np.clip(valid_frac, 0.0, 1.0),
    ])

    y_state = np.stack([
        clean,
        motion_contam,
        posture_shift,
        rot_instability,
        trans_instability,
        np.clip(valid_frac, 0.0, 1.0),
    ], axis=1)

    y_state = np.nan_to_num(y_state, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    state_valid = valid & (valid_frac > 0.5)

    meta = {
        "quality_state_names": QUALITY_STATE_NAMES,
        "quality_state_valid_rows": int(state_valid.sum()),
        "quality_state_mean": dict(zip(QUALITY_STATE_NAMES, y_state.mean(axis=0).tolist())),
    }
    return y_state.astype(np.float32), state_valid, meta
```

## Recommended first implementation variant

Do the smallest clean implementation first:

```text
Variant C:
  marker_quality_state_gate
```

That means:

```text
expert blocks:
  resp_dyn
  activity_dyn
  resp_activity
  papa_state

gate context:
  respiratory quality
  IMU activity quality
  predicted marker quality states
  predicted state uncertainty

no marker-state expert blocks
no direct marker feature proxy
no full marker vector regression
```

This directly tests:

```text
Does marker-derived motion/posture reliability improve expert weighting?
```

Then add HMM:

```text
Variant D:
  marker_quality_state_gate + quality-state HMM
```

Only after that should you test:

```text
Variant E:
  marker quality state as feature block
```

## Expected result if the idea is correct

You should see:

```text
marker_quality_state_gate
  > imu_activity_gate_no_marker
```

especially in:

```text
high-motion windows
posture-shift windows
high RR/STFT disagreement windows
subjects with high motion confounding
```

You should also see:

```text
quality_state_oracle_gate
  > quality_state_predicted_gate
  > imu_activity_gate_no_marker
```

If the oracle quality state does not help, marker reliability is not useful for this dataset.

If the oracle helps but predicted quality state does not, the target is good but the AR→state predictor is weak.

If shuffled/time-only controls match the true state, the state is still contaminated by protocol chronology.

## Bottom line

The implementation should evolve from:

```text
marker as high-dimensional proxy feature
```

to:

```text
marker as low-dimensional reliability-state teacher
```

The current code already has nearly all the infrastructure: marker loading, marker feature extraction, CUDA ridge/logistic helpers, source-inner-LOSO gate, respiratory dynamics, and summaries. The main change is to insert a **marker quality-state distillation layer** between `marker_motion_features(...)` and `build_marker_rd_feature_blocks(...)`, then use its predicted states only inside the gate and optional HMM reliability logic.


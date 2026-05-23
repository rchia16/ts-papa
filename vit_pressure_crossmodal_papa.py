#!/usr/bin/env python3
"""
PAPA: Physiology-Aligned Prototype Adaptation for IMU mental workload.

This script keeps the original ViT pressure/RR pretraining path, but adds a
small respiration-state bottleneck and a simple prototype-based MWL head.

Idea:
  IMU -> pressure/RR grounded ViT -> respiration-state vector r
      -> subject adapter A_s(r)
      -> distance to workload-conditioned respiration prototypes.

At test time, PAPA adapts only the tiny subject adapter with unlabeled target
windows. The encoder and pressure/RR heads remain fixed so the adapted space
stays tied to the learned respiratory geometry.

Ablations:
  --papa-no-bottleneck
      Use the raw rich physiological state vector directly:
        mean/std/max hidden + RR + pressure-STFT summary stats.
      This tests whether an explicit learned respiration-state bottleneck helps.

  --papa-no-adapter
      Disable the subject adapter.
      This tests whether subject-specific test-time alignment helps.

Expected files next to this script:
  - vit_pressure_crossmodal_stft_rr_core.py
  - vit_pressure_crossmodal_stft_rr_mwl_tta_main.py


  
Explicit ablation settings for:

```bash
--papa-no-bottleneck
--papa-no-adapter
```

Can run four variants:

```bash
# Full PAPA
--eval-papa --papa-tta papa

# No respiration-state bottleneck, adapter kept
--eval-papa --papa-tta papa --papa-no-bottleneck

# Bottleneck kept, no subject adapter / no adaptation
--eval-papa --papa-tta none --papa-no-adapter

# Neither bottleneck nor adapter
--eval-papa --papa-tta none --papa-no-bottleneck --papa-no-adapter
```

"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import vit_pressure_crossmodal_stft_rr_core as core
from config import SBJ_PROCESSED_DIR
from vit_pressure_crossmodal_stft_rr_mwl_tta_main import (
    SUBJECTS,
    _build_frozen_embedding_loaders,
    frozen_embedding_hook,
    infer_embed_data_group_from_labels,
    parse_mwl_labels,
)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class IdentityAdapter(nn.Module):
    """Adapter ablation: no subject-specific alignment."""

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return r


class ResidualAffineAdapter(nn.Module):
    """
    Tiny subject adapter:

        r_out = r + scale * W(LN(r))

    Initialized as identity, so pre-TTA predictions are unchanged.
    """

    def __init__(self, dim: int, init_scale: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.delta = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return r + self.scale * self.delta(self.norm(r))

    @torch.no_grad()
    def reset_identity(self, init_scale: float = 0.05) -> None:
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.scale.fill_(float(init_scale))


class PAPAIMU2PressureViT(core.TinyIMU2PressureViT):
    """
    Original pressure/RR ViT plus optional respiration-state bottleneck and
    optional subject adapter.

    forward() intentionally keeps the same return signature as the core model:

        pressure_logmag, rr_pred, hidden

    so all existing pressure/RR pretraining and evaluation utilities still work.
    """

    # These are set by finalize_args() before the core runner creates models.
    default_use_bottleneck: bool = True
    default_use_subject_adapter: bool = True
    default_resp_state_dim: int = 48
    default_adapter_init_scale: float = 0.05

    def __init__(
        self,
        *args,
        resp_state_dim: int | None = None,
        use_bottleneck: bool | None = None,
        use_subject_adapter: bool | None = None,
        adapter_init_scale: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_bottleneck = (
            bool(self.default_use_bottleneck)
            if use_bottleneck is None
            else bool(use_bottleneck)
        )
        self.use_subject_adapter = (
            bool(self.default_use_subject_adapter)
            if use_subject_adapter is None
            else bool(use_subject_adapter)
        )
        self.adapter_init_scale = (
            float(self.default_adapter_init_scale)
            if adapter_init_scale is None
            else float(adapter_init_scale)
        )

        requested_state_dim = (
            int(self.default_resp_state_dim)
            if resp_state_dim is None
            else int(resp_state_dim)
        )

        # Raw physiological state:
        #   hidden mean/std/max + RR + pressure-STFT mean/std/max
        self.raw_state_dim = self.d_model * 3 + 4

        if self.use_bottleneck:
            self.resp_state_dim = requested_state_dim
            self.resp_state_head = nn.Sequential(
                nn.LayerNorm(self.raw_state_dim),
                nn.Linear(self.raw_state_dim, self.d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, self.resp_state_dim),
                nn.LayerNorm(self.resp_state_dim),
            )
        else:
            # Bottleneck ablation: use the raw physiological state directly.
            self.resp_state_dim = self.raw_state_dim
            self.resp_state_head = nn.Identity()

        self.resp_rr_head = nn.Linear(self.resp_state_dim, 1)

        # Auxiliary decoder: force the bottleneck to preserve the rich
        # respiration/pressure morphology summary, not only RR.
        self.resp_recon_head = nn.Sequential(
            nn.LayerNorm(self.resp_state_dim),
            nn.Linear(self.resp_state_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.raw_state_dim),
        )

        if self.use_subject_adapter:
            self.subject_adapter = ResidualAffineAdapter(
                self.resp_state_dim,
                init_scale=self.adapter_init_scale,
            )
        else:
            self.subject_adapter = IdentityAdapter()

    def raw_respiration_state_from_outputs(
        self,
        pred_logmag: torch.Tensor,
        rr_pred: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        h_mean = hidden.mean(dim=1)
        h_std = hidden.std(dim=1, unbiased=False)
        h_max = hidden.max(dim=1).values

        spec_mean = pred_logmag.mean(dim=(1, 2), keepdim=False).view(-1, 1)
        spec_std = pred_logmag.std(dim=(1, 2), unbiased=False, keepdim=False).view(-1, 1)
        spec_max = pred_logmag.amax(dim=(1, 2), keepdim=False).view(-1, 1)
        rr = rr_pred.view(-1, 1)

        return torch.cat(
            [h_mean, h_std, h_max, rr, spec_mean, spec_std, spec_max],
            dim=1,
        )

    def raw_and_respiration_state_from_outputs(
        self,
        pred_logmag: torch.Tensor,
        rr_pred: torch.Tensor,
        hidden: torch.Tensor,
        adapt: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.raw_respiration_state_from_outputs(pred_logmag, rr_pred, hidden)
        r = self.resp_state_head(raw)
        if adapt:
            r = self.subject_adapter(r)
        return raw, r

    def respiration_state_from_outputs(
        self,
        pred_logmag: torch.Tensor,
        rr_pred: torch.Tensor,
        hidden: torch.Tensor,
        adapt: bool = False,
    ) -> torch.Tensor:
        _raw, r = self.raw_and_respiration_state_from_outputs(
            pred_logmag, rr_pred, hidden, adapt=adapt)
        return r

    def respiration_state(
        self,
        imu: torch.Tensor,
        adapt: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_logmag, rr_pred, hidden = self(imu)
        r = self.respiration_state_from_outputs(
            pred_logmag,
            rr_pred,
            hidden,
            adapt=adapt,
        )
        return r, {
            "pred_logmag": pred_logmag,
            "rr_pred": rr_pred,
            "hidden": hidden,
        }

    @torch.no_grad()
    def reset_subject_adapter(self) -> None:
        if isinstance(self.subject_adapter, ResidualAffineAdapter):
            self.subject_adapter.reset_identity(init_scale=self.adapter_init_scale)


# Monkey-patch the core runner so it builds PAPAIMU2PressureViT instead of
# TinyIMU2PressureViT. The constructor keeps the same default call signature.
core.TinyIMU2PressureViT = PAPAIMU2PressureViT

# -----------------------------------------------------------------------------
# PAPA-aware pretraining
# -----------------------------------------------------------------------------
def train_one_epoch_papa(model: nn.Module, loader, optimizer, device: str, args, lambda_contrast: float) -> core.EpochMetrics:
    """
    Same as core.train_one_epoch, plus an optional RR-from-respiration-state
    auxiliary loss. This trains the bottleneck to preserve RR information.
    """
    model.train()
    totals = {
        "loss": [],
        "stft": [],
        "rr": [],
        "contrast": [],
        "resp_rr": [],
        "resp_recon": [],
    }
    for batch in loader:
        imu, pressure, _, br, _ = core.unpack_batch(batch, device)

        if not args.use_device_br:
            br = None    

        optimizer.zero_grad(set_to_none=True)

        pred_logmag, rr_pred, hidden = model(imu)

        loss, parts = core.pressure_stft_recon_loss(
            model,
            pred_logmag,
            pressure,
            rr_pred,
            br,
            lambda_stft=args.lambda_stft,
            lambda_rr=args.lambda_rr,
        )

        l_resp_rr = pred_logmag.new_tensor(0.0)
        l_resp_recon = pred_logmag.new_tensor(0.0)

        has_papa_state = (
            hasattr(model, "raw_and_respiration_state_from_outputs")
            and hasattr(model, "resp_rr_head")
        )

        if has_papa_state:
            raw_state, r = model.raw_and_respiration_state_from_outputs(
                pred_logmag,
                rr_pred,
                hidden,
                adapt=False,
            )

            if float(getattr(args, "lambda_resp_rr", 0.0)) > 0:
                rr_true = core.rr_targets_from_batch(pressure, br).view(-1)
                rr_from_r = model.resp_rr_head(r).view(-1)
                l_resp_rr = F.smooth_l1_loss(rr_from_r, rr_true)
                loss = loss + float(args.lambda_resp_rr) * l_resp_rr

            if (
                hasattr(model, "resp_recon_head")
                and bool(getattr(args, "papa_use_bottleneck", True))
                and float(getattr(args, "lambda_resp_recon", 0.0)) > 0
            ):
                raw_hat = model.resp_recon_head(r)

                # Batch-standardize the raw target so hidden dimensions and
                # pressure/RR summary dimensions contribute comparably.
                raw_target = raw_state.detach()
                raw_mu = raw_target.mean(dim=0, keepdim=True)
                raw_sd = raw_target.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)

                raw_target_z = (raw_target - raw_mu) / raw_sd
                raw_hat_z = (raw_hat - raw_mu) / raw_sd

                l_resp_recon = F.smooth_l1_loss(raw_hat_z, raw_target_z)
                loss = loss + float(args.lambda_resp_recon) * l_resp_recon

        lc = pred_logmag.new_tensor(0.0)
        if lambda_contrast > 0:
            h_imu = model.encode(core.augment_imu(imu, shift_max=args.shift_max))
            h_pressure = model.encode_pressure(pressure, target_tokens=h_imu.size(1))
            lc = core.token_contrastive_loss(
                h_imu,
                h_pressure,
                temperature=args.temperature,
            )
            loss = loss + lambda_contrast * lc

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        totals["loss"].append(float(loss.detach().cpu()))
        totals["stft"].append(parts["stft"])
        totals["rr"].append(parts["rr"])
        totals["contrast"].append(float(lc.detach().cpu()))
        totals["resp_rr"].append(float(l_resp_rr.detach().cpu()))
        totals["resp_recon"].append(float(l_resp_recon.detach().cpu()))

    # core.EpochMetrics only has loss/stft/rr/contrast, so keep return compatible.
    # The resp_rr values are included in total train loss, but not printed unless
    # you extend core.EpochMetrics/history.
    return core.EpochMetrics(
        loss=float(np.mean(totals["loss"])) if totals["loss"] else float("nan"),
        stft=float(np.mean(totals["stft"])) if totals["stft"] else float("nan"),
        rr=float(np.mean(totals["rr"])) if totals["rr"] else float("nan"),
        contrast=float(np.mean(totals["contrast"])) if totals["contrast"] else float("nan"),
    )


core.train_one_epoch = train_one_epoch_papa


# -----------------------------------------------------------------------------
# Prototype utilities
# -----------------------------------------------------------------------------
def _to_device_stats(stats: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in stats.items()
    }


def _safe_cov_diag(x: torch.Tensor) -> torch.Tensor:
    return x.var(dim=0, unbiased=False).clamp_min(1e-4)


@torch.no_grad()
def collect_papa_states(
    model: PAPAIMU2PressureViT,
    loader,
    device: str,
    adapt: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    states, labels, rrs = [], [], []

    for batch in loader:
        imu, _pressure, cond, _br, _tlx = core.unpack_batch(batch, device)
        r, aux = model.respiration_state(imu, adapt=adapt)
        states.append(r.detach().cpu().numpy())
        labels.append(cond.detach().cpu().numpy().astype(int).reshape(-1))
        rrs.append(aux["rr_pred"].detach().cpu().numpy().reshape(-1))

    if not states:
        raise RuntimeError("No batches available for PAPA state collection.")

    return (
        np.concatenate(states),
        np.concatenate(labels),
        np.concatenate(rrs),
    )


def fit_papa_prototypes(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Fit class prototypes and a source distribution in standardized
    respiration-state space.
    """
    scaler = StandardScaler()
    x = scaler.fit_transform(x_train).astype(np.float32)
    y = y_train.astype(int)

    classes = np.array(sorted(np.unique(y).tolist()), dtype=np.int64)
    protos, vars_ = [], []

    for c in classes:
        xc = torch.tensor(x[y == c], dtype=torch.float32)
        protos.append(xc.mean(dim=0))
        vars_.append(_safe_cov_diag(xc))

    xt = torch.tensor(x, dtype=torch.float32)

    return {
        "classes": torch.tensor(classes, dtype=torch.long),
        "prototypes": torch.stack(protos, dim=0),
        "proto_var": torch.stack(vars_, dim=0),
        "source_mu": xt.mean(dim=0),
        "source_sd": xt.std(dim=0, unbiased=False).clamp_min(1e-4),
        "scaler_mean": torch.tensor(scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.tensor(scaler.scale_.astype(np.float32)).clamp_min(1e-6),
    }


def standardize_states(
    r: torch.Tensor,
    stats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    return (r - stats["scaler_mean"].to(r.device)) / stats["scaler_scale"].to(r.device)


def papa_logits_from_state(
    r: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    z = standardize_states(r, stats)
    p = stats["prototypes"].to(z.device)
    v = stats["proto_var"].to(z.device)

    # Diagonal Mahalanobis distance. Negative distance is the class logit.
    dist = ((z[:, None, :] - p[None, :, :]) ** 2 / v[None, :, :]).mean(dim=-1)
    return -dist / max(float(temperature), 1e-6)


def predict_papa(
    model: PAPAIMU2PressureViT,
    loader,
    stats: Dict[str, torch.Tensor],
    device: str,
    args,
    adapt: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    classes = stats["classes"].cpu().numpy()

    with torch.no_grad():
        for batch in loader:
            imu, _pressure, cond, _br, _tlx = core.unpack_batch(batch, device)
            r, _aux = model.respiration_state(imu, adapt=adapt)
            logits = papa_logits_from_state(
                r,
                stats,
                temperature=args.papa_temperature,
            )
            pred = classes[logits.argmax(dim=1).detach().cpu().numpy()]
            preds.append(pred.astype(int))
            labels.append(cond.detach().cpu().numpy().astype(int).reshape(-1))

    return np.concatenate(labels), np.concatenate(preds)


# -----------------------------------------------------------------------------
# PAPA test-time adaptation losses
# -----------------------------------------------------------------------------
def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=1).clamp_min(1e-8)
    return -(p * p.log()).sum(dim=1).mean()


def diversity_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Batch-level anti-collapse term.

    This is negative entropy of the mean prediction. Minimizing it encourages
    the batch to use more than one class.
    """
    p = F.softmax(logits, dim=1).mean(dim=0).clamp_min(1e-8)
    return (p * p.log()).sum()


def prototype_attraction_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Soft prototype attraction without hard pseudo-labels.

    Since logits are negative distances, minimizing -E_p[logit] pulls samples
    toward the currently most plausible prototype.
    """
    p = F.softmax(logits.detach(), dim=1)
    return -(p * logits).sum(dim=1).mean()


def nrc_loss(
    r: torch.Tensor,
    logits: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Small target-neighborhood consistency loss in respiration-state space.

    This is a simple NRC-style baseline/component:
      - build nearest-neighbor graph in current respiration-state space
      - encourage each sample's distribution to match its neighbors' average
    """
    if r.size(0) <= 2:
        return r.new_tensor(0.0)

    k = max(1, min(int(k), r.size(0) - 1))
    z = F.normalize(r.detach(), dim=1)
    sim = z @ z.T
    sim.fill_diagonal_(-1.0)
    nn_idx = sim.topk(k=k, dim=1).indices

    logp = F.log_softmax(logits, dim=1)
    p_nb = F.softmax(logits.detach(), dim=1)[nn_idx].mean(dim=1)
    return F.kl_div(logp, p_nb, reduction="batchmean")


def papa_alignment_loss(
    r: torch.Tensor,
    stats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Align target subject's adapted respiration-state distribution to the
    source respiration-state distribution.

    This is intentionally simple: match standardized mean and log-std.
    """
    z = standardize_states(r, stats)
    mu = z.mean(dim=0)
    sd = z.std(dim=0, unbiased=False).clamp_min(1e-4)

    return F.mse_loss(mu, stats["source_mu"].to(z.device)) + F.mse_loss(
        torch.log(sd),
        torch.log(stats["source_sd"].to(z.device)),
    )


def temporal_smoothness_loss(
    r: torch.Tensor,
    rr: torch.Tensor,
) -> torch.Tensor:
    """
    CoTTA-lite temporal stabilizer.

    Consecutive windows should not jump sharply in respiration-state space
    or RR prediction.
    """
    if r.size(0) < 2:
        return r.new_tensor(0.0)

    r_step = (
        F.normalize(r[1:], dim=1)
        - F.normalize(r[:-1], dim=1)
    ).pow(2).mean()

    rr_step = F.smooth_l1_loss(rr[1:].view(-1), rr[:-1].view(-1))
    return r_step + 0.05 * rr_step


def run_papa_tta(
    model: PAPAIMU2PressureViT,
    target_loader,
    stats: Dict[str, torch.Tensor],
    device: str,
    args,
    out_dir: Path,
) -> Dict[str, float]:
    """
    Adapt only the subject adapter using unlabeled target windows.

    If --papa-no-adapter is used, there are no trainable adapter parameters and
    this function returns a skipped TTA row.
    """
    if not bool(getattr(args, "papa_use_adapter", True)):
        row = {
            "papa_epoch": 0,
            "papa_method": str(args.papa_tta).lower(),
            "papa_tta_skipped": 1,
            "papa_skip_reason": "subject_adapter_disabled",
        }
        pd.DataFrame([row]).to_csv(out_dir / "papa_tta_history.csv", index=False)
        print("[PAPA] TTA skipped because --papa-no-adapter was set.")
        return row

    model.train()
    model.reset_subject_adapter()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.subject_adapter.parameters():
        p.requires_grad = True

    params = [p for p in model.subject_adapter.parameters() if p.requires_grad]
    if not params:
        row = {
            "papa_epoch": 0,
            "papa_method": str(args.papa_tta).lower(),
            "papa_tta_skipped": 1,
            "papa_skip_reason": "no_trainable_adapter_params",
        }
        pd.DataFrame([row]).to_csv(out_dir / "papa_tta_history.csv", index=False)
        print("[PAPA] TTA skipped because no adapter parameters are trainable.")
        return row

    opt = torch.optim.AdamW(
        params,
        lr=args.papa_lr,
        weight_decay=args.papa_weight_decay,
    )

    rows: List[Dict[str, float]] = []
    stats = _to_device_stats(stats, device)

    for epoch in range(1, int(args.papa_epochs) + 1):
        totals: Dict[str, List[float]] = {
            "loss": [],
            "align": [],
            "proto": [],
            "ent": [],
            "div": [],
            "nrc": [],
            "smooth": [],
        }

        for batch in target_loader:
            imu, _pressure, _cond, _br, _tlx = core.unpack_batch(batch, device)

            opt.zero_grad(set_to_none=True)
            r, aux = model.respiration_state(imu, adapt=True)
            logits = papa_logits_from_state(
                r,
                stats,
                temperature=args.papa_temperature,
            )

            l_align = papa_alignment_loss(r, stats)
            l_proto = prototype_attraction_loss(logits)
            l_ent = entropy_loss(logits)
            l_div = diversity_loss(logits)
            l_nrc = nrc_loss(r, logits, k=args.papa_nrc_k)
            l_smooth = temporal_smoothness_loss(r, aux["rr_pred"])

            method = str(args.papa_tta).lower()

            if method == "tent":
                # TENT baseline: confidence adaptation only.
                loss = l_ent

            elif method == "nrc":
                # NRC-style baseline in respiration-state space.
                loss = (
                    args.lambda_papa_entropy * l_ent
                    + args.lambda_papa_diversity * l_div
                    + args.lambda_papa_nrc * l_nrc
                )

            elif method == "cotta":
                # CoTTA-lite baseline: confidence + diversity + temporal smoothness.
                # This intentionally keeps implementation readable; no EMA teacher.
                loss = (
                    args.lambda_papa_entropy * l_ent
                    + args.lambda_papa_diversity * l_div
                    + args.lambda_papa_smooth * l_smooth
                )

            else:
                # Full PAPA:
                #   physiology distribution alignment
                # + soft prototype comparison
                # + entropy/diversity
                # + neighborhood consistency
                # + temporal smoothness
                loss = (
                    args.lambda_papa_align * l_align
                    + args.lambda_papa_proto * l_proto
                    + args.lambda_papa_entropy * l_ent
                    + args.lambda_papa_diversity * l_div
                    + args.lambda_papa_nrc * l_nrc
                    + args.lambda_papa_smooth * l_smooth
                )

            if not torch.isfinite(loss):
                continue

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            opt.step()

            for key, value in [
                ("loss", loss),
                ("align", l_align),
                ("proto", l_proto),
                ("ent", l_ent),
                ("div", l_div),
                ("nrc", l_nrc),
                ("smooth", l_smooth),
            ]:
                totals[key].append(float(value.detach().cpu()))

        row = {
            f"papa_{k}": float(np.mean(v)) if v else float("nan")
            for k, v in totals.items()
        }
        row["papa_epoch"] = epoch
        row["papa_method"] = str(args.papa_tta).lower()
        row["papa_tta_skipped"] = 0
        rows.append(row)

        print(
            f"PAPA epoch {epoch:03d} | "
            f"loss {row['papa_loss']:.4f} "
            f"align {row['papa_align']:.4f} "
            f"proto {row['papa_proto']:.4f} "
            f"nrc {row['papa_nrc']:.4f} "
            f"smooth {row['papa_smooth']:.4f}"
        )

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "papa_tta_history.csv", index=False)
        return rows[-1]

    return {}


# -----------------------------------------------------------------------------
# Evaluation hook
# -----------------------------------------------------------------------------
def papa_hook(
    model,
    sbj: str,
    subjects: List[str],
    _train_loader,
    _test_loader,
    device: str,
    args,
    sbj_dir: Path,
):
    if not bool(getattr(args, "eval_papa", False)):
        return []

    if not isinstance(model, PAPAIMU2PressureViT):
        raise TypeError("PAPA hook expected PAPAIMU2PressureViT.")

    train_loader, test_loader = _build_frozen_embedding_loaders(
        sbj,
        subjects,
        args,
    )

    x_train, y_train, _rr_train = collect_papa_states(
        model,
        train_loader,
        device,
        adapt=False,
    )
    stats = fit_papa_prototypes(x_train, y_train)

    out = sbj_dir / "papa"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "papa_prototype_classes.json", "w") as f:
        json.dump(
            {
                "classes": stats["classes"].cpu().numpy().astype(int).tolist(),
                "use_bottleneck": bool(args.papa_use_bottleneck),
                "use_adapter": bool(args.papa_use_adapter),
                "state_dim": int(x_train.shape[1]),
            },
            f,
            indent=2,
        )

    y_true_pre, y_pred_pre = predict_papa(
        model,
        test_loader,
        stats,
        device,
        args,
        adapt=False,
    )

    pre = {
        "papa_acc": float(accuracy_score(y_true_pre, y_pred_pre)),
        "papa_f1_macro": float(f1_score(y_true_pre, y_pred_pre, average="macro", zero_division=0)),
        "papa_f1_weighted": float(f1_score(y_true_pre, y_pred_pre, average="weighted", zero_division=0)),
        "papa_n_train": int(y_train.shape[0]),
        "papa_n_test": int(y_true_pre.shape[0]),
        "papa_state_dim": int(x_train.shape[1]),
        "papa_use_bottleneck": int(bool(args.papa_use_bottleneck)),
        "papa_use_adapter": int(bool(args.papa_use_adapter)),
    }

    pd.DataFrame(
        {
            "y_true": y_true_pre.astype(int),
            "y_pred": y_pred_pre.astype(int),
        }
    ).to_csv(out / "papa_predictions_pre_tta.csv", index=False)

    rows = [
        {
            "__summary_name__": "papa_summary",
            "subject": sbj,
            "tag": "pre_tta",
            **pre,
        }
    ]

    print(f"PAPA_PRE_TTA {sbj}: {pre}")

    if str(args.papa_tta).lower() != "none" and int(args.papa_epochs) > 0:
        last = run_papa_tta(
            model,
            test_loader,
            stats,
            device,
            args,
            out,
        )

        y_true_post, y_pred_post = predict_papa(
            model,
            test_loader,
            stats,
            device,
            args,
            adapt=bool(args.papa_use_adapter),
        )

        post = {
            "papa_acc": float(accuracy_score(y_true_post, y_pred_post)),
            "papa_f1_macro": float(f1_score(y_true_post, y_pred_post, average="macro", zero_division=0)),
            "papa_f1_weighted": float(f1_score(y_true_post, y_pred_post, average="weighted", zero_division=0)),
            "papa_n_train": int(y_train.shape[0]),
            "papa_n_test": int(y_true_post.shape[0]),
            "papa_state_dim": int(x_train.shape[1]),
            "papa_use_bottleneck": int(bool(args.papa_use_bottleneck)),
            "papa_use_adapter": int(bool(args.papa_use_adapter)),
            **last,
        }

        pd.DataFrame(
            {
                "y_true": y_true_post.astype(int),
                "y_pred": y_pred_post.astype(int),
            }
        ).to_csv(out / "papa_predictions_post_tta.csv", index=False)

        rows.append(
            {
                "__summary_name__": "papa_summary",
                "subject": sbj,
                "tag": "post_tta",
                **post,
            }
        )

        print(f"PAPA_POST_TTA {sbj}: {post}")

    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def finalize_args(args) -> None:
    if args.eval_frozen_tlx:
        args.eval_frozen_embeddings = True

    args.embed_labels = parse_mwl_labels(args.embed_labels)

    if args.embed_data_group is None:
        args.embed_data_group = infer_embed_data_group_from_labels(args.embed_labels)

    # Explicit ablation switches.
    args.papa_use_bottleneck = not bool(args.papa_no_bottleneck)
    args.papa_use_adapter = not bool(args.papa_no_adapter)

    # Configure class defaults before the core runner instantiates the model.
    PAPAIMU2PressureViT.default_use_bottleneck = bool(args.papa_use_bottleneck)
    PAPAIMU2PressureViT.default_use_subject_adapter = bool(args.papa_use_adapter)
    PAPAIMU2PressureViT.default_resp_state_dim = int(args.papa_state_dim)
    PAPAIMU2PressureViT.default_adapter_init_scale = float(args.papa_adapter_init_scale)

    if args.eval_papa:
        # PAPA uses the same downstream loaders and label choices as frozen embeddings.
        args.include_tlx = bool(args.include_tlx or args.eval_frozen_tlx)


def main() -> None:
    parser = core.build_base_parser(
        SUBJECTS,
        str(Path(SBJ_PROCESSED_DIR) / "vit_pressure_crossmodal_papa"),
    )

    # Existing frozen-embedding evaluation, retained for comparison.
    parser.add_argument("--eval-frozen-embeddings", action="store_true")
    parser.add_argument("--eval-frozen-tlx", action="store_true")
    parser.add_argument("--tlx-ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--embed-data-group",
        default=None,
        choices=["mr", "level", "levels", "mr_levels"],
    )
    parser.add_argument("--embed-labels", default="L0,L1,L3")
    parser.add_argument(
        "--embed-classifier",
        default="lda",
        choices=["lda", "logreg", "linear_probe"],
    )
    parser.add_argument(
        "--embed-pooling",
        default="rich",
        choices=["mean", "max", "cls_last", "mean_std", "mean_std_max", "rich"],
    )
    parser.add_argument("--embed-stft-profile", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--linear-probe-epochs", type=int, default=30)
    parser.add_argument("--linear-probe-lr", type=float, default=1e-3)
    parser.add_argument("--linear-probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--linear-probe-batch-size", type=int, default=64)

    # PAPA model knobs.
    parser.add_argument(
        "--eval-papa",
        action="store_true",
        help="Evaluate respiration-state prototype MWL classifier.",
    )
    parser.add_argument(
        "--papa-state-dim",
        type=int,
        default=48,
        help="Respiration-state bottleneck dimension when bottleneck is enabled.",
    )
    parser.add_argument(
        "--papa-adapter-init-scale",
        type=float,
        default=0.05,
        help="Initial residual scale for the subject adapter.",
    )

    # Ablations.
    parser.add_argument(
        "--papa-no-bottleneck",
        action="store_true",
        help=(
            "Ablation: disable the learned respiration-state bottleneck and use "
            "the raw rich physiological state directly."
        ),
    )
    parser.add_argument(
        "--papa-no-adapter",
        action="store_true",
        help=(
            "Ablation: disable the subject adapter. PAPA TTA will be skipped "
            "because there is no subject-specific alignment module to update."
        ),
    )

    # PAPA TTA knobs.
    parser.add_argument(
        "--papa-tta",
        default="papa",
        choices=["none", "tent", "nrc", "cotta", "papa"],
        help="Classification TTA preset for the PAPA subject adapter.",
    )
    parser.add_argument("--papa-epochs", type=int, default=3)
    parser.add_argument("--papa-lr", type=float, default=5e-4)
    parser.add_argument("--papa-weight-decay", type=float, default=0.0)
    parser.add_argument("--papa-temperature", type=float, default=1.0)
    parser.add_argument("--papa-nrc-k", type=int, default=5)

    parser.add_argument("--lambda-papa-align", type=float, default=0.20)
    parser.add_argument("--lambda-papa-proto", type=float, default=0.20)
    parser.add_argument("--lambda-papa-entropy", type=float, default=0.05)
    parser.add_argument("--lambda-papa-diversity", type=float, default=0.05)
    parser.add_argument("--lambda-papa-nrc", type=float, default=0.10)
    parser.add_argument("--lambda-papa-smooth", type=float, default=0.05)
    parser.add_argument(
        "--lambda-resp-rr",
        type=float,
        default=0.05,
        help="Auxiliary RR-from-respiration-state bottleneck loss weight.",
    )
    parser.add_argument(
        "--lambda-resp-recon",
        type=float,
        default=0.05,
        help=(
            "Auxiliary bottleneck reconstruction loss weight. "
            "Reconstructs the raw physiological summary from the respiration-state bottleneck."
        ),
    )
    parser.add_argument("--use-device-br", action="store_true")

    args = parser.parse_args()

    core.run_loocv_experiment(
        args,
        post_eval_hooks=[frozen_embedding_hook, papa_hook],
        config_mutator=finalize_args,
    )


if __name__ == "__main__":
    main()

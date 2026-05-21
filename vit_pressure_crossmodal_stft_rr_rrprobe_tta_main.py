from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pressure_crossmodal_stft_rr_core import (
    build_base_parser,
    default_subjects,
    pooled_features,
    rr_targets_from_batch,
    run_loocv_experiment,
    unpack_batch,
)

IMU_ISSUES_MR = [17, 26, 30]
IMU_ISSUES_L = [17, 21, 26, 30]
SUBJECTS = default_subjects(IMU_ISSUES_MR, IMU_ISSUES_L)


class RRLinearProbe(nn.Module):
    def __init__(self, d_in: int, adapter_scale: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.adapter = nn.Linear(d_in, d_in)
        self.head = nn.Linear(d_in, 1)
        self.adapter_scale = float(adapter_scale)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def features(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.adapter_scale * self.adapter(self.norm(z))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        za = self.features(z)
        rr = self.head(za).squeeze(-1)
        return rr, za


@torch.no_grad()
def _collect_rr_probe_arrays(model: nn.Module, loader, device: str, max_batches: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    xs, ys = [], []
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        imu, pressure, _, br, _ = unpack_batch(batch, device)
        hidden = model.encode(imu)
        z = pooled_features(hidden)
        rr_true = rr_targets_from_batch(pressure, br)
        xs.append(z.detach().cpu().numpy())
        ys.append(rr_true.detach().cpu().numpy().reshape(-1))
    if not xs:
        raise RuntimeError("No batches available for RR probe feature extraction.")
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)


def _rr_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    err = y_pred.reshape(-1) - y_true.reshape(-1)
    corr = float(np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0, 1]) if y_true.size > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8 else float("nan")
    return {
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(err ** 2))),
        f"{prefix}_corr": corr,
        f"{prefix}_n": int(y_true.shape[0]),
    }


def _train_rr_probe_source(probe: RRLinearProbe, x_train: np.ndarray, y_train: np.ndarray, args, device: str) -> RRLinearProbe:
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(probe.parameters(), lr=float(args.rr_probe_lr), weight_decay=float(args.rr_probe_weight_decay))
    bs = int(args.rr_probe_batch_size)
    for _epoch in range(int(args.rr_probe_epochs)):
        perm = torch.randperm(x.size(0), device=device)
        for st in range(0, x.size(0), bs):
            idx = perm[st : st + bs]
            opt.zero_grad(set_to_none=True)
            pred, _ = probe(x[idx])
            loss = F.smooth_l1_loss(pred, y[idx])
            loss.backward()
            opt.step()
    return probe


@torch.no_grad()
def _predict_rr_probe(probe: RRLinearProbe, x: np.ndarray, device: str, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    probe.eval()
    preds, feats = [], []
    for st in range(0, x.shape[0], batch_size):
        xb = torch.tensor(x[st : st + batch_size], dtype=torch.float32, device=device)
        pred, za = probe(xb)
        preds.append(pred.detach().cpu().numpy())
        feats.append(za.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(feats, axis=0)


def _compute_rr_ssa_stats(probe: RRLinearProbe, x_source: np.ndarray, y_source: np.ndarray, args, device: str) -> Dict[str, torch.Tensor]:
    _, z_np = _predict_rr_probe(probe, x_source, device)
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_source, dtype=torch.float32, device=device)
    mu = z.mean(dim=0)
    zc = z - mu
    rank = max(1, min(int(args.rr_ssa_rank), zc.size(0) - 1, zc.size(1)))
    try:
        _, _, v = torch.pca_lowrank(zc, q=rank, center=False)
        basis = v[:, :rank]
    except Exception:
        _, _, vh = torch.linalg.svd(zc, full_matrices=False)
        basis = vh[:rank].T
    proj = zc @ basis
    src_mu = proj.mean(dim=0)
    src_sd = proj.std(dim=0, unbiased=False).clamp_min(1e-6)
    yc = y - y.mean()
    pc = proj - proj.mean(dim=0, keepdim=True)
    cov = (pc * yc.view(-1, 1)).mean(dim=0).abs()
    denom = pc.std(dim=0, unbiased=False).clamp_min(1e-6) * y.std(unbiased=False).clamp_min(1e-6)
    weights = (cov / denom).clamp_min(0.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return {"mu": mu.detach(), "basis": basis.detach(), "src_mu": src_mu.detach(), "src_sd": src_sd.detach(), "weights": weights.detach()}


def _rr_ssa_loss(probe: RRLinearProbe, x_target: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    _, zt = probe(x_target)
    basis = stats["basis"].to(x_target.device)
    mu = stats["mu"].to(x_target.device)
    src_mu = stats["src_mu"].to(x_target.device)
    src_sd = stats["src_sd"].to(x_target.device)
    weights = stats["weights"].to(x_target.device)
    proj = (zt - mu) @ basis
    tgt_mu = proj.mean(dim=0)
    tgt_sd = proj.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (weights * (tgt_mu - src_mu).pow(2)).mean() + (weights * (torch.log(tgt_sd) - torch.log(src_sd)).pow(2)).mean()


def _fewshot_indices(n: int, k: int, seed: int = 0) -> np.ndarray:
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n), size=min(k, n), replace=False))


def _adapt_rr_probe_tta(probe: RRLinearProbe, x_source: np.ndarray, y_source: np.ndarray, x_target: np.ndarray, y_target: np.ndarray, args, device: str) -> Tuple[RRLinearProbe, Dict[str, float]]:
    mode = str(args.rr_tta).lower()
    if mode == "none" or int(args.rr_tta_epochs) <= 0:
        return probe, {}
    source_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    stats = _compute_rr_ssa_stats(probe, x_source, y_source, args, device) if mode in {"ssa", "ssa_cmt"} else None
    few_idx = _fewshot_indices(x_target.shape[0], int(args.rr_cmt_fewshot), seed=int(args.seed))
    xtf = torch.tensor(x_target[few_idx], dtype=torch.float32, device=device) if few_idx.size else None
    ytf = torch.tensor(y_target[few_idx], dtype=torch.float32, device=device) if few_idx.size else None
    xt = torch.tensor(x_target, dtype=torch.float32, device=device)
    xs = torch.tensor(x_source, dtype=torch.float32, device=device)
    ys = torch.tensor(y_source, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(probe.parameters(), lr=float(args.rr_tta_lr), weight_decay=float(args.rr_tta_weight_decay))
    rows = []
    bs = int(args.rr_probe_batch_size)
    for ep in range(1, int(args.rr_tta_epochs) + 1):
        losses, ssa_losses, cmt_losses, src_losses, l2_losses = [], [], [], [], []
        for _ in range(max(1, math.ceil(x_target.shape[0] / max(1, bs)))):
            opt.zero_grad(set_to_none=True)
            total = xt.new_tensor(0.0)
            l_ssa = xt.new_tensor(0.0)
            l_cmt = xt.new_tensor(0.0)
            l_src = xt.new_tensor(0.0)
            l_l2 = xt.new_tensor(0.0)
            if mode in {"ssa", "ssa_cmt"} and stats is not None:
                idx_t = torch.randint(0, xt.size(0), (min(bs, xt.size(0)),), device=device)
                l_ssa = _rr_ssa_loss(probe, xt[idx_t], stats)
                total = total + float(args.lambda_rr_tta_ssa) * l_ssa
            if mode in {"cmt", "ssa_cmt"} and xtf is not None and xtf.numel() > 0:
                idx_f = torch.randint(0, xtf.size(0), (min(bs, xtf.size(0)),), device=device)
                pred_f, _ = probe(xtf[idx_f])
                l_cmt = F.smooth_l1_loss(pred_f, ytf[idx_f])
                total = total + float(args.lambda_rr_tta_cmt) * l_cmt
                if float(args.lambda_rr_tta_source) > 0:
                    idx_s = torch.randint(0, xs.size(0), (min(bs, xs.size(0)),), device=device)
                    pred_s, _ = probe(xs[idx_s])
                    l_src = F.smooth_l1_loss(pred_s, ys[idx_s])
                    total = total + float(args.lambda_rr_tta_source) * l_src
            if float(args.lambda_rr_tta_l2) > 0:
                for name, param in probe.state_dict().items():
                    if param.dtype.is_floating_point:
                        l_l2 = l_l2 + (param - source_state[name].to(param.device)).pow(2).mean()
                total = total + float(args.lambda_rr_tta_l2) * l_l2
            if not total.requires_grad:
                continue
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(total.detach().cpu()))
            ssa_losses.append(float(l_ssa.detach().cpu()))
            cmt_losses.append(float(l_cmt.detach().cpu()))
            src_losses.append(float(l_src.detach().cpu()))
            l2_losses.append(float(l_l2.detach().cpu()))
        rows.append({
            "rr_tta_epoch": ep,
            "rr_tta_loss": float(np.mean(losses)) if losses else float("nan"),
            "rr_tta_ssa": float(np.mean(ssa_losses)) if ssa_losses else float("nan"),
            "rr_tta_cmt": float(np.mean(cmt_losses)) if cmt_losses else float("nan"),
            "rr_tta_source": float(np.mean(src_losses)) if src_losses else float("nan"),
            "rr_tta_l2": float(np.mean(l2_losses)) if l2_losses else float("nan"),
        })
    return probe, (rows[-1] if rows else {})


def rr_probe_evaluate(model: nn.Module, train_loader, test_loader, subject: str, device: str, args, out_dir: Optional[Path] = None) -> Dict[str, float]:
    for p in model.parameters():
        p.requires_grad = False
    x_source, y_source = _collect_rr_probe_arrays(model, train_loader, device, max_batches=int(args.rr_probe_source_batches))
    x_target, y_target = _collect_rr_probe_arrays(model, test_loader, device, max_batches=0)
    probe = RRLinearProbe(x_source.shape[1], adapter_scale=float(args.rr_probe_adapter_scale)).to(device)
    probe = _train_rr_probe_source(probe, x_source, y_source, args, device)
    pred_pre, _ = _predict_rr_probe(probe, x_target, device)
    metrics = _rr_metrics(y_target, pred_pre, prefix="rr_probe_pre")
    if str(args.rr_tta).lower() != "none":
        probe, tta_last = _adapt_rr_probe_tta(probe, x_source, y_source, x_target, y_target, args, device)
        pred_post, _ = _predict_rr_probe(probe, x_target, device)
        metrics.update(_rr_metrics(y_target, pred_post, prefix="rr_probe_post"))
        metrics.update(tta_last)
    else:
        pred_post = pred_pre
    metrics.update({
        "rr_probe_n_source": int(x_source.shape[0]),
        "rr_probe_n_target": int(x_target.shape[0]),
        "rr_probe_n_features": int(x_source.shape[1]),
        "rr_tta_mode": str(args.rr_tta),
        "rr_cmt_fewshot": int(args.rr_cmt_fewshot),
    })
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"rr_true": y_target, "rr_pred_pre": pred_pre, "rr_pred_post": pred_post}).to_csv(out_dir / f"rr_probe_predictions_{subject}.csv", index=False)
        with open(out_dir / f"rr_probe_metrics_{subject}.json", "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def rr_probe_hook(model, sbj: str, train_loader, test_loader, device: str, args, sbj_dir: Path):
    if not bool(getattr(args, "eval_rr_probe", False)):
        return {}
    metrics = rr_probe_evaluate(model, train_loader, test_loader, sbj, device, args, out_dir=sbj_dir / "rr_probe")
    print(f"RR_PROBE {sbj}: {metrics}")
    return {"__summary_name__": "rr_probe_summary", "__summary_row__": {"subject": sbj, **metrics}}


def main() -> None:
    parser = build_base_parser(SUBJECTS, "smoke_vit_pressure_rich_linear_probe")
    parser.add_argument("--eval-rr-probe", action="store_true")
    parser.add_argument("--rr-probe-epochs", type=int, default=100)
    parser.add_argument("--rr-probe-lr", type=float, default=1e-3)
    parser.add_argument("--rr-probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--rr-probe-batch-size", type=int, default=256)
    parser.add_argument("--rr-probe-adapter-scale", type=float, default=0.1)
    parser.add_argument("--rr-probe-source-batches", type=int, default=0)
    parser.add_argument("--rr-tta", default="none", choices=["none", "ssa", "cmt", "ssa_cmt"])
    parser.add_argument("--rr-tta-epochs", type=int, default=20)
    parser.add_argument("--rr-tta-lr", type=float, default=1e-4)
    parser.add_argument("--rr-tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--rr-ssa-rank", type=int, default=32)
    parser.add_argument("--lambda-rr-tta-ssa", type=float, default=1.0)
    parser.add_argument("--rr-cmt-fewshot", type=int, default=32)
    parser.add_argument("--lambda-rr-tta-cmt", type=float, default=1.0)
    parser.add_argument("--lambda-rr-tta-source", type=float, default=0.1)
    parser.add_argument("--lambda-rr-tta-l2", type=float, default=1e-3)
    args = parser.parse_args()
    run_loocv_experiment(args, pre_eval_hooks=[rr_probe_hook])


if __name__ == "__main__":
    main()

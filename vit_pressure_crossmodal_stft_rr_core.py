from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils import _filter_subjects_with_data
from dataloader import loocv_generator
from config import BR_FS, SBJ_PROCESSED_DIR, M_DIR

TLX_CSV = "/projects/BLVMob/imu-rr-seated/Data/seated_tlx.csv"
DEFAULT_IMU_ISSUES_MR = [17, 26, 30]
DEFAULT_IMU_ISSUES_LEVELS = [17, 21, 26, 30]


def default_subjects(
    imu_issues_mr: Optional[List[int]] = None,
    imu_issues_levels: Optional[List[int]] = None,
) -> List[str]:
    issues_mr = DEFAULT_IMU_ISSUES_MR if imu_issues_mr is None else imu_issues_mr
    issues_levels = DEFAULT_IMU_ISSUES_LEVELS if imu_issues_levels is None else imu_issues_levels
    return _filter_subjects_with_data(
        ["S" + str(i).zfill(2) for i in range(12, 31)],
        excluded_subject_nums=issues_mr + issues_levels,
        data_dir=SBJ_PROCESSED_DIR,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyIMU2PressureViT(nn.Module):
    def __init__(
        self,
        input_channels: int = 6,
        d_model: int = 128,
        pred_len: int = 360,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.pred_len = pred_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.spec_proj: Optional[nn.Linear] = None
        self.pos = PositionalEncoding(d_model)
        enc_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dec_rnn = nn.GRU(d_model, d_model, batch_first=True)

        pressure_n_fft = min(self.n_fft, self.pred_len)
        self.pressure_freq_bins = pressure_n_fft // 2 + 1
        self.pressure_mag_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.pressure_freq_bins),
        )
        self.rr_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.pressure_proj: Optional[nn.Linear] = None
        self.pressure_pos = PositionalEncoding(d_model)
        pressure_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.pressure_encoder = TransformerEncoder(
            pressure_layer, num_layers=max(1, num_layers // 2)
        )

    def _imu_stft_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected IMU tensor (B,T,C), got {tuple(x.shape)}")
        if x.size(1) < x.size(2):
            x = x.transpose(1, 2)

        b, _, c = x.shape
        if c != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {c}")

        x = x.transpose(1, 2)
        window = torch.hann_window(self.win_length, device=x.device)
        mags = []
        for ch in range(c):
            spec = torch.stft(
                x[:, ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=True,
            )
            mags.append(spec.abs().unsqueeze(1))
        mag = torch.cat(mags, dim=1)
        b, c, f, tf = mag.shape
        feat = mag.reshape(b, c * f, tf).transpose(1, 2)

        if self.spec_proj is None:
            self.spec_proj = nn.Linear(c * f, self.d_model).to(x.device)
        return feat

    def _pressure_stft_tokens(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 3:
            y = y.squeeze(-1)
        if y.ndim != 2:
            raise ValueError(f"Expected pressure tensor (B,T) or (B,T,1), got {tuple(y.shape)}")

        t = y.size(1)
        n_fft = min(self.n_fft, t)
        win_length = min(self.win_length, t)
        hop_length = min(self.hop_length, max(1, win_length // 4))
        window = torch.hann_window(win_length, device=y.device)

        spec = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).abs()
        return torch.log1p(spec).transpose(1, 2)

    def encode_pressure(self, y: torch.Tensor, target_tokens: Optional[int] = None) -> torch.Tensor:
        tokens = self._pressure_stft_tokens(y)
        if self.pressure_proj is None:
            self.pressure_proj = nn.Linear(tokens.size(-1), self.d_model).to(tokens.device)
        h = self.pressure_proj(tokens)
        h = self.pressure_encoder(self.pressure_pos(h))

        if target_tokens is not None and h.size(1) != target_tokens:
            h = F.interpolate(
                h.transpose(1, 2),
                size=target_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return h

    def pressure_stft_target(self, y: torch.Tensor, target_tokens: Optional[int] = None) -> torch.Tensor:
        target = self._pressure_stft_tokens(y)
        if target_tokens is not None and target.size(1) != target_tokens:
            target = F.interpolate(
                target.transpose(1, 2),
                size=target_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        if target.size(-1) != self.pressure_freq_bins:
            target = F.interpolate(
                target.transpose(1, 2),
                size=self.pressure_freq_bins,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return target

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._imu_stft_tokens(x)
        h = self.spec_proj(tokens)
        h = self.encoder(self.pos(h))
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        h_dec, _ = self.dec_rnn(h)
        pressure_logmag = F.softplus(self.pressure_mag_head(h_dec))
        rr = self.rr_head(h.mean(dim=1)).squeeze(-1)
        return pressure_logmag, rr, h


def rr_targets_from_batch(pressure: torch.Tensor, br: Optional[torch.Tensor]) -> torch.Tensor:
    if br is not None and isinstance(br, torch.Tensor) and br.numel() >= pressure.size(0):
        return br.view(-1)[: pressure.size(0)].float()

    if pressure.ndim == 3:
        pressure = pressure.squeeze(-1)
    t = pressure.size(1)
    n_fft = min(256, t)
    win_length = min(256, t)
    hop_length = min(64, max(1, win_length // 4))
    window = torch.hann_window(win_length, device=pressure.device)
    spec = torch.stft(
        pressure,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    ).abs().mean(dim=-1)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / BR_FS).to(pressure.device)
    mask = (freqs >= 0.05) & (freqs <= 0.75)
    if not mask.any():
        return pressure.new_zeros(pressure.size(0))
    local_idx = spec[:, mask].argmax(dim=1)
    return freqs[mask][local_idx] * 60.0


def pressure_stft_recon_loss(
    model: TinyIMU2PressureViT,
    pred_logmag: torch.Tensor,
    pressure: torch.Tensor,
    rr_pred: Optional[torch.Tensor],
    br: Optional[torch.Tensor],
    lambda_stft: float,
    lambda_rr: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_logmag = model.pressure_stft_target(pressure, target_tokens=pred_logmag.size(1))
    l_stft = F.l1_loss(pred_logmag, target_logmag)

    l_rr = pred_logmag.new_tensor(0.0)
    if rr_pred is not None and lambda_rr > 0:
        rr_true = rr_targets_from_batch(pressure, br)
        l_rr = F.smooth_l1_loss(rr_pred.view(-1), rr_true.view(-1))

    loss = lambda_stft * l_stft + lambda_rr * l_rr
    return loss, {"stft": float(l_stft.detach().cpu()), "rr": float(l_rr.detach().cpu())}


def augment_imu(x: torch.Tensor, noise_std: float = 0.03, gain_std: float = 0.10, shift_max: int = 24) -> torch.Tensor:
    y = x
    gain = torch.randn(y.size(0), 1, y.size(2), device=y.device, dtype=y.dtype) * gain_std + 1.0
    y = y * gain
    if shift_max > 0:
        shifts = torch.randint(-shift_max, shift_max + 1, (y.size(0),), device=y.device)
        out = []
        for i, s in enumerate(shifts.tolist()):
            out.append(torch.roll(y[i], shifts=s, dims=0))
        y = torch.stack(out, dim=0)
    sd = y.std(dim=1, keepdim=True).clamp_min(1e-6)
    y = y + torch.randn_like(y) * sd * noise_std
    return y


def token_contrastive_loss(h1: torch.Tensor, h2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(h1.reshape(-1, h1.size(-1)), dim=-1)
    z2 = F.normalize(h2.reshape(-1, h2.size(-1)), dim=-1)
    logits = (z1 @ z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def unpack_batch(
    batch: Iterable[torch.Tensor],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if len(batch) == 4:
        imu, pressure, conds, br = batch
        tlx = None
    elif len(batch) == 5:
        imu, pressure, conds, br, tlx = batch
    else:
        raise ValueError(f"Expected 4 or 5 tensors from dataloader, got {len(batch)}")

    imu = imu.float().to(device)
    pressure = pressure.float().to(device)
    if pressure.ndim == 3:
        pressure = pressure.squeeze(-1)
    conds = conds.to(device)
    br = br.float().to(device)
    if tlx is not None:
        tlx = tlx.float().to(device)
    return imu, pressure, conds, br, tlx


@dataclass
class EpochMetrics:
    loss: float
    stft: float
    rr: float
    contrast: float


def contrast_weight_for_epoch(args, epoch: int) -> float:
    warmup_epochs = max(0, int(args.contrast_warmup_epochs))
    ramp_end_epoch = max(warmup_epochs + 1, int(args.contrast_ramp_end_epoch))
    min_weight = float(args.lambda_contrast_min)
    max_weight = float(args.lambda_contrast)

    if epoch <= warmup_epochs or max_weight <= 0:
        return 0.0
    if epoch >= ramp_end_epoch:
        return max_weight
    progress = (epoch - warmup_epochs - 1) / max(1, ramp_end_epoch - warmup_epochs - 1)
    return min_weight + progress * (max_weight - min_weight)


def train_one_epoch(
    model:nn.Module, loader, optimizer, device:str, args, 
    lambda_contrast:float,
) -> EpochMetrics:
    model.train()
    totals = {"loss": [], "stft": [], "rr": [], "contrast": []}

    for batch in loader:
        imu, pressure, _, br, _ = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if not args.use_device_br:
            br = None

        pred_logmag, rr_pred, _hidden = model(imu)

        loss, parts = pressure_stft_recon_loss(
            model,
            pred_logmag,
            pressure,
            rr_pred,
            br,
            lambda_stft=args.lambda_stft,
            lambda_rr=args.lambda_rr,
        )

        lc = pred_logmag.new_tensor(0.0)
        if lambda_contrast > 0:
            h_imu = model.encode(augment_imu(imu, shift_max=args.shift_max))
            h_pressure = model.encode_pressure(pressure, target_tokens=h_imu.size(1))
            lc = token_contrastive_loss(h_imu, h_pressure, temperature=args.temperature)
            loss = loss + lambda_contrast * lc

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        totals["loss"].append(float(loss.detach().cpu()))
        totals["stft"].append(parts["stft"])
        totals["rr"].append(parts["rr"])
        totals["contrast"].append(float(lc.detach().cpu()))

    return EpochMetrics(**{k: float(np.mean(v)) if v else float("nan") for k, v in totals.items()})


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, args, save_arrays: Optional[Path] = None) -> Dict[str, float]:
    model.eval()
    losses, parts_all = [], {"stft": [], "rr": []}
    pred_specs, true_specs = [], []
    rr_preds, rr_trues = [], []
 
    for batch in loader:
        imu, pressure, _, br, _ = unpack_batch(batch, device)

        if not args.use_device_br:
            br = None

        pred_logmag, rr_pred, _ = model(imu)
        loss, parts = pressure_stft_recon_loss(
            model,
            pred_logmag,
            pressure,
            rr_pred,
            br,
            lambda_stft=args.lambda_stft,
            lambda_rr=args.lambda_rr,
        )
        losses.append(float(loss.cpu()))
        for k in parts_all:
            parts_all[k].append(parts[k])

        true_logmag = model.pressure_stft_target(pressure, target_tokens=pred_logmag.size(1))
        pred_specs.append(pred_logmag.detach().cpu().numpy())
        true_specs.append(true_logmag.detach().cpu().numpy())

        rr_true = rr_targets_from_batch(pressure, br)
        rr_preds.append(rr_pred.detach().cpu().numpy().reshape(-1))
        rr_trues.append(rr_true.detach().cpu().numpy().reshape(-1))

    spec_true = np.concatenate(true_specs, axis=0)
    spec_pred = np.concatenate(pred_specs, axis=0)
    spec_err = spec_pred - spec_true
    rr_true = np.concatenate(rr_trues, axis=0)
    rr_pred = np.concatenate(rr_preds, axis=0)
    rr_err = rr_pred - rr_true

    spec_corr = float(np.corrcoef(spec_true.reshape(-1), spec_pred.reshape(-1))[0, 1]) if spec_true.size > 1 else float("nan")
    rr_corr = (
        float(np.corrcoef(rr_true.reshape(-1), rr_pred.reshape(-1))[0, 1])
        if rr_true.size > 1 and np.std(rr_true) > 1e-8 and np.std(rr_pred) > 1e-8
        else float("nan")
    )

    metrics = {
        "loss": float(np.mean(losses)),
        "stft": float(np.mean(parts_all["stft"])),
        "rr_loss": float(np.mean(parts_all["rr"])),
        "spec_mae": float(np.mean(np.abs(spec_err))),
        "spec_rmse": float(np.sqrt(np.mean(spec_err ** 2))),
        "spec_corr": spec_corr,
        "rr_mae": float(np.mean(np.abs(rr_err))),
        "rr_rmse": float(np.sqrt(np.mean(rr_err ** 2))),
        "rr_corr": rr_corr,
        "n_windows": int(spec_true.shape[0]),
    }

    if save_arrays is not None:
        save_arrays.mkdir(parents=True, exist_ok=True)
        np.save(save_arrays / "pressure_stft_true.npy", spec_true)
        np.save(save_arrays / "pressure_stft_pred.npy", spec_pred)
        np.save(save_arrays / "rr_true.npy", rr_true)
        np.save(save_arrays / "rr_pred.npy", rr_pred)

    return metrics


def pooled_features(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.mean(dim=1)


@torch.no_grad()
def collect_source_tta_stats(model: nn.Module, loader, device: str, args) -> Dict[str, torch.Tensor]:
    model.eval()
    zs, rrs = [], []
    max_batches = int(args.tta_source_batches)

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        imu, pressure, _, br, _ = unpack_batch(batch, device)
        _, _rr_pred, hidden = model(imu)
        z = pooled_features(hidden)

        if not args.use_device_br:
            br = None    

        rr_true = rr_targets_from_batch(pressure, br)
        zs.append(z.detach())
        rrs.append(rr_true.detach())

    if not zs:
        raise RuntimeError("Could not collect source TTA statistics: empty train loader.")

    z_all = torch.cat(zs, dim=0).float()
    rr = torch.cat(rrs, dim=0).float()
    mu = z_all.mean(dim=0)
    sd = z_all.std(dim=0, unbiased=False).clamp_min(1e-6)

    zc = z_all - mu
    rank = max(1, min(int(args.ssa_rank), zc.size(0) - 1, zc.size(1)))
    try:
        _, _, v = torch.pca_lowrank(zc, q=rank, center=False)
        basis = v[:, :rank]
    except Exception:
        _, _, vh = torch.linalg.svd(zc, full_matrices=False)
        basis = vh[:rank].T
    src_proj = zc @ basis
    src_proj_mu = src_proj.mean(dim=0)
    src_proj_sd = src_proj.std(dim=0, unbiased=False).clamp_min(1e-6)

    k = max(1, int(args.proto_k))
    order = torch.argsort(rr)
    chunks = torch.chunk(order, k)
    protos, centers = [], []
    for idx in chunks:
        if idx.numel() == 0:
            continue
        protos.append(z_all[idx].mean(dim=0))
        centers.append(rr[idx].mean())
    prototypes = torch.stack(protos, dim=0)
    rr_centers = torch.stack(centers, dim=0)

    return {
        "mu": mu.detach(),
        "sd": sd.detach(),
        "basis": basis.detach(),
        "src_proj_mu": src_proj_mu.detach(),
        "src_proj_sd": src_proj_sd.detach(),
        "prototypes": prototypes.detach(),
        "rr_centers": rr_centers.detach(),
    }


def rr_from_predicted_stft(pred_logmag: torch.Tensor, fs: float = BR_FS, min_hz: float = 0.05, max_hz: float = 0.75) -> torch.Tensor:
    spec = torch.expm1(pred_logmag).clamp_min(0.0).mean(dim=1)
    n_freq = spec.size(-1)
    n_fft = max(2, (n_freq - 1) * 2)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs).to(spec.device)
    freqs = freqs[:n_freq]
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not mask.any():
        return pred_logmag.new_zeros(pred_logmag.size(0))
    f = freqs[mask]
    s = spec[:, mask].clamp_min(1e-8)
    hz = (s * f.view(1, -1)).sum(dim=1) / s.sum(dim=1).clamp_min(1e-8)
    return hz * 60.0


def ssa_alignment_loss(z: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    basis = stats["basis"].to(z.device)
    src_mu = stats["src_proj_mu"].to(z.device)
    src_sd = stats["src_proj_sd"].to(z.device)
    zc = z - stats["mu"].to(z.device)
    proj = zc @ basis
    mu = proj.mean(dim=0)
    sd = proj.std(dim=0, unbiased=False).clamp_min(1e-6)
    return F.mse_loss(mu, src_mu) + F.mse_loss(torch.log(sd), torch.log(src_sd))


def augmentation_consistency_loss(model: nn.Module, imu: torch.Tensor, pred_logmag: torch.Tensor, rr_pred: torch.Tensor, z: torch.Tensor, args) -> torch.Tensor:
    imu_aug = augment_imu(imu, shift_max=args.shift_max)
    pred_aug, rr_aug, h_aug = model(imu_aug)
    z_aug = pooled_features(h_aug)
    spec_loss = F.smooth_l1_loss(pred_aug, pred_logmag.detach())
    rr_loss = F.smooth_l1_loss(rr_aug.view(-1), rr_pred.detach().view(-1))
    z_loss = F.mse_loss(F.normalize(z_aug, dim=-1), F.normalize(z.detach(), dim=-1))
    return spec_loss + rr_loss + z_loss


def rr_stft_consistency_loss(pred_logmag: torch.Tensor, rr_pred: torch.Tensor) -> torch.Tensor:
    rr_spec = rr_from_predicted_stft(pred_logmag)
    return F.smooth_l1_loss(rr_pred.view(-1), rr_spec.detach().view(-1))


def gated_temporal_smoothness_loss(z: torch.Tensor, rr_pred: torch.Tensor, pred_logmag: torch.Tensor, gate_scale: float = 1.0) -> torch.Tensor:
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    dz = (z[1:] - z[:-1]).pow(2).mean(dim=1).sqrt()
    gate = torch.exp(-dz.detach() / max(float(gate_scale), 1e-6)).clamp(0.0, 1.0)
    rr_step = F.smooth_l1_loss(rr_pred[1:], rr_pred[:-1], reduction="none")
    spec_step = (pred_logmag[1:] - pred_logmag[:-1]).abs().mean(dim=(1, 2))
    z_step = (F.normalize(z[1:], dim=-1) - F.normalize(z[:-1], dim=-1)).pow(2).mean(dim=1)
    return (gate * (rr_step + spec_step + z_step)).mean()


def prototype_alignment_loss(z: torch.Tensor, rr_pred: torch.Tensor, stats: Dict[str, torch.Tensor], temperature: float = 2.0) -> torch.Tensor:
    protos = stats["prototypes"].to(z.device)
    centers = stats["rr_centers"].to(z.device)
    if protos.numel() == 0:
        return z.new_tensor(0.0)
    dist_rr = (rr_pred.view(-1, 1) - centers.view(1, -1)).abs()
    weights = torch.softmax(-dist_rr / max(float(temperature), 1e-6), dim=1)
    target_proto = weights @ protos
    return F.mse_loss(F.normalize(z, dim=-1), F.normalize(target_proto.detach(), dim=-1))


def set_tta_trainable(model: nn.Module, mode: str = "norm_proj_rr") -> None:
    for p in model.parameters():
        p.requires_grad = False
    mode = str(mode).lower()
    for name, module in model.named_modules():
        allow = False
        if isinstance(module, nn.LayerNorm) and "norm" in mode:
            allow = True
        if "proj" in mode and any(k in name for k in ("spec_proj", "pressure_proj")):
            allow = True
        if "head" in mode and any(k in name for k in ("rr_head", "pressure_mag_head")):
            allow = True
        if "rr" in mode and "rr_head" in name:
            allow = True
        if allow:
            for p in module.parameters(recurse=False):
                p.requires_grad = True
    if "proj" in mode:
        for attr in ("spec_proj", "pressure_proj"):
            module = getattr(model, attr, None)
            if isinstance(module, nn.Module):
                for p in module.parameters():
                    p.requires_grad = True


def tta_loss_for_batch(
    model: nn.Module,
    imu: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    args,
    prev_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    pred_logmag, rr_pred, hidden = model(imu)
    z = pooled_features(hidden)

    l_ssa = ssa_alignment_loss(z, stats) if args.lambda_tta_ssa > 0 else z.new_tensor(0.0)
    l_cons = augmentation_consistency_loss(model, imu, pred_logmag, rr_pred, z, args) if args.lambda_tta_cons > 0 else z.new_tensor(0.0)
    l_rrspec = rr_stft_consistency_loss(pred_logmag, rr_pred) if args.lambda_tta_rrspec > 0 else z.new_tensor(0.0)
    l_proto = prototype_alignment_loss(z, rr_pred, stats, temperature=args.proto_temperature) if args.lambda_tta_proto > 0 else z.new_tensor(0.0)
    l_smooth = z.new_tensor(0.0)
    if args.lambda_tta_smooth > 0:
        l_smooth = gated_temporal_smoothness_loss(z, rr_pred, pred_logmag, gate_scale=args.smooth_gate_scale)
        if prev_state is not None and prev_state.get("z") is not None:
            z_cat = torch.cat([prev_state["z"].to(z.device), z], dim=0)
            rr_cat = torch.cat([prev_state["rr"].to(z.device), rr_pred], dim=0)
            spec_cat = torch.cat([prev_state["spec"].to(z.device), pred_logmag], dim=0)
            l_smooth = 0.5 * (l_smooth + gated_temporal_smoothness_loss(z_cat, rr_cat, spec_cat, gate_scale=args.smooth_gate_scale))

    loss = (
        args.lambda_tta_ssa * l_ssa
        + args.lambda_tta_cons * l_cons
        + args.lambda_tta_rrspec * l_rrspec
        + args.lambda_tta_smooth * l_smooth
        + args.lambda_tta_proto * l_proto
    )
    parts = {
        "tta_loss": float(loss.detach().cpu()),
        "ssa": float(l_ssa.detach().cpu()),
        "cons": float(l_cons.detach().cpu()),
        "rrspec": float(l_rrspec.detach().cpu()),
        "smooth": float(l_smooth.detach().cpu()),
        "proto": float(l_proto.detach().cpu()),
    }
    next_state = {"z": z[-1:].detach(), "rr": rr_pred[-1:].detach(), "spec": pred_logmag[-1:].detach()}
    return loss, parts, next_state


def run_subject_tta(model: nn.Module, source_loader, target_loader, device: str, args, out_dir: Optional[Path] = None) -> Dict[str, float]:
    if str(args.tta).lower() == "none" or int(args.tta_epochs) <= 0:
        return {}

    active_tta_weight = (
        abs(float(args.lambda_tta_ssa))
        + abs(float(args.lambda_tta_cons))
        + abs(float(args.lambda_tta_rrspec))
        + abs(float(args.lambda_tta_smooth))
        + abs(float(args.lambda_tta_proto))
    )
    if active_tta_weight <= 0.0:
        print("[TTA] All TTA lambda weights are zero; skipping TTA.")
        return {}

    stats = collect_source_tta_stats(model, source_loader, device, args)
    set_tta_trainable(model, args.tta_adapt)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = int(sum(p.numel() for p in params))
    print(f"[TTA] Trainable parameters: {n_trainable} | adapt={args.tta_adapt}")
    if not params:
        print("[TTA] No trainable parameters selected; skipping TTA.")
        return {}
    opt = torch.optim.AdamW(params, lr=args.tta_lr, weight_decay=args.tta_weight_decay)

    rows = []
    for epoch in range(1, int(args.tta_epochs) + 1):
        model.train()
        prev_state = None
        totals: Dict[str, List[float]] = {"tta_loss": [], "ssa": [], "cons": [], "rrspec": [], "smooth": [], "proto": []}
        for batch in target_loader:
            imu, _, _, _, _ = unpack_batch(batch, device)
            for _ in range(max(1, int(args.tta_steps_per_batch))):
                opt.zero_grad(set_to_none=True)
                loss, parts, next_state = tta_loss_for_batch(model, imu, stats, args, prev_state=prev_state)
                if not torch.isfinite(loss):
                    continue
                if not loss.requires_grad:
                    print("[TTA] Loss has no grad_fn; skipping this update. Check nonzero --lambda-tta-* weights and --tta-adapt.")
                    continue
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                prev_state = next_state
                for k in totals:
                    totals[k].append(parts[k])
        row = {f"tta_{k}": float(np.mean(v)) if v else float("nan") for k, v in totals.items()}
        row["tta_epoch"] = epoch
        rows.append(row)
        print(
            f"TTA epoch {epoch:03d} | loss {row['tta_tta_loss']:.4f} "
            f"ssa {row['tta_ssa']:.4f} cons {row['tta_cons']:.4f} "
            f"rrspec {row['tta_rrspec']:.4f} smooth {row['tta_smooth']:.4f} proto {row['tta_proto']:.4f}"
        )

    if out_dir is not None and rows:
        pd.DataFrame(rows).to_csv(out_dir / "tta_history.csv", index=False)
    return rows[-1] if rows else {}


def infer_n_channels(loader) -> int:
    batch = next(iter(loader))[0]
    if batch.ndim != 3:
        raise ValueError(f"Expected 3D IMU batch, got {tuple(batch.shape)}")
    return int(min(batch.shape[1], batch.shape[2]))


def _checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, args, epoch: int, best_val: float, hist: List[Dict[str, float]]) -> Dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "history": hist,
    }


def save_last_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, args, epoch: int, best_val: float, hist: List[Dict[str, float]]) -> None:
    torch.save(_checkpoint_payload(model, optimizer, args, epoch, best_val, hist), path)


def load_resume_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> Tuple[int, float, List[Dict[str, float]]]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    hist = list(ckpt.get("history", []))
    return epoch, best_val, hist


def read_summary_row(summary_path: Path, subject: str) -> Optional[Dict[str, float]]:
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if "subject" not in df.columns:
        return None
    rows = df[df["subject"] == subject]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def read_history_rows(history_path: Path) -> List[Dict[str, float]]:
    if not history_path.exists():
        return []
    return pd.read_csv(history_path).to_dict("records")


def subject_outputs_complete(sbj_dir: Path) -> bool:
    required = [
        sbj_dir / "best_model.pt",
        sbj_dir / "history.csv",
        sbj_dir / "pressure_stft_true.npy",
        sbj_dir / "pressure_stft_pred.npy",
        sbj_dir / "rr_true.npy",
        sbj_dir / "rr_pred.npy",
    ]
    return all(path.exists() for path in required)


@torch.no_grad()
def frozen_embedding_from_batch(model: nn.Module, imu: torch.Tensor, args) -> torch.Tensor:
    pred_logmag, rr_pred, hidden = model(imu)

    mode = str(args.embed_pooling).lower()
    parts: List[torch.Tensor] = []

    if mode in {"mean", "mean_std", "mean_std_max", "rich"}:
        parts.append(hidden.mean(dim=1))
    if mode in {"mean_std", "mean_std_max", "rich"}:
        parts.append(hidden.std(dim=1, unbiased=False))
    if mode in {"mean_std_max", "rich"}:
        parts.append(hidden.max(dim=1).values)
    if mode == "max":
        parts.append(hidden.max(dim=1).values)
    if mode == "cls_last":
        parts.append(hidden[:, -1, :])
    if mode == "rich":
        parts.append(rr_pred.view(-1, 1))
        spec_mean = pred_logmag.mean(dim=(1, 2), keepdim=False).view(-1, 1)
        spec_std = pred_logmag.std(dim=(1, 2), unbiased=False, keepdim=False).view(-1, 1)
        spec_max = pred_logmag.amax(dim=(1, 2), keepdim=False).view(-1, 1)
        parts.extend([spec_mean, spec_std, spec_max])
        if bool(args.embed_stft_profile):
            parts.append(pred_logmag.mean(dim=1))

    if not parts:
        raise ValueError(f"Unsupported --embed-pooling={args.embed_pooling!r}")
    return torch.cat(parts, dim=1)


@torch.no_grad()
def collect_frozen_embeddings(model: nn.Module, loader, device: str, args) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    tlxs: List[np.ndarray] = []
    saw_tlx = False
    for batch in loader:
        imu, _, cond, _, tlx = unpack_batch(batch, device)
        z = frozen_embedding_from_batch(model, imu, args)
        xs.append(z.detach().cpu().numpy())
        ys.append(cond.detach().cpu().numpy().astype(int).reshape(-1))
        if tlx is not None:
            saw_tlx = True
            tlxs.append(tlx.detach().cpu().numpy().astype(np.float32).reshape(-1))
    if not xs:
        raise RuntimeError("No batches available for frozen-embedding collection.")
    tlx_out = np.concatenate(tlxs, axis=0) if saw_tlx and tlxs else None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), tlx_out


def build_base_parser(default_subjects_list: List[str], default_out_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=default_subjects_list, help="Subjects for LOSO")
    parser.add_argument("--data-str", default="imu_filt", choices=["imu_filt", "imu_ica"])
    parser.add_argument("--data-dir", default=SBJ_PROCESSED_DIR, help="Directory containing processed subject .pkl files")
    parser.add_argument("--data-group", default="mr", choices=["mr", "levels", "mr_levels"], help="Processed split to load; use mr for M/R pretraining or levels for L0-L3.")
    parser.add_argument("--include-tlx", action="store_true", help="Ask the latest dataloader for TLX values in pretraining loaders. Training ignores these unless downstream TLX probing is enabled.")
    parser.add_argument("--tlx-csv-path", default=TLX_CSV, help="Optional path to seated_tlx.csv for dataloader TLX mapping.")
    parser.add_argument("--mdl-dir", default=None, help="Model parent directory. Defaults to project dir")
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lambda-time", type=float, default=0.0, help="Deprecated; waveform reconstruction is disabled in this STFT/RR variant.")
    parser.add_argument("--lambda-stft", type=float, default=1.0, help="Weight for pressure log-STFT magnitude reconstruction.")
    parser.add_argument("--lambda-rr", type=float, default=0.1, help="Weight for auxiliary RR regression from pooled IMU features.")
    parser.add_argument("--lambda-band", type=float, default=0.0, help="Deprecated; band-energy loss is disabled in this STFT/RR variant.")
    parser.add_argument("--lambda-contrast", type=float, default=0.05, help="Final weight for IMU-token <-> pressure-token contrastive loss after warmup/ramp")
    parser.add_argument("--lambda-contrast-min", type=float, default=0.0, help="Initial nonzero contrastive weight after warmup")
    parser.add_argument("--contrast-warmup-epochs", type=int, default=5, help="Number of initial epochs with contrastive loss disabled")
    parser.add_argument("--contrast-ramp-end-epoch", type=int, default=10, help="Epoch by which contrastive weight reaches --lambda-contrast")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--shift-max", type=int, default=24)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume each subject from last_model.pt in --out-dir when available")

    parser.add_argument("--tta", default="none", choices=["none", "physio"], help="Run physiology-aware TTA after loading the best checkpoint.")
    parser.add_argument("--tta-epochs", type=int, default=0)
    parser.add_argument("--tta-steps-per-batch", type=int, default=1)
    parser.add_argument("--tta-lr", type=float, default=1e-5)
    parser.add_argument("--tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--tta-adapt", default="norm_proj_rr", help="Small parameter set to adapt: e.g. norm, norm_proj, norm_proj_rr, norm_proj_head.")
    parser.add_argument("--tta-source-batches", type=int, default=0, help="Max source train batches for TTA stats/prototypes; 0 uses all.")
    parser.add_argument("--ssa-rank", type=int, default=32)
    parser.add_argument("--proto-k", type=int, default=6)
    parser.add_argument("--proto-temperature", type=float, default=2.0)
    parser.add_argument("--smooth-gate-scale", type=float, default=1.0)
    parser.add_argument("--lambda-tta-ssa", type=float, default=0.05)
    parser.add_argument("--lambda-tta-cons", type=float, default=0.10)
    parser.add_argument("--lambda-tta-rrspec", type=float, default=0.10)
    parser.add_argument("--lambda-tta-smooth", type=float, default=0.02)
    parser.add_argument("--lambda-tta-proto", type=float, default=0.05)
    parser.add_argument("--use-device-br", action="store_true")
    return parser


PreEvalHook = Callable[[nn.Module, str, Any, Any, str, Any, Path], Dict[str, float]]
PostEvalHook = Callable[[nn.Module, str, List[str], Any, Any, str, Any, Path], List[Dict[str, Any]]]


def run_loocv_experiment(
    args,
    pre_eval_hooks: Optional[List[PreEvalHook]] = None,
    post_eval_hooks: Optional[List[PostEvalHook]] = None,
    config_mutator: Optional[Callable[[Any], None]] = None,
) -> Dict[str, pd.DataFrame]:
    pre_eval_hooks = pre_eval_hooks or []
    post_eval_hooks = post_eval_hooks or []
    if config_mutator is not None:
        config_mutator(args)

    set_seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mdl_dir is None:
        args.mdl_dir = f"{M_DIR}/{args.data_str}/loocv"

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Subjects: {args.subjects}")
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Model dir: {args.mdl_dir}")
    print(f"Output dir: {out_dir}")
    if bool(getattr(args, "eval_frozen_tlx", False)) or bool(getattr(args, "include_tlx", False)):
        print(f"TLX CSV: {args.tlx_csv_path}")

    rows: List[Dict[str, float]] = []
    extra_rows_by_name: Dict[str, List[Dict[str, Any]]] = {}
    generator = loocv_generator(
        args.subjects,
        args.data_str,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        data_dir=args.data_dir,
        mdl_dir=args.mdl_dir,
        autoencoder=None,
        data_group=args.data_group,
        include_tlx=bool(args.include_tlx),
        tlx_csv_path=args.tlx_csv_path,
    )

    for sbj, train_loader, val_loader, test_loader in generator:
        print(f"\n=== Held-out subject {sbj} ===")
        sbj_dir = out_dir / sbj
        sbj_dir.mkdir(parents=True, exist_ok=True)
        best_path = sbj_dir / "best_model.pt"
        last_path = sbj_dir / "last_model.pt"

        n_channels = infer_n_channels(train_loader)
        pred_len = int(round(20 * BR_FS))
        model = TinyIMU2PressureViT(
            input_channels=n_channels,
            d_model=args.d_model,
            pred_len=pred_len,
            nhead=args.heads,
            num_layers=args.layers,
        ).to(device)

        with torch.no_grad():
            warm_batch = next(iter(train_loader))
            warm_imu, warm_pressure, _, _, _ = unpack_batch(warm_batch, device)
            warm_spec, _warm_rr, warm_h = model(warm_imu[:1])
            _ = model.pressure_stft_target(warm_pressure[:1], target_tokens=warm_spec.size(1))
            _ = model.encode_pressure(warm_pressure[:1], target_tokens=warm_h.size(1))

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        hist = []
        best_val = float("inf")
        start_epoch = 1

        if args.resume:
            if last_path.exists():
                last_epoch, best_val, hist = load_resume_checkpoint(last_path, model, opt, device)
                start_epoch = last_epoch + 1
                print(f"[RESUME] Loaded {last_path} at epoch {last_epoch}; target epochs={args.epochs}, best_val={best_val:.4f}")
            elif subject_outputs_complete(sbj_dir):
                hist = read_history_rows(sbj_dir / "history.csv")
                summary_row = read_summary_row(out_dir / "summary.csv", sbj)
                if summary_row is not None:
                    print(f"[RESUME] Complete outputs found for {sbj}; reusing summary row.")
                    rows.append(summary_row)
                    continue
                print(f"[RESUME] Complete outputs found for {sbj}; summary row missing, rerunning test only.")
                start_epoch = args.epochs + 1

        if args.resume and subject_outputs_complete(sbj_dir) and start_epoch > args.epochs:
            summary_row = read_summary_row(out_dir / "summary.csv", sbj)
            if summary_row is not None:
                print(f"[RESUME] {sbj} already complete through epoch {start_epoch - 1}; reusing summary row.")
                rows.append(summary_row)
                continue
            print(f"[RESUME] {sbj} training complete; summary row missing, rerunning test only.")

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_lambda_contrast = contrast_weight_for_epoch(args, epoch)
            tr = train_one_epoch(model, train_loader, opt, device, args, epoch_lambda_contrast)
            val = evaluate(model, val_loader, device, args)
            hist.append({
                "epoch": epoch,
                "lambda_contrast": epoch_lambda_contrast,
                **{f"train_{k}": v for k, v in asdict(tr).items()},
                **{f"val_{k}": v for k, v in val.items()},
            })
            print(
                f"epoch {epoch:03d} | lambda_contrast {epoch_lambda_contrast:.4f} | "
                f"train loss {tr.loss:.4f} stft {tr.stft:.4f} rr {tr.rr:.4f} con {tr.contrast:.4f} | "
                f"val loss {val['loss']:.4f} spec_mae {val['spec_mae']:.4f} "
                f"spec_corr {val['spec_corr']:.3f} rr_mae {val['rr_mae']:.3f} rr_corr {val['rr_corr']:.3f}"
            )
            if val["loss"] < best_val:
                best_val = val["loss"]
                torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "epoch": epoch, "val": val}, best_path)
            save_last_checkpoint(last_path, model, opt, args, epoch, best_val, hist)
            pd.DataFrame(hist).to_csv(sbj_dir / "history.csv", index=False)

        pd.DataFrame(hist).to_csv(sbj_dir / "history.csv", index=False)

        if not best_path.exists() and last_path.exists():
            print(f"[CKPT] No best_model.pt found for {sbj}; using last checkpoint for test.")
            ckpt = torch.load(last_path, map_location=device)
            torch.save(
                {
                    "model_state_dict": ckpt["model_state_dict"],
                    "args": ckpt.get("args", vars(args)),
                    "epoch": ckpt.get("epoch", args.epochs),
                    "val": {"loss": ckpt.get("best_val", float('nan'))},
                },
                best_path,
            )

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_pre = evaluate(model, test_loader, device, args, save_arrays=sbj_dir / "pre_tta")
        print(f"TEST_PRE_TTA {sbj}: {test_pre}")

        for hook in pre_eval_hooks:
            hook_metrics = hook(model, sbj, train_loader, test_loader, device, args, sbj_dir)
            if hook_metrics:
                name = hook_metrics.pop("__summary_name__", None)
                row = hook_metrics.pop("__summary_row__", None)
                if name is not None and row is not None:
                    extra_rows_by_name.setdefault(name, []).append(row)

        tta_last = {}
        test = test_pre
        if str(args.tta).lower() != "none" and int(args.tta_epochs) > 0:
            tta_last = run_subject_tta(model, train_loader, test_loader, device, args, out_dir=sbj_dir)
            test = evaluate(model, test_loader, device, args, save_arrays=sbj_dir / "post_tta")
            print(f"TEST_POST_TTA {sbj}: {test}")

        _ = evaluate(model, test_loader, device, args, save_arrays=sbj_dir)
        row = {"subject": sbj, **test}
        if str(args.tta).lower() != "none" and int(args.tta_epochs) > 0:
            row.update({f"pre_{k}": v for k, v in test_pre.items()})
            row.update(tta_last)
        rows.append(row)

        for hook in post_eval_hooks:
            for item in hook(model, sbj, list(args.subjects), train_loader, test_loader, device, args, sbj_dir):
                name = item.pop("__summary_name__")
                extra_rows_by_name.setdefault(name, []).append(item)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== Summary ===")
    print(df)
    print("\nMean metrics:")
    print(df.drop(columns=["subject"]).mean(numeric_only=True))

    outputs: Dict[str, pd.DataFrame] = {"summary": df}
    for name, row_list in extra_rows_by_name.items():
        extra_df = pd.DataFrame(row_list)
        extra_df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(extra_df)
        drop_cols = [c for c in ("subject", "tag") if c in extra_df.columns]
        print(f"\n{name} mean metrics:")
        print(extra_df.drop(columns=drop_cols).mean(numeric_only=True))
        outputs[name] = extra_df
    return outputs

from os.path import join, exists
import os
import csv
import numpy as np
import pickle
import ipdb
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import random
from typing import Optional

from sklearn.model_selection import train_test_split
from digitalsignalprocessing import reject_artefact
from augmentations import dom_shuffle, scaling, spectral_augment
from config import IMU_FS

# DataLoader performance defaults (overridable via env vars)
_DEFAULT_WORKERS = int(os.environ.get("IMU_DATALOADER_WORKERS", "4"))
_DEFAULT_PREFETCH = int(os.environ.get("IMU_DATALOADER_PREFETCH", "2"))
_DEFAULT_PIN = os.environ.get("IMU_DATALOADER_PIN_MEMORY", "1") != "0"
DEFAULT_TLX_CSV_PATH = "/projects/BLVMob/imu-rr-seated/Data/seated_tlx.csv"

def _loader_kwargs():
    num_workers = max(0, _DEFAULT_WORKERS)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(_DEFAULT_PIN),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, _DEFAULT_PREFETCH)
    return kwargs


def normalize_data_group(data_group: Optional[str]) -> Optional[str]:
    if data_group is None:
        return None
    g = str(data_group).strip().lower()
    if g in {"", "all", "combined", "root", "none"}:
        return None
    if g == "mr":
        return "mr"
    if g in {"level", "levels"}:
        return "levels"
    if g in {"mr_levels", "levels_mr", "sixclass", "all_conditions"}:
        return "mr_levels"
    raise ValueError(f"Unsupported data_group '{data_group}'. Expected one of: mr, levels, mr_levels.")


def _data_group_candidate_dirs(data_dir: str, data_group: Optional[str]) -> list[str]:
    canonical = normalize_data_group(data_group)
    if canonical is None:
        return [data_dir]
    if canonical == "levels":
        return [join(data_dir, "levels"), join(data_dir, "level"), data_dir]
    if canonical == "mr_levels":
        return [join(data_dir, "mr_levels"), join(data_dir, "combined"), data_dir]
    return [join(data_dir, canonical), data_dir]


def _resolve_subject_path(sbj: str, data_dir: str, data_group: Optional[str] = None) -> str:
    for base in _data_group_candidate_dirs(data_dir, data_group):
        sbj_fname = join(base, sbj + '.pkl')
        if exists(sbj_fname):
            return sbj_fname
    candidates = [join(base, sbj + ".pkl") for base in _data_group_candidate_dirs(data_dir, data_group)]
    raise FileNotFoundError(
        f"Could not find subject pickle for {sbj}. Tried: {candidates}"
    )
# ---------------------------------------------------------------------
# Example LOOCV generator (simple version)
# If you already have this defined, you can keep your original one.
# ---------------------------------------------------------------------
def load_data(sbj: str, data_dir='/scratch/data', data_group: Optional[str] = None):
    sbj_fname = _resolve_subject_path(sbj, data_dir, data_group=data_group)
    with open(sbj_fname, 'rb') as f:
        sbj_processed = pickle.load(f)
    # Keep the subject id with the loaded dict so downstream targets that are
    # stored per subject (for example seated_tlx.csv) can be mapped per window.
    if isinstance(sbj_processed, dict):
        sbj_processed = dict(sbj_processed)
        sbj_processed.setdefault('subject', sbj)
        sbj_processed.setdefault('subject_id', sbj)
    return sbj_processed


# ---------------------------------------------------------------------
# Optional NASA-TLX mapping
# ---------------------------------------------------------------------
def _canonical_subject_id(subject) -> str:
    """Normalize subject ids like 12, '12', 'S12', 'S012' to 'S12'."""
    if subject is None:
        return ""
    txt = str(subject).strip()
    if not txt:
        return ""
    if txt.lower().startswith("pilot"):
        return txt
    if txt.upper().startswith("S"):
        digits = "".join(ch for ch in txt[1:] if ch.isdigit())
    else:
        digits = "".join(ch for ch in txt if ch.isdigit())
    if digits:
        return f"S{int(digits):02d}"
    return txt


def _resolve_tlx_csv_path(data_dir: Optional[str] = None, tlx_csv_path: Optional[str] = None) -> Optional[str]:
    """Find seated_tlx.csv, preferring an explicit path then the data root."""
    candidates = []
    if tlx_csv_path:
        candidates.append(os.path.expanduser(str(tlx_csv_path)))
    if data_dir:
        d = os.path.expanduser(str(data_dir))
        candidates.extend([
            join(d, "seated_tlx.csv"),
            join(d, "tlx.csv"),
            join(os.path.dirname(d), "seated_tlx.csv"),
        ])
    candidates.append(DEFAULT_TLX_CSV_PATH)
    candidates.append("seated_tlx.csv")
    for path in candidates:
        if path and exists(path):
            return path
    return None


def _clean_csv_row(row: dict) -> dict:
    """Strip whitespace/BOM from CSV column names and string values."""
    out = {}
    for k, v in row.items():
        kk = str(k).strip().lstrip("\ufeff")
        vv = v.strip() if isinstance(v, str) else v
        out[kk] = vv
    return out


def _get_row_value_case_insensitive(row: dict, *names):
    lower = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        key = str(name).strip().lower()
        if key in lower:
            return lower[key]
    return None


def load_tlx_table(data_dir: Optional[str] = None, tlx_csv_path: Optional[str] = None) -> dict:
    """
    Load seated NASA-TLX scores from CSV.

    Expected columns:
      Subject,L0,L1,L2,L3

    This is tolerant to case/whitespace in column names and defaults to:
      /projects/BLVMob/imu-rr-seated/Data/seated_tlx.csv

    Returns:
      {"S12": {"L0": 82.0, "L1": 62.0, ...}, ...}
    """
    path = _resolve_tlx_csv_path(data_dir=data_dir, tlx_csv_path=tlx_csv_path)
    if path is None:
        raise FileNotFoundError(
            "Could not find seated_tlx.csv. Put it in the processed data root "
            "or pass tlx_csv_path to make_dataset/build_loocv_loaders. "
            f"Default tried: {DEFAULT_TLX_CSV_PATH}"
        )
    table = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = _clean_csv_row(raw_row)
            sid = _canonical_subject_id(
                _get_row_value_case_insensitive(row, 'Subject', 'subject', 'subject_id', 'sid')
            )
            if not sid:
                continue
            table[sid] = {}
            for cond in ('L0', 'L1', 'L2', 'L3'):
                val = _get_row_value_case_insensitive(row, cond, cond.lower(), cond.replace('L', 'level'))
                try:
                    table[sid][cond] = float(str(val).strip())
                except Exception:
                    table[sid][cond] = np.nan
    return table

def _subject_from_dict(data: dict) -> str:
    for key in ('subject', 'subject_id', 'sbj', 'sid'):
        if key in data:
            return _canonical_subject_id(data[key])
    return ""


def _canonical_tlx_condition(cond) -> str:
    """Normalize condition labels to L0-L3 when possible."""
    if isinstance(cond, bytes):
        cond = cond.decode(errors="ignore")
    txt = str(cond).strip().upper()
    if txt in {"L0", "L1", "L2", "L3"}:
        return txt
    if txt in {"0", "1", "2", "3"}:
        return f"L{txt}"
    try:
        f = float(txt)
        if f.is_integer() and int(f) in (0, 1, 2, 3):
            return f"L{int(f)}"
    except Exception:
        pass
    return txt


def tlx_for_subject_conditions(subject, conds, tlx_table: dict) -> np.ndarray:
    """
    Map each window condition to that subject's TLX score.

    M/R do not have TLX task-load scores in seated_tlx.csv, so they are mapped
    to NaN. Downstream TLX losses/regressors should mask non-finite values.
    """
    sid = _canonical_subject_id(subject)
    subject_scores = tlx_table.get(sid, {})
    out = []
    for c in np.asarray(conds):
        key = _canonical_tlx_condition(c)
        out.append(float(subject_scores.get(key, np.nan)))
    return np.asarray(out, dtype=np.float32)

def _load_label_encoder(data_dir: Optional[str] = None, data_group: Optional[str] = None):
    if data_dir is None:
        raise ValueError("label_encoder_dir/data_dir must not be None")

    candidates = []
    for base in _data_group_candidate_dirs(data_dir, data_group):
        candidates.append(join(base, 'label_encoder.pkl'))

    canonical = normalize_data_group(data_group)
    if canonical == "levels":
        candidates.extend([
            join(data_dir, "levels", "label_encoder.pkl"),
            join(data_dir, "level", "label_encoder.pkl"),
            join(data_dir, "label_encoder_levels.pkl"),
        ])
    elif canonical == "mr":
        candidates.extend([
            join(data_dir, "mr", "label_encoder.pkl"),
            join(data_dir, "label_encoder_mr.pkl"),
        ])
    elif canonical == "mr_levels":
        candidates.extend([
            join(data_dir, "mr_levels", "label_encoder.pkl"),
            join(data_dir, "combined", "label_encoder.pkl"),
            join(data_dir, "label_encoder_mr_levels.pkl"),
            join(data_dir, "label_encoder.pkl"),
        ])
    else:
        candidates.append(join(data_dir, "label_encoder.pkl"))

    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]
    for path in candidates:
        if exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError(
        "Could not find label_encoder.pkl. Tried:\n  " + "\n  ".join(candidates)
    )


def load_normwear_embedding_subject(
    sbj: str,
    embed_root: str,
    embedding_kind: str = "pooled_fp32",
):
    subject_dir = join(embed_root, sbj)
    meta_path = join(subject_dir, "meta.json")
    if not exists(meta_path):
        raise FileNotFoundError(f"Missing NormWear meta for {sbj}: {meta_path}")

    import json
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    source_pkl = meta.get("source_pkl", None)
    if not source_pkl or not exists(source_pkl):
        raise FileNotFoundError(f"Missing source_pkl for {sbj} in {meta_path}")

    with open(source_pkl, 'rb') as f:
        raw = pickle.load(f)

    embed_path = join(subject_dir, f"{embedding_kind}.npy")
    if not exists(embed_path):
        raise FileNotFoundError(f"Missing NormWear embedding file for {sbj}: {embed_path}")

    out = dict(raw)
    out[embedding_kind] = np.load(embed_path, mmap_mode='r')

    cond_path = join(subject_dir, "conds.npy")
    br_path = join(subject_dir, "br.npy")
    if exists(cond_path):
        out["conds"] = np.load(cond_path, allow_pickle=True)
    if exists(br_path):
        out["br"] = np.load(br_path, allow_pickle=False)
    return out


def make_dataset(
    data_list,
    data_str,
    reject=False,
    label_encoder=None,
    label_encoder_dir: Optional[str] = None,
    data_group: Optional[str] = None,
    include_tlx: bool = False,
    include_ecg: bool = False,
    include_marker: bool = False,
    tlx_csv_path: Optional[str] = None,
    tlx_table: Optional[dict] = None,
):
    if len(data_list) == 0:
        raise ValueError(
            f"make_dataset received no subject arrays for data_str={data_str}. "
            "This usually means the LOSO train cohort is empty."
        )
    le = label_encoder or _load_label_encoder(
        label_encoder_dir, data_group=data_group)

    x = np.concatenate(
        [data[data_str] for data in data_list], axis=0
    )
    pss = np.concatenate(
        [data['pss_filt'] for data in data_list], axis=0
    )
    ecg = None
    if include_ecg:
        ecg_parts = [data.get('ecg_filt') for data in data_list]
        if any(part is None for part in ecg_parts):
            raise ValueError(
                "include_ecg=True but at least one subject is missing 'ecg_filt'. "
                "Re-run preprocessing for those subjects or disable include_ecg."
            )
        ecg = np.concatenate(ecg_parts, axis=0)
    marker = None
    if include_marker:
        marker_parts = [data.get('marker_filt') for data in data_list]
        if any(part is None for part in marker_parts):
            raise ValueError(
                "include_marker=True but at least one subject is missing 'marker_filt'. "
                "Re-run preprocessing for those subjects with marker data or disable include_marker."
            )
        marker = np.concatenate(marker_parts, axis=0)
    br = np.concatenate(
        [data['br'] for data in data_list], axis=0
    )
    cond_raw = np.concatenate(
        [data['conds'] for data in data_list], axis=0
    )

    tlx = None
    if include_tlx:
        table = tlx_table or load_tlx_table(
            data_dir=label_encoder_dir, tlx_csv_path=tlx_csv_path
        )
        tlx_parts = []
        for data in data_list:
            subject = _subject_from_dict(data)
            if not subject:
                raise ValueError(
                    "Cannot map TLX because subject id is missing from a loaded "
                    "subject dict. Use load_data(), which now adds subject ids, "
                    "or add a 'subject' key to each dict."
                )
            tlx_parts.append(tlx_for_subject_conditions(subject, data['conds'], table))
        tlx = np.concatenate(tlx_parts, axis=0)

    cond = le.transform(cond_raw)


    if reject:
        idxs = [
            i for i, data in enumerate(pss) if not reject_artefact(data)
        ]
        x = x[idxs]
        pss = pss[idxs]
        if ecg is not None:
            ecg = ecg[idxs]
        if marker is not None:
            marker = marker[idxs]
        br = br[idxs]
        cond = cond[idxs]
        if tlx is not None:
            tlx = tlx[idxs]

    out = [x, pss, br, cond]
    if include_tlx:
        out.append(tlx)
    if include_ecg:
        out.append(ecg)
    if include_marker:
        out.append(marker)
    return tuple(out)

# Load data to dataloader
class LoadDataset(Dataset):
    def __init__(self, x, y=None, cond=None, br=None, tlx=None, ecg=None,
                 marker=None, aug_ratio=0.3, preserve_layout: bool = False):
        self.len = len(x)
        self.preserve_layout = bool(preserve_layout)
        if isinstance(x, np.ndarray):
            self.x = torch.from_numpy(x)
        else:
            self.x = x

        # make sure the Channels in second dim
        if (not self.preserve_layout) and self.x.shape.index(min(self.x.shape)) != 2:
            self.x = self.x.permute(0, 2, 1)

        if y is not None and isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).float()
        elif y is not None and not isinstance(y, np.ndarray):
            self.y = y.float()
        else:
            self.y = y

        if cond is not None and isinstance(cond, np.ndarray):
            self.cond = torch.from_numpy(cond).float()
        elif cond is not None and not isinstance(cond, np.ndarray):
            self.cond = cond.float()
        else:
            self.cond = cond

        if br is not None and isinstance(br, np.ndarray):
            self.br = torch.from_numpy(br).float()
        elif br is not None and not isinstance(br, np.ndarray):
            self.br = br.float()
        else:
            self.br = br

        if tlx is not None and isinstance(tlx, np.ndarray):
            self.tlx = torch.from_numpy(tlx).float()
        elif tlx is not None and not isinstance(tlx, np.ndarray):
            self.tlx = tlx.float()
        else:
            self.tlx = tlx

        if ecg is not None and isinstance(ecg, np.ndarray):
            self.ecg = torch.from_numpy(ecg).float()
        elif ecg is not None and not isinstance(ecg, np.ndarray):
            self.ecg = ecg.float()
        else:
            self.ecg = ecg

        if marker is not None and isinstance(marker, np.ndarray):
            self.marker = torch.from_numpy(marker).float()
        elif marker is not None and not isinstance(marker, np.ndarray):
            self.marker = marker.float()
        else:
            self.marker = marker

        if aug_ratio > 0:
            if self.preserve_layout:
                raise ValueError("Augmentation is not supported when preserve_layout=True.")
            # Choose ratio random percentage
            aug_idxs = random.sample(range(len(x)), int(aug_ratio*len(x)))
            x_aug, y_aug = self.x[aug_idxs], self.y[aug_idxs]

            if cond is not None:
                c_aug = self.cond[aug_idxs]
                self.cond = torch.cat((self.cond, c_aug), dim=0)
            if br is not None:
                br_aug = self.br[aug_idxs]
                self.br = torch.cat((self.br, br_aug), dim=0)
            if tlx is not None:
                tlx_aug = self.tlx[aug_idxs]
                self.tlx = torch.cat((self.tlx, tlx_aug), dim=0)
            if ecg is not None:
                ecg_aug = self.ecg[aug_idxs]
                self.ecg = torch.cat((self.ecg, ecg_aug), dim=0)
            if marker is not None:
                marker_aug = self.marker[aug_idxs]
                self.marker = torch.cat((self.marker, marker_aug), dim=0)

            # channels first
            x_aug_tmp = x_aug.permute(0, 2, 1)
            # x_aug = dom_shuffle(x_aug, rate=3, dim=-1)
            x_aug = torch.from_numpy(
                scaling(x_aug_tmp, sigma=1.1, device=self.x.device))
            x_aug = spectral_augment(
                x_aug,
                fs=IMU_FS,           # 120 Hz
                max_phase=np.pi/2,   # phase jitter up to ±90°
                max_shift_hz=0.5,    # small ±0.5 Hz frequency shift
                dim=-1,
                p_phase=0.5,
                p_shift=0.5,
            )
            x_aug = x_aug.permute(0, 2, 1)
            if aug_ratio == 1:
                self.x = x_aug
                self.y = y_aug
            else:
                self.x = torch.cat((self.x, x_aug), dim=0)
                self.y = torch.cat((self.y, y_aug), dim=0)
            self.len = len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            item = {
                'past_values': self.x[index].float(),
                'past_observed_mask': torch.ones_like(self.x[index]).float(),
            }
            return item
        out = [self.x[index], self.y[index]]
        if self.cond is not None:
            out.append(self.cond[index])
        if self.br is not None:
            out.append(self.br[index])
        if self.tlx is not None:
            out.append(self.tlx[index])
        if self.ecg is not None:
            out.append(self.ecg[index])
        if self.marker is not None:
            out.append(self.marker[index])
        return tuple(out)

    def __len__(self):
        return self.len


def build_loocv_loaders(
    sbj,
    subjects,
    data_str,
    val_split=0.25,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    data_dir='/scratch/raqchia/',
    mdl_dir='/data/raqchia/',
    autoencoder=None,
    subject_loader=load_data,
    dataset_builder=make_dataset,
    train_aug_ratio=0.2,
    preserve_layout: bool = False,
    label_encoder_dir: Optional[str] = None,
    data_group: Optional[str] = None,
    include_tlx: bool = False,
    include_ecg: bool = False,
    include_marker: bool = False,
    tlx_csv_path: Optional[str] = None,
):
    def _call_subject_loader(loader_fn, subject_name: str):
        try:
            return loader_fn(subject_name, data_dir=data_dir, data_group=data_group)
        except TypeError:
            return loader_fn(subject_name, data_dir=data_dir)

    train_list = [_call_subject_loader(subject_loader, sbj_str) for sbj_str in
                  subjects if sbj_str != sbj]
    test_list = [_call_subject_loader(subject_loader, sbj)]

    train_out = dataset_builder(
        train_list, data_str,
        label_encoder_dir=(label_encoder_dir or data_dir),
        data_group=data_group,
        include_tlx=include_tlx,
        include_ecg=include_ecg,
        include_marker=include_marker,
        tlx_csv_path=tlx_csv_path,
    )
    test_out = dataset_builder(
        test_list, data_str,
        label_encoder_dir=(label_encoder_dir or data_dir),
        data_group=data_group,
        include_tlx=include_tlx,
        include_ecg=include_ecg,
        include_marker=include_marker,
        tlx_csv_path=tlx_csv_path,
    )
    x_train, y_train, br_train, cond_train = train_out[:4]
    x_test,  y_test,  br_test,  cond_test  = test_out[:4]
    train_extra = list(train_out[4:])
    test_extra = list(test_out[4:])
    tlx_train, tlx_test = None, None
    ecg_train, ecg_test = None, None
    marker_train, marker_test = None, None
    if include_tlx:
        tlx_train, tlx_test = train_extra.pop(0), test_extra.pop(0)
    if include_ecg:
        ecg_train, ecg_test = train_extra.pop(0), test_extra.pop(0)
    if include_marker:
        marker_train, marker_test = train_extra.pop(0), test_extra.pop(0)

    if autoencoder is not None:
        prefix = f'{sbj}_{data_str}_'
        ae_ckpt = join(mdl_dir, sbj, 'autoencode', prefix+'ckp_last.pt')
        assert exists(ae_ckpt), f"Error: No autencoder for {sbj}"
        ae_state_dict = torch.load(ae_ckpt, weights_only=True)
        autoencoder.load_state_dict(ae_state_dict)
        autoencoder = autoencoder.to('cpu')
        x_test = autoencoder(
            torch.Tensor(x_test).permute(0, 2, 1)
        ).permute(0, 2, 1).detach().numpy()

    train_idxs, val_idxs = train_test_split(
        np.arange(len(x_train)), test_size=val_split
    )
    x_val = x_train[val_idxs]
    y_val = y_train[val_idxs]
    cond_val = cond_train[val_idxs]
    br_val = br_train[val_idxs]
    if include_tlx:
        tlx_val = tlx_train[val_idxs]
    if include_ecg:
        ecg_val = ecg_train[val_idxs]
    if include_marker:
        marker_val = marker_train[val_idxs]

    x_train = x_train[train_idxs]
    y_train = y_train[train_idxs]
    cond_train = cond_train[train_idxs]
    br_train = br_train[train_idxs]
    if include_tlx:
        tlx_train = tlx_train[train_idxs]
    if include_ecg:
        ecg_train = ecg_train[train_idxs]
    if include_marker:
        marker_train = marker_train[train_idxs]

    def _dataset(x, y, cond, br, tlx=None, ecg=None, marker=None, aug_ratio=0.0):
        return LoadDataset(
            x, y, cond=cond, br=br, tlx=tlx, ecg=ecg, marker=marker,
            aug_ratio=aug_ratio, preserve_layout=preserve_layout,
        )

    train_dataloader = DataLoader(
        _dataset(
            x_train, y_train, cond_train, br_train,
            tlx=tlx_train, ecg=ecg_train, marker=marker_train,
            aug_ratio=train_aug_ratio,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **_loader_kwargs(),
    )
    val_dataloader = DataLoader(
        _dataset(
            x_val, y_val, cond_val, br_val,
            tlx=tlx_val if include_tlx else None,
            ecg=ecg_val if include_ecg else None,
            marker=marker_val if include_marker else None,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **_loader_kwargs(),
    )
    test_dataloader = DataLoader(
        _dataset(
            x_test, y_test, cond_test, br_test,
            tlx=tlx_test, ecg=ecg_test, marker=marker_test,
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        **_loader_kwargs(),
    )
    return train_dataloader, val_dataloader, test_dataloader


def loocv_generator(
    subjects,
    data_str,
    val_split=0.25,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    data_dir='/scratch/raqchia/',
    mdl_dir='/data/raqchia/',
    autoencoder=None,
    data_group: Optional[str] = None,
    include_tlx: bool = False,
    include_ecg: bool = False,
    include_marker: bool = False,
    tlx_csv_path: Optional[str] = None,
):
    for sbj in subjects:
        train_dataloader, val_dataloader, test_dataloader = build_loocv_loaders(
            sbj,
            subjects,
            data_str,
            val_split=val_split,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            data_dir=data_dir,
            mdl_dir=mdl_dir,
            autoencoder=autoencoder,
            data_group=data_group,
            include_tlx=include_tlx,
            include_ecg=include_ecg,
            include_marker=include_marker,
            tlx_csv_path=tlx_csv_path,
        )

        yield sbj, train_dataloader, val_dataloader, test_dataloader

def make_fewshot_loader_from_test(
    test_dataloader,
    k: int,
    seed: int = 0,
    stratify_by_cond: bool = False,
):
    """
    Build a K-shot loader from the *existing* test_dataloader's dataset.

    Returns:
        fewshot_loader, remaining_loader

    Notes:
      - Uses Subset so it doesn't depend on dataset internals.
      - If stratify_by_cond=True, will try to sample evenly per condition label
        if the dataset exposes labels in __getitem__ as (imu, chest, cond, br).
    """
    ds = test_dataloader.dataset
    n = len(ds)
    if n == 0:
        return None, test_dataloader

    k = min(k, n)
    g = torch.Generator().manual_seed(seed)

    if not stratify_by_cond:
        idx = torch.randperm(n, generator=g)[:k].tolist()
    else:
        # Try to stratify by cond from __getitem__ tuple (imu, chest, cond, br)
        # This can be slower but K is small.
        conds = []
        for i in range(n):
            item = ds[i]
            # expecting (imu, chest, cond, br)
            conds.append(int(item[2]))
        conds = np.asarray(conds)

        # sample_equal_per_condition is already in your file
        idx = sample_equal_per_condition(conds, n_per_cond=max(1, k // max(1, len(np.unique(conds)))),
                                         rng=np.random.default_rng(seed))
        idx = idx[:k]

    idx_set = set(idx)
    rest_idx = [i for i in range(n) if i not in idx_set]

    fewshot_ds = Subset(ds, idx)
    rest_ds = Subset(ds, rest_idx)

    # Small batch size for few-shot
    fewshot_loader = DataLoader(
        fewshot_ds,
        batch_size=min(k, getattr(test_dataloader, "batch_size", 64)),
        shuffle=True,
        drop_last=False,
        **_loader_kwargs(),
    )
    remaining_loader = DataLoader(
        rest_ds,
        batch_size=getattr(test_dataloader, "batch_size", 64),
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(),
    )
    return fewshot_loader, remaining_loader

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataloader import LoadDataset, load_data, make_dataset
from config import SBJ_PROCESSED_DIR
from vit_pressure_crossmodal_stft_rr_core import (
    build_base_parser,
    collect_frozen_embeddings,
    default_subjects,
    run_loocv_experiment,
)

IMU_ISSUES_MR = [17, 26, 30]
IMU_ISSUES_L = [15, 17, 21, 26, 28, 30]
SUBJECTS = default_subjects(IMU_ISSUES_MR, IMU_ISSUES_L)
VALID_MWL_LABELS = ("M", "R", "L0", "L1", "L2", "L3")


def parse_mwl_labels(s: str) -> List[str]:
    if isinstance(s, (list, tuple, np.ndarray)):
        out = [str(c).strip().upper() for c in s if str(c).strip()]
    else:
        out = [c.strip().upper() for c in str(s).split(",") if c.strip()]
    if not out:
        raise ValueError("embed labels list is empty; pass labels like M,R,L1,L3")
    bad = [c for c in out if c not in VALID_MWL_LABELS]
    if bad:
        raise ValueError(f"Unsupported embed labels {bad}. Use any of: {VALID_MWL_LABELS}")
    return out


def infer_embed_data_group_from_labels(labels: List[str]) -> str:
    s = set(labels)
    has_mr = bool(s & {"M", "R"})
    has_levels = bool(s & {"L0", "L1", "L2", "L3"})
    if has_mr and has_levels:
        return "mr_levels"
    if has_mr:
        return "mr"
    return "levels"


def _canonical_mwl_class_id(label: str) -> int:
    mapping = {"M": 0, "R": 1, "L0": 2, "L1": 3, "L2": 4, "L3": 5}
    return mapping[str(label).strip().upper()]


def _is_level_label(label: str) -> bool:
    return str(label).strip().upper() in {"L0", "L1", "L2", "L3"}


def _grouped_embed_labels(labels: List[str]) -> Dict[str, List[str]]:
    grouped = {"mr": [], "levels": []}
    for lbl in labels:
        key = "levels" if _is_level_label(lbl) else "mr"
        grouped[key].append(str(lbl).strip().upper())
    return {k: v for k, v in grouped.items() if v}


def _filter_subject_dict_for_downstream_group(d: Dict[str, Any], data_group: str, keep_values: set[str]) -> Dict[str, Any]:
    conds = np.asarray(d.get("conds"))
    if conds.size == 0:
        return d
    mask = np.asarray([str(c) in keep_values for c in conds], dtype=bool)
    if not mask.any():
        raise RuntimeError(
            f"No labels {sorted(keep_values)} found for subject {d.get('subject', '<unknown>')} "
            f"while building frozen-embedding downstream split data_group={data_group!r}."
        )
    out = dict(d)
    n = len(conds)
    for k, v in d.items():
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.shape[:1] == (n,):
            out[k] = arr[mask]
    return out


def _load_group_subjects(subject: str, subjects: List[str], data_group: str, keep_values: set[str], args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_subjects = [s for s in subjects if s != subject]
    train_list = [
        _filter_subject_dict_for_downstream_group(load_data(s, data_dir=args.data_dir, data_group=data_group), data_group, keep_values)
        for s in train_subjects
    ]
    test_list = [
        _filter_subject_dict_for_downstream_group(load_data(subject, data_dir=args.data_dir, data_group=data_group), data_group, keep_values)
    ]
    return train_list, test_list


def _build_frozen_embedding_loaders(subject: str, subjects: List[str], args) -> Tuple[DataLoader, DataLoader]:
    keep_labels = parse_mwl_labels(getattr(args, "embed_labels", []))
    grouped = _grouped_embed_labels(keep_labels)
    include_tlx = bool(getattr(args, "eval_frozen_tlx", False))

    def _build_split(split_lists):
        xs, ys, brs, conds, tlxs = [], [], [], [], []
        saw_tlx = False
        keep_ids = np.asarray([_canonical_mwl_class_id(lbl) for lbl in keep_labels], dtype=int)
        for group, data_list in split_lists:
            if not data_list:
                continue
            out = make_dataset(
                data_list,
                args.data_str,
                label_encoder_dir=args.data_dir,
                data_group=group,
                include_tlx=include_tlx,
                tlx_csv_path=getattr(args, "tlx_csv_path", None),
            )
            if include_tlx:
                x, pressure, br, cond, tlx = out
            else:
                x, pressure, br, cond = out
                tlx = None
            cond = np.asarray(cond, dtype=int).reshape(-1)
            if group == "levels":
                cond = cond + 2
            mask = np.isin(cond, keep_ids)
            xs.append(x[mask])
            ys.append(pressure[mask])
            brs.append(br[mask])
            conds.append(cond[mask])
            if include_tlx and tlx is not None:
                tlxs.append(tlx[mask])
                saw_tlx = True
        if not xs:
            raise RuntimeError(f"No labels {keep_labels} found while building frozen-embedding split for subject {subject}.")
        tlx = np.concatenate(tlxs, axis=0) if include_tlx and saw_tlx and tlxs else None
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(brs, axis=0), np.concatenate(conds, axis=0), tlx

    train_split_lists = []
    test_split_lists = []
    for group, group_labels in grouped.items():
        train_list, test_list = _load_group_subjects(subject, subjects, group, set(group_labels), args)
        train_split_lists.append((group, train_list))
        test_split_lists.append((group, test_list))

    x_train, y_train, br_train, cond_train, tlx_train = _build_split(train_split_lists)
    x_test, y_test, br_test, cond_test, tlx_test = _build_split(test_split_lists)

    train_ds = LoadDataset(x_train, y_train, cond_train, br_train, tlx_train, aug_ratio=0.0) if include_tlx else LoadDataset(x_train, y_train, cond_train, br_train, aug_ratio=0.0)
    test_ds = LoadDataset(x_test, y_test, cond_test, br_test, tlx_test, aug_ratio=0.0) if include_tlx else LoadDataset(x_test, y_test, cond_test, br_test, aug_ratio=0.0)
    train_loader = DataLoader(train_ds, batch_size=int(args.embed_batch_size), shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(args.embed_batch_size), shuffle=False, drop_last=False, num_workers=0)
    return train_loader, test_loader


def _make_embedding_classifier(kind: str):
    kind = str(kind).lower()
    if kind == "lda":
        return LinearDiscriminantAnalysis()
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", multi_class="auto")
    if kind == "linear":
        return None
    raise ValueError(f"Unsupported --embed-classifier={kind!r}")


def _train_linear_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, args, device: str) -> np.ndarray:
    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    xte = torch.tensor(x_test, dtype=torch.float32, device=device)
    n_classes = int(np.unique(y_train).max()) + 1
    model = nn.Linear(xtr.size(1), n_classes).to(device)
    counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.linear_probe_lr), weight_decay=float(args.linear_probe_weight_decay))
    bs = int(args.linear_probe_batch_size)
    for _epoch in range(int(args.linear_probe_epochs)):
        perm = torch.randperm(xtr.size(0), device=device)
        for st in range(0, xtr.size(0), bs):
            idx = perm[st : st + bs]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        return model(xte).argmax(dim=1).detach().cpu().numpy()


def frozen_embedding_evaluate(model: nn.Module, subject: str, subjects: List[str], device: str, args, out_dir: Optional[Path] = None, tag: str = "pre_tta") -> Dict[str, float]:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, tlx_train = collect_frozen_embeddings(model, _build_frozen_embedding_loaders(subject, subjects, args)[0], device, args)
    x_test, y_test, tlx_test = collect_frozen_embeddings(model, _build_frozen_embedding_loaders(subject, subjects, args)[1], device, args)
    requested_labels = [str(lbl) for lbl in getattr(args, "embed_labels", [])]
    requested_label_indices = [_canonical_mwl_class_id(lbl) for lbl in requested_labels]
    canonical_label_names = ["M", "R", "L0", "L1", "L2", "L3"]
    id_to_label = {idx: name for idx, name in enumerate(canonical_label_names)}
    resolved_embed_data_group = infer_embed_data_group_from_labels(requested_labels)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    if str(args.embed_classifier).lower() == "linear":
        pred = _train_linear_probe(x_train_s, y_train, x_test_s, args, device)
    else:
        clf = _make_embedding_classifier(args.embed_classifier)
        clf.fit(x_train_s, y_train)
        pred = clf.predict(x_test_s)

    metrics: Dict[str, float] = {
        "embed_acc": float(accuracy_score(y_test, pred)),
        "embed_f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "embed_f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "embed_n_train": int(y_train.shape[0]),
        "embed_n_test": int(y_test.shape[0]),
        "embed_n_features": int(x_train.shape[1]),
        "embed_n_classes_train": int(np.unique(y_train).shape[0]),
        "embed_n_classes_test": int(np.unique(y_test).shape[0]),
        "embed_pooling": str(args.embed_pooling),
        "embed_classifier": str(args.embed_classifier),
        "embed_data_group": str(resolved_embed_data_group),
        "embed_label_names": requested_labels,
        "embed_label_indices": requested_label_indices,
        "embed_label_encoder_classes": canonical_label_names,
    }
    labels = sorted(set(np.unique(y_train).tolist()) | set(np.unique(y_test).tolist()))
    per_class = f1_score(y_test, pred, labels=labels, average=None, zero_division=0)
    for lbl, val in zip(labels, per_class):
        metrics[f"embed_f1_class_{int(lbl)}"] = float(val)

    if bool(getattr(args, "eval_frozen_tlx", False)):
        if tlx_train is None or tlx_test is None:
            metrics.update({"tlx_available": 0, "tlx_n_train": 0, "tlx_n_test": 0})
        else:
            train_mask = np.isfinite(tlx_train)
            test_mask = np.isfinite(tlx_test)
            metrics["tlx_available"] = int(train_mask.any() and test_mask.any())
            metrics["tlx_n_train"] = int(train_mask.sum())
            metrics["tlx_n_test"] = int(test_mask.sum())
            if train_mask.any() and test_mask.any():
                tlx_model = Ridge(alpha=float(getattr(args, "tlx_ridge_alpha", 1.0)))
                tlx_model.fit(x_train_s[train_mask], tlx_train[train_mask])
                tlx_pred = tlx_model.predict(x_test_s[test_mask])
                tlx_true = tlx_test[test_mask]
                metrics.update({
                    "tlx_mae": float(mean_absolute_error(tlx_true, tlx_pred)),
                    "tlx_rmse": float(np.sqrt(mean_squared_error(tlx_true, tlx_pred))),
                    "tlx_r2": float(r2_score(tlx_true, tlx_pred)) if len(np.unique(tlx_true)) > 1 else float("nan"),
                    "tlx_corr": float(np.corrcoef(tlx_true.reshape(-1), tlx_pred.reshape(-1))[0, 1]) if tlx_true.size > 1 and np.std(tlx_true) > 1e-8 and np.std(tlx_pred) > 1e-8 else float("nan"),
                })
                if out_dir is not None:
                    pd.DataFrame({"tlx_true": tlx_true.astype(float), "tlx_pred": tlx_pred.astype(float)}).to_csv(out_dir / f"frozen_tlx_predictions_{tag}.csv", index=False)

    if out_dir is not None:
        np.save(out_dir / f"embeddings_train_{tag}.npy", x_train)
        np.save(out_dir / f"embeddings_test_{tag}.npy", x_test)
        pd.DataFrame({
            "y_true": y_test.astype(int),
            "y_true_label": [id_to_label.get(int(v), str(int(v))) for v in y_test.astype(int)],
            "y_pred": pred.astype(int),
            "y_pred_label": [id_to_label.get(int(v), str(int(v))) for v in pred.astype(int)],
        }).to_csv(out_dir / f"frozen_embedding_predictions_{tag}.csv", index=False)
        with open(out_dir / f"frozen_embedding_labels_{tag}.json", "w") as f:
            json.dump(
                {
                    "embed_label_names": requested_labels,
                    "embed_label_indices": requested_label_indices,
                    "embed_data_group": str(resolved_embed_data_group),
                    "embed_label_encoder_classes": canonical_label_names,
                },
                f,
                indent=2,
            )
        with open(out_dir / f"frozen_embedding_metrics_{tag}.json", "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def frozen_embedding_hook(model, sbj: str, subjects: List[str], _train_loader, _test_loader, device: str, args, sbj_dir: Path):
    if not bool(getattr(args, "eval_frozen_embeddings", False)):
        return []
    pre = frozen_embedding_evaluate(model, sbj, subjects, device, args, out_dir=sbj_dir / "frozen_embeddings", tag="pre_tta")
    rows = [{"__summary_name__": "frozen_embedding_summary", "subject": sbj, "tag": "pre_tta", **pre}]
    print(f"FROZEN_EMBED_PRE_TTA {sbj}: {pre}")
    if str(args.tta).lower() != "none" and int(args.tta_epochs) > 0:
        post = frozen_embedding_evaluate(model, sbj, subjects, device, args, out_dir=sbj_dir / "frozen_embeddings", tag="post_tta_summary")
        rows.append({"__summary_name__": "frozen_embedding_summary", "subject": sbj, "tag": "post_tta", **post})
    return rows


def finalize_args(args) -> None:
    if args.eval_frozen_tlx:
        args.eval_frozen_embeddings = True
    args.embed_labels = parse_mwl_labels(args.embed_labels)
    if args.embed_data_group is None:
        args.embed_data_group = infer_embed_data_group_from_labels(args.embed_labels)


def main() -> None:
    parser = build_base_parser(SUBJECTS, str(Path(SBJ_PROCESSED_DIR) / "smoke_vit_pressure"))
    parser.add_argument("--eval-frozen-embeddings", action="store_true")
    parser.add_argument("--eval-frozen-tlx", action="store_true")
    parser.add_argument("--tlx-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--embed-data-group", default=None, choices=["mr", "level", "levels", "mr_levels"])
    parser.add_argument("--embed-labels", default="M,R,L1,L3")
    parser.add_argument("--embed-classifier", default="lda", choices=["lda", "logreg", "linear"])
    parser.add_argument("--embed-pooling", default="mean_std_max", choices=["mean", "max", "cls_last", "mean_std", "mean_std_max", "rich"])
    parser.add_argument("--embed-stft-profile", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--linear-probe-epochs", type=int, default=30)
    parser.add_argument("--linear-probe-lr", type=float, default=1e-3)
    parser.add_argument("--linear-probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--linear-probe-batch-size", type=int, default=64)
    args = parser.parse_args()
    run_loocv_experiment(args, post_eval_hooks=[frozen_embedding_hook], config_mutator=finalize_args)


if __name__ == "__main__":
    main()

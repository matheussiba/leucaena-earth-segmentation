"""
Per-class evaluation of a trained experiment on the patch dataset.

Unlike the macro F1 printed during training, this reports Precision / Recall /
F1 **per class** (background vs. leucaena) plus the full confusion matrix
(TP/FP/FN/TN) and IoU. The leucaena line — especially its Precision — is the
number tied to the false-positive problem you see in the full-scene map.

It loads the checkpoint saved by train.py (experiments/exp_<N>/models/model.pt),
runs the chosen split through the same PatchFileDataset used in training
(honouring lidar_bands and the COMPUTE_NDVI flag of the experiment), and
accumulates a global confusion matrix over all non-IGNORE pixels.

IMPORTANT — what this does and does NOT measure
-----------------------------------------------
Pixels outside the annotated polygons are IGNORE (255) in the labels, so they
are excluded here too. This report therefore measures performance *inside the
polygons*. It does not capture false positives in the open landscape (pasture,
roofs) unless those areas were annotated. For that you need a fully-annotated
test tile or visual inspection of the predicted map.

Examples
--------
# Evaluate exp_4 on the test split (default)
python eval-patches.py -e 4

# Evaluate on validation, point at the refined dataset explicitly
python eval-patches.py -e 4 --split val \
    --patches-dir /data/patches --manifest /data/patches/manifest.csv
"""
from __future__ import annotations

import argparse
import importlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from conf import default, general, paths
from utils.dataloader import PatchFileDataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--experiment", type=int, required=True, help="Experiment id (matches conf/model_<e>.py).")
    p.add_argument("--split", choices=("val", "test", "train"), default="test", help="Patch split (default: %(default)s).")
    p.add_argument("--experiments-path", default=str(paths.PATH_EXPERIMENTS), help="Root with exp_<N> folders.")
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path (default: experiments/exp_<e>/models/model.pt).")
    p.add_argument("--patches-dir", default=str(paths.PATH_PATCHES_DIR), help="Folder with opt/, lbl/, lidar/, manifest.csv.")
    p.add_argument("--manifest", default=None, help="Manifest CSV (default: <patches-dir>/manifest.csv).")
    p.add_argument("-b", "--batch-size", type=int, default=default.PREDICTION_BATCH_SIZE, help="Batch size (default: %(default)s).")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu (default: auto).")
    return p.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if requested not in ("cuda", "cpu"):
        raise ValueError(f"Unknown device {requested!r}; use auto, cuda, or cpu.")
    return requested


def _resolve_checkpoint(exp_dir: str, explicit: str | None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return explicit
    candidates = [
        os.path.join(exp_dir, "models", "model.pt"),
        os.path.join(exp_dir, "models", "best_model.pt"),
        os.path.join(exp_dir, "model.pt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "No checkpoint found. Checked:\n  - " + "\n  - ".join(candidates) +
        "\nPass --checkpoint if it lives elsewhere."
    )


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _report(tp: int, fp: int, fn: int, tn: int) -> str:
    # leucaena (positive class = 1)
    leuc_p = _safe_div(tp, tp + fp)
    leuc_r = _safe_div(tp, tp + fn)
    leuc_f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    leuc_iou = _safe_div(tp, tp + fp + fn)
    # background (treat bg as the positive class: its TP=tn, FP=fn, FN=fp)
    bg_p = _safe_div(tn, tn + fn)
    bg_r = _safe_div(tn, tn + fp)
    bg_f1 = _safe_div(2 * tn, 2 * tn + fn + fp)
    bg_iou = _safe_div(tn, tn + fn + fp)

    macro_f1 = (leuc_f1 + bg_f1) / 2.0
    total = tp + fp + fn + tn
    accuracy = _safe_div(tp + tn, total)
    leuc_support = tp + fn
    bg_support = tn + fp

    lines = [
        "",
        "================ PER-CLASS REPORT (inside polygons only) ================",
        f"{'class':<12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'IoU':>12}{'Support(px)':>16}",
        "-" * 76,
        f"{'background':<12}{bg_p:>12.4f}{bg_r:>12.4f}{bg_f1:>12.4f}{bg_iou:>12.4f}{bg_support:>16,}",
        f"{'leucaena':<12}{leuc_p:>12.4f}{leuc_r:>12.4f}{leuc_f1:>12.4f}{leuc_iou:>12.4f}{leuc_support:>16,}",
        "-" * 76,
        f"{'macro F1':<12}{macro_f1:>48.4f}",
        f"{'accuracy':<12}{accuracy:>48.4f}",
        "",
        "Confusion matrix (positive = leucaena):",
        f"  TP (leucaena correct)        = {tp:,}",
        f"  FP (background -> leucaena)   = {fp:,}   <- false positives",
        f"  FN (leucaena -> background)   = {fn:,}   <- missed leucaena",
        f"  TN (background correct)       = {tn:,}",
        "=========================================================================",
        "",
        "Reading it: low leucaena Precision = the model calls too much background",
        "'leucaena' (the false-positive problem). Low Recall = it misses real",
        "leucaena. F1 balances both. NOTE: pixels outside polygons are excluded.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    exp_dir = os.path.join(args.experiments_path, f"exp_{args.experiment}")
    manifest_path = args.manifest or os.path.join(args.patches_dir, "manifest.csv")
    checkpoint_path = _resolve_checkpoint(exp_dir, args.checkpoint)

    model_module = importlib.import_module(f"conf.model_{args.experiment}")
    model, lidar_bands = model_module.get_model()
    compute_ndvi = getattr(model_module, "COMPUTE_NDVI", False)
    model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    dataset = PatchFileDataset(
        manifest_path=manifest_path,
        split=args.split,
        device=device,
        patches_dir=args.patches_dir,
        data_aug=False,
        lidar_bands=lidar_bands,
        cache_in_ram=False,
        compute_ndvi=compute_ndvi,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Experiment   : {args.experiment}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Manifest     : {manifest_path}")
    print(f"Split        : {args.split}  ({len(dataset):,} patches)")
    print(f"Compute NDVI : {compute_ndvi}")
    print(f"Device       : {device}")
    print("Evaluating...")

    ignore = general.IGNORE_INDEX
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)  # (N, H, W)
            valid = y != ignore
            gt_pos = (y == 1) & valid
            gt_neg = (y == 0) & valid
            pred_pos = (pred == 1) & valid
            pred_neg = (pred == 0) & valid
            tp += int((pred_pos & gt_pos).sum().item())
            fp += int((pred_pos & gt_neg).sum().item())
            fn += int((pred_neg & gt_pos).sum().item())
            tn += int((pred_neg & gt_neg).sum().item())

    report = _report(tp, fp, fn, tn)
    print(report)

    logs_dir = os.path.join(exp_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, f"eval_patches_{args.split}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment {args.experiment} | split={args.split} | checkpoint={checkpoint_path}\n")
        f.write(report + "\n")
    print(f"\nSaved report -> {out_path}")


if __name__ == "__main__":
    main()

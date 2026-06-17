"""
Inspect where a trained segmentation experiment fails on patch splits.

The script reuses the tile-based patch dataset produced by
prep-patches-from-tiles.py and the same model configuration used by train.py.
It writes:

- ranked CSV with one row per evaluated patch;
- PNG panels for the selected worst patches;
- GeoJSON containing the selected patch footprints, enriched with metrics.

It does not require QGIS at runtime; the GeoJSON can be opened there later.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from conf import general, paths
from utils.dataloader import PatchFileDataset


CSV_FIELDS = [
    "rank",
    "patch_id",
    "split",
    "loss",
    "f1",
    "iou",
    "tp",
    "fp",
    "fn",
    "tn",
    "valid_pixels",
    "gt_positive_pixels",
    "pred_positive_pixels",
    "leucaena_fraction",
    "polygon_fraction",
    "tile_name",
    "row",
    "col",
    "xoff",
    "yoff",
    "has_lidar",
    "panel_path",
]


@dataclass
class PatchMetrics:
    patch_id: str
    split: str
    loss: float
    f1: float
    iou: float
    tp: int
    fp: int
    fn: int
    tn: int
    valid_pixels: int
    gt_positive_pixels: int
    pred_positive_pixels: int
    record: dict
    panel_path: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank validation/test/train patches by segmentation errors and write "
            "diagnostic PNG panels, CSV, and GeoJSON."
        )
    )
    parser.add_argument("-e", "--experiment", type=int, required=True)
    parser.add_argument(
        "--experiments-path",
        default=str(paths.PATH_EXPERIMENTS),
        help="Root folder containing exp_<N> folders (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path. Default searches under experiments/exp_<N>/models/.",
    )
    parser.add_argument(
        "--patches-dir",
        default=str(paths.PATH_PATCHES_DIR),
        help="Folder containing opt/, lbl/, optional lidar/, manifest.csv, and patch_footprints.geojson.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest CSV. Default: <patches-dir>/manifest.csv.",
    )
    parser.add_argument(
        "--footprints",
        default=None,
        help="Patch footprints GeoJSON. Default: <patches-dir>/patch_footprints.geojson.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test", "train"),
        default="val",
        help="Patch split to inspect (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of worst patches to render/export to GeoJSON (default: %(default)s).",
    )
    parser.add_argument(
        "--rank-by",
        choices=("loss", "iou", "f1"),
        default="loss",
        help="How to choose worst patches. loss ranks descending; iou/f1 rank ascending.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output folder. Default: experiments/exp_<N>/diagnostics/<split>.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cuda | cpu (default: auto).",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in ("cuda", "cpu"):
        raise ValueError(f"Unknown device {requested!r}; use auto, cuda, or cpu.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return requested


def _checkpoint_candidates(exp_dir: str) -> list[str]:
    return [
        os.path.join(exp_dir, "models", "model.pt"),
        os.path.join(exp_dir, "models", "best_model.pt"),
        os.path.join(exp_dir, "model.pt"),
        os.path.join(exp_dir, "checkpoints", "model.pt"),
        os.path.join(exp_dir, "checkpoints", "best_model.pt"),
    ]


def _resolve_checkpoint(exp_dir: str, explicit: str | None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return explicit
    candidates = _checkpoint_candidates(exp_dir)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    formatted = "\n  - ".join(candidates)
    raise FileNotFoundError(
        "Could not find a trained checkpoint. Checked:\n"
        f"  - {formatted}\n"
        "Pass --checkpoint if the experiment uses a different filename."
    )


def _load_state_dict(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def _safe_float(value: str | float | int | None, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_metrics(
    outputs: torch.Tensor,
    label: torch.Tensor,
    class_weights: torch.Tensor,
    ignore_index: int,
) -> tuple[float, dict[str, int | float]]:
    loss_map = F.cross_entropy(
        outputs,
        label.unsqueeze(0),
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = label != ignore_index
    valid_pixels = int(valid.sum().item())
    if valid_pixels == 0:
        return math.nan, {
            "f1": math.nan,
            "iou": math.nan,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "valid_pixels": 0,
            "gt_positive_pixels": 0,
            "pred_positive_pixels": 0,
        }

    loss = float(loss_map.squeeze(0)[valid].mean().item())
    pred = outputs.argmax(dim=1).squeeze(0)

    gt_pos = (label == 1) & valid
    gt_neg = (label == 0) & valid
    pred_pos = (pred == 1) & valid
    pred_neg = (pred == 0) & valid

    tp = int((pred_pos & gt_pos).sum().item())
    fp = int((pred_pos & gt_neg).sum().item())
    fn = int((pred_neg & gt_pos).sum().item())
    tn = int((pred_neg & gt_neg).sum().item())
    f1_den = (2 * tp) + fp + fn
    iou_den = tp + fp + fn
    f1 = 1.0 if f1_den == 0 else (2 * tp) / f1_den
    iou = 1.0 if iou_den == 0 else tp / iou_den

    return loss, {
        "f1": float(f1),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "valid_pixels": valid_pixels,
        "gt_positive_pixels": int(gt_pos.sum().item()),
        "pred_positive_pixels": int(pred_pos.sum().item()),
    }


def _rank_key(metric: PatchMetrics, rank_by: str) -> tuple:
    value = getattr(metric, rank_by)
    if math.isnan(value):
        return (1, 0.0)
    if rank_by == "loss":
        return (0, -value)
    return (0, value)


def _add_batch_dim(inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return inputs[0].unsqueeze(0), inputs[1].unsqueeze(0)


def _format_float(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def _write_csv(metrics: list[PatchMetrics], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rank, metric in enumerate(metrics, 1):
            rec = metric.record
            writer.writerow(
                {
                    "rank": rank,
                    "patch_id": metric.patch_id,
                    "split": metric.split,
                    "loss": _format_float(metric.loss),
                    "f1": _format_float(metric.f1),
                    "iou": _format_float(metric.iou),
                    "tp": metric.tp,
                    "fp": metric.fp,
                    "fn": metric.fn,
                    "tn": metric.tn,
                    "valid_pixels": metric.valid_pixels,
                    "gt_positive_pixels": metric.gt_positive_pixels,
                    "pred_positive_pixels": metric.pred_positive_pixels,
                    "leucaena_fraction": rec.get("leucaena_fraction", ""),
                    "polygon_fraction": rec.get("polygon_fraction", ""),
                    "tile_name": rec.get("tile_name", ""),
                    "row": rec.get("row", ""),
                    "col": rec.get("col", ""),
                    "xoff": rec.get("xoff", ""),
                    "yoff": rec.get("yoff", ""),
                    "has_lidar": rec.get("has_lidar", ""),
                    "panel_path": metric.panel_path,
                }
            )


def _optical_rgb(opt_bgrn: np.ndarray) -> np.ndarray:
    rgb = opt_bgrn[:, :, [2, 1, 0]].astype(np.float32)
    if rgb.max(initial=0) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _error_overlay(label: np.ndarray, pred: np.ndarray, ignore_index: int) -> np.ndarray:
    valid = label != ignore_index
    gt_pos = (label == 1) & valid
    pred_pos = (pred == 1) & valid

    overlay = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.float32)
    overlay[~valid] = (0.55, 0.55, 0.55)      # ignore
    overlay[pred_pos & gt_pos] = (0.0, 0.65, 0.0)   # true positive
    overlay[pred_pos & ~gt_pos & valid] = (1.0, 0.0, 0.0)  # false positive
    overlay[~pred_pos & gt_pos] = (0.0, 0.25, 1.0)  # false negative
    return overlay


def _load_raw_arrays(patches_dir: str, rec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    patch_id = rec["patch_id"]
    opt = np.load(os.path.join(patches_dir, "opt", f"{patch_id}.npy"))
    label = np.load(os.path.join(patches_dir, "lbl", f"{patch_id}.npy"))
    lidar_path = os.path.join(patches_dir, "lidar", f"{patch_id}.npy")
    lidar = np.load(lidar_path) if os.path.isfile(lidar_path) else None
    return opt, label, lidar


def _render_panel(
    patches_dir: str,
    out_path: str,
    metric: PatchMetrics,
    pred: np.ndarray,
    ignore_index: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to write PNG panels. Install project requirements "
            "or rerun diagnostics where matplotlib is available."
        ) from exc

    opt, label, lidar = _load_raw_arrays(patches_dir, metric.record)
    rgb = _optical_rgb(opt)
    chm = lidar[:, :, 0] if lidar is not None and lidar.ndim == 3 and lidar.shape[-1] > 0 else None
    overlay = _error_overlay(label, pred, ignore_index)

    label_cmap = ListedColormap(["black", "lime", "lightgray"])
    label_show = np.where(label == ignore_index, 2, label)
    pred_show = pred

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(18, 4))
    fig.suptitle(
        f"{metric.patch_id} | loss={_format_float(metric.loss)} "
        f"F1={_format_float(metric.f1)} IoU={_format_float(metric.iou)}",
        fontsize=10,
    )
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    if chm is None:
        axes[1].imshow(np.zeros(label.shape), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("CHM missing")
    else:
        im = axes[1].imshow(chm, cmap="viridis")
        axes[1].set_title("CHM")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(label_show, cmap=label_cmap, vmin=0, vmax=2)
    axes[2].set_title("Ground truth")
    axes[3].imshow(pred_show, cmap=label_cmap, vmin=0, vmax=2)
    axes[3].set_title("Prediction")
    axes[4].imshow(overlay)
    axes[4].set_title("Errors: TP green, FP red, FN blue")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_geojson(
    source_geojson: str,
    out_geojson: str,
    selected: Iterable[PatchMetrics],
) -> int:
    selected = list(selected)
    by_patch = {m.patch_id: m for m in selected}
    if not os.path.isfile(source_geojson):
        print(f"[WARN] footprint GeoJSON not found: {source_geojson}")
        print("       Skipping worst patch GeoJSON export.")
        return 0

    with open(source_geojson, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        patch_id = props.get("patch_id")
        metric = by_patch.get(patch_id)
        if metric is None:
            continue
        copied = dict(feature)
        copied_props = dict(props)
        copied_props.update(
            {
                "diag_rank": selected.index(metric) + 1,
                "diag_loss": None if math.isnan(metric.loss) else metric.loss,
                "diag_f1": None if math.isnan(metric.f1) else metric.f1,
                "diag_iou": None if math.isnan(metric.iou) else metric.iou,
                "diag_tp": metric.tp,
                "diag_fp": metric.fp,
                "diag_fn": metric.fn,
                "diag_tn": metric.tn,
                "diag_valid_pixels": metric.valid_pixels,
                "diag_panel": metric.panel_path,
            }
        )
        copied["properties"] = copied_props
        features.append(copied)

    out = dict(data)
    out["name"] = "worst_patch_footprints"
    out["features"] = features
    os.makedirs(os.path.dirname(out_geojson) or ".", exist_ok=True)
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return len(features)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    exp_dir = os.path.join(args.experiments_path, f"exp_{args.experiment}")
    out_dir = args.out_dir or os.path.join(exp_dir, "diagnostics", args.split)
    panels_dir = os.path.join(out_dir, "panels")
    manifest_path = args.manifest or os.path.join(args.patches_dir, "manifest.csv")
    footprints_path = args.footprints or os.path.join(args.patches_dir, "patch_footprints.geojson")
    checkpoint_path = _resolve_checkpoint(exp_dir, args.checkpoint)

    print(f"Experiment      : {args.experiment}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Manifest        : {manifest_path}")
    print(f"Patches dir     : {args.patches_dir}")
    print(f"Split           : {args.split}")
    print(f"Rank by         : {args.rank_by}")
    print(f"Top K           : {args.top_k}")
    print(f"Out dir         : {out_dir}")
    print(f"Device          : {device}")

    model_module = importlib.import_module(f"conf.model_{args.experiment}")
    model, lidar_bands = model_module.get_model()
    compute_ndvi = getattr(model_module, "COMPUTE_NDVI", False)
    model.to(device)
    _load_state_dict(model, checkpoint_path, device)
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
    class_weights = torch.tensor(general.CLASSES_WEIGHTS, dtype=torch.float32, device=device)

    print(f"Evaluating {len(dataset):,} patch(es)...")
    metrics: list[PatchMetrics] = []
    with torch.no_grad():
        for i in range(len(dataset)):
            (inputs, label), rec = dataset[i], dataset.records[i]
            inputs = _add_batch_dim(inputs)
            outputs = model(inputs)
            loss, values = _compute_metrics(
                outputs=outputs,
                label=label,
                class_weights=class_weights,
                ignore_index=general.IGNORE_INDEX,
            )
            patch_id = rec["patch_id"]
            metrics.append(
                PatchMetrics(
                    patch_id=patch_id,
                    split=args.split,
                    loss=loss,
                    f1=float(values["f1"]),
                    iou=float(values["iou"]),
                    tp=int(values["tp"]),
                    fp=int(values["fp"]),
                    fn=int(values["fn"]),
                    tn=int(values["tn"]),
                    valid_pixels=int(values["valid_pixels"]),
                    gt_positive_pixels=int(values["gt_positive_pixels"]),
                    pred_positive_pixels=int(values["pred_positive_pixels"]),
                    record=rec,
                )
            )

    metrics.sort(key=lambda m: _rank_key(m, args.rank_by))
    selected = metrics[: max(0, args.top_k)]
    index_by_patch = {rec["patch_id"]: i for i, rec in enumerate(dataset.records)}

    with torch.no_grad():
        for rank, metric in enumerate(selected, 1):
            filename = f"{rank:03d}_{metric.patch_id}.png"
            panel_path = os.path.join(panels_dir, filename)
            inputs, _label = dataset[index_by_patch[metric.patch_id]]
            inputs = _add_batch_dim(inputs)
            pred = model(inputs).argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            _render_panel(
                patches_dir=args.patches_dir,
                out_path=panel_path,
                metric=metric,
                pred=pred,
                ignore_index=general.IGNORE_INDEX,
            )
            metric.panel_path = panel_path

    csv_path = os.path.join(out_dir, f"worst_patches_{args.split}.csv")
    geojson_path = os.path.join(out_dir, f"worst_patches_{args.split}.geojson")
    _write_csv(metrics, csv_path)
    n_geo = _write_geojson(footprints_path, geojson_path, selected)

    print("\nDone.")
    print(f"Ranking CSV     : {csv_path}")
    print(f"Panels          : {panels_dir}")
    print(f"Worst GeoJSON   : {geojson_path} ({n_geo} feature(s))")


if __name__ == "__main__":
    main()

"""
Visual sanity-check: run the trained model on one GeoTIFF tile (sliding windows).

Writes PNGs under ``experiments/exp_<e>/predicted/``:
  - ``preview_<stem>_rgb.png`` — R,G,B stretch from the tile
  - ``preview_<stem>_pred_class.png`` — argmax class (0/1)
  - ``preview_<stem>_pred_overlay.png`` — RGB + leucaena in red
  - ``preview_<stem>_triptych.png`` — RGB | ground-truth | prediction

If ``--masks`` is given, rasterizes intersecting polygons (same as prep-patches) and
prints pixel accuracy and F1.

Example (Docker)::

    python predict-tile-preview.py -e 1 \\
        --tile /data/rgbir/SF-23-Y-A-IV-2-SE-C.tif \\
        --masks /data/masks/polygons.geojson \\
        --band-order RGBN -b 16
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

import numpy as np
import torch
from osgeo import gdal, gdalconst
from torchmetrics.functional.classification import multiclass_f1_score

from conf import general, paths
from utils.ops import rasterize_geojson_for_tile


def _band_reorder_indices(band_order: str) -> list[int]:
    if band_order == "BGRN":
        return [0, 1, 2, 3]
    if band_order == "RGBN":
        return [2, 1, 0, 3]
    raise ValueError(f"Unsupported band order: {band_order}")


def _crop_box(fh: int, fw: int, max_side: int) -> tuple[int, int, int, int]:
    """Return (y0, x0, crop_h, crop_w) center crop; max_side<=0 means full image."""
    if max_side <= 0 or (fh <= max_side and fw <= max_side):
        return 0, 0, fh, fw
    ch = min(max_side, fh)
    cw = min(max_side, fw)
    y0 = (fh - ch) // 2
    x0 = (fw - cw) // 2
    return y0, x0, ch, cw


def _rgb_preview_from_bgrn(chw: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """chw float [0,1], BGRN -> uint8 RGB H W 3 using percentile stretch."""
    b, g, r = chw[0], chw[1], chw[2]
    rgb = np.stack([r, g, b], axis=-1)
    lo = np.quantile(rgb.reshape(-1, 3), alpha, axis=0)
    hi = np.quantile(rgb.reshape(-1, 3), 1.0 - alpha, axis=0)
    out = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (out * 255).astype(np.uint8)


def _save_pred_overlay(rgb: np.ndarray, pred_hw: np.ndarray, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_rgb = np.zeros((*pred_hw.shape, 3), dtype=np.float32)
    pred_rgb[..., 0] = (pred_hw == 1).astype(np.float32) * 255
    pred_rgb[..., 1] = np.where(pred_hw == 0, rgb[..., 1], pred_rgb[..., 1]).astype(np.float32)
    pred_rgb[..., 2] = np.where(pred_hw == 0, rgb[..., 2], pred_rgb[..., 2]).astype(np.float32)
    blend = 0.45 * rgb.astype(np.float32) + 0.55 * pred_rgb
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    plt.imsave(out_path, blend)


def _save_triptych(
    rgb: np.ndarray,
    gt: np.ndarray | None,
    pred_hw: np.ndarray,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (tile)")
    axes[0].axis("off")
    if gt is not None:
        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Label (GeoJSON)")
    else:
        axes[1].text(0.5, 0.5, "no --masks", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")
    axes[2].imshow(pred_hw.astype(np.float32), cmap="RdGy", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Preview segmentation on one GeoTIFF tile.")
    p.add_argument("-e", "--experiment", type=int, default=1)
    p.add_argument("--tile", type=str, required=True, help="Path to 4-band GeoTIFF")
    p.add_argument(
        "--masks",
        type=str,
        default=None,
        help="GeoJSON with polygons (optional; for triptych + metrics)",
    )
    p.add_argument(
        "--band-order",
        choices=("RGBN", "BGRN"),
        default="RGBN",
        help="Source band order in the GeoTIFF (must match training)",
    )
    p.add_argument("-b", "--batch-size", type=int, default=8)
    p.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding-window overlap (same idea as prep-patches)",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=4096,
        help="Center-crop to this max height/width for speed (0 = full tile; needs RAM)",
    )
    p.add_argument(
        "--experiments-path",
        type=str,
        default=paths.PATH_EXPERIMENTS,
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output folder (default: experiments/exp_<e>/predicted)",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    exp_path = os.path.join(args.experiments_path, f"exp_{args.experiment}")
    models_path = os.path.join(exp_path, "models")
    out_dir = args.out_dir or os.path.join(exp_path, "predicted")
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.tile))[0]

    ds = gdal.Open(args.tile, gdalconst.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(args.tile)
    if ds.RasterCount != 4:
        raise ValueError(f"Expected 4 bands, got {ds.RasterCount}")

    print("Loading tile into RAM...")
    bands = ds.ReadAsArray().astype(np.uint8)
    ds = None
    _, fh, fw = bands.shape
    y0, x0, ch, cw = _crop_box(fh, fw, args.max_side)
    idx = _band_reorder_indices(args.band_order)
    bgrn_full = np.stack([bands[i] for i in idx], axis=0)
    bgrn = bgrn_full[:, y0 : y0 + ch, x0 : x0 + cw]
    del bands, bgrn_full
    _, h, w = bgrn.shape
    if args.max_side > 0:
        print(f"Center crop offset=({y0},{x0}) size={h}x{w}")

    ps = general.PATCH_SIZE
    step = max(1, int((1.0 - args.overlap) * ps))
    if h < ps or w < ps:
        raise ValueError(f"Tile/crop {h}x{w} is smaller than patch size {ps}")

    ys = sorted(set(list(range(0, h - ps + 1, step)) + ([h - ps] if (h - ps) % step != 0 else [])))
    xs = sorted(set(list(range(0, w - ps + 1, step)) + ([w - ps] if (w - ps) % step != 0 else [])))
    coords = [(y, x) for y in ys for x in xs]
    print(f"Patches: {len(coords)}  patch_size={ps}  step={step}  overlap={args.overlap}")

    pred_sum = np.zeros((general.N_CLASSES, h, w), dtype=np.float32)
    pred_count = np.zeros((h, w), dtype=np.float32)

    model_m = importlib.import_module(f"conf.model_{args.experiment}")
    model, lidar_bands = model_m.get_model()
    model.to(device)
    model.eval()
    model_path = os.path.join(models_path, "model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded weights: {model_path}")

    t0 = time.perf_counter()
    bs = args.batch_size
    for i in range(0, len(coords), bs):
        batch_coords = coords[i : i + bs]
        opt_b = []
        for y, x in batch_coords:
            patch = bgrn[:, y : y + ps, x : x + ps].astype(np.float32) / 255.0
            opt_b.append(torch.from_numpy(patch))
        opt_t = torch.stack(opt_b, dim=0).to(device)
        lidar_t = torch.zeros(opt_t.shape[0], 1, ps, ps, dtype=torch.float32, device=device)
        with torch.no_grad():
            prob = model((opt_t, lidar_t))
        prob_np = prob.cpu().numpy()
        for k, (y, x) in enumerate(batch_coords):
            pred_sum[:, y : y + ps, x : x + ps] += prob_np[k]
            pred_count[y : y + ps, x : x + ps] += 1.0
    mean_prob = pred_sum / np.maximum(pred_count[np.newaxis, ...], 1e-6)
    pred_hw = np.argmax(mean_prob, axis=0).astype(np.uint8)
    elapsed_min = (time.perf_counter() - t0) / 60.0
    print(f"Inference done in {elapsed_min:.2f} min")

    chw01 = bgrn.astype(np.float32) / 255.0
    rgb_u8 = _rgb_preview_from_bgrn(chw01)

    p_rgb = os.path.join(out_dir, f"preview_{stem}_rgb.png")
    p_pred = os.path.join(out_dir, f"preview_{stem}_pred_class.png")
    p_tri = os.path.join(out_dir, f"preview_{stem}_triptych.png")
    p_over = os.path.join(out_dir, f"preview_{stem}_pred_overlay.png")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.imsave(p_rgb, rgb_u8)
    plt.imsave(p_pred, pred_hw.astype(np.float32), cmap="RdGy", vmin=0, vmax=1)
    _save_pred_overlay(rgb_u8, pred_hw, p_over)
    print(f"Saved:\n  {p_rgb}\n  {p_pred}\n  {p_over}")

    if args.masks:
        print("Rasterizing labels on full tile (then crop)...")
        lbl_full, n_feat = rasterize_geojson_for_tile(args.masks, args.tile)
        lbl = lbl_full[y0 : y0 + h, x0 : x0 + w]
        print(f"Features intersecting tile (OGR count on full path): {n_feat}")
        _save_triptych(rgb_u8, lbl, pred_hw, p_tri)

        pred_flat = torch.from_numpy(pred_hw.ravel()).long()
        tgt_flat = torch.from_numpy(lbl.ravel()).long()
        valid = tgt_flat != general.IGNORE_INDEX
        if valid.any():
            p = pred_flat[valid]
            t = tgt_flat[valid]
            acc = (p == t).float().mean()
            f1_macro = multiclass_f1_score(
                p, t, num_classes=general.N_CLASSES, average="macro"
            )
            f1_per = multiclass_f1_score(
                p, t, num_classes=general.N_CLASSES, average=None
            )
            print(f"Pixel accuracy (valid pixels): {acc:.4f}")
            print(f"Macro F1: {f1_macro:.4f}")
            print(f"Per-class F1 [background, leucaena]: {f1_per.tolist()}")
        else:
            print("No valid label pixels in crop (unexpected).")
    else:
        _save_triptych(rgb_u8, None, pred_hw, p_tri)
    print(f"  {p_tri}")


if __name__ == "__main__":
    main()

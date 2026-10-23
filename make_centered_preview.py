"""
Build a paper-ready RGB | label | prediction triptych centered on a map point.

Example (Docker)::

    python make_centered_preview.py -e 1 \\
        --tile /data/rgbir/SF-23-Y-A-IV-2-SO-D.tif \\
        --masks /data/masks/leucaena-polygons.geojson \\
        --center-x 230170.1 --center-y 7487070.8 \\
        --window-m 900 \\
        --experiments-path /experiments \\
        --out /out/pred_triptych_centered.jpg
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

import numpy as np
import torch
from osgeo import gdal, gdalconst

from conf import general
from utils.ops import rasterize_geojson_for_tile


def _band_reorder_indices(band_order: str) -> list[int]:
    if band_order == "BGRN":
        return [0, 1, 2, 3]
    if band_order == "RGBN":
        return [2, 1, 0, 3]
    raise ValueError(f"Unsupported band order: {band_order}")


def _rgb_preview_from_bgrn(chw: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    b, g, r = chw[0], chw[1], chw[2]
    rgb = np.stack([r, g, b], axis=-1)
    lo = np.quantile(rgb.reshape(-1, 3), alpha, axis=0)
    hi = np.quantile(rgb.reshape(-1, 3), 1.0 - alpha, axis=0)
    out = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (out * 255).astype(np.uint8)


def _window_around_point(
    gt: tuple[float, ...],
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    window_m: float,
) -> tuple[int, int, int, int]:
    """Return (x0, y0, w, h) pixel window clamped to the tile."""
    origin_x, px_w, _, origin_y, _, px_h = gt
    # GDAL geotransform: x = origin_x + col*px_w; y = origin_y + row*px_h (px_h < 0)
    col = int(round((center_x - origin_x) / px_w))
    row = int(round((center_y - origin_y) / px_h))
    half = int(round((window_m / abs(px_w)) / 2.0))
    x0 = max(0, col - half)
    y0 = max(0, row - half)
    x1 = min(width, col + half)
    y1 = min(height, row + half)
    # If we hit a border, expand the opposite side to keep ~window size when possible
    target = 2 * half
    if x1 - x0 < target:
        need = target - (x1 - x0)
        x0 = max(0, x0 - need)
        x1 = min(width, x0 + target)
        x0 = max(0, x1 - target)
    if y1 - y0 < target:
        need = target - (y1 - y0)
        y0 = max(0, y0 - need)
        y1 = min(height, y0 + target)
        y0 = max(0, y1 - target)
    return x0, y0, x1 - x0, y1 - y0


def _overlay(rgb: np.ndarray, mask: np.ndarray, color=(255, 40, 40), alpha: float = 0.55) -> np.ndarray:
    """Tint mask==1 pixels over RGB."""
    out = rgb.astype(np.float32).copy()
    m = mask == 1
    if m.any():
        tint = np.zeros_like(out)
        tint[..., 0], tint[..., 1], tint[..., 2] = color
        out[m] = (1.0 - alpha) * out[m] + alpha * tint[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_triptych(rgb, gt, pred, out_path: str, center_note: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ref_over = _overlay(rgb, gt, color=(40, 220, 80), alpha=0.55)   # green = reference
    pred_over = _overlay(rgb, pred, color=(255, 40, 40), alpha=0.55)  # red = prediction

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.35))
    for ax, img, title in zip(
        axes,
        [rgb, ref_over, pred_over],
        ["RGB", "Reference (green)", "Prediction (red)"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, pad=4)
        ax.axis("off")

    plt.tight_layout(pad=0.3)
    tmp = out_path.rsplit(".", 1)[0] + "_tmp.png"
    fig.savefig(tmp, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)

    img = Image.open(tmp).convert("RGB")
    max_w = 1800
    if img.width > max_w:
        h = int(img.height * max_w / img.width)
        img = img.resize((max_w, h), Image.Resampling.LANCZOS)
    jpg = out_path if out_path.lower().endswith((".jpg", ".jpeg")) else out_path + ".jpg"
    img.save(jpg, "JPEG", quality=92, optimize=True)
    if os.path.isfile(tmp):
        os.remove(tmp)
    print(f"Saved {jpg} ({img.size[0]}x{img.size[1]}) | {center_note}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--experiment", type=int, default=1)
    p.add_argument("--tile", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--center-x", type=float, required=True)
    p.add_argument("--center-y", type=float, required=True)
    p.add_argument("--window-m", type=float, default=900.0, help="Window side length in metres")
    p.add_argument("--band-order", choices=("RGBN", "BGRN"), default="RGBN")
    p.add_argument("-b", "--batch-size", type=int, default=16)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--lidar", type=str, default=None, help="2-band CHM+INTENSITY GeoTIFF (needed for exp 2/4)")
    p.add_argument("--experiments-path", type=str, default="experiments")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds = gdal.Open(args.tile, gdalconst.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(args.tile)
    gt = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    x0, y0, cw, ch = _window_around_point(
        gt, width, height, args.center_x, args.center_y, args.window_m
    )
    print(f"Window px: x0={x0} y0={y0} w={cw} h={ch}  (~{cw * abs(gt[1]):.0f} m x {ch * abs(gt[5]):.0f} m)")
    print(f"Center pixel approx: col={(args.center_x - gt[0]) / gt[1]:.0f} row={(args.center_y - gt[3]) / gt[5]:.0f}")

    # Read only the window (4 bands)
    bands = ds.ReadAsArray(x0, y0, cw, ch).astype(np.uint8)
    ds = None
    idx = _band_reorder_indices(args.band_order)
    bgrn = np.stack([bands[i] for i in idx], axis=0)
    del bands
    _, h, w = bgrn.shape

    # Optional LiDAR window (CHM, INTENSITY), scaled like training
    lidar_chw = None
    if args.lidar:
        lds = gdal.Open(args.lidar, gdalconst.GA_ReadOnly)
        if lds is None:
            raise FileNotFoundError(args.lidar)
        lraw = lds.ReadAsArray(x0, y0, cw, ch).astype(np.float32)
        lds = None
        if lraw.ndim == 2:
            lraw = lraw[np.newaxis, ...]
        # scale: CHM / 50, INTENSITY / 32768
        scaled = np.empty_like(lraw, dtype=np.float32)
        for b in range(lraw.shape[0]):
            denom = float(general.LIDAR_CHM_MAX_M) if b == 0 else float(general.LIDAR_INTENSITY_MAX)
            scaled[b] = np.clip(lraw[b] / denom, 0.0, 1.0)
        lidar_chw = scaled
        print(f"LiDAR window bands={lidar_chw.shape[0]}")

    ps = general.PATCH_SIZE
    step = max(1, int((1.0 - args.overlap) * ps))
    if h < ps or w < ps:
        raise ValueError(f"Window {h}x{w} smaller than patch {ps}")

    ys = sorted(set(list(range(0, h - ps + 1, step)) + ([h - ps] if (h - ps) % step != 0 else [])))
    xs = sorted(set(list(range(0, w - ps + 1, step)) + ([w - ps] if (w - ps) % step != 0 else [])))
    coords = [(y, x) for y in ys for x in xs]
    print(f"Patches: {len(coords)}")

    exp_path = os.path.join(args.experiments_path, f"exp_{args.experiment}")
    model_path = os.path.join(exp_path, "models", "model.pt")
    model_m = importlib.import_module(f"conf.model_{args.experiment}")
    model, lidar_bands = model_m.get_model()
    compute_ndvi = bool(getattr(model_m, "COMPUTE_NDVI", False))
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded {model_path} | NDVI={compute_ndvi} | lidar_bands={lidar_bands}")

    if lidar_bands is not None and lidar_chw is None:
        raise ValueError("This experiment needs --lidar")

    pred_sum = np.zeros((general.N_CLASSES, h, w), dtype=np.float32)
    pred_count = np.zeros((h, w), dtype=np.float32)
    t0 = time.perf_counter()
    bs = args.batch_size
    for i in range(0, len(coords), bs):
        batch_coords = coords[i : i + bs]
        opt_b = []
        lidar_b = []
        for y, x in batch_coords:
            patch = bgrn[:, y : y + ps, x : x + ps].astype(np.float32) / 255.0
            if compute_ndvi:
                # BGRN -> R=2, NIR=3
                red, nir = patch[2], patch[3]
                ndvi = (nir - red) / np.clip(nir + red, 1e-6, None)
                patch = np.concatenate([patch, ndvi[np.newaxis, ...]], axis=0)
            opt_b.append(torch.from_numpy(patch))
            if lidar_chw is not None:
                lp = lidar_chw[:, y : y + ps, x : x + ps]
                lidar_b.append(torch.from_numpy(lp.copy()))
            else:
                lidar_b.append(torch.zeros(1, ps, ps, dtype=torch.float32))
        opt_t = torch.stack(opt_b, dim=0).to(device)
        lidar_t = torch.stack(lidar_b, dim=0).to(device)
        with torch.no_grad():
            prob = model((opt_t, lidar_t))
        prob_np = prob.cpu().numpy()
        for k, (y, x) in enumerate(batch_coords):
            pred_sum[:, y : y + ps, x : x + ps] += prob_np[k]
            pred_count[y : y + ps, x : x + ps] += 1.0
    pred_hw = np.argmax(pred_sum / np.maximum(pred_count[np.newaxis, ...], 1e-6), axis=0).astype(np.uint8)
    print(f"Inference { (time.perf_counter()-t0)/60:.2f} min | leucaena px={(pred_hw==1).sum():,}")

    rgb = _rgb_preview_from_bgrn(bgrn.astype(np.float32) / 255.0)

    print("Rasterizing labels on full tile then cropping...")
    lbl_full, n_feat = rasterize_geojson_for_tile(args.masks, args.tile)
    lbl = lbl_full[y0 : y0 + h, x0 : x0 + w]
    print(f"Features on tile: {n_feat} | leucaena in window={(lbl==1).sum():,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    note = f"Center EPSG:31983 ({args.center_x:.1f}, {args.center_y:.1f}) | window ~{args.window_m:.0f} m"
    _save_triptych(rgb, lbl, pred_hw, args.out, note)


if __name__ == "__main__":
    main()

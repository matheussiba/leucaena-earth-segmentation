"""
Visualise prepared patches as multi-layer panels (no QGIS, no GDAL needed).

For each selected patch it renders one PNG with:

    [ True-colour RGB ]  [ False-colour CIR ]  [ NDVI ]
    [ CHM (LiDAR) ]      [ Refined label ]     [ stats ]

- True-colour RGB     : bands R,G,B (stored order is BGRN -> indices 2,1,0).
- False-colour CIR    : "infravermelho em RGB" = NIR,R,G (indices 3,2,1).
                        Vigorous vegetation glows bright red — the classic way
                        to spot canopy vs. bare ground / man-made surfaces.
- NDVI                : (NIR - RED) / (NIR + RED), computed from the optical
                        patch. The LEUCAENA_NDVI_MIN threshold is drawn as a
                        contour so you can see which pixels pass the rule.
- CHM                 : canopy height (LiDAR band 1). LEUCAENA_CHM_MIN_M drawn
                        as a contour.
- Refined label       : 0=background (black), 1=leucaena (lime),
                        255=IGNORE/outside polygon (grey).

This reads the patch triplets written by ``prep-patches-from-tiles.py``::

    <patches-dir>/opt/<patch_id>.npy     uint8   (H, W, 4)  BGRN
    <patches-dir>/lbl/<patch_id>.npy     uint8   (H, W)     {0,1,255}
    <patches-dir>/lidar/<patch_id>.npy   float32 (H, W, k)  [CHM, INTENSITY]
    <patches-dir>/manifest.csv

Examples
--------
# 10 patches with the most confirmed leucaena, from the refined dataset on disk
python viz-patches.py --patches-dir C:/00_DATASETS_AI/260515-piracicaba-aoi/patches \
    --select most-leucaena --top-k 10 --out-dir viz_patches

# random sample, capped at 3 per tile so it spreads across the scene
python viz-patches.py --select random --top-k 12 --per-tile-cap 3

# only patches from one tile
python viz-patches.py --select tile --tile SF-23-Y-A-IV-2-SE-C --top-k 8
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np

from conf import general, paths

# Band indices inside the stored optical patch (BGRN order).
B_IDX, G_IDX, R_IDX, NIR_IDX = 0, 1, 2, 3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--patches-dir",
        default=str(paths.PATH_PATCHES_DIR),
        help="Folder with opt/, lbl/, lidar/, manifest.csv (default: %(default)s).",
    )
    p.add_argument("--manifest", default=None, help="Manifest CSV (default: <patches-dir>/manifest.csv).")
    p.add_argument("--out-dir", default="viz_patches", help="Where to write the PNG panels (default: %(default)s).")
    p.add_argument(
        "--select",
        choices=("most-leucaena", "hard", "random", "tile"),
        default="most-leucaena",
        help=(
            "Which patches to show. most-leucaena: highest leucaena_fraction. "
            "hard: most 'background-inside-polygon' (refinement acting). "
            "random: random sample. tile: all from --tile. (default: %(default)s)"
        ),
    )
    p.add_argument("--top-k", type=int, default=10, help="How many patches to render (default: %(default)s).")
    p.add_argument("--tile", default=None, help="Tile name to filter on when --select tile.")
    p.add_argument(
        "--per-tile-cap",
        type=int,
        default=None,
        help="Optional: at most N patches per tile, to spread the sample (default: no cap).",
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Restrict to one split (default: all).",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for --select random (default: %(default)s).")
    p.add_argument("--dpi", type=int, default=130, help="PNG resolution (default: %(default)s).")
    return p.parse_args()


def _read_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _select_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    rows = records
    if args.split != "all":
        rows = [r for r in rows if r.get("split") == args.split]

    if args.select == "tile":
        if not args.tile:
            raise SystemExit("--select tile requires --tile <tile_name>.")
        rows = [r for r in rows if r.get("tile_name") == args.tile]
        rows = sorted(rows, key=lambda r: _to_float(r.get("leucaena_fraction")), reverse=True)
    elif args.select == "most-leucaena":
        rows = sorted(rows, key=lambda r: _to_float(r.get("leucaena_fraction")), reverse=True)
    elif args.select == "hard":
        # background-inside-polygon fraction = polygon_fraction - leucaena_fraction.
        # High value = the refinement turned a lot of polygon interior into "not leucaena".
        def bg_in_poly(r: dict) -> float:
            return _to_float(r.get("polygon_fraction"), 1.0) - _to_float(r.get("leucaena_fraction"))

        rows = sorted(rows, key=bg_in_poly, reverse=True)
    elif args.select == "random":
        rows = list(rows)
        random.Random(args.seed).shuffle(rows)

    if args.per_tile_cap:
        capped: list[dict] = []
        seen: dict[str, int] = {}
        for r in rows:
            t = r.get("tile_name", "")
            if seen.get(t, 0) >= args.per_tile_cap:
                continue
            seen[t] = seen.get(t, 0) + 1
            capped.append(r)
        rows = capped

    return rows[: max(0, args.top_k)]


def _load_patch(patches_dir: str, patch_id: str):
    opt = np.load(os.path.join(patches_dir, "opt", f"{patch_id}.npy"))
    lbl = np.load(os.path.join(patches_dir, "lbl", f"{patch_id}.npy"))
    lidar_path = os.path.join(patches_dir, "lidar", f"{patch_id}.npy")
    lidar = np.load(lidar_path) if os.path.isfile(lidar_path) else None
    return opt, lbl, lidar


def _stretch(channel: np.ndarray) -> np.ndarray:
    """Percentile contrast stretch to [0,1] so faint patches are still readable."""
    c = channel.astype(np.float32)
    lo, hi = np.percentile(c, 2), np.percentile(c, 98)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((c - lo) / (hi - lo), 0.0, 1.0)


def _rgb(opt: np.ndarray) -> np.ndarray:
    return np.dstack([_stretch(opt[:, :, R_IDX]), _stretch(opt[:, :, G_IDX]), _stretch(opt[:, :, B_IDX])])


def _cir(opt: np.ndarray) -> np.ndarray:
    # NIR -> red channel, R -> green, G -> blue. Vegetation = bright red.
    return np.dstack([_stretch(opt[:, :, NIR_IDX]), _stretch(opt[:, :, R_IDX]), _stretch(opt[:, :, G_IDX])])


def _ndvi(opt: np.ndarray) -> np.ndarray:
    red = opt[:, :, R_IDX].astype(np.float32)
    nir = opt[:, :, NIR_IDX].astype(np.float32)
    denom = nir + red
    denom = np.where(denom == 0.0, 1e-6, denom)
    ndvi = (nir - red) / denom
    ndvi[~np.isfinite(ndvi)] = 0.0
    return ndvi


def _render(out_path: str, rec: dict, opt, lbl, lidar, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    patch_id = rec["patch_id"]
    ndvi = _ndvi(opt)
    chm = lidar[:, :, 0] if (lidar is not None and lidar.ndim == 3 and lidar.shape[-1] > 0) else None

    ignore = general.IGNORE_INDEX
    in_poly = lbl != ignore
    is_leuc = lbl == 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"{patch_id}   |   tile={rec.get('tile_name','?')}   split={rec.get('split','?')}",
        fontsize=12,
    )

    # --- Row 1 -----------------------------------------------------------
    axes[0, 0].imshow(_rgb(opt))
    axes[0, 0].set_title("True-colour RGB (R,G,B)")

    axes[0, 1].imshow(_cir(opt))
    axes[0, 1].set_title("False-colour CIR (NIR,R,G) — vegetação = vermelho")

    im_ndvi = axes[0, 2].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes[0, 2].set_title(f"NDVI (linha = limiar {general.LEUCAENA_NDVI_MIN})")
    axes[0, 2].contour(ndvi, levels=[general.LEUCAENA_NDVI_MIN], colors="black", linewidths=0.8)
    fig.colorbar(im_ndvi, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # --- Row 2 -----------------------------------------------------------
    if chm is None:
        axes[1, 0].imshow(np.zeros_like(ndvi), cmap="gray", vmin=0, vmax=1)
        axes[1, 0].set_title("CHM ausente (patch sem LiDAR)")
    else:
        vmax = max(float(np.percentile(chm, 99)), 10.0)
        im_chm = axes[1, 0].imshow(chm, cmap="viridis", vmin=0, vmax=vmax)
        axes[1, 0].set_title(f"CHM — altura (linha = limiar {general.LEUCAENA_CHM_MIN_M} m)")
        axes[1, 0].contour(chm, levels=[general.LEUCAENA_CHM_MIN_M], colors="red", linewidths=0.8)
        fig.colorbar(im_chm, ax=axes[1, 0], fraction=0.046, pad=0.04)

    label_cmap = ListedColormap(["black", "lime", "lightgray"])
    label_show = np.where(lbl == ignore, 2, lbl)
    axes[1, 1].imshow(label_show, cmap=label_cmap, vmin=0, vmax=2)
    axes[1, 1].set_title("Label refinado (preto=0, verde=leucaena, cinza=IGNORE)")

    # --- Stats panel -----------------------------------------------------
    axes[1, 2].axis("off")
    n_poly = int(in_poly.sum())
    n_leuc = int(is_leuc.sum())
    total = lbl.size
    lines = [
        "Resumo do patch",
        "",
        f"leucaena_fraction : {_to_float(rec.get('leucaena_fraction')):.3f}",
        f"polygon_fraction  : {_to_float(rec.get('polygon_fraction'), 1.0):.3f}",
        f"has_lidar         : {rec.get('has_lidar','?')}",
        "",
        f"px no polígono    : {n_poly:,} ({100*n_poly/total:.1f}%)",
        f"px leucaena (=1)  : {n_leuc:,} ({100*n_leuc/total:.1f}%)",
        f"px bg no polígono : {n_poly-n_leuc:,}",
    ]
    if n_poly > 0:
        lines.append(f"  -> {100*n_leuc/n_poly:.1f}% do polígono virou leucaena")
    if chm is not None and n_leuc > 0:
        lines += [
            "",
            f"CHM na leucaena   : {chm[is_leuc].mean():.1f} m (méd), {chm[is_leuc].max():.1f} m (máx)",
            f"NDVI na leucaena  : {ndvi[is_leuc].mean():.2f} (méd)",
        ]
    axes[1, 2].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest or os.path.join(args.patches_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"Manifest not found: {manifest_path}")

    records = _read_manifest(manifest_path)
    selected = _select_records(records, args)
    if not selected:
        raise SystemExit("No patches matched the selection.")

    print(f"Patches dir : {args.patches_dir}")
    print(f"Manifest    : {manifest_path}  ({len(records):,} rows)")
    print(f"Selection   : {args.select} | top_k={args.top_k} | split={args.split}")
    print(f"Out dir     : {args.out_dir}")
    print(f"Rendering {len(selected)} patch(es)...\n")

    os.makedirs(args.out_dir, exist_ok=True)
    for i, rec in enumerate(selected, 1):
        patch_id = rec["patch_id"]
        opt, lbl, lidar = _load_patch(args.patches_dir, patch_id)
        out_path = os.path.join(args.out_dir, f"{i:02d}_{patch_id}.png")
        _render(out_path, rec, opt, lbl, lidar, args.dpi)
        leuc = _to_float(rec.get("leucaena_fraction"))
        print(f"  [{i:02d}] {patch_id}  leucaena={leuc:.3f}  -> {out_path}")

    print(f"\nDone. {len(selected)} PNG(s) in {args.out_dir}/")


if __name__ == "__main__":
    main()

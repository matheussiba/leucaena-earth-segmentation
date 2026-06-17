"""
Compara label com CHM=4.5m (atual) vs CHM=3.0m (proposto) nas patches existentes.

Não toca nenhum arquivo — recalcula o label na hora a partir do CHM bruto
já armazenado nas patches de lidar.

Uso (dentro do container):
    python viz-chm-threshold.py
    python viz-chm-threshold.py --top-k 5 --patches-dir /data/patches
    python viz-chm-threshold.py --chm-new 3.0 --select hard
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from conf import general, paths

B, G, R, NIR = 0, 1, 2, 3
CHM_MAX_M = general.LIDAR_CHM_MAX_M   # 50.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patches-dir", default=str(paths.PATH_PATCHES_DIR))
    p.add_argument("--out-dir", default="viz_chm_compare")
    p.add_argument("--top-k", type=int, default=4, help="Número de patches a comparar (default: 4)")
    p.add_argument("--chm-new", type=float, default=3.0, help="Novo limiar CHM em metros (default: 3.0)")
    p.add_argument("--ndvi-min", type=float, default=general.LEUCAENA_NDVI_MIN)
    p.add_argument(
        "--select",
        choices=("hard", "most-leucaena"),
        default="hard",
        help="hard = onde o filtro atuou mais (melhor pra ver diferença). "
             "most-leucaena = patches com mais leucaena no label atual. (default: hard)",
    )
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def _load_manifest(patches_dir: str) -> list[dict]:
    with open(os.path.join(patches_dir, "manifest.csv"), newline="") as f:
        return list(csv.DictReader(f))


def _select(records: list[dict], select: str, top_k: int) -> list[dict]:
    def _f(r: dict, key: str) -> float:
        try:
            return float(r.get(key, 0))
        except ValueError:
            return 0.0

    if select == "hard":
        # patches where refinement turned the most polygon area into background
        rows = sorted(records, key=lambda r: _f(r, "polygon_fraction") - _f(r, "leucaena_fraction"), reverse=True)
    else:
        rows = sorted(records, key=lambda r: _f(r, "leucaena_fraction"), reverse=True)

    # filter to patches that actually have lidar
    rows = [r for r in rows if r.get("has_lidar", "").lower() in ("true", "1")]
    return rows[:top_k]


def _label_color(label: np.ndarray) -> np.ndarray:
    """Map label {0,1,255} → RGB image for display."""
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    rgb[label == 255] = [128, 128, 128]  # grey  = IGNORE (outside polygon)
    rgb[label == 0]   = [30,  30,  30]   # dark  = background inside polygon
    rgb[label == 1]   = [0,   220,  60]  # green = leucaena
    return rgb


def _compute_new_label(
    label_old: np.ndarray,
    lidar: np.ndarray,
    opt: np.ndarray,
    chm_min: float,
    ndvi_min: float,
) -> np.ndarray:
    """Recompute label with new CHM threshold from raw patch arrays."""
    # CHM stored as float32 in [0,1]; convert back to metres
    chm_m = lidar[:, :, 0] * CHM_MAX_M

    opt_f = opt.astype(np.float32) / 255.0
    denom = opt_f[:, :, NIR] + opt_f[:, :, R]
    denom = np.where(denom == 0, 1e-6, denom)
    ndvi = np.clip((opt_f[:, :, NIR] - opt_f[:, :, R]) / denom, -1.0, 1.0)

    in_poly   = label_old != 255
    is_tall   = chm_m >= chm_min
    is_veg    = ndvi >= ndvi_min

    new_label = np.full(label_old.shape, 255, dtype=np.uint8)
    new_label[in_poly] = 0
    new_label[in_poly & is_tall & is_veg] = 1
    return new_label


def _pixel_counts(label: np.ndarray) -> str:
    n_ignore = int((label == 255).sum())
    n_bg     = int((label == 0).sum())
    n_leuc   = int((label == 1).sum())
    n_poly   = n_bg + n_leuc
    frac = n_leuc / n_poly if n_poly else 0
    return f"leucaena={n_leuc:,} ({frac:.1%})\nbg={n_bg:,} | ignore={n_ignore:,}"


def render_patch(
    patch_id: str,
    patches_dir: str,
    chm_new: float,
    ndvi_min: float,
    dpi: int,
    out_dir: str,
) -> None:
    opt   = np.load(os.path.join(patches_dir, "opt",   f"{patch_id}.npy"))
    label = np.load(os.path.join(patches_dir, "lbl",   f"{patch_id}.npy"))
    lidar = np.load(os.path.join(patches_dir, "lidar", f"{patch_id}.npy"))

    label_new = _compute_new_label(label, lidar, opt, chm_new, ndvi_min)

    # RGB (true colour): stored BGRN → show R,G,B
    rgb = opt[:, :, [R, G, B]]
    # CIR false colour (infravermelho): NIR,R,G → vegetação saudável fica VERMELHA
    cir = opt[:, :, [NIR, R, G]]

    # NDVI from the optical patch
    opt_f = opt.astype(np.float32) / 255.0
    denom = opt_f[:, :, NIR] + opt_f[:, :, R]
    denom = np.where(denom == 0, 1e-6, denom)
    ndvi = np.clip((opt_f[:, :, NIR] - opt_f[:, :, R]) / denom, -1.0, 1.0)

    # CHM in metres for display
    chm_m = lidar[:, :, 0] * CHM_MAX_M

    fig, axes = plt.subplots(1, 6, figsize=(22, 4.2))
    fig.suptitle(f"Patch: {patch_id}", fontsize=10)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB (true colour)", fontsize=9)

    axes[1].imshow(cir)
    axes[1].set_title("CIR infravermelho\n(vegetação = vermelho)", fontsize=9)

    im2 = axes[2].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[2].set_title(f"NDVI (limiar ≥{ndvi_min})", fontsize=9)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[2].contour(ndvi, levels=[ndvi_min], colors="black", linewidths=1.2)

    im3 = axes[3].imshow(chm_m, cmap="YlGn", vmin=0, vmax=20)
    axes[3].set_title("CHM (metros)", fontsize=9)
    fig.colorbar(im3, ax=axes[3], fraction=0.046)
    axes[3].contour(chm_m, levels=[general.LEUCAENA_CHM_MIN_M], colors="red", linewidths=1.0, linestyles="--")

    axes[4].imshow(_label_color(label))
    axes[4].set_title(f"Label ATUAL (CHM≥{general.LEUCAENA_CHM_MIN_M}m)\n{_pixel_counts(label)}", fontsize=8)

    axes[5].imshow(_label_color(label_new))
    axes[5].set_title(f"Label NOVO (CHM≥{chm_new}m)\n{_pixel_counts(label_new)}", fontsize=8)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{patch_id}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    # --- Decomposição: quem passa em cada filtro (só dentro do polígono) ---
    in_poly = label != 255
    n_poly = int(in_poly.sum())
    if n_poly:
        pass_chm  = int(((chm_m >= general.LEUCAENA_CHM_MIN_M) & in_poly).sum())
        pass_ndvi = int(((ndvi >= ndvi_min) & in_poly).sum())
        pass_both = int(((chm_m >= general.LEUCAENA_CHM_MIN_M) & (ndvi >= ndvi_min) & in_poly).sum())
        print(f"    dentro do poligono: {n_poly:,} px")
        print(f"      passa CHM≥{general.LEUCAENA_CHM_MIN_M}m : {pass_chm:,} ({pass_chm/n_poly:.0%})")
        print(f"      passa NDVI≥{ndvi_min}  : {pass_ndvi:,} ({pass_ndvi/n_poly:.0%})   <- gargalo se for baixo")
        print(f"      passa AMBOS (=leucaena): {pass_both:,} ({pass_both/n_poly:.0%})")
        print(f"      NDVI mediano no poligono: {float(np.median(ndvi[in_poly])):.3f}")


def main() -> None:
    args = _parse_args()
    records = _load_manifest(args.patches_dir)
    selected = _select(records, args.select, args.top_k)

    if not selected:
        print("Nenhum patch encontrado. Verifique --patches-dir e se has_lidar=True.")
        return

    print(f"Comparando CHM≥{general.LEUCAENA_CHM_MIN_M}m (atual) vs CHM≥{args.chm_new}m (novo)")
    print(f"Patches: {len(selected)} | Saída: {args.out_dir}/\n")

    for rec in selected:
        pid = rec["patch_id"]
        print(f"  {pid}")
        render_patch(pid, args.patches_dir, args.chm_new, args.ndvi_min, args.dpi, args.out_dir)

    print(f"\nPronto! Abra os PNGs em {args.out_dir}/")
    print("Cores: cinza=IGNORE | preto=background | verde=leucaena")
    print("Linha vermelha=limiar atual (4.5m) | linha verde=limiar novo (3.0m) no CHM")


if __name__ == "__main__":
    main()

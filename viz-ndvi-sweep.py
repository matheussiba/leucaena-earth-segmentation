"""
Varredura de limiar de NDVI — mostra como o label muda em vários thresholds.

A ordem das bandas está correta (banda3=NIR), mas o NDVI desta imagem é
"comprimido" (vegetação ~0.2-0.36). O limiar 0.3 corta no meio da copa e
gera o efeito sal-e-pimenta. Este script mostra o label do patch com a
altura fixada em >=4m (critério biológico) e o NDVI varrido em vários
valores, pra você escolher o limiar que deixa a copa contínua.

Não toca nenhum arquivo.

Uso (dentro do container):
    python viz-ndvi-sweep.py --patches-dir /data/patches --top-k 4
    python viz-ndvi-sweep.py --chm-min 4.0 --thresholds 0.30 0.25 0.20 0.15 0.10
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from conf import general, paths

B, G, R, NIR = 0, 1, 2, 3
CHM_MAX_M = general.LIDAR_CHM_MAX_M


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patches-dir", default=str(paths.PATH_PATCHES_DIR))
    p.add_argument("--out-dir", default="viz_ndvi_sweep")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--chm-min", type=float, default=4.0, help="Altura mínima fixa em metros (default: 4.0)")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.25, 0.20, 0.15, 0.10],
                   help="Limiares de NDVI a testar (default: 0.30 0.25 0.20 0.15 0.10)")
    p.add_argument(
        "--select", choices=("most-leucaena", "hard"), default="most-leucaena",
        help="most-leucaena = patches com copa densa (melhor pra ver). (default: most-leucaena)",
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
        rows = sorted(records, key=lambda r: _f(r, "polygon_fraction") - _f(r, "leucaena_fraction"), reverse=True)
    else:
        rows = sorted(records, key=lambda r: _f(r, "leucaena_fraction"), reverse=True)
    rows = [r for r in rows if r.get("has_lidar", "").lower() in ("true", "1")]
    return rows[:top_k]


def _label_color(label: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    rgb[label == 255] = [128, 128, 128]
    rgb[label == 0]   = [30, 30, 30]
    rgb[label == 1]   = [0, 220, 60]
    return rgb


def render_patch(pid: str, patches_dir: str, chm_min: float, thresholds: list[float], dpi: int, out_dir: str) -> None:
    opt   = np.load(os.path.join(patches_dir, "opt",   f"{pid}.npy"))
    label = np.load(os.path.join(patches_dir, "lbl",   f"{pid}.npy"))
    lidar = np.load(os.path.join(patches_dir, "lidar", f"{pid}.npy"))

    opt_f = opt.astype(np.float32) / 255.0
    denom = opt_f[:, :, NIR] + opt_f[:, :, R]
    denom = np.where(denom == 0, 1e-6, denom)
    ndvi = np.clip((opt_f[:, :, NIR] - opt_f[:, :, R]) / denom, -1.0, 1.0)
    chm_m = lidar[:, :, 0] * CHM_MAX_M

    in_poly = label != 255
    is_tall = chm_m >= chm_min
    n_poly = int(in_poly.sum())

    n_panels = 1 + len(thresholds)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 4.0))
    fig.suptitle(f"Patch: {pid}  |  altura fixa >={chm_min}m", fontsize=10)

    axes[0].imshow(opt[:, :, [R, G, B]])
    axes[0].set_title("RGB", fontsize=9)

    print(f"  {pid}")
    for i, thr in enumerate(thresholds, start=1):
        new_label = np.full(label.shape, 255, dtype=np.uint8)
        new_label[in_poly] = 0
        new_label[in_poly & is_tall & (ndvi >= thr)] = 1
        n_leuc = int((new_label == 1).sum())
        frac = n_leuc / n_poly if n_poly else 0
        axes[i].imshow(_label_color(new_label))
        axes[i].set_title(f"NDVI >= {thr:.2f}\nleucaena {frac:.0%}", fontsize=9)
        print(f"    NDVI>={thr:.2f}: leucaena {n_leuc:,} px ({frac:.0%})")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pid}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {out_path}")


def main() -> None:
    args = _parse_args()
    records = _load_manifest(args.patches_dir)
    selected = _select(records, args.select, args.top_k)
    if not selected:
        print("Nenhum patch com lidar encontrado.")
        return

    print(f"Altura fixa >={args.chm_min}m | varrendo NDVI: {args.thresholds}")
    print(f"Patches: {len(selected)} | Saída: {args.out_dir}/\n")
    for rec in selected:
        render_patch(rec["patch_id"], args.patches_dir, args.chm_min, args.thresholds, args.dpi, args.out_dir)

    print(f"\nPronto! Abra os PNGs em {args.out_dir}/")
    print("Escolha o limiar onde a copa fica CONTÍNUA (sem sal-e-pimenta).")


if __name__ == "__main__":
    main()

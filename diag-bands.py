"""
Diagnóstico de ordem de bandas — descobre qual banda é o NIR de verdade.

As patches são salvas em ordem BGRN (B=0, G=1, R=2, NIR=3). Se o NDVI sai
baixo (~0.2) sobre vegetação densa, provavelmente Red e NIR estão trocados.

Este script lê patches existentes e, para cada candidato a NIR, calcula o
NDVI mediano sobre a área de vegetação. O candidato que der NDVI ALTO é o
NIR verdadeiro. Não toca nenhum arquivo.

Uso (dentro do container):
    python diag-bands.py --patches-dir /data/patches --top-k 5
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from conf import paths

BAND_NAMES = ["banda0 (B?)", "banda1 (G?)", "banda2 (R?)", "banda3 (NIR?)"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patches-dir", default=str(paths.PATH_PATCHES_DIR))
    p.add_argument("--top-k", type=int, default=5, help="Quantos patches analisar (default: 5)")
    return p.parse_args()


def _load_manifest(patches_dir: str) -> list[dict]:
    with open(os.path.join(patches_dir, "manifest.csv"), newline="") as f:
        return list(csv.DictReader(f))


def _select(records: list[dict], top_k: int) -> list[dict]:
    def _f(r: dict, key: str) -> float:
        try:
            return float(r.get(key, 0))
        except ValueError:
            return 0.0
    rows = sorted(records, key=lambda r: _f(r, "leucaena_fraction"), reverse=True)
    rows = [r for r in rows if r.get("has_lidar", "").lower() in ("true", "1")]
    return rows[:top_k]


def _ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    denom = np.where(denom == 0, 1e-6, denom)
    return np.clip((nir - red) / denom, -1.0, 1.0)


def main() -> None:
    args = _parse_args()
    records = _load_manifest(args.patches_dir)
    selected = _select(records, args.top_k)
    if not selected:
        print("Nenhum patch com lidar encontrado.")
        return

    # Accumulate band means and NDVI medians across patches (inside polygon only)
    band_sums = np.zeros(4, dtype=np.float64)
    band_px = 0
    # NDVI median per (nir_candidate, red_candidate) pair — we fix red=band2 and vary nir,
    # plus the reverse (red=band3) to catch a straight swap.
    print(f"Analisando {len(selected)} patches (área dentro do polígono)\n")
    print("=" * 70)

    ndvi_medians: dict[str, list[float]] = {}

    for rec in selected:
        pid = rec["patch_id"]
        opt = np.load(os.path.join(args.patches_dir, "opt", f"{pid}.npy"))
        lbl = np.load(os.path.join(args.patches_dir, "lbl", f"{pid}.npy"))
        in_poly = lbl != 255
        if not in_poly.any():
            continue

        # Per-band mean inside polygon
        for b in range(4):
            band_sums[b] += float(opt[:, :, b][in_poly].sum())
        band_px += int(in_poly.sum())

        # Test NDVI for each NIR candidate, using band2 as RED (the assumed red)
        for nir_b in range(4):
            if nir_b == 2:
                continue
            key = f"NIR=banda{nir_b} vs RED=banda2"
            nd = _ndvi(opt[:, :, nir_b], opt[:, :, 2])
            ndvi_medians.setdefault(key, []).append(float(np.median(nd[in_poly])))

        # Also test the straight swap: assume real order has NIR where we think RED is
        nd_swap = _ndvi(opt[:, :, 2], opt[:, :, 3])  # NIR=banda2, RED=banda3
        ndvi_medians.setdefault("NIR=banda2 vs RED=banda3 (swap)", []).append(
            float(np.median(nd_swap[in_poly]))
        )

    print("\nMÉDIA DE CADA BANDA (dentro do polígono):")
    print("  (o NIR verdadeiro costuma ser a banda MAIS BRILHANTE sobre vegetação)")
    for b in range(4):
        print(f"    {BAND_NAMES[b]:<16}: {band_sums[b] / band_px:7.1f}")

    print("\nNDVI MEDIANO por combinação testada:")
    print("  (vegetação saudável → NDVI alto, 0.6–0.9. O combo certo é o de NDVI ALTO)")
    for key, vals in sorted(ndvi_medians.items(), key=lambda kv: -np.mean(kv[1])):
        mean_v = float(np.mean(vals))
        flag = "  <== NIR VERDADEIRO" if mean_v > 0.45 else ""
        print(f"    {key:<35}: {mean_v:+.3f}{flag}")

    print("=" * 70)
    print("\nLeitura:")
    print("  - Atual (código): NIR=banda3 vs RED=banda2.")
    print("  - Se OUTRO combo der NDVI bem mais alto, as bandas estão trocadas na origem")
    print("    e o flag --band-order do prep-patches está errado.")


if __name__ == "__main__":
    main()

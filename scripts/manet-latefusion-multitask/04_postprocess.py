# -*- coding: utf-8 -*-
"""
postprocess_dist_map.py
=======================
Pós-processamento do mapa de distância predito (pred_dist.tif) para gerar uma
máscara binária dos maciços:

  1. filtro Gaussiano com sigma e janela ajustáveis (px);
  2. binarização pelo limiar escolhido (filtrado >= limiar → 1);
  3. rotulação dos objetos (componentes conexos) e exclusão, via regionprops,
     dos objetos com menos de --min_size pixels.

Pixels NaN (nodata) do mapa de entrada entram como 0 no filtro e ficam 0 na
máscara. Georreferência e CRS são preservados nas saídas.

Saídas (na pasta --out_dir; o nome leva os parâmetros, então rodadas com
parâmetros diferentes não se sobrescrevem):
  <nome>_gauss_s<sigma>_w<janela>.tif           mapa filtrado (float32)
  <nome>_bin_t<limiar>_min<min_size>.tif        máscara final (uint8, 0/1)

Uso (PowerShell):
    python postprocess_dist_map.py
    python postprocess_dist_map.py --input ...\\pred_dist.tif --sigma 3 --window 15 ^
        --threshold 0.5 --min_size 500
"""
import argparse
import os
import time

# o PROJ_LIB desta máquina aponta p/ um proj.db de outra versão; sem isso o
# rasterio não grava o CRS (mesmo cuidado dos outros scripts)
for _v in ("PROJ_LIB", "PROJ_DATA"):
    os.environ.pop(_v, None)

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops

ENTRADA = (r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\codes\multitask_PAN_MULTI_FUSION_without_background_macicos_only_with_dist_map\prediction\pred_dist.tif")


def area_pixel_m2(src):
    """Área de um pixel em m² (aproximada na latitude central se o CRS for geográfico)."""
    px, py = abs(src.transform.a), abs(src.transform.e)
    if src.crs is not None and src.crs.is_geographic:
        lat = np.deg2rad(src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2)
        return px * 111320 * np.cos(lat) * py * 110574
    return px * py


def _fmt(v):
    """Número curto p/ nome de arquivo: 0.5 → '0.5', 2.0 → '2'."""
    return f"{v:g}"


def main():
    p = argparse.ArgumentParser(description="Filtro Gaussiano + limiar + remoção de "
                                            "objetos pequenos sobre o mapa de distância.")
    p.add_argument("--input", default=ENTRADA, help="Mapa de distância (GeoTIFF, 1 banda).")
    p.add_argument("--out_dir", default=None,
                   help="Pasta de saída. Padrão: a mesma do --input.")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="Sigma do filtro Gaussiano, em pixels. 0 = sem filtro.")
    p.add_argument("--window", type=int, default=None,
                   help="Tamanho da janela do filtro (px, ímpar; ex.: 9 = 9x9). "
                        "Padrão: 2*ceil(4*sigma)+1 (o padrão do scipy).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Limiar sobre o mapa filtrado: >= limiar vira 1.")
    p.add_argument("--min_size", type=int, default=100,
                   help="Objetos com MENOS que este nº de pixels são removidos.")
    p.add_argument("--connectivity", type=int, default=2, choices=[1, 2],
                   help="Vizinhança dos objetos: 1 = 4-vizinhos, 2 = 8-vizinhos (padrão).")
    p.add_argument("--save_filtered", type=lambda v: str(v).lower() in ("1", "true", "t", "sim", "s", "yes"),
                   default=True, help="Salva também o mapa filtrado (float32).")
    a = p.parse_args()

    if a.sigma < 0:
        raise SystemExit("--sigma deve ser >= 0.")
    if a.window is not None:
        if a.window < 1:
            raise SystemExit("--window deve ser >= 1.")
        if a.window % 2 == 0:
            print(f"⚠️  --window {a.window} é par; usando {a.window + 1} (janela centrada).")
            a.window += 1
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.input))
    os.makedirs(out_dir, exist_ok=True)
    nome = os.path.splitext(os.path.basename(a.input))[0]
    t0 = time.time()

    # ── 1. leitura ─────────────────────────────────────────────
    with rasterio.open(a.input) as src:
        perfil = src.profile.copy()
        dist = src.read(1).astype(np.float32, copy=False)
        a_px = area_pixel_m2(src)
    nan = ~np.isfinite(dist)
    dist[nan] = 0.0
    print(f"Entrada: {a.input}")
    print(f"  {dist.shape[1]} x {dist.shape[0]} px | pixel ≈ {a_px:.2f} m² | "
          f"NaN: {100 * nan.mean():.2f}% (tratados como 0)")

    # ── 2. filtro Gaussiano ────────────────────────────────────
    if a.sigma > 0:
        raio = a.window // 2 if a.window is not None else int(np.ceil(4 * a.sigma))
        janela = 2 * raio + 1
        print(f"Filtro Gaussiano: sigma = {_fmt(a.sigma)} px | janela = {janela} x {janela} px")
        filt = gaussian_filter(dist, sigma=a.sigma, radius=raio, mode="nearest")
    else:
        janela = 1
        print("Filtro Gaussiano: desligado (sigma = 0)")
        filt = dist
    del dist

    if a.save_filtered:
        caminho_f = os.path.join(out_dir, f"{nome}_gauss_s{_fmt(a.sigma)}_w{janela}.tif")
        pf = perfil.copy()
        pf.update(dtype="float32", count=1, nodata=np.nan, compress="deflate",
                  tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        saida = filt.copy()
        saida[nan] = np.nan
        with rasterio.open(caminho_f, "w", **pf) as dst:
            dst.write(saida, 1)
        del saida
        print(f"  salvo: {caminho_f}")

    # ── 3. limiar → binário ────────────────────────────────────
    binario = filt >= a.threshold
    binario[nan] = False
    del filt
    print(f"Limiar >= {_fmt(a.threshold)}: {int(binario.sum()):,} px "
          f"({100 * binario.mean():.2f}% da imagem)")

    # ── 4. objetos pequenos fora (regionprops) ─────────────────
    rot = label(binario, connectivity=a.connectivity)
    props = regionprops(rot)
    manter = np.zeros(rot.max() + 1, dtype=bool)
    areas = np.array([r.area for r in props], dtype=np.int64)
    rotulos = np.array([r.label for r in props], dtype=np.int64)
    grandes = areas >= a.min_size
    manter[rotulos[grandes]] = True
    final = manter[rot]
    del rot
    n_rem, px_rem = int((~grandes).sum()), int(areas[~grandes].sum())
    print(f"Objetos ({'8' if a.connectivity == 2 else '4'}-vizinhos): {len(props):,} | "
          f"removidos (< {a.min_size} px ≈ {a.min_size * a_px / 1e4:.3f} ha): {n_rem:,} "
          f"({px_rem:,} px) | mantidos: {int(grandes.sum()):,}")
    if grandes.any():
        ar = areas[grandes] * a_px / 1e4
        print(f"  área dos mantidos (ha): min {ar.min():.2f} | mediana {np.median(ar):.2f} | "
              f"max {ar.max():.2f} | total {ar.sum():.1f}")

    # ── 5. máscara final ───────────────────────────────────────
    caminho_b = os.path.join(out_dir, f"{nome}_bin_t{_fmt(a.threshold)}_min{a.min_size}.tif")
    pb = perfil.copy()
    pb.update(dtype="uint8", count=1, nodata=None, compress="deflate",
              tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(caminho_b, "w", **pb) as dst:
        dst.write(final.astype(np.uint8), 1)
    print(f"  salvo: {caminho_b}")
    print(f"Concluído em {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()

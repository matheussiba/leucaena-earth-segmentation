# =============================================================================
# build-overviews-qgis.py
#
# Cria piramides / overviews externos (.tif.ovr) para todos os GeoTIFFs
# de uma pasta. O .ovr fica ao lado do .tif e NAO altera o raster original.
#
# COMO USAR NO QGIS:
#   1. Edite TIF_FOLDER abaixo
#   2. No QGIS: Plugins -> Python Console
#   3. Clique no icone "Show Editor" (folha de papel)
#   4. Abra este arquivo OU cole o codigo
#   5. Clique em Run (botao play)
#
# Tambem funciona fora do QGIS, se o ambiente tiver GDAL (osgeo).
#
# Niveis: 2 4 8 16 32 64
# =============================================================================

from pathlib import Path
from time import perf_counter

from osgeo import gdal

# >>> EDITE SO ESTA LINHA <<<
TIF_FOLDER = r"C:\00_DATASETS_AI\260515-piracicaba-aoi\opt\rgbnir\deleteme"

# True = inclui subpastas | False = so a pasta raiz
RECURSE = False

# True = pula se ja existir .ovr | False = refaz
SKIP_EXISTING = True

OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64]


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {secs:04.1f}s"


def build_overview(tif_path: Path) -> None:
    gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
    gdal.SetConfigOption("PREDICTOR_OVERVIEW", "2")
    gdal.SetConfigOption("BIGTIFF_OVERVIEW", "IF_SAFER")
    gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", "512")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

    # Abrir em modo leitura => overview EXTERNO (.tif.ovr)
    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Nao foi possivel abrir: {tif_path}")

    result = ds.BuildOverviews("AVERAGE", OVERVIEW_LEVELS)
    ds = None  # fecha e libera o arquivo

    if result != 0:
        raise RuntimeError(f"BuildOverviews falhou (codigo {result}): {tif_path}")


def main() -> None:
    folder = Path(TIF_FOLDER)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Pasta nao encontrada: {folder}\n"
            "Edite TIF_FOLDER no inicio deste script."
        )

    pattern = "**/*.tif" if RECURSE else "*.tif"
    files = sorted(folder.glob(pattern))
    # Em alguns Windows os GeoTIFF vem como .TIF
    if RECURSE:
        files += sorted(folder.glob("**/*.TIF"))
    else:
        files += sorted(folder.glob("*.TIF"))
    # remove duplicatas preservando ordem
    seen = set()
    unique = []
    for f in files:
        key = str(f.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    files = unique

    total = len(files)
    print("")
    print("=" * 60)
    print("INICIO — geracao de overviews (.tif.ovr)")
    print("=" * 60)
    print(f"Pasta:    {folder}")
    print(f"Tiles:    {total}")
    print(f"Niveis:   {OVERVIEW_LEVELS}")
    print(f"Recurse:  {RECURSE}")
    print(f"Pular .ovr existente: {SKIP_EXISTING}")
    print("=" * 60)
    print("")

    if total == 0:
        print("Nenhum .tif encontrado. Nada a fazer.")
        print("FIM.")
        return

    ok = skip = fail = 0
    processed_times = []
    t0_all = perf_counter()

    for i, tif in enumerate(files, start=1):
        prefix = f"[{i}/{total}]"
        ovr = Path(str(tif) + ".ovr")

        if SKIP_EXISTING and ovr.exists():
            print(f"{prefix} PULADO (ja existe .ovr) — {tif.name}")
            skip += 1
            continue

        print(f"{prefix} PROCESSANDO — {tif.name}")
        t0 = perf_counter()
        try:
            build_overview(tif)
            elapsed = perf_counter() - t0
            processed_times.append(elapsed)
            ok += 1
            print(f"{prefix} CONCLUIDO — {tif.name}  |  tempo: {format_duration(elapsed)}")
        except Exception as exc:
            elapsed = perf_counter() - t0
            fail += 1
            print(
                f"{prefix} ERRO — {tif.name}  |  tempo: {format_duration(elapsed)}  |  {exc}"
            )

    total_elapsed = perf_counter() - t0_all
    avg = (sum(processed_times) / len(processed_times)) if processed_times else 0.0

    print("")
    print("=" * 60)
    print("FIM — processamento concluido")
    print("=" * 60)
    print(f"Tiles no total:     {total}")
    print(f"Gerados com sucesso:{ok:>4}")
    print(f"Pulados:            {skip:>4}")
    print(f"Com erro:           {fail:>4}")
    if processed_times:
        print(f"Tempo medio/tile:   {format_duration(avg)}")
        print(f"Tile mais rapido:   {format_duration(min(processed_times))}")
        print(f"Tile mais lento:    {format_duration(max(processed_times))}")
    print(f"Tempo TOTAL:         {format_duration(total_elapsed)}")
    print("=" * 60)
    print("Pode fechar / parar aqui. O script terminou.")
    print("")


main()

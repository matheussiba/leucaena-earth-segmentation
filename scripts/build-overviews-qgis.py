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

from osgeo import gdal

# >>> EDITE SO ESTA LINHA <<<
TIF_FOLDER = r"C:\00_DATASETS_AI\260515-piracicaba-aoi\opt\rgbnir\deleteme"

# True = inclui subpastas | False = so a pasta raiz
RECURSE = False

# True = pula se ja existir .ovr | False = refaz
SKIP_EXISTING = True

OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64]


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

    print(f"Pasta: {folder}")
    print(f"Arquivos encontrados: {len(files)}")
    print("")

    ok = skip = fail = 0
    for i, tif in enumerate(files, start=1):
        ovr = Path(str(tif) + ".ovr")
        if SKIP_EXISTING and ovr.exists():
            print(f"[{i}/{len(files)}] pulando (ja tem .ovr): {tif.name}")
            skip += 1
            continue

        print(f"[{i}/{len(files)}] criando overview: {tif}")
        try:
            build_overview(tif)
            ok += 1
        except Exception as exc:
            print(f"  FALHOU: {tif.name} -> {exc}")
            fail += 1

    print("")
    print("======= RESUMO =======")
    print(f"OK:      {ok}")
    print(f"Pulados: {skip}")
    print(f"Falhas:  {fail}")


main()

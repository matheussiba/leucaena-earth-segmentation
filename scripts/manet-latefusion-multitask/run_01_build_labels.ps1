# Etapa 1 - labels + distancia (CHM>2 so onde INT>0; gap LiDAR mantem poligono).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\_load_aoi.ps1
$AOI = $AOI_HOST

$OUT = Join-Path $AOI "experiments\$MANET_EXP_NAME\masks"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

python .\01_build_labels.py `
  --img_path  "$AOI\opt\rgbnir\SF-23-Y-A-IV-4-SO-A.tif" `
  --shp_path  "$AOI\annotations\261030-classes.gpkg" `
  --chm_path  "$AOI\lidar\rasters\SF-23-Y-A-IV-4-SO-A.tif" `
  --chm_band 1 `
  --intensity_band 2 `
  --chm_min_m 2 `
  --chm_filter_class 1 `
  --out_base_dir $OUT `
  --train_ratio 0.8 `
  --seed 27 `
  --calc_distance True `
  --dist_map_as_cluster True `
  --cluster_edge_m 10 `
  --apply_gaussian True `
  --gaussian_sigma 2 `
  --classes_dist_map 1
if ($LASTEXITCODE -ne 0) { throw "01_build_labels.py falhou (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Pronto. Veja: $OUT"
Write-Host "  (CHM>2 so com LiDAR; gaps mantem classe 1)"

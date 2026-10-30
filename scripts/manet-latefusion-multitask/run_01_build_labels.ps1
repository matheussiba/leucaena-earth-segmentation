# Etapa 1 - rasteriza poligonos multiclasse + mapa de pertinencia (macico) da leucaena.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\_load_aoi.ps1
$AOI = $AOI_HOST

$OUT = Join-Path $AOI "experiments\manet-latefusion-multitask\masks"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

python .\01_build_labels.py `
  --img_path  "$AOI\opt\rgbnir\SF-23-Y-A-IV-4-SO-A.tif" `
  --shp_path  "$AOI\annotations\261030-classes.gpkg" `
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
Write-Host "  ITCs_classes_report.txt"
Write-Host "  ITCs_labels_train.tif / ITCs_labels_test.tif"
Write-Host "  ITCs_distance_train.tif / ITCs_distance_test.tif"

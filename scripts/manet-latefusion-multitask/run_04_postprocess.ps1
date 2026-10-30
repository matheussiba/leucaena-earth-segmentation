# Etapa 4 — Gauss + limiar + remove objetos pequenos sobre pred_dist.tif
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$AOI = "C:\00_DATASETS_AI\261002-riodaspedras-aoi"
$IN  = Join-Path $AOI "experiments\manet-latefusion-multitask\prediction\pred_dist.tif"
$OUT = Join-Path $AOI "experiments\manet-latefusion-multitask\prediction"

if (-not (Test-Path $IN)) {
    throw "Falta $IN — rode run_03_predict_docker.ps1 primeiro."
}

# Em 0.25 m: 500 px ≈ 0.031 ha — ajuste --min_size conforme o menor maciço esperado.
python .\04_postprocess.py `
  --input $IN `
  --out_dir $OUT `
  --sigma 2 `
  --threshold 0.5 `
  --min_size 500 `
  --connectivity 2 `
  --save_filtered True

Write-Host ""
Write-Host "Mascara binaria dos macicos em: $OUT"

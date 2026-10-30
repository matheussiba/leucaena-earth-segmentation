# Etapa 4 - Gauss + limiar + remove objetos pequenos sobre pred_dist.tif
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\_load_aoi.ps1
$AOI = $AOI_HOST

$IN  = Join-Path $AOI "experiments\manet-latefusion-multitask\prediction\pred_dist.tif"
$OUT = Join-Path $AOI "experiments\manet-latefusion-multitask\prediction"

if (-not (Test-Path $IN)) {
    throw "Falta $IN - rode run_03_predict_docker.ps1 primeiro."
}

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

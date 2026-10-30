# Etapa 4 - pos-processamento do pred_dist (EXPERIMENT_NAME).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\_load_aoi.ps1
$AOI = $AOI_HOST

$IN  = Join-Path $AOI "experiments\$EXPERIMENT_NAME\prediction\pred_dist.tif"
$OUT = Join-Path $AOI "experiments\$EXPERIMENT_NAME\prediction"

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
Write-Host "Mascara binaria em: $OUT"

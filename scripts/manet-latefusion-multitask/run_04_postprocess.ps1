# Etapa 4 - pos-processamento do pred_dist via Docker (mesma imagem do treino).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\_load_aoi.ps1

$IN_HOST  = Join-Path $AOI_HOST "experiments\$EXPERIMENT_NAME\prediction\pred_dist.tif"
$OUT_HOST = Join-Path $AOI_HOST "experiments\$EXPERIMENT_NAME\prediction"

if (-not (Test-Path $IN_HOST)) {
    throw "Falta $IN_HOST - rode run_03_predict_docker.ps1 primeiro."
}

$IN  = "/data/aoi/experiments/$EXPERIMENT_NAME/prediction/pred_dist.tif"
$OUT = "/data/aoi/experiments/$EXPERIMENT_NAME/prediction"

Write-Host "Postprocess via Docker: $IN"

docker compose run --rm manet-latefusion-multitask python scripts/manet-latefusion-multitask/04_postprocess.py `
  --input $IN `
  --out_dir $OUT `
  --sigma 2 `
  --threshold 0.5 `
  --min_size 500 `
  --connectivity 2 `
  --save_filtered True
if ($LASTEXITCODE -ne 0) { throw "04_postprocess.py falhou (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Mascara binaria em: $OUT_HOST"
Write-Host "  pred_dist_gauss_s2_w*.tif"
Write-Host "  pred_dist_bin_t0.5_min500.tif"

# Etapa 3 - predicao da cena inteira (janela deslizante + overlaps).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\_load_aoi.ps1

$WEIGHTS  = Join-Path $AOI_HOST "experiments\manet-latefusion-multitask\models\weights"
$OUT_HOST = Join-Path $AOI_HOST "experiments\manet-latefusion-multitask\prediction"
New-Item -ItemType Directory -Force -Path $OUT_HOST | Out-Null

$ckptLeaf = $null
foreach ($name in @("best_model.pth", "last_model.pth")) {
    if (Test-Path (Join-Path $WEIGHTS $name)) { $ckptLeaf = $name; break }
}
if (-not $ckptLeaf) { throw "Checkpoint nao encontrado em $WEIGHTS. Rode run_02_train_docker.ps1." }

$IMG  = "/data/aoi/opt/rgbnir/SF-23-Y-A-IV-4-SO-A.tif"
$CHM  = "/data/aoi/lidar/rasters/SF-23-Y-A-IV-4-SO-A.tif"
$CKPT = "/data/aoi/experiments/manet-latefusion-multitask/models/weights/$ckptLeaf"
$OUT  = "/data/aoi/experiments/manet-latefusion-multitask/prediction"

Write-Host "Predizendo com: $ckptLeaf"

docker compose run --rm manet-latefusion-multitask python scripts/manet-latefusion-multitask/03_predict.py `
  --image_path $IMG `
  --chm_path $CHM `
  --checkpoint $CKPT `
  --out_dir $OUT `
  --output_crs EPSG:31983 `
  --batch_size 16 `
  --overlaps 0.3 0.5 0.7

Write-Host ""
Write-Host "Saidas em: $OUT_HOST"
Write-Host "  pred_class.tif | pred_prob.tif | pred_dist.tif"

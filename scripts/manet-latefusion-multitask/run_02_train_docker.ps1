# Etapa 2 - treino MANet late fusion. Pasta = MANET_EXP_NAME (v2 nao sobrescreve v1).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\_load_aoi.ps1

$EXP_HOST = Join-Path $AOI_HOST "experiments\$MANET_EXP_NAME"
$MASKS_HOST = Join-Path $EXP_HOST "masks"
$MODEL_HOST = Join-Path $EXP_HOST "models"

if (-not (Test-Path (Join-Path $MASKS_HOST "ITCs_labels_train.tif"))) {
    throw "Rode primeiro run_01_build_labels.ps1 (falta ITCs_labels_train.tif em $MASKS_HOST)"
}
New-Item -ItemType Directory -Force -Path $MODEL_HOST | Out-Null

$IMG  = "/data/aoi/opt/rgbnir/SF-23-Y-A-IV-4-SO-A.tif"
$CHM  = "/data/aoi/lidar/rasters/SF-23-Y-A-IV-4-SO-A.tif"
$REF  = "/data/aoi/experiments/$MANET_EXP_NAME/masks/ITCs_labels_train.tif"
$DIST = "/data/aoi/experiments/$MANET_EXP_NAME/masks/ITCs_distance_train.tif"
$MDIR = "/data/aoi/experiments/$MANET_EXP_NAME/models"

# patch 512: batch menor que no v1 (256)
$patchSize = 512
if ($TRAIN_PROFILE -eq "lab") {
    $batchSize = 4
    $numWorkers = 8
    $prefetch = 4
} else {
    $batchSize = 2
    $numWorkers = 2
    $prefetch = 2
}

Write-Host "Subindo container e iniciando treino..."
Write-Host "  exp=$MANET_EXP_NAME | arch=manet | latefusion | multitask"
Write-Host "  profile=$TRAIN_PROFILE | patch=$patchSize | batch=$batchSize | workers=$numWorkers"

docker compose run --rm manet-latefusion-multitask python scripts/manet-latefusion-multitask/02_train.py `
  --on_the_fly True `
  --image_path $IMG `
  --ref_path $REF `
  --depth_path $DIST `
  --chm_path $CHM `
  --model_dir $MDIR `
  --arch manet `
  --pretrained `
  --channels 4 `
  --lidar_channels 2 `
  --latefusion True `
  --fusion concat `
  --task_multitask True `
  --task_classification False `
  --task_regression False `
  --dist_guided False `
  --num_classes 6 `
  --ignore_index True `
  --ignore_class_value 0 `
  --rgb_normalize True `
  --rgb_norm_method minmax `
  --lidar_norm minmax `
  --rgb_resolution 0.25 `
  --chm_resolution 0.25 `
  --patch_size $patchSize `
  --batch_size $batchSize `
  --samples 10000 `
  --val_samples 2000 `
  --val_fraction 0.2 `
  --val_block_size 1024 `
  --oversample True `
  --epochs 60 `
  --save_every 15 `
  --lr 1e-4 `
  --amp `
  --num_workers $numWorkers `
  --prefetch_factor $prefetch `
  --cudnn_benchmark True `
  --resume True `
  --w_reg 1.5 `
  --w_seg 1.0 `
  --reg_loss l1 `
  --reg_include_bg False `
  --reg_mask_source label `
  --class_weights inverse `
  --learn_task_weights True

Write-Host ""
Write-Host "Modelo em: $MODEL_HOST"

# Wrapper na raiz: fluxo PhD (lidar/resolve/labels/train/predict/post/report/all).
# Uso:  .\run.cmd all -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
#       .\run.cmd all ... -mask
#       .\run.cmd report
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("lidar", "resolve", "labels", "train", "predict", "post", "report", "all")]
    [string]$phase,
    [string]$aoi,
    [string]$experiment,
    [string]$run,
    [string]$profile,
    [ValidateSet("docker", "host")][string]$runtime = "docker",
    [ValidateSet("python", "r", "geotrees")][string]$lidarEngine = "python",
    [string]$root,
    [string]$initFrom,
    [string]$parentRun,
    [string]$maskGpkg,
    [switch]$mask,
    [switch]$forceLidar,
    [switch]$skipLidar,
    [switch]$forceResolve,
    [switch]$dryRun
)
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "scripts\phd-run\run.ps1") @PSBoundParameters
exit $LASTEXITCODE

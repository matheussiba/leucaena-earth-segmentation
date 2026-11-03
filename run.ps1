# Wrapper na raiz: fluxo PhD (labels/train/predict/post/report/all).
# Uso:  .\run.ps1 all -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
#       .\run.ps1 report
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("resolve", "labels", "train", "predict", "post", "report", "all")]
    [string]$phase,
    [string]$aoi,
    [string]$experiment,
    [string]$run,
    [string]$profile,
    [ValidateSet("docker", "host")][string]$runtime = "docker",
    [string]$root,
    [string]$initFrom,
    [string]$parentRun,
    [switch]$dryRun
)
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "scripts\phd-run\run.ps1") @PSBoundParameters
exit $LASTEXITCODE

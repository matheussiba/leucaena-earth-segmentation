# Wrapper na raiz: LiDAR do AOI (pool PHD_RUN_ROOT via .env).
# Uso:  .\run_lidar.ps1 -aoi 261002-riodaspedras
#       .\run_lidar.ps1 -aoi 261002-riodaspedras -Mode products
param(
    [Parameter(Mandatory = $true)]
    [string]$aoi,
    [ValidateSet("all", "products", "align")]
    [string]$Mode = "all",
    [switch]$Force,
    [switch]$NoOptical,
    [switch]$NoIntensity,
    [double]$Res = 0.5,
    [double]$ChmMax = 120,
    [double]$MaxEdge = 50,
    [int]$Workers = 4,
    [int]$Threads = 12
)
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "scripts\lidar-products\run_lidar_pipeline.ps1") @PSBoundParameters
exit $LASTEXITCODE

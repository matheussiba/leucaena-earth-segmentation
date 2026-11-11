# Wrapper na raiz: LiDAR do AOI (pool PHD_RUN_ROOT via .env).
# Uso:  .\run_lidar.ps1 -aoi 261002-riodaspedras                 # Engine python (default)
#       .\run_lidar.ps1 -aoi 261002-riodaspedras -Engine r        # lidR
#       .\run_lidar.ps1 -aoi 261002-riodaspedras -Engine geotrees # GeoTrees/GCA
param(
    [Parameter(Mandatory = $true)]
    [string]$aoi,
    [ValidateSet("r", "python", "geotrees")]
    [string]$Engine = "python",
    [ValidateSet("all", "products", "align")]
    [string]$Mode = "all",
    [switch]$Force,
    [switch]$NoOptical,
    [switch]$NoIntensity,
    [double]$Res = -1,
    [double]$ChmMax = 120,
    [double]$MaxEdge = 50,
    [int]$Workers = 4,
    [int]$Threads = 12,
    [ValidateSet("perfile", "catalog")]
    [string]$GeotreesProtocol = "perfile"
)
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "scripts\lidar-products\run_lidar_pipeline.ps1") @PSBoundParameters
exit $LASTEXITCODE

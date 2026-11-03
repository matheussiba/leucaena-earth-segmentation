# =============================================================================
# Orquestrador LiDAR (abordagem C): R (lidR) -> Python (align ao optico)
# -----------------------------------------------------------------------------
# Seleciona LAZ/tiles pelo AOI vetorial (NOMENC_* no aois/<aoi>.gpkg) sobre o
# POOL central (PHD_RUN_ROOT/tiles). Sem pasta temporaria: R/align filtram por
# --names / --laz-names / --chm-glob.
#
# Exemplo:
#   .\run_lidar_pipeline.ps1 -aoi 261002-riodaspedras
#   .\run_lidar_pipeline.ps1 -aoi 261002-riodaspedras -Mode products
#   .\run_lidar_pipeline.ps1 -aoi 261002-riodaspedras -Mode align -Force
#
# .env (raiz do repo): PHD_RUN_ROOT, R_BIN (se Rscript nao estiver no PATH)
# =============================================================================
param(
    [Parameter(Mandatory = $true)]
    [string]$aoi,
    [ValidateSet("all", "products", "align")]
    [string]$Mode = "all",
    [switch]$Force,
    [switch]$NoOptical,
    [switch]$NoIntensity,
    [double]$Res = 0.5,
    [double]$ChmMax = 120
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ---- .env -------------------------------------------------------------------
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$envFile = Join-Path $repoRoot ".env"
if (-not (Test-Path $envFile)) { throw "Falta $envFile (copie de .env.example)." }
$vars = @{}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
        $vars[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
    }
}
if (-not $vars.ContainsKey("PHD_RUN_ROOT") -or [string]::IsNullOrWhiteSpace($vars["PHD_RUN_ROOT"])) {
    throw "PHD_RUN_ROOT ausente no .env"
}
$root = ($vars["PHD_RUN_ROOT"] -replace '/', '\').TrimEnd('\')
$Rbin = "Rscript"
if ($vars.ContainsKey("R_BIN") -and -not [string]::IsNullOrWhiteSpace($vars["R_BIN"])) {
    $Rbin = $vars["R_BIN"].Trim()
}

# ---- resolve AOI -> optical tiles + laz stems (NOMENC) ----------------------
Write-Host "Resolvendo AOI '$aoi' em $root ..."
$resolveJson = & python (Join-Path $PSScriptRoot "resolve_aoi_lidar.py") --root $root --aoi $aoi
if ($LASTEXITCODE -ne 0) { throw "resolve_aoi_lidar.py falhou" }
$info = $resolveJson | ConvertFrom-Json
if ($info.n_optical -lt 1) { throw "AOI nao resolveu nenhum tile optico (cheque NOMENC_* vs tiles/rgbnir)." }
if ($info.n_laz -lt 1) { throw "AOI nao resolveu nenhum LAZ (cheque NOMENC_* vs tiles/lidar/raw)." }

$optical = @($info.optical_tiles)
$lazStems = @($info.laz_stems)
$lazNamesCsv = ($lazStems -join ",")
$LAZ_HOST  = $info.paths.laz  -replace '/', '\'
$PROD_HOST = $info.paths.products -replace '/', '\'
$RGB_HOST  = $info.paths.rgbnir -replace '/', '\'
$RAS_HOST  = $info.paths.rasters -replace '/', '\'

Write-Host "AOI: $aoi | via $($info.resolved_via)"
Write-Host "  optical ($($optical.Count)): $($optical -join ', ')"
Write-Host "  laz     ($($lazStems.Count)): $($lazStems -join ', ')"
Write-Host "Mode: $Mode | Force=$Force | NoOptical=$NoOptical | NoIntensity=$NoIntensity"
Write-Host "R: $Rbin"

New-Item -ItemType Directory -Force -Path $PROD_HOST, $RAS_HOST | Out-Null

# ---- PASSO 1: R (lidR) — filtra pool por --names ----------------------------
function Aoi-Products-Exist {
    foreach ($stem in $lazStems) {
        if (-not (Test-Path (Join-Path $PROD_HOST "${stem}_CHM.tif"))) { return $false }
    }
    return $true
}

if ($Mode -eq "products" -or $Mode -eq "all") {
    $have = Aoi-Products-Exist
    if ($have -and -not $Force -and $Mode -eq "all") {
        Write-Host "CHM do AOI ja existem em $PROD_HOST -> pulando R (use -Force p/ refazer)."
    } else {
        if (-not (Test-Path $Rbin) -and -not (Get-Command $Rbin -ErrorAction SilentlyContinue)) {
            throw "Rscript nao encontrado ('$Rbin'). Ajuste R_BIN no .env."
        }
        $rargs = @(".\make_lidar_products.R",
                   "--laz-dir", $LAZ_HOST,
                   "--out-dir", $PROD_HOST,
                   "--names", $lazNamesCsv,
                   "--res", $Res,
                   "--chm-max", $ChmMax)
        if ($Force) { $rargs += "--force" }
        Write-Host "`n[PASSO 1 - R] $Rbin $($rargs -join ' ')"
        & $Rbin @rargs
        if ($LASTEXITCODE -ne 0) { throw "make_lidar_products.R falhou (exit $LASTEXITCODE)" }
    }
}

if ($Mode -eq "products") {
    Write-Host "`nSo passo R concluido. Produtos em: $PROD_HOST"
    return
}

if (-not (Aoi-Products-Exist)) {
    throw "Sem CHM do AOI em $PROD_HOST. Rode -Mode products (ou all) primeiro."
}

# ---- PASSO 2: Docker align - 1 raster por tile optico do AOI ----------------
$env:TILES_HOST = ($root + "\tiles") -replace '\\', '/'
Write-Host ("`n[PASSO 2 - Docker] align por tile optico (TILES_HOST={0})" -f $env:TILES_HOST)

foreach ($tile in $optical) {
    $outHost = Join-Path $RAS_HOST ("{0}.tif" -f $tile)
    if ((Test-Path $outHost) -and -not $Force -and -not $NoOptical) {
        Write-Host ("  ja existe {0}.tif -> pulando align (use -Force p/ refazer)" -f $tile)
        continue
    }
    # CHM/LAZ do tile: stem == tile OU stem comeca com "tile-"
    $tileLaz = @($lazStems | Where-Object { $_ -eq $tile -or $_.StartsWith(("{0}-" -f $tile)) })
    if ($tileLaz.Count -eq 0) {
        Write-Host ("  [WARN] nenhum LAZ para tile {0} - pulando" -f $tile)
        continue
    }
    $tileLazCsv = ($tileLaz -join ",")
    $chmGlob = "{0}*_CHM.tif" -f $tile

    $pyargs = @(
        "scripts/lidar-products/align_lidar_to_optical.py",
        "--chm-dir", "/data/tiles/lidar/products",
        "--chm-glob", $chmGlob,
        "--laz-dir", "/data/tiles/lidar/raw",
        "--laz-names", $tileLazCsv,
        "--out", ("/data/tiles/lidar/rasters/{0}.tif" -f $tile)
    )
    if ($NoOptical) { $pyargs += "--no-optical" }
    else { $pyargs += @("--optical", ("/data/tiles/rgbnir/{0}.tif" -f $tile)) }
    if ($NoIntensity) { $pyargs += "--no-intensity" }

    Write-Host ("  align {0} ({1} LAZ)" -f $tile, $tileLaz.Count)
    docker compose run --rm lidar-align python @pyargs
    if ($LASTEXITCODE -ne 0) { throw ("align falhou para {0} (exit {1})" -f $tile, $LASTEXITCODE) }
    Write-Host ("  -> {0}" -f $outHost)
}

Write-Host ("`nLiDAR do AOI '{0}' pronto em: {1}" -f $aoi, $RAS_HOST)

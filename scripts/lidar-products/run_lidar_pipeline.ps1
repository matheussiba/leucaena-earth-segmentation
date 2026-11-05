# =============================================================================
# Orquestrador LiDAR: AOI (NOMENC) -> [CHM, INTENSITY] no pool
# -----------------------------------------------------------------------------
# Engines (-Engine):
#   python  (default) prep-lidar-scene.py (PDAL: 1st return + p99) via Docker
#   r       R/lidR (MDT TIN + CHM lspikefree) -> align Docker
#
# Exemplo:
#   .\run_lidar_pipeline.ps1 -aoi 261002-riodaspedras
#   .\run_lidar_pipeline.ps1 -aoi 261002-riodaspedras -Engine r
#   .\run_lidar_pipeline.ps1 -aoi 261002-riodaspedras -Engine r -Mode products
#
# .env: PHD_RUN_ROOT, R_BIN (so para -Engine r)
# =============================================================================
param(
    [Parameter(Mandatory = $true)]
    [string]$aoi,
    [ValidateSet("r", "python")]
    [string]$Engine = "python",
    [ValidateSet("all", "products", "align")]
    [string]$Mode = "all",
    [switch]$Force,
    [switch]$NoOptical,
    [switch]$NoIntensity,
    [double]$Res = -1,       # -1 = default por engine (r:0.5, python:1.0)
    [double]$ChmMax = 120,
    [double]$MaxEdge = 50,   # so Engine=r (TIN)
    [int]$Workers = 4,       # so Engine=r
    [int]$Threads = 12       # so Engine=r
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
    throw "PHD_RUN_ROOT ausente no .env ($envFile)"
}
$root = ($vars["PHD_RUN_ROOT"] -replace '/', '\').TrimEnd('\')
Write-Host "env: $envFile"
Write-Host "PHD_RUN_ROOT: $root"
if (-not (Test-Path $root)) {
    throw "PHD_RUN_ROOT nao existe: $root (ajuste no .env)"
}
$Rbin = "Rscript"
if ($vars.ContainsKey("R_BIN") -and -not [string]::IsNullOrWhiteSpace($vars["R_BIN"])) {
    $Rbin = $vars["R_BIN"].Trim()
}

if ($Res -lt 0) {
    $Res = if ($Engine -eq "python") { 1.0 } else { 0.5 }
}

# ---- resolve AOI -> optical tiles + laz stems (NOMENC) ----------------------
Write-Host "Resolvendo AOI '$aoi' em $root ..."
$resolveJson = & python (Join-Path $PSScriptRoot "resolve_aoi_lidar.py") --root $root --aoi $aoi
if ($LASTEXITCODE -ne 0) { throw "resolve_aoi_lidar.py falhou" }
$info = $resolveJson | ConvertFrom-Json
if ($info.n_optical -lt 1) {
    throw "AOI nao resolveu nenhum tile optico. Cheque NOMENC_* vs tiles/optical/rgbnir/*.tif."
}
# paths.rgbnir = PHD_RUN_ROOT/tiles/optical/rgbnir
$RGB_HOST = ($info.paths.rgbnir -replace '/', '\')
$RGB_CONT = ($info.paths.rgbnir -replace '\\', '/')
# container: /data/tiles/...  (TILES_HOST monta tiles/)
if ($RGB_CONT -match '[/\\]tiles[/\\](.+)$') {
    $RGB_CONT = "/data/tiles/" + $Matches[1].Replace('\', '/')
} else {
    $RGB_CONT = "/data/tiles/optical/rgbnir"
}
if ($info.n_laz -lt 1) { throw "AOI nao resolveu nenhum LAZ (cheque NOMENC_* vs tiles/lidar/raw)." }

$optical = @($info.optical_tiles)
$lazStems = @($info.laz_stems)
$lazNamesCsv = ($lazStems -join ",")
$LAZ_HOST  = $info.paths.laz  -replace '/', '\'
$PROD_HOST = $info.paths.products -replace '/', '\'
$RAS_HOST  = $info.paths.rasters -replace '/', '\'

Write-Host "AOI: $aoi | via $($info.resolved_via) | Engine=$Engine"
Write-Host "  optical ($($optical.Count)): $($optical -join ', ')"
Write-Host "  laz     ($($lazStems.Count)): $($lazStems -join ', ')"
Write-Host "Mode: $Mode | Force=$Force | Res=$Res | ChmMax=$ChmMax"
if ($Engine -eq "r") {
    Write-Host "Workers=$Workers | Threads=$Threads | MaxEdge=$MaxEdge m (TIN)"
    Write-Host "R: $Rbin"
}

New-Item -ItemType Directory -Force -Path $PROD_HOST, $RAS_HOST | Out-Null
$env:TILES_HOST = ($root + "\tiles") -replace '\\', '/'

function Get-TileLaz([string]$tile) {
    return @($lazStems | Where-Object { $_ -eq $tile -or $_.StartsWith(("{0}-" -f $tile)) })
}

function Test-Aoi-R-Products {
    foreach ($stem in $lazStems) {
        if (-not (Test-Path (Join-Path $PROD_HOST "${stem}_CHM.tif"))) { return $false }
    }
    return $true
}

# =============================================================================
# Engine = python  (prep-lidar-scene.py — PDAL antigo, DSM-DTM min)
# =============================================================================
if ($Engine -eq "python") {
    if ($NoOptical) {
        throw "-Engine python exige optico (prep-lidar-scene.py). Sem -NoOptical."
    }
    if ($Mode -eq "align") {
        Write-Host "[WARN] -Engine python nao tem passo align separado (tudo em prep-lidar-scene). Rodando pipeline completo."
    }
    if ($NoIntensity) {
        Write-Host "[WARN] -NoIntensity ignorado em -Engine python (prep sempre gera INTENSITY)."
    }

    Write-Host ("`n[Engine python] prep-lidar-scene.py via Docker (TILES_HOST={0})" -f $env:TILES_HOST)

    foreach ($tile in $optical) {
        $outHost = Join-Path $RAS_HOST ("{0}.tif" -f $tile)
        if ((Test-Path $outHost) -and -not $Force) {
            Write-Host ("  ja existe {0}.tif -> pulando (use -Force p/ refazer)" -f $tile)
            continue
        }
        $tileLaz = Get-TileLaz $tile
        if ($tileLaz.Count -eq 0) {
            Write-Host ("  [WARN] nenhum LAZ para tile {0} - pulando" -f $tile)
            continue
        }
        $lazArgs = @()
        foreach ($stem in $tileLaz) {
            $lazArgs += "/data/tiles/lidar/raw/{0}.laz" -f $stem
        }
        $pyargs = @(
            "prep-lidar-scene.py",
            "--optical", ("{0}/{1}.tif" -f $RGB_CONT.TrimEnd('/'), $tile),
            "--out", ("/data/tiles/lidar/rasters/{0}.tif" -f $tile),
            "--resolution", ("{0}" -f $Res).Replace(",", "."),
            "--chm-max-m", $ChmMax,
            "--laz"
        ) + $lazArgs

        Write-Host ("  prep-lidar-scene {0} (opt={1}/{0}.tif, {2} LAZ, res={3} m)" -f $tile, $RGB_CONT.TrimEnd('/'), $tileLaz.Count, $Res)
        docker compose run --rm lidar-align python @pyargs
        if ($LASTEXITCODE -ne 0) { throw ("prep-lidar-scene falhou para {0} (exit {1})" -f $tile, $LASTEXITCODE) }
        Write-Host ("  -> {0}" -f $outHost)
    }

    Write-Host ("`nLiDAR do AOI '{0}' pronto (engine=python) em: {1}" -f $aoi, $RAS_HOST)
    return
}

# =============================================================================
# Engine = r: make_lidar_products.R -> align_lidar_to_optical.py
# =============================================================================
if ($Mode -eq "products" -or $Mode -eq "all") {
    $have = Test-Aoi-R-Products
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
                   "--res", ("{0}" -f $Res).Replace(",", "."),
                   "--max-edge", ("{0}" -f $MaxEdge).Replace(",", "."),
                   "--chm-max", $ChmMax,
                   "--workers", $Workers,
                   "--threads", $Threads)
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

if (-not (Test-Aoi-R-Products)) {
    throw "Sem CHM do AOI em $PROD_HOST. Rode -Mode products (ou all) primeiro."
}

Write-Host ("`n[PASSO 2 - Docker] align por tile optico (TILES_HOST={0})" -f $env:TILES_HOST)

foreach ($tile in $optical) {
    $outHost = Join-Path $RAS_HOST ("{0}.tif" -f $tile)
    if ((Test-Path $outHost) -and -not $Force -and -not $NoOptical) {
        Write-Host ("  ja existe {0}.tif -> pulando align (use -Force p/ refazer)" -f $tile)
        continue
    }
    $tileLaz = Get-TileLaz $tile
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
    else { $pyargs += @("--optical", ("{0}/{1}.tif" -f $RGB_CONT.TrimEnd('/'), $tile)) }
    if ($NoIntensity) { $pyargs += "--no-intensity" }

    Write-Host ("  align {0} ({1} LAZ)" -f $tile, $tileLaz.Count)
    docker compose run --rm lidar-align python @pyargs
    if ($LASTEXITCODE -ne 0) { throw ("align falhou para {0} (exit {1})" -f $tile, $LASTEXITCODE) }
    Write-Host ("  -> {0}" -f $outHost)
}

Write-Host ("`nLiDAR do AOI '{0}' pronto (engine=r) em: {1}" -f $aoi, $RAS_HOST)

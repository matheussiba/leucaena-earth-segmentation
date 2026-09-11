# =============================================================================
# build-overviews-standalone.ps1
#
# Cria piramides / overviews externos (.tif.ovr) para todos os GeoTIFFs de uma
# pasta. O arquivo .ovr fica AO LADO do .tif e NAO altera o raster original.
# Isso deixa o QGIS abrir tiles grandes bem mais rapido.
#
# COMO USAR (Windows PowerShell):
#   1. Edite APENAS a linha $TifFolder abaixo, apontando para a pasta com .tif
#   2. Clique com o botao direito no arquivo -> "Executar com PowerShell"
#      OU abra o PowerShell e rode:
#         cd "CAMINHO\ONDE\ESTA\ESTE\SCRIPT"
#         .\build-overviews-standalone.ps1
#
# REQUISITOS (um dos dois):
#   A) QGIS instalado (recomendado) — o script acha o gdaladdo sozinho
#   B) Docker Desktop + imagem leucaena-segmentation:cuda (opcional)
#
# Niveis gerados: 2 4 8 16 32 64
# =============================================================================

# >>> EDITE SO ESTA LINHA <<<
$TifFolder = "D:\rgb"

# Procura tambem em subpastas? $true = sim | $false = so a pasta raiz
$Recurse = $false

# Se ja existir .ovr, pular o arquivo? $true = pular | $false = refazer
$SkipExisting = $true

# -----------------------------------------------------------------------------
# Nao precisa editar abaixo
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

function Find-GdalAddo {
    $candidates = @(
        "C:\Program Files\QGIS 3.40*\bin\gdaladdo.exe",
        "C:\Program Files\QGIS 3.38*\bin\gdaladdo.exe",
        "C:\Program Files\QGIS 3.36*\bin\gdaladdo.exe",
        "C:\Program Files\QGIS 3.34*\bin\gdaladdo.exe",
        "C:\Program Files\QGIS 3.28*\bin\gdaladdo.exe",
        "C:\OSGeo4W64\bin\gdaladdo.exe",
        "C:\OSGeo4W\bin\gdaladdo.exe"
    )
    foreach ($pattern in $candidates) {
        $hit = Get-Item $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    $fromPath = Get-Command gdaladdo -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }
    return $null
}

if (-not (Test-Path -LiteralPath $TifFolder)) {
    Write-Host "ERRO: pasta nao encontrada: $TifFolder" -ForegroundColor Red
    Write-Host "Edite a variavel `$TifFolder no inicio deste script."
    exit 1
}

$gdaladdo = Find-GdalAddo
if (-not $gdaladdo) {
    Write-Host "ERRO: gdaladdo nao encontrado." -ForegroundColor Red
    Write-Host "Instale o QGIS (https://qgis.org) e rode de novo,"
    Write-Host "ou use o script Docker do projeto (build-overviews-qgis.ps1)."
    exit 1
}

Write-Host "Usando: $gdaladdo"
Write-Host "Pasta:  $TifFolder"
Write-Host ""

$search = @{
    Path    = $TifFolder
    Filter  = "*.tif"
    File    = $true
}
if ($Recurse) { $search["Recurse"] = $true }

$files = @(Get-ChildItem @search)
if ($files.Count -eq 0) {
    Write-Host "Nenhum .tif encontrado em: $TifFolder" -ForegroundColor Yellow
    exit 0
}

Write-Host "Arquivos encontrados: $($files.Count)"
Write-Host ""

$ok = 0
$skip = 0
$fail = 0
$i = 0

foreach ($file in $files) {
    $i++
    $ovr = "$($file.FullName).ovr"

    if ($SkipExisting -and (Test-Path -LiteralPath $ovr)) {
        Write-Host "[$i/$($files.Count)] pulando (ja tem .ovr): $($file.Name)"
        $skip++
        continue
    }

    Write-Host "[$i/$($files.Count)] criando overview: $($file.FullName)"

    $env:COMPRESS_OVERVIEW = "DEFLATE"
    $env:PREDICTOR_OVERVIEW = "2"
    $env:BIGTIFF_OVERVIEW = "IF_SAFER"
    $env:GDAL_TIFF_OVR_BLOCKSIZE = "512"
    $env:GDAL_NUM_THREADS = "ALL_CPUS"

    & $gdaladdo `
        -ro `
        -r average `
        --config COMPRESS_OVERVIEW DEFLATE `
        --config PREDICTOR_OVERVIEW 2 `
        --config BIGTIFF_OVERVIEW IF_SAFER `
        --config GDAL_TIFF_OVR_BLOCKSIZE 512 `
        --config GDAL_NUM_THREADS ALL_CPUS `
        $file.FullName `
        2 4 8 16 32 64

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FALHOU: $($file.Name)" -ForegroundColor Red
        $fail++
    } else {
        $ok++
    }
}

Write-Host ""
Write-Host "======= RESUMO ======="
Write-Host "OK:      $ok"
Write-Host "Pulados: $skip"
Write-Host "Falhas:  $fail"
Write-Host ""
Write-Host "Para conferir:"
Write-Host "  Get-ChildItem `"$TifFolder`" -Filter *.ovr"
if ($fail -gt 0) { exit 1 }

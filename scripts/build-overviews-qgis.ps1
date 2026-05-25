# build-overviews-qgis.ps1
#
# Build external GeoTIFF overviews (.tif.ovr) for every tile in $root and in
# $root\bkp, so QGIS opens the rasters smoothly even on a moderate laptop.
# The .ovr file lives next to the .tif and never modifies the original raster.
#
# Runs gdaladdo through the project's Docker image (no local GDAL needed).
# Requirements: Docker Desktop running, WSL2/Ubuntu enabled.
#
# Usage:
#   Edit $root for a different dataset (and $repo only if you moved the repo),
#   then paste the whole block into a Windows PowerShell prompt.

$repo = "/mnt/c/Users/mathe/OneDrive/Documents/0-GITHUB/leucaena-earth-segmentation"
$root = "C:\00_DATASETS_AI\260515-piracicaba-aoi\tiles"

$files = @(Get-ChildItem -Path $root -Filter *.tif -File) +
         @(Get-ChildItem -Path (Join-Path $root "bkp") -Filter *.tif -File -ErrorAction SilentlyContinue)

Write-Host "Arquivos encontrados: $($files.Count)"

$i = 0
foreach ($file in $files) {
    $i++

    $rel = $file.FullName.Substring($root.Length).TrimStart('\') -replace '\\','/'
    $containerPath = "/data/rgbir/$rel"

    Write-Host "[$i/$($files.Count)] criando overview: $containerPath"

    $cmd = "cd $repo && docker compose run --rm --no-TTY segmentation gdaladdo -ro --config COMPRESS_OVERVIEW DEFLATE --config PREDICTOR_OVERVIEW 2 --config BIGTIFF_OVERVIEW IF_SAFER --config GDAL_TIFF_OVR_BLOCKSIZE 512 --config GDAL_NUM_THREADS ALL_CPUS -r average '$containerPath' 2 4 8 16 32 64"

    wsl.exe -d Ubuntu -e bash -lc $cmd

    if ($LASTEXITCODE -ne 0) {
        throw "gdaladdo failed for $containerPath"
    }
}

Write-Host "OK - overviews criados"

# To verify afterwards:
#   Get-ChildItem "C:\00_DATASETS_AI\260515-piracicaba-aoi\tiles" -Filter *.ovr
#   Get-ChildItem "C:\00_DATASETS_AI\260515-piracicaba-aoi\tiles\bkp" -Filter *.ovr

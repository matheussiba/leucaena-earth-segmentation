# Load DIR_DATASETS_AI_PATH + LEUCAENA_DATASET_ID (+ TRAIN_PROFILE, EXPERIMENT_NAME)
# from repo-root .env. Each PC has its own .env (gitignored).
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$envFile = Join-Path $repoRoot ".env"
if (-not (Test-Path $envFile)) {
    throw "Missing $envFile - copy .env.example to .env and set DIR_DATASETS_AI_PATH and LEUCAENA_DATASET_ID."
}

$vars = @{}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
        $vars[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
    }
}

if (-not $vars.ContainsKey("DIR_DATASETS_AI_PATH") -or [string]::IsNullOrWhiteSpace($vars["DIR_DATASETS_AI_PATH"])) {
    throw "DIR_DATASETS_AI_PATH not set in $envFile (e.g. C:/00_DATASETS_AI or E:/dados-msb/00_DATASETS_AI)."
}
if (-not $vars.ContainsKey("LEUCAENA_DATASET_ID") -or [string]::IsNullOrWhiteSpace($vars["LEUCAENA_DATASET_ID"])) {
    throw "LEUCAENA_DATASET_ID not set in $envFile (e.g. 261002-riodaspedras-aoi)."
}

$base = ($vars["DIR_DATASETS_AI_PATH"] -replace '/', '\').TrimEnd('\')
$id = $vars["LEUCAENA_DATASET_ID"].Trim()
$AOI_HOST = Join-Path $base $id

if (-not (Test-Path $AOI_HOST)) {
    throw "AOI not found: $AOI_HOST. Check DIR_DATASETS_AI_PATH and LEUCAENA_DATASET_ID in $envFile"
}

# home = PC pessoal; lab = laboratorio
$TRAIN_PROFILE = "home"
if ($vars.ContainsKey("TRAIN_PROFILE") -and -not [string]::IsNullOrWhiteSpace($vars["TRAIN_PROFILE"])) {
    $TRAIN_PROFILE = $vars["TRAIN_PROFILE"].Trim().ToLowerInvariant()
}
if ($TRAIN_PROFILE -notin @("home", "lab")) {
    throw "TRAIN_PROFILE must be 'home' or 'lab' (got '$TRAIN_PROFILE' in $envFile)."
}

# v1 = manet-latefusion-multitask (treino antigo, nao apagar)
# v2 = manet-latefusion-multitask-v2 (patch 512 + filtro CHM so com LiDAR)
$EXPERIMENT_NAME = "manet-latefusion-multitask-v2"
if ($vars.ContainsKey("EXPERIMENT_NAME") -and -not [string]::IsNullOrWhiteSpace($vars["EXPERIMENT_NAME"])) {
    $EXPERIMENT_NAME = $vars["EXPERIMENT_NAME"].Trim()
}

$env:AOI_HOST = ($AOI_HOST -replace '\\', '/')
if ($TRAIN_PROFILE -eq "lab") {
    $env:DOCKER_SHM_SIZE = "16gb"
} else {
    $env:DOCKER_SHM_SIZE = "8gb"
}

Write-Host "AOI: $AOI_HOST"
Write-Host "EXPERIMENT_NAME: $EXPERIMENT_NAME"
Write-Host "TRAIN_PROFILE: $TRAIN_PROFILE (shm=$($env:DOCKER_SHM_SIZE))"

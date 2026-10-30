# Le DIR_DATASETS_AI_PATH + LEUCAENA_DATASET_ID do .env na RAIZ do repo.
# Cada PC tem o seu .env (gitignored); o .env.example vai no git.
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$envFile = Join-Path $repoRoot ".env"
if (-not (Test-Path $envFile)) {
    throw "Falta $envFile — copie .env.example para .env e defina DIR_DATASETS_AI_PATH e LEUCAENA_DATASET_ID."
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
    throw "DIR_DATASETS_AI_PATH nao definido em $envFile (ex.: C:/00_DATASETS_AI ou E:/dados-msb/00_DATASETS_AI)."
}
if (-not $vars.ContainsKey("LEUCAENA_DATASET_ID") -or [string]::IsNullOrWhiteSpace($vars["LEUCAENA_DATASET_ID"])) {
    throw "LEUCAENA_DATASET_ID nao definido em $envFile (ex.: 261002-riodaspedras-aoi)."
}

$base = ($vars["DIR_DATASETS_AI_PATH"] -replace '/', '\').TrimEnd('\')
$id = $vars["LEUCAENA_DATASET_ID"].Trim()
$AOI_HOST = Join-Path $base $id

if (-not (Test-Path $AOI_HOST)) {
    throw "AOI nao existe: $AOI_HOST`nConfira DIR_DATASETS_AI_PATH e LEUCAENA_DATASET_ID em $envFile"
}

# Docker Compose substitui ${AOI_HOST} a partir do ambiente do processo.
$env:AOI_HOST = ($AOI_HOST -replace '\\', '/')
Write-Host "AOI: $AOI_HOST"

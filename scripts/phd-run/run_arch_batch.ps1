# =============================================================================
# Batch: mesma AOI (+ mask) x todas as redes, patch 256, 30 ep, run-001
# -----------------------------------------------------------------------------
# Gera experiments/<rede>/latefusion-multitask-p256/experiment.yaml (se faltar)
# a partir do manet de referencia, depois:
#   1) lidar + resolve + labels UMA vez
#   2) train + predict + post por rede
#
# Uso (raiz do repo):
#   .\run_arch_batch.cmd -aoi 261002-riodaspedras -mask
#   .\run_arch_batch.cmd -aoi 261002-riodaspedras -mask -SkipLidar
#   .\run_arch_batch.cmd -aoi 261002-riodaspedras -mask -Architectures manet,unet,resunet
# =============================================================================
param(
    [Parameter(Mandatory = $true)]
    [string]$aoi,
    [switch]$mask,
    [string]$maskGpkg,
    [string]$run = "run-001",
    [string]$expName = "latefusion-multitask-p256",
    [string]$refExperiment = "manet/latefusion-multitask-p256",
    [string]$Architectures = "unet,unetplusplus,manet,linknet,BPATUNet,resunet",
    [ValidateSet("docker", "host")][string]$runtime = "docker",
    [ValidateSet("python", "r", "geotrees")][string]$lidarEngine = "python",
    [switch]$SkipLidar,
    [switch]$ForceLidar,
    [switch]$ForceResolve,
    [switch]$DryRun,
    [switch]$SkipPredict,
    [switch]$SkipPost
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

# ---- .env → PHD_RUN_ROOT ----------------------------------------------------
$envFile = Join-Path $repoRoot ".env"
if (-not (Test-Path $envFile)) { throw "Falta .env na raiz do repo." }
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
$expRoot = Join-Path $root "experiments"
$refParts = $refExperiment -split '/'
$refYaml = Join-Path (Join-Path $expRoot $refParts[0]) (Join-Path $refParts[1] "experiment.yaml")
if (-not (Test-Path $refYaml)) {
    throw "YAML de referencia nao encontrado: $refYaml`nCrie experiments/manet/latefusion-multitask-p256/experiment.yaml no PHD_RUN_ROOT."
}

$archList = @($Architectures.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($archList.Count -lt 1) { throw "-Architectures vazio." }

Write-Host "============================================================"
Write-Host " ARCH BATCH  aoi=$aoi  run=$run  exp=$expName"
Write-Host " PHD_RUN_ROOT=$root"
Write-Host " redes: $($archList -join ', ')"
Write-Host " mask=$mask  skipLidar=$SkipLidar  dryRun=$DryRun"
Write-Host "============================================================"

# ---- garante experiment.yaml por rede (so muda network:) --------------------
$refText = Get-Content -Raw -Encoding UTF8 $refYaml
foreach ($arch in $archList) {
    $destDir = Join-Path (Join-Path $expRoot $arch) $expName
    $destYaml = Join-Path $destDir "experiment.yaml"
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    if ($refText -match '(?m)^network:\s*\S+') {
        $newText = [regex]::Replace($refText, '(?m)^network:\s*\S+', "network: $arch", 1)
    } else {
        $newText = "network: $arch`r`n" + $refText
    }
    if ($newText -notmatch '(?m)^report_class:') {
        $newText = [regex]::Replace($newText, '(?m)^(num_classes:\s*\d+)', "`$1`r`nreport_class: 1", 1)
    }
    $write = $true
    if (Test-Path $destYaml) {
        $old = Get-Content -Raw -Encoding UTF8 $destYaml
        if ($old -eq $newText) { $write = $false }
    }
    if ($write) {
        [System.IO.File]::WriteAllText($destYaml, $newText)
        Write-Host "[yaml] $arch/$expName <- $refExperiment (network=$arch)"
    } else {
        Write-Host "[yaml] ok $arch/$expName"
    }
}

$runPs1 = Join-Path $repoRoot "run.ps1"
function Invoke-Phd([string[]]$ArgsList) {
    Write-Host "`n>> .\run.cmd $($ArgsList -join ' ')"
    if ($DryRun) { return }
    & $runPs1 @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Falha (rc=$LASTEXITCODE): run.cmd $($ArgsList -join ' ')"
    }
}

$common = @("-aoi", $aoi, "-run", $run, "-runtime", $runtime)
if ($mask) { $common += "-mask" }
if ($maskGpkg) { $common += @("-maskGpkg", $maskGpkg) }

$first = $archList[0]
$firstExp = "$first/$expName"

# ---- 1) preparar dados UMA vez ----------------------------------------------
Write-Host "`n=== PREP (lidar / resolve / labels) via $firstExp ==="

if (-not $SkipLidar) {
    $la = @("lidar", "-aoi", $aoi, "-lidarEngine", $lidarEngine)
    if ($ForceLidar) { $la += "-forceLidar" }
    Invoke-Phd $la
}

$ra = @("resolve", "-experiment", $firstExp) + $common
if ($ForceResolve) { $ra += "-forceResolve" }
Invoke-Phd $ra

Invoke-Phd (@("labels", "-experiment", $firstExp) + $common)

# ---- 2) train (+ predict/post) por rede -------------------------------------
$logDir = Join-Path $expRoot "_batch_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$summary = Join-Path $logDir "arch_batch_${aoi}_${stamp}.txt"
"arch_batch aoi=$aoi run=$run exp=$expName started=$(Get-Date -Format o)" | Set-Content $summary -Encoding UTF8

foreach ($arch in $archList) {
    $exp = "$arch/$expName"
    Write-Host "`n=== TRAIN $exp  run=$run ==="
    try {
        Invoke-Phd (@("train", "-experiment", $exp) + $common)
        if (-not $SkipPredict) {
            Invoke-Phd (@("predict", "-experiment", $exp) + $common)
        }
        if (-not $SkipPost) {
            Invoke-Phd (@("post", "-experiment", $exp) + $common)
        }
        "OK  $exp" | Add-Content $summary -Encoding UTF8
        Write-Host "[OK] $exp"
    } catch {
        "FAIL $exp :: $($_.Exception.Message)" | Add-Content $summary -Encoding UTF8
        Write-Host "[FAIL] $exp — $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Continuando com a proxima rede..."
    }
}

"finished=$(Get-Date -Format o)" | Add-Content $summary -Encoding UTF8
Write-Host "`n============================================================"
Write-Host " Batch concluido. Resumo: $summary"
Write-Host " Modelos: $expRoot\<rede>\$expName\$run\models"
Write-Host "============================================================"

# =============================================================================
# Batch: mesma AOI (+ mask) x todas as redes (config da referencia YAML)
# -----------------------------------------------------------------------------
# Referencia unica: PHD_RUN_ROOT/experiments/experiment-batch.yaml
# (se nao existir, copia de scripts/phd-run/templates/experiment-batch.yaml)
#
# Nome da pasta = slug derivado do YAML (nao e rotulo fixo):
#   latefusion|single + task + p<patch>
#   ex.: latefusion:true, task:multitask, patch_size:256
#        -> experiments/<rede>/latefusion-multitask-p256/
# Override manual: -expName meu-nome
#
# Depois:
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
    [string]$expName = "",
    [string]$RefYaml = "",
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

function Get-YamlScalar([string]$text, [string]$key, [string]$default = "") {
    $pat = '(?m)^\s*' + [regex]::Escape($key) + '\s*:\s*(.+?)\s*$'
    $m = [regex]::Match($text, $pat)
    if (-not $m.Success) { return $default }
    $v = $m.Groups[1].Value.Trim()
    if ($v -match '^["''](.*)["'']$') { $v = $Matches[1] }
    if ($v -match '^\#') { return $default }
    $hash = $v.IndexOf('#')
    if ($hash -ge 0) { $v = $v.Substring(0, $hash).Trim() }
    return $v
}

function Get-ExperimentSlugFromYaml([string]$yamlText) {
    # latefusion:true  -> latefusion
    # latefusion:false -> single  (RGB single-source; nao ha early-fusion no motor)
    $lfRaw = (Get-YamlScalar $yamlText "latefusion" "true").ToLowerInvariant()
    $fusionMode = if ($lfRaw -in @("true", "1", "yes", "on")) { "latefusion" } else { "single" }
    $task = (Get-YamlScalar $yamlText "task" "multitask").ToLowerInvariant()
    if ($task -notmatch '^[a-z0-9_-]+$') { $task = "multitask" }
    $patchRaw = Get-YamlScalar $yamlText "patch_size" "256"
    if (-not ($patchRaw -match '^\d+$')) { $patchRaw = "256" }
    return ("{0}-{1}-p{2}" -f $fusionMode, $task, $patchRaw)
}

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

# ---- .env -> PHD_RUN_ROOT ----------------------------------------------------
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
New-Item -ItemType Directory -Force -Path $expRoot | Out-Null

$templateYaml = Join-Path $PSScriptRoot "templates\experiment-batch.yaml"
if ([string]::IsNullOrWhiteSpace($RefYaml)) {
    $refYaml = Join-Path $expRoot "experiment-batch.yaml"
} else {
    $refYaml = $RefYaml
    if (-not [System.IO.Path]::IsPathRooted($refYaml)) {
        $refYaml = Join-Path $expRoot $refYaml
    }
}

if (-not (Test-Path $refYaml)) {
    if (-not (Test-Path $templateYaml)) {
        throw "Falta referencia $refYaml e template $templateYaml"
    }
    Copy-Item $templateYaml $refYaml -Force
    Write-Host "[yaml] criado $refYaml a partir do template do repo"
}

$archList = @($Architectures.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($archList.Count -lt 1) { throw "-Architectures vazio." }

# ---- slug da pasta a partir do YAML (ou -expName) ---------------------------
$refText = Get-Content -Raw -Encoding UTF8 $refYaml
$slugAuto = Get-ExperimentSlugFromYaml $refText
if ([string]::IsNullOrWhiteSpace($expName)) {
    $expName = $slugAuto
    Write-Host "[slug] derivado do YAML: $expName"
} elseif ($expName -ne $slugAuto) {
    Write-Host "[slug] -expName=$expName (YAML sugeriria $slugAuto)"
}

Write-Host "============================================================"
Write-Host " ARCH BATCH  aoi=$aoi  run=$run  exp=$expName"
Write-Host " PHD_RUN_ROOT=$root"
Write-Host " ref: $refYaml"
Write-Host " redes: $($archList -join ', ')"
Write-Host " mask=$mask  skipLidar=$SkipLidar  dryRun=$DryRun"
Write-Host "============================================================"

# ---- garante experiment.yaml por rede (so muda network:) --------------------
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
        Write-Host "[yaml] $arch/$expName <- experiment-batch.yaml (network=$arch)"
    } else {
        Write-Host "[yaml] ok $arch/$expName"
    }
}

$runCmd = Join-Path $repoRoot "run.cmd"
function Invoke-Phd {
    # Nao usar & run.ps1 @array: splat de array no PowerShell e POSICIONAL
    # e quebra (-aoi vira arg posicional). run.cmd %* preserva nomes.
    param([Parameter(Mandatory = $true)][string[]]$ArgsList)
    Write-Host "`n>> .\run.cmd $($ArgsList -join ' ')"
    if ($DryRun) { return }
    $p = Start-Process -FilePath $runCmd -ArgumentList $ArgsList `
        -WorkingDirectory $repoRoot -Wait -PassThru -NoNewWindow
    if ($null -eq $p.ExitCode -or $p.ExitCode -ne 0) {
        throw "Falha (rc=$($p.ExitCode)): run.cmd $($ArgsList -join ' ')"
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
    Invoke-Phd -ArgsList $la
}

$ra = @("resolve", "-experiment", $firstExp) + $common
if ($ForceResolve) { $ra += "-forceResolve" }
Invoke-Phd -ArgsList $ra

Invoke-Phd -ArgsList (@("labels", "-experiment", $firstExp) + $common)

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
        Invoke-Phd -ArgsList (@("train", "-experiment", $exp) + $common)
        if (-not $SkipPredict) {
            Invoke-Phd -ArgsList (@("predict", "-experiment", $exp) + $common)
        }
        if (-not $SkipPost) {
            Invoke-Phd -ArgsList (@("post", "-experiment", $exp) + $common)
        }
        "OK  $exp" | Add-Content $summary -Encoding UTF8
        Write-Host "[OK] $exp"
    } catch {
        "FAIL $exp :: $($_.Exception.Message)" | Add-Content $summary -Encoding UTF8
        Write-Host "[FAIL] $exp - $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Continuando com a proxima rede..."
    }
}

"finished=$(Get-Date -Format o)" | Add-Content $summary -Encoding UTF8
Write-Host "`n============================================================"
Write-Host " Batch concluido. Resumo: $summary"
Write-Host (" Modelos: {0}\<rede>\{1}\{2}\models" -f $expRoot, $expName, $run)
Write-Host " Edite a referencia em: $refYaml"
Write-Host "============================================================"

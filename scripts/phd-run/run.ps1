# run.ps1 - wrapper PowerShell do orquestrador (comandos em minusculo).
#
# Exemplos:
#   .\run.ps1 all     -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
#   .\run.ps1 train   -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
#   .\run.ps1 predict -aoi 260515-piracicaba   -experiment manet/latefusion-multitask -run run-001
#   .\run.ps1 report
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("resolve", "labels", "train", "predict", "post", "report", "all")]
    [string]$phase,
    [string]$aoi,
    [string]$experiment,
    [string]$run,
    [string]$profile,
    [ValidateSet("docker", "host")][string]$runtime = "docker",
    [string]$root,
    [string]$initFrom,
    [string]$parentRun,
    [switch]$dryRun
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$args = @($phase)
if ($aoi)        { $args += @("-aoi", $aoi) }
if ($experiment) { $args += @("-experiment", $experiment) }
if ($run)        { $args += @("-run", $run) }
if ($profile)    { $args += @("-profile", $profile) }
if ($runtime)    { $args += @("-runtime", $runtime) }
if ($root)       { $args += @("-root", $root) }
if ($initFrom)   { $args += @("--init-from", $initFrom) }
if ($parentRun)  { $args += @("--parent-run", $parentRun) }
if ($dryRun)     { $args += @("--dry-run") }

python (Join-Path $PSScriptRoot "run.py") @args
exit $LASTEXITCODE

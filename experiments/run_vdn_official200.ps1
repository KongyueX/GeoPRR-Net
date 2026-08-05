#requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'Probe', 'Train', 'Verify', 'Cohort')]
    [string]$Mode = 'Preflight',
    [int[]]$Seeds = @(20260720, 20260721, 20260722),
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$Manifest = 'artifacts\manifests\syncg_train.jsonl',
    [string]$VdnSource = 'artifacts\vendor\VectorDetectionNetwork',
    [string]$ContentInventory = (
        'artifacts\protocols\' +
        'vdn_phase2_syncg_train_content_inventory_v1.json'
    ),
    [string]$PreflightReport = (
        'artifacts\protocols\vdn_official200_preflight_v1.json'
    ),
    [string]$DeterminismReport = (
        'artifacts\protocols\' +
        'vdn_official200_determinism_probe_v1.json'
    ),
    [string]$RunRoot = 'artifacts\runs\vdn_syncg_official200',
    [string]$CohortReport = (
        'artifacts\protocols\' +
        'vdn_official200_three_seed_cohort_v1.json'
    ),
    [int]$InventoryWorkers = 8,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw 'Formal VDN official-200 requires PowerShell 7 or newer.'
}
if ($InventoryWorkers -le 0) {
    throw '-InventoryWorkers must be positive.'
}
if ($Resume -and $Mode -ne 'Train') {
    throw '-Resume is valid only in Train mode.'
}

$FormalSeeds = @(20260720, 20260721, 20260722)
foreach ($Seed in $Seeds) {
    if ($Seed -notin $FormalSeeds) {
        throw "Unsupported formal VDN official-200 seed: $Seed"
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = (Resolve-Path -LiteralPath (
    Join-Path $ProjectRoot $Python
)).Path
$ManifestPath = (Resolve-Path -LiteralPath (
    Join-Path $ProjectRoot $Manifest
)).Path
$VdnSourcePath = (Resolve-Path -LiteralPath (
    Join-Path $ProjectRoot $VdnSource
)).Path
$ContentInventoryPath = (Resolve-Path -LiteralPath (
    Join-Path $ProjectRoot $ContentInventory
)).Path
$PreflightPath = Join-Path $ProjectRoot $PreflightReport
$DeterminismPath = Join-Path $ProjectRoot $DeterminismReport
$RunRootPath = Join-Path $ProjectRoot $RunRoot
$CohortPath = Join-Path $ProjectRoot $CohortReport

$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

Push-Location $ProjectRoot
try {
    if ($Mode -eq 'Preflight') {
        $env:PYTHONHASHSEED = "$($FormalSeeds[0])"
        $Arguments = @(
            '-m',
            'experiments.preflight_vdn_official200',
            '--manifest',
            $ManifestPath,
            '--vdn-source',
            $VdnSourcePath,
            '--content-inventory',
            $ContentInventoryPath,
            '--run-root',
            $RunRootPath,
            '--output',
            $PreflightPath,
            '--inventory-workers',
            "$InventoryWorkers"
        )
        & $PythonPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw 'VDN official-200 Preflight failed.'
        }
        return
    }

    if (-not (Test-Path -LiteralPath $PreflightPath -PathType Leaf)) {
        throw "Frozen official-200 preflight is absent: $PreflightPath"
    }

    if ($Mode -eq 'Probe') {
        $env:PYTHONHASHSEED = "$($FormalSeeds[0])"
        $Arguments = @(
            '-m',
            'experiments.probe_vdn_official200_determinism',
            '--manifest',
            $ManifestPath,
            '--vdn-source',
            $VdnSourcePath,
            '--content-inventory',
            $ContentInventoryPath,
            '--preflight',
            $PreflightPath,
            '--run-root',
            $RunRootPath,
            '--output',
            $DeterminismPath
        )
        & $PythonPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw 'VDN official-200 Probe failed closed.'
        }
        return
    }

    if ($Mode -eq 'Cohort') {
        $env:PYTHONHASHSEED = "$($FormalSeeds[0])"
        $Arguments = @(
            '-m',
            'experiments.verify_vdn_official200_cohort',
            '--run-root',
            $RunRootPath,
            '--manifest',
            $ManifestPath,
            '--vdn-source',
            $VdnSourcePath,
            '--content-inventory',
            $ContentInventoryPath,
            '--preflight',
            $PreflightPath,
            '--output',
            $CohortPath
        )
        & $PythonPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw 'VDN official-200 Cohort failed closed.'
        }
        return
    }

    if (-not (Test-Path -LiteralPath $DeterminismPath -PathType Leaf)) {
        throw (
            'Frozen official-200 determinism report is absent: ' +
            $DeterminismPath
        )
    }

    foreach ($Seed in $Seeds) {
        $env:PYTHONHASHSEED = "$Seed"
        $RunDirectory = Join-Path $RunRootPath "seed_$Seed"
        if ($Mode -eq 'Train') {
            $Arguments = @(
                '-m',
                'experiments.train_vdn_official200',
                '--seed',
                "$Seed",
                '--manifest',
                $ManifestPath,
                '--vdn-source',
                $VdnSourcePath,
                '--content-inventory',
                $ContentInventoryPath,
                '--preflight',
                $PreflightPath,
                '--determinism-report',
                $DeterminismPath,
                '--run-root',
                $RunRootPath,
                '--output-dir',
                $RunDirectory
            )
            if ($Resume) {
                $Arguments += '--resume'
            }
        }
        else {
            $VerificationOutput = Join-Path (
                $RunDirectory
            ) 'verification_v1.json'
            $Arguments = @(
                '-m',
                'experiments.verify_vdn_official200',
                '--run-dir',
                $RunDirectory,
                '--run-root',
                $RunRootPath,
                '--manifest',
                $ManifestPath,
                '--vdn-source',
                $VdnSourcePath,
                '--content-inventory',
                $ContentInventoryPath,
                '--preflight',
                $PreflightPath,
                '--determinism-report',
                $DeterminismPath,
                '--output',
                $VerificationOutput
            )
        }
        & $PythonPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "VDN official-200 $Mode failed for seed $Seed."
        }
    }
}
finally {
    Pop-Location
}

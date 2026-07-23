param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Oof = "artifacts\runs\uncertainty_fusion_syncg\probabilistic_oof_clean.jsonl",
    [string]$CalibratorRoot = "artifacts\runs\reference_conditioned_progress_calibrator_syncg",
    [string]$RouterRoot = "artifacts\runs\reference_conditioned_router_syncg",
    [string]$BaselineCalibratorRoot = "artifacts\runs\progress_calibrator_syncg",
    [string]$BaselineRouterRoot = "artifacts\runs\calibrated_progress_router_syncg",
    [int]$BootstrapIterations = 5000,
    [int]$Seed = 20260722
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

$CalibratorModelDir = Join-Path $CalibratorRoot "model"
$RouterModelDir = Join-Path $RouterRoot "model"
$BaselineCalibratorModelDir = Join-Path $BaselineCalibratorRoot "model"
$BaselineRouterModelDir = Join-Path $BaselineRouterRoot "model"

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.train_reference_conditioned_progress_calibrator",
    "--oof-pairs", $Oof,
    "--output-dir", $CalibratorModelDir,
    "--baseline-diagnostics", (Join-Path $BaselineCalibratorModelDir "nested_oof_predictions.jsonl"),
    "--baseline-summary", (Join-Path $BaselineCalibratorModelDir "training_summary.json"),
    "--bootstrap-iterations", "$BootstrapIterations",
    "--seed", "$Seed",
    "--overwrite"
)

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.train_reference_conditioned_router",
    "--oof-pairs", $Oof,
    "--calibration-diagnostics", (Join-Path $CalibratorModelDir "strict_nested_oof_predictions.jsonl"),
    "--calibration-summary", (Join-Path $CalibratorModelDir "training_summary.json"),
    "--calibrator", (Join-Path $CalibratorModelDir "reference_conditioned_calibrator.joblib"),
    "--output-dir", $RouterModelDir,
    "--baseline-router-diagnostics", (Join-Path $BaselineRouterModelDir "nested_oof_routing.jsonl"),
    "--baseline-router-summary", (Join-Path $BaselineRouterModelDir "training_summary.json"),
    "--bootstrap-iterations", "$BootstrapIterations",
    "--seed", "$Seed",
    "--overwrite"
)

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.verify_reference_conditioned_training",
    "--oof-pairs", $Oof,
    "--calibrator-root", $CalibratorRoot,
    "--router-root", $RouterRoot
)

Write-Host "Reference-conditioned train-only experiment completed: $RouterRoot"

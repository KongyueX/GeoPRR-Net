param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$CalibratorRoot = "artifacts\runs\progress_calibrator_syncg",
    [string]$RouterRoot = "artifacts\runs\calibrated_progress_router_syncg",
    [string]$Oof = "artifacts\runs\uncertainty_fusion_syncg\probabilistic_oof_clean.jsonl",
    [string]$DirectionRunDir = "artifacts\runs\probabilistic_pivot_direction_syncg\seed_20260722",
    [string]$VdnRunDir = "artifacts\runs\vdn_syncg\seed_20260720",
    [string]$QualityRunDir = "artifacts\runs\quality_router_syncg",
    [string]$UncertaintyRouterRunDir = "artifacts\runs\uncertainty_router_syncg",
    [int]$BootstrapIterations = 5000,
    [int]$Seed = 20260722,
    [switch]$SkipTraining
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

if (-not $SkipTraining) {
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.train_progress_calibrator",
        "--oof-pairs", $Oof,
        "--output-dir", (Join-Path $CalibratorRoot "model"),
        "--seed", "$Seed",
        "--overwrite"
    )
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.train_calibrated_progress_router",
        "--oof-pairs", $Oof,
        "--calibration-diagnostics", (Join-Path (Join-Path $CalibratorRoot "model") "nested_oof_predictions.jsonl"),
        "--calibration-summary", (Join-Path (Join-Path $CalibratorRoot "model") "training_summary.json"),
        "--output-dir", (Join-Path $RouterRoot "model"),
        "--seed", "$Seed",
        "--overwrite"
    )
}

$Calibrator = Join-Path (Join-Path $CalibratorRoot "model") "progress_calibrator.joblib"
$Router = Join-Path (Join-Path $RouterRoot "model") "calibrated_progress_router.joblib"
$Conditions = @(
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe"
)
foreach ($Condition in $Conditions) {
    $Vector = Join-Path (Join-Path $DirectionRunDir "evaluations") "$Condition.jsonl"
    $Reference = Join-Path (Join-Path $VdnRunDir "evaluations") "$Condition.jsonl"
    $Base = "artifacts\runs\robustness\$Condition\predictions.jsonl"
    $Quality = Join-Path (Join-Path (Join-Path $QualityRunDir "evaluations") $Condition) "predictions.jsonl"
    $UncertaintyRouter = Join-Path (Join-Path (Join-Path $UncertaintyRouterRunDir "evaluations") $Condition) "predictions.jsonl"
    $CalibratedDir = Join-Path (Join-Path $CalibratorRoot "evaluations") $Condition
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.evaluate_progress_calibrator",
        "--vector-predictions", $Vector,
        "--reference-predictions", $Reference,
        "--base-predictions", $Base,
        "--quality-predictions", $Quality,
        "--uncertainty-router-predictions", $UncertaintyRouter,
        "--calibrator", $Calibrator,
        "--output-dir", $CalibratedDir,
        "--condition", $Condition,
        "--bootstrap-iterations", "$BootstrapIterations",
        "--seed", "$Seed",
        "--overwrite"
    )
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.evaluate_calibrated_progress_router",
        "--raw-predictions", "artifacts\predictions\robustness\syncg_test_$Condition.jsonl",
        "--base-predictions", $Base,
        "--vector-predictions", $Vector,
        "--reference-predictions", $Reference,
        "--calibrated-predictions", (Join-Path $CalibratedDir "predictions.jsonl"),
        "--quality-predictions", $Quality,
        "--uncertainty-router-predictions", $UncertaintyRouter,
        "--router", $Router,
        "--output-dir", (Join-Path (Join-Path $RouterRoot "evaluations") $Condition),
        "--condition", $Condition,
        "--bootstrap-iterations", "$BootstrapIterations",
        "--seed", "$Seed",
        "--overwrite"
    )
}

$RpmVector = Join-Path (Join-Path $DirectionRunDir "evaluations") "rpm10k.jsonl"
$RpmReference = Join-Path (Join-Path $VdnRunDir "evaluations") "rpm10k.jsonl"
$RpmBase = "artifacts\runs\rpm10k_single_pointer_zero_shot\predictions.jsonl"
$RpmQuality = Join-Path (Join-Path (Join-Path $QualityRunDir "evaluations") "rpm10k") "predictions.jsonl"
$RpmUncertaintyRouter = Join-Path (Join-Path (Join-Path $UncertaintyRouterRunDir "evaluations") "rpm10k") "predictions.jsonl"
$RpmCalibratedDir = Join-Path (Join-Path $CalibratorRoot "evaluations") "rpm10k"
Invoke-CheckedPython -Arguments @(
    "-m", "experiments.evaluate_progress_calibrator",
    "--vector-predictions", $RpmVector,
    "--reference-predictions", $RpmReference,
    "--base-predictions", $RpmBase,
    "--quality-predictions", $RpmQuality,
    "--uncertainty-router-predictions", $RpmUncertaintyRouter,
    "--calibrator", $Calibrator,
    "--output-dir", $RpmCalibratedDir,
    "--condition", "rpm10k",
    "--bootstrap-iterations", "$BootstrapIterations",
    "--seed", "$Seed",
    "--overwrite"
)
Invoke-CheckedPython -Arguments @(
    "-m", "experiments.evaluate_calibrated_progress_router",
    "--raw-predictions", "artifacts\predictions\rpm10k_single_pointer_test.jsonl",
    "--base-predictions", $RpmBase,
    "--vector-predictions", $RpmVector,
    "--reference-predictions", $RpmReference,
    "--calibrated-predictions", (Join-Path $RpmCalibratedDir "predictions.jsonl"),
    "--quality-predictions", $RpmQuality,
    "--uncertainty-router-predictions", $RpmUncertaintyRouter,
    "--router", $Router,
    "--output-dir", (Join-Path (Join-Path $RouterRoot "evaluations") "rpm10k"),
    "--condition", "rpm10k",
    "--bootstrap-iterations", "$BootstrapIterations",
    "--seed", "$Seed",
    "--overwrite"
)

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.summarize_calibrated_progress",
    "--calibrator-root", $CalibratorRoot,
    "--router-root", $RouterRoot,
    "--overwrite"
)
Invoke-CheckedPython -Arguments @(
    "-m", "experiments.verify_calibrated_progress_run",
    "--calibrator-root", $CalibratorRoot,
    "--router-root", $RouterRoot
)

Write-Host "Calibrated-progress experiment completed: $RouterRoot"

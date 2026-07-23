param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunRoot = "artifacts\runs\uncertainty_router_syncg",
    [string]$Oof = "artifacts\runs\uncertainty_fusion_syncg\probabilistic_oof_clean.jsonl",
    [string]$DirectionRunDir = "artifacts\runs\probabilistic_pivot_direction_syncg\seed_20260722",
    [string]$VdnRunDir = "artifacts\runs\vdn_syncg\seed_20260720",
    [string]$QualityRunDir = "artifacts\runs\quality_router_syncg",
    [string]$FusionRunDir = "artifacts\runs\uncertainty_fusion_syncg",
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
        "-m", "experiments.train_uncertainty_router",
        "--oof-pairs", $Oof,
        "--output-dir", (Join-Path $RunRoot "model"),
        "--seed", "$Seed",
        "--overwrite"
    )
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.ablate_uncertainty_router_features",
        "--oof-pairs", $Oof,
        "--output", (Join-Path $RunRoot "feature_ablation.json"),
        "--seed", "$Seed",
        "--overwrite"
    )
}

$Router = Join-Path (Join-Path $RunRoot "model") "uncertainty_router.joblib"
$Conditions = @(
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe"
)
foreach ($Condition in $Conditions) {
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.evaluate_uncertainty_router",
        "--raw-predictions", "artifacts\predictions\robustness\syncg_test_$Condition.jsonl",
        "--base-predictions", "artifacts\runs\robustness\$Condition\predictions.jsonl",
        "--vector-predictions", (Join-Path (Join-Path $DirectionRunDir "evaluations") "$Condition.jsonl"),
        "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "$Condition.jsonl"),
        "--quality-predictions", (Join-Path (Join-Path (Join-Path $QualityRunDir "evaluations") $Condition) "predictions.jsonl"),
        "--fusion-predictions", (Join-Path (Join-Path (Join-Path $FusionRunDir "evaluations") $Condition) "predictions.jsonl"),
        "--router", $Router,
        "--output-dir", (Join-Path (Join-Path $RunRoot "evaluations") $Condition),
        "--condition", $Condition,
        "--bootstrap-iterations", "$BootstrapIterations",
        "--seed", "$Seed",
        "--overwrite"
    )
}

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.evaluate_uncertainty_router",
    "--raw-predictions", "artifacts\predictions\rpm10k_single_pointer_test.jsonl",
    "--base-predictions", "artifacts\runs\rpm10k_single_pointer_zero_shot\predictions.jsonl",
    "--vector-predictions", (Join-Path (Join-Path $DirectionRunDir "evaluations") "rpm10k.jsonl"),
    "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "rpm10k.jsonl"),
    "--quality-predictions", (Join-Path (Join-Path (Join-Path $QualityRunDir "evaluations") "rpm10k") "predictions.jsonl"),
    "--fusion-predictions", (Join-Path (Join-Path (Join-Path $FusionRunDir "evaluations") "rpm10k") "predictions.jsonl"),
    "--router", $Router,
    "--output-dir", (Join-Path (Join-Path $RunRoot "evaluations") "rpm10k"),
    "--condition", "rpm10k",
    "--bootstrap-iterations", "$BootstrapIterations",
    "--seed", "$Seed",
    "--overwrite"
)

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.summarize_uncertainty_router",
    "--run-root", $RunRoot,
    "--overwrite"
)
Invoke-CheckedPython -Arguments @(
    "-m", "experiments.verify_uncertainty_router_run",
    "--run-root", $RunRoot
)

Write-Host "Uncertainty-router experiment completed: $RunRoot"

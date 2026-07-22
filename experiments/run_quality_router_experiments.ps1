param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunRoot = "artifacts\runs\quality_router_syncg",
    [string]$DirectionRunDir = "artifacts\runs\pivot_direction_syncg\seed_20260722",
    [string]$VdnRunDir = "artifacts\runs\vdn_syncg\seed_20260720",
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
        "-m", "experiments.collect_quality_router_oof",
        "--output", (Join-Path $RunRoot "oof_clean.jsonl"),
        "--overwrite"
    )
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.train_quality_router",
        "--oof-pairs", (Join-Path $RunRoot "oof_clean.jsonl"),
        "--output-dir", (Join-Path $RunRoot "model"),
        "--seed", "$Seed",
        "--overwrite"
    )
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.ablate_quality_router_features",
        "--oof-pairs", (Join-Path $RunRoot "oof_clean.jsonl"),
        "--output", (Join-Path $RunRoot "feature_ablation.json"),
        "--seed", "$Seed",
        "--overwrite"
    )
}

$Router = Join-Path (Join-Path $RunRoot "model") "quality_router.joblib"
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
        "-m", "experiments.evaluate_quality_router",
        "--raw-predictions", "artifacts\predictions\robustness\syncg_test_$Condition.jsonl",
        "--base-predictions", "artifacts\runs\robustness\$Condition\predictions.jsonl",
        "--vector-predictions", (Join-Path (Join-Path $DirectionRunDir "evaluations") "$Condition.jsonl"),
        "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "$Condition.jsonl"),
        "--router", $Router,
        "--output-dir", (Join-Path (Join-Path $RunRoot "evaluations") $Condition),
        "--condition", $Condition,
        "--bootstrap-iterations", "$BootstrapIterations",
        "--seed", "$Seed",
        "--overwrite"
    )
}

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.evaluate_quality_router",
    "--raw-predictions", "artifacts\predictions\rpm10k_single_pointer_test.jsonl",
    "--base-predictions", "artifacts\runs\rpm10k_single_pointer_zero_shot\predictions.jsonl",
    "--vector-predictions", (Join-Path (Join-Path $DirectionRunDir "evaluations") "rpm10k.jsonl"),
    "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "rpm10k.jsonl"),
    "--router", $Router,
    "--output-dir", (Join-Path (Join-Path $RunRoot "evaluations") "rpm10k"),
    "--condition", "rpm10k",
    "--bootstrap-iterations", "$BootstrapIterations",
    "--seed", "$Seed",
    "--overwrite"
)

foreach ($ReplicateSeed in @(20260720, 20260721)) {
    $ReplicateDirectionDir = "artifacts\runs\pivot_direction_syncg\seed_$ReplicateSeed"
    foreach ($Condition in @("clean", "rpm10k")) {
        if ($Condition -eq "clean") {
            $RawPredictions = "artifacts\predictions\robustness\syncg_test_clean.jsonl"
            $BasePredictions = "artifacts\runs\robustness\clean\predictions.jsonl"
        }
        else {
            $RawPredictions = "artifacts\predictions\rpm10k_single_pointer_test.jsonl"
            $BasePredictions = "artifacts\runs\rpm10k_single_pointer_zero_shot\predictions.jsonl"
        }
        Invoke-CheckedPython -Arguments @(
            "-m", "experiments.evaluate_quality_router",
            "--raw-predictions", $RawPredictions,
            "--base-predictions", $BasePredictions,
            "--vector-predictions", (Join-Path (Join-Path $ReplicateDirectionDir "evaluations") "$Condition.jsonl"),
            "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "$Condition.jsonl"),
            "--router", $Router,
            "--output-dir", (Join-Path (Join-Path $RunRoot "evaluations_seed_$ReplicateSeed") $Condition),
            "--condition", $Condition,
            "--bootstrap-iterations", "$BootstrapIterations",
            "--seed", "$Seed",
            "--overwrite"
        )
    }
}

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.summarize_quality_router",
    "--run-root", $RunRoot,
    "--overwrite"
)
Invoke-CheckedPython -Arguments @(
    "-m", "experiments.verify_quality_router_run",
    "--run-root", $RunRoot
)

Write-Host "Quality-router experiment completed: $RunRoot"

param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [int[]]$Seeds = @(20260720, 20260721, 20260722),
    [string]$OutputRoot = "artifacts\runs\seed_stability"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = (Resolve-Path (Join-Path $ProjectRoot $Python)).Path
$OutputRootPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputRoot))
$TrainPredictions = Join-Path $ProjectRoot "artifacts\predictions\syncg_train.jsonl"

$Evaluations = [ordered]@{
    "syncg_clean" = "artifacts\predictions\robustness\syncg_test_clean.jsonl"
    "blur_moderate" = "artifacts\predictions\robustness\syncg_test_blur_moderate.jsonl"
    "blur_severe" = "artifacts\predictions\robustness\syncg_test_blur_severe.jsonl"
    "perspective_moderate" = "artifacts\predictions\robustness\syncg_test_perspective_moderate.jsonl"
    "perspective_severe" = "artifacts\predictions\robustness\syncg_test_perspective_severe.jsonl"
    "combined_severe" = "artifacts\predictions\robustness\syncg_test_combined_severe.jsonl"
    "rpm10k" = "artifacts\predictions\rpm10k_single_pointer_test.jsonl"
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Force -Path $OutputRootPath | Out-Null
Push-Location $ProjectRoot
try {
    foreach ($Seed in $Seeds) {
        $SeedRoot = Join-Path $OutputRootPath ("seed_{0}" -f $Seed)
        New-Item -ItemType Directory -Force -Path $SeedRoot | Out-Null

        if ($Seed -eq 20260720) {
            $Calibrator = Join-Path $ProjectRoot "artifacts\runs\syncg_full\calibrator.joblib"
        }
        else {
            $Calibrator = Join-Path $SeedRoot "calibrator.joblib"
            $TrainingSummary = Join-Path $SeedRoot "training_summary.json"
            if (-not (Test-Path $TrainingSummary)) {
                Invoke-CheckedPython @(
                    "-m", "experiments.selective_experiment", "fit",
                    "--train-predictions", $TrainPredictions,
                    "--output-dir", $SeedRoot,
                    "--seed", [string]$Seed,
                    "--bootstrap-iterations", "500"
                )
            }
        }

        if (-not (Test-Path $Calibrator)) {
            throw "Missing calibrator: $Calibrator"
        }
        foreach ($Entry in $Evaluations.GetEnumerator()) {
            $EvaluationRoot = Join-Path $SeedRoot $Entry.Key
            $Metrics = Join-Path $EvaluationRoot "metrics.json"
            if (Test-Path $Metrics) {
                continue
            }
            $Predictions = Join-Path $ProjectRoot $Entry.Value
            Invoke-CheckedPython @(
                "-m", "experiments.selective_experiment", "evaluate",
                "--predictions", $Predictions,
                "--calibrator", $Calibrator,
                "--output-dir", $EvaluationRoot,
                "--seed", [string]$Seed,
                "--bootstrap-iterations", "500"
            )
        }
    }
}
finally {
    Pop-Location
}

Invoke-CheckedPython @(
    "-m", "experiments.summarize_seed_stability",
    "--root", $OutputRootPath
)

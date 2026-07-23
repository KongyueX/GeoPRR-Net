param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunRoot = "artifacts\runs\probabilistic_direction_ablations",
    [string]$VdnRunDir = "artifacts\runs\vdn_syncg\seed_20260720",
    [int]$BatchSize = 24,
    [int]$EvaluationBatchSize = 64,
    [int]$BootstrapIterations = 5000,
    [int]$Seed = 20260722,
    [int]$DegradationSeed = 20260720,
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

$Ablations = @(
    @{
        Name = "no_equivariance_loss"
        EquivarianceWeight = "0.0"
        PerspectiveProbability = "0.8"
    },
    @{
        Name = "no_projective_pair"
        EquivarianceWeight = "0.0"
        PerspectiveProbability = "0.0"
    }
)
$Conditions = @("clean", "blur_severe", "perspective_severe", "combined_severe")

foreach ($Ablation in $Ablations) {
    $RunDir = Join-Path (Join-Path $RunRoot $Ablation.Name) "seed_$Seed"
    if (-not $SkipTraining) {
        Invoke-CheckedPython -Arguments @(
            "-m", "experiments.train_probabilistic_pivot_direction_syncg",
            "--manifest", "artifacts\manifests\syncg_train.jsonl",
            "--output-dir", $RunDir,
            "--epochs", "30",
            "--batch-size", "$BatchSize",
            "--workers", "4",
            "--seed", "$Seed",
            "--equivariance-weight", $Ablation.EquivarianceWeight,
            "--perspective-probability", $Ablation.PerspectiveProbability
        )
    }
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.verify_probabilistic_direction_ablation",
        "--run-dir", $RunDir,
        "--ablation-name", $Ablation.Name,
        "--expected-equivariance-weight", $Ablation.EquivarianceWeight,
        "--expected-perspective-probability", $Ablation.PerspectiveProbability,
        "--expected-batch-size", "$BatchSize",
        "--expected-seed", "$Seed"
    )
    $EvaluationDir = Join-Path $RunDir "evaluations"
    foreach ($Condition in $Conditions) {
        Invoke-CheckedPython -Arguments @(
            "-m", "experiments.evaluate_probabilistic_pivot_direction",
            "--manifest", "artifacts\manifests\syncg_test.jsonl",
            "--checkpoint", (Join-Path $RunDir "best.pt"),
            "--verification", (Join-Path $RunDir "verification.json"),
            "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "$Condition.jsonl"),
            "--output", (Join-Path $EvaluationDir "$Condition.jsonl"),
            "--condition", $Condition,
            "--degradation-seed", "$DegradationSeed",
            "--batch-size", "$EvaluationBatchSize",
            "--bootstrap-iterations", "$BootstrapIterations",
            "--seed", "$Seed",
            "--overwrite"
        )
    }
}

Write-Host "Probabilistic direction training ablations completed: $RunRoot"

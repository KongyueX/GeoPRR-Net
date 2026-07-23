param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunDir = "artifacts\runs\probabilistic_pivot_direction_syncg\seed_20260722",
    [string]$VdnRunDir = "artifacts\runs\vdn_syncg\seed_20260720",
    [int]$BatchSize = 64,
    [int]$BootstrapIterations = 5000,
    [int]$Seed = 20260722,
    [int]$DegradationSeed = 20260720,
    [switch]$Overwrite
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

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.verify_probabilistic_pivot_direction_run",
    "--run-dir", $RunDir,
    "--expected-epochs", "30",
    "--expected-batch-size", "24",
    "--expected-seed", "$Seed"
)

$Checkpoint = Join-Path $RunDir "best.pt"
$Verification = Join-Path $RunDir "verification.json"
$EvaluationDir = Join-Path $RunDir "evaluations"
New-Item -ItemType Directory -Path $EvaluationDir -Force | Out-Null
$Conditions = @(
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe"
)

foreach ($Condition in $Conditions) {
    $Output = Join-Path $EvaluationDir "$Condition.jsonl"
    $Arguments = @(
        "-m", "experiments.evaluate_probabilistic_pivot_direction",
        "--manifest", "artifacts\manifests\syncg_test.jsonl",
        "--checkpoint", $Checkpoint,
        "--verification", $Verification,
        "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "$Condition.jsonl"),
        "--output", $Output,
        "--condition", $Condition,
        "--degradation-seed", "$DegradationSeed",
        "--batch-size", "$BatchSize",
        "--bootstrap-iterations", "$BootstrapIterations",
        "--seed", "$Seed"
    )
    if ($Overwrite) {
        $Arguments += "--overwrite"
    }
    elseif (Test-Path $Output) {
        $Arguments += "--resume"
    }
    Invoke-CheckedPython -Arguments $Arguments
}

$RpmOutput = Join-Path $EvaluationDir "rpm10k.jsonl"
$RpmArguments = @(
    "-m", "experiments.evaluate_probabilistic_pivot_direction",
    "--manifest", "artifacts\manifests\rpm10k_single_pointer_test.jsonl",
    "--checkpoint", $Checkpoint,
    "--verification", $Verification,
    "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "rpm10k.jsonl"),
    "--output", $RpmOutput,
    "--condition", "clean",
    "--degradation-seed", "$DegradationSeed",
    "--batch-size", "$BatchSize",
    "--bootstrap-iterations", "$BootstrapIterations",
    "--seed", "$Seed"
)
if ($Overwrite) {
    $RpmArguments += "--overwrite"
}
elseif (Test-Path $RpmOutput) {
    $RpmArguments += "--resume"
}
Invoke-CheckedPython -Arguments $RpmArguments

Write-Host "Probabilistic direction evaluation completed: $RunDir"

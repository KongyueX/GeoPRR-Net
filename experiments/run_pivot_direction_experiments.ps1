param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunDir = "artifacts\runs\pivot_direction_syncg\seed_20260722",
    [string]$VdnRunDir = "artifacts\runs\vdn_syncg\seed_20260720",
    [int]$BatchSize = 64,
    [int]$BootstrapIterations = 2000,
    [int]$Seed = 20260722,
    [int]$DegradationSeed = 20260720
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
    "-m", "experiments.verify_pivot_direction_run",
    "--run-dir", $RunDir,
    "--expected-epochs", "30",
    "--expected-batch-size", "48",
    "--expected-seed", "$Seed"
)

$Conditions = @(
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe"
)
$Checkpoint = Join-Path $RunDir "best.pt"
$Verification = Join-Path $RunDir "verification.json"
$EvaluationDir = Join-Path $RunDir "evaluations"
New-Item -ItemType Directory -Path $EvaluationDir -Force | Out-Null

foreach ($Condition in $Conditions) {
    $Output = Join-Path $EvaluationDir "$Condition.jsonl"
    $Reference = Join-Path (Join-Path $VdnRunDir "evaluations") "$Condition.jsonl"
    $Arguments = @(
        "-m", "experiments.evaluate_pivot_direction_fallback",
        "--manifest", "artifacts\manifests\syncg_test.jsonl",
        "--checkpoint", $Checkpoint,
        "--verification", $Verification,
        "--reference-predictions", $Reference,
        "--output", $Output,
        "--condition", $Condition,
        "--degradation-seed", "$DegradationSeed",
        "--batch-size", "$BatchSize",
        "--bootstrap-iterations", "$BootstrapIterations",
        "--seed", "$Seed"
    )
    if (Test-Path $Output) {
        $Arguments += "--resume"
    }
    Invoke-CheckedPython -Arguments $Arguments
}

$RpmOutput = Join-Path $EvaluationDir "rpm10k.jsonl"
$RpmReference = Join-Path (Join-Path $VdnRunDir "evaluations") "rpm10k.jsonl"
$RpmArguments = @(
    "-m", "experiments.evaluate_pivot_direction_fallback",
    "--manifest", "artifacts\manifests\rpm10k_single_pointer_test.jsonl",
    "--checkpoint", $Checkpoint,
    "--verification", $Verification,
    "--reference-predictions", $RpmReference,
    "--output", $RpmOutput,
    "--condition", "clean",
    "--degradation-seed", "$DegradationSeed",
    "--batch-size", "$BatchSize",
    "--bootstrap-iterations", "$BootstrapIterations",
    "--seed", "$Seed"
)
if (Test-Path $RpmOutput) {
    $RpmArguments += "--resume"
}
Invoke-CheckedPython -Arguments $RpmArguments

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.summarize_dual_route",
    "--fallback-root", $RunDir,
    "--vdn-root", $VdnRunDir,
    "--bootstrap-iterations", "$BootstrapIterations",
    "--seed", "$Seed"
)

Write-Host "Pivot-direction fallback experiment completed: $RunDir"

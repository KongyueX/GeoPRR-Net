param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunRoot = "artifacts\runs\probabilistic_pivot_direction_syncg",
    [string]$VdnRunDir = "artifacts\runs\vdn_syncg\seed_20260720",
    [int[]]$Seeds = @(20260720, 20260721, 20260722),
    [int]$BatchSize = 64,
    [int]$BootstrapIterations = 5000,
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

foreach ($Seed in $Seeds) {
    $RunDir = Join-Path $RunRoot "seed_$Seed"
    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.verify_probabilistic_pivot_direction_run",
        "--run-dir", $RunDir,
        "--expected-epochs", "30",
        "--expected-batch-size", "24",
        "--expected-seed", "$Seed"
    )
    foreach ($Condition in @("clean", "rpm10k")) {
        if ($Condition -eq "clean") {
            $Manifest = "artifacts\manifests\syncg_test.jsonl"
            $EvaluationCondition = "clean"
        }
        else {
            $Manifest = "artifacts\manifests\rpm10k_single_pointer_test.jsonl"
            $EvaluationCondition = "clean"
        }
        $Output = Join-Path (Join-Path $RunDir "evaluations") "$Condition.jsonl"
        $Arguments = @(
            "-m", "experiments.evaluate_probabilistic_pivot_direction",
            "--manifest", $Manifest,
            "--checkpoint", (Join-Path $RunDir "best.pt"),
            "--verification", (Join-Path $RunDir "verification.json"),
            "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "$Condition.jsonl"),
            "--output", $Output,
            "--condition", $EvaluationCondition,
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
}

Invoke-CheckedPython -Arguments @(
    "-m", "experiments.summarize_probabilistic_direction_replicates",
    "--run-root", $RunRoot,
    "--overwrite"
)

Write-Host "Probabilistic direction replicate evaluation completed: $RunRoot"

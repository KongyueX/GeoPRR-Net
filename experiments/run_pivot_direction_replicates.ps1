param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [int[]]$Seeds = @(20260720, 20260721),
    [int]$Epochs = 30,
    [int]$TrainBatchSize = 48,
    [int]$EvalBatchSize = 64,
    [int]$Workers = 4,
    [int]$BootstrapIterations = 2000,
    [int]$BootstrapSeed = 20260722,
    [int]$DegradationSeed = 20260720,
    [string]$VdnRunDir = "artifacts\runs\vdn_syncg\seed_20260720",
    [int[]]$AggregateSeeds = @(20260720, 20260721, 20260722)
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
    $RunDir = "artifacts\runs\pivot_direction_syncg\seed_$Seed"
    $Summary = Join-Path $RunDir "summary.json"
    $TrainingComplete = $false
    if (Test-Path $Summary) {
        $SummaryData = Get-Content $Summary -Raw | ConvertFrom-Json
        $TrainingComplete = $SummaryData.status -eq "complete"
    }
    if (-not $TrainingComplete) {
        $TrainArguments = @(
            "-m", "experiments.train_pivot_direction_syncg",
            "--output-dir", $RunDir,
            "--epochs", "$Epochs",
            "--batch-size", "$TrainBatchSize",
            "--workers", "$Workers",
            "--seed", "$Seed"
        )
        if (Test-Path (Join-Path $RunDir "last.pt")) {
            $TrainArguments += "--resume"
        }
        Invoke-CheckedPython -Arguments $TrainArguments
    }

    Invoke-CheckedPython -Arguments @(
        "-m", "experiments.verify_pivot_direction_run",
        "--run-dir", $RunDir,
        "--expected-epochs", "$Epochs",
        "--expected-batch-size", "$TrainBatchSize",
        "--expected-seed", "$Seed"
    )

    $EvaluationDir = Join-Path $RunDir "evaluations"
    New-Item -ItemType Directory -Path $EvaluationDir -Force | Out-Null
    $Checkpoint = Join-Path $RunDir "best.pt"
    $Verification = Join-Path $RunDir "verification.json"

    $CleanOutput = Join-Path $EvaluationDir "clean.jsonl"
    $CleanArguments = @(
        "-m", "experiments.evaluate_pivot_direction_fallback",
        "--manifest", "artifacts\manifests\syncg_test.jsonl",
        "--checkpoint", $Checkpoint,
        "--verification", $Verification,
        "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "clean.jsonl"),
        "--output", $CleanOutput,
        "--condition", "clean",
        "--degradation-seed", "$DegradationSeed",
        "--batch-size", "$EvalBatchSize",
        "--bootstrap-iterations", "$BootstrapIterations",
        "--seed", "$Seed"
    )
    if (Test-Path $CleanOutput) {
        $CleanArguments += "--resume"
    }
    Invoke-CheckedPython -Arguments $CleanArguments

    $RpmOutput = Join-Path $EvaluationDir "rpm10k.jsonl"
    $RpmArguments = @(
        "-m", "experiments.evaluate_pivot_direction_fallback",
        "--manifest", "artifacts\manifests\rpm10k_single_pointer_test.jsonl",
        "--checkpoint", $Checkpoint,
        "--verification", $Verification,
        "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "rpm10k.jsonl"),
        "--output", $RpmOutput,
        "--condition", "clean",
        "--degradation-seed", "$DegradationSeed",
        "--batch-size", "$EvalBatchSize",
        "--bootstrap-iterations", "$BootstrapIterations",
        "--seed", "$Seed"
    )
    if (Test-Path $RpmOutput) {
        $RpmArguments += "--resume"
    }
    Invoke-CheckedPython -Arguments $RpmArguments
}

$SummaryArguments = @(
    "-m", "experiments.summarize_pivot_direction_replicates",
    "--root", "artifacts\runs\pivot_direction_syncg",
    "--vdn-root", $VdnRunDir,
    "--bootstrap-iterations", "$BootstrapIterations",
    "--bootstrap-seed", "$BootstrapSeed",
    "--seeds"
)
$SummaryArguments += @($AggregateSeeds | ForEach-Object { "$_" })
Invoke-CheckedPython -Arguments $SummaryArguments

Write-Host "Pivot-direction replicate runs completed: $($Seeds -join ', ')"

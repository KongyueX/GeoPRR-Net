param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunDir = "artifacts\runs\probabilistic_pivot_direction_syncg\seed_20260722",
    [string]$VdnRunDir = "artifacts\runs\vdn_syncg\seed_20260720",
    [int]$BatchSize = 64,
    [int]$BootstrapIterations = 5000,
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

$Conditions = @("clean", "blur_severe", "perspective_severe", "combined_severe")
foreach ($Decoder in @("direct", "circular")) {
    $OutputRoot = Join-Path (Join-Path $RunDir "decoder_ablations") $Decoder
    foreach ($Condition in $Conditions) {
        Invoke-CheckedPython -Arguments @(
            "-m", "experiments.evaluate_probabilistic_pivot_direction",
            "--manifest", "artifacts\manifests\syncg_test.jsonl",
            "--checkpoint", (Join-Path $RunDir "best.pt"),
            "--verification", (Join-Path $RunDir "verification.json"),
            "--reference-predictions", (Join-Path (Join-Path $VdnRunDir "evaluations") "$Condition.jsonl"),
            "--output", (Join-Path $OutputRoot "$Condition.jsonl"),
            "--condition", $Condition,
            "--direction-decoder", $Decoder,
            "--degradation-seed", "$DegradationSeed",
            "--batch-size", "$BatchSize",
            "--bootstrap-iterations", "$BootstrapIterations",
            "--seed", "$Seed",
            "--overwrite"
        )
    }
}

Write-Host "Frozen decoder ablations completed: $RunDir"

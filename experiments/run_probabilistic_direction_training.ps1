param(
    [int[]]$Seeds = @(20260720, 20260721, 20260722),
    [int]$Epochs = 30,
    [int]$BatchSize = 24,
    [int]$Workers = 4,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Manifest = Join-Path $ProjectRoot "artifacts\manifests\syncg_train.jsonl"
$RunRoot = Join-Path $ProjectRoot "artifacts\runs\probabilistic_pivot_direction_syncg"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Training interpreter does not exist: $Python"
}

Push-Location $ProjectRoot
try {
    foreach ($Seed in $Seeds) {
        $RunDir = Join-Path $RunRoot "seed_$Seed"
        $Arguments = @(
            "-m", "experiments.train_probabilistic_pivot_direction_syncg",
            "--manifest", $Manifest,
            "--output-dir", $RunDir,
            "--epochs", "$Epochs",
            "--batch-size", "$BatchSize",
            "--workers", "$Workers",
            "--seed", "$Seed"
        )
        if ($Resume) {
            $Arguments += "--resume"
        }
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Probabilistic direction training failed for seed $Seed"
        }
        & $Python -m experiments.verify_probabilistic_pivot_direction_run `
            --run-dir $RunDir `
            --manifest $Manifest `
            --expected-epochs $Epochs `
            --expected-batch-size $BatchSize `
            --expected-seed $Seed
        if ($LASTEXITCODE -ne 0) {
            throw "Probabilistic direction verification failed for seed $Seed"
        }
    }
}
finally {
    Pop-Location
}

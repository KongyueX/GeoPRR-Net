param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [int[]]$Seeds = @(20260721, 20260722),
    [int]$PollSeconds = 30,
    [double]$WaitTimeoutHoursPerSeed = 8.0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = (Resolve-Path (Join-Path $ProjectRoot $Python)).Path
$EvaluationScript = Join-Path $PSScriptRoot "run_vdn_evaluations.ps1"

function Get-CompletedSummary {
    param([string]$RunRoot)

    $SummaryPath = Join-Path $RunRoot "summary.json"
    if (-not (Test-Path $SummaryPath)) {
        return $null
    }
    $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    if ($Summary.status -eq "complete") {
        return $Summary
    }
    return $null
}

function Wait-ForExistingTraining {
    param(
        [string]$RunRoot,
        [DateTime]$Deadline
    )

    $PidPath = Join-Path $RunRoot "train.pid"
    if (-not (Test-Path $PidPath)) {
        return
    }
    $TrainingPid = [int](Get-Content $PidPath -Raw)
    while ((Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) -and
           -not (Get-CompletedSummary -RunRoot $RunRoot)) {
        if ([DateTime]::UtcNow -ge $Deadline) {
            throw "Timed out waiting for VDN training process $TrainingPid"
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

Push-Location $ProjectRoot
try {
    foreach ($Seed in $Seeds) {
        $RelativeRunRoot = "artifacts\runs\vdn_syncg\seed_$Seed"
        $RunRoot = [System.IO.Path]::GetFullPath(
            (Join-Path $ProjectRoot $RelativeRunRoot)
        )
        New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
        $Deadline = [DateTime]::UtcNow.AddHours($WaitTimeoutHoursPerSeed)

        Wait-ForExistingTraining -RunRoot $RunRoot -Deadline $Deadline
        if (-not (Get-CompletedSummary -RunRoot $RunRoot)) {
            $Arguments = @(
                "-m", "experiments.train_vdn_syncg",
                "--epochs", "100",
                "--batch-size", "8",
                "--workers", "4",
                "--seed", "$Seed",
                "--output-dir", $RunRoot
            )
            if (Test-Path (Join-Path $RunRoot "last.pt")) {
                $Arguments += "--resume"
            }
            & $PythonPath @Arguments
            if ($LASTEXITCODE -ne 0) {
                throw "VDN seed $Seed training failed with exit code $LASTEXITCODE"
            }
        }

        & powershell -NoProfile -ExecutionPolicy Bypass -File $EvaluationScript `
            -Python $Python `
            -RunRoot $RelativeRunRoot `
            -ExpectedSeed $Seed `
            -PollSeconds $PollSeconds `
            -WaitTimeoutHours $WaitTimeoutHoursPerSeed
        if ($LASTEXITCODE -ne 0) {
            throw "VDN seed $Seed evaluation failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}

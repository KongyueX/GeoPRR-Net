param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunRoot = "artifacts\runs\vdn_syncg\seed_20260720",
    [int]$PollSeconds = 30,
    [double]$WaitTimeoutHours = 8.0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = (Resolve-Path (Join-Path $ProjectRoot $Python)).Path
$RunRootPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $RunRoot))
$TrainingSummary = Join-Path $RunRootPath "summary.json"
$Checkpoint = Join-Path $RunRootPath "best.pt"
$EvaluationRoot = Join-Path $RunRootPath "evaluations"

$Evaluations = @(
    @(
        "clean",
        "artifacts\manifests\syncg_test.jsonl",
        "artifacts\predictions\robustness\syncg_test_clean.jsonl",
        "clean"
    ),
    @(
        "blur_moderate",
        "artifacts\manifests\syncg_test.jsonl",
        "artifacts\predictions\robustness\syncg_test_blur_moderate.jsonl",
        "blur_moderate"
    ),
    @(
        "blur_severe",
        "artifacts\manifests\syncg_test.jsonl",
        "artifacts\predictions\robustness\syncg_test_blur_severe.jsonl",
        "blur_severe"
    ),
    @(
        "perspective_moderate",
        "artifacts\manifests\syncg_test.jsonl",
        "artifacts\predictions\robustness\syncg_test_perspective_moderate.jsonl",
        "perspective_moderate"
    ),
    @(
        "perspective_severe",
        "artifacts\manifests\syncg_test.jsonl",
        "artifacts\predictions\robustness\syncg_test_perspective_severe.jsonl",
        "perspective_severe"
    ),
    @(
        "combined_severe",
        "artifacts\manifests\syncg_test.jsonl",
        "artifacts\predictions\robustness\syncg_test_combined_severe.jsonl",
        "combined_severe"
    ),
    @(
        "rpm10k",
        "artifacts\manifests\rpm10k_single_pointer_test.jsonl",
        "artifacts\predictions\rpm10k_single_pointer_test.jsonl",
        "clean"
    )
)

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

$Deadline = [DateTime]::UtcNow.AddHours($WaitTimeoutHours)
while ($true) {
    if (Test-Path $TrainingSummary) {
        $Summary = Get-Content $TrainingSummary -Raw | ConvertFrom-Json
        if ($Summary.status -eq "complete") {
            break
        }
    }
    if ([DateTime]::UtcNow -ge $Deadline) {
        throw "Timed out waiting for completed VDN training: $TrainingSummary"
    }
    Start-Sleep -Seconds $PollSeconds
}
if (-not (Test-Path $Checkpoint)) {
    throw "Completed training has no best checkpoint: $Checkpoint"
}

New-Item -ItemType Directory -Force -Path $EvaluationRoot | Out-Null
Push-Location $ProjectRoot
try {
    Invoke-CheckedPython @(
        "-m", "experiments.verify_vdn_run",
        "--run-dir", $RunRootPath,
        "--expected-epochs", "100",
        "--expected-batch-size", "8",
        "--expected-seed", "20260720"
    )
    foreach ($Evaluation in $Evaluations) {
        $Name = $Evaluation[0]
        $Manifest = Join-Path $ProjectRoot $Evaluation[1]
        $Shared = Join-Path $ProjectRoot $Evaluation[2]
        $Condition = $Evaluation[3]
        $Output = Join-Path $EvaluationRoot ("{0}.jsonl" -f $Name)
        $SummaryOutput = Join-Path $EvaluationRoot ("{0}.summary.json" -f $Name)
        if (Test-Path $SummaryOutput) {
            continue
        }
        $Arguments = @(
            "-m", "experiments.evaluate_vdn_baseline",
            "--manifest", $Manifest,
            "--checkpoint", $Checkpoint,
            "--shared-predictions", $Shared,
            "--output", $Output,
            "--condition", $Condition,
            "--batch-size", "16",
            "--bootstrap-iterations", "2000"
        )
        if (Test-Path $Output) {
            $Arguments += "--resume"
        }
        Invoke-CheckedPython $Arguments
    }
    Invoke-CheckedPython @(
        "-m", "experiments.summarize_vdn_comparison",
        "--vdn-root", $RunRootPath,
        "--bootstrap-iterations", "2000"
    )
}
finally {
    Pop-Location
}

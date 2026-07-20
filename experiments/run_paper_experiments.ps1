[CmdletBinding()]
param(
    [string]$SyncGRoot = "datasets\SyncG",
    [string]$RpmImageRoot = "datasets\RPM10K\images",
    [string]$RpmLabels = "datasets\RPM10K\labels\test.json",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [ValidateRange(1, 200)]
    [int]$SegmentationEpochs = 15,
    [ValidateRange(1, 128)]
    [int]$SegmentationBatchSize = 8,
    [ValidateRange(0, 64)]
    [int]$Workers = 4,
    [ValidateRange(2, 20)]
    [int]$Folds = 5,
    [int]$Seed = 20260720,
    [switch]$RunFeatureAblations
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Host ""
    Write-Host ("python " + ($Arguments -join " "))
    & $script:PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Collect-Predictions {
    param(
        [Parameter(Mandatory = $true)][string]$Manifest,
        [Parameter(Mandatory = $true)][string]$Output,
        [Parameter(Mandatory = $true)][string]$SegmentationWeights
    )
    $arguments = @(
        "-m", "experiments.collect_predictions",
        "--manifest", $Manifest,
        "--output", $Output,
        "--segmentation-weights", $SegmentationWeights,
        "--correction-mode", "off",
        "--device", $Device
    )
    if (Test-Path -LiteralPath $Output -PathType Leaf) {
        $arguments += "--resume"
    }
    Invoke-CheckedPython @arguments
}

$script:PythonPath = Resolve-ProjectPath $Python
$syncgRootPath = Resolve-ProjectPath $SyncGRoot
$rpmImageRootPath = Resolve-ProjectPath $RpmImageRoot
$rpmLabelsPath = Resolve-ProjectPath $RpmLabels
$releasedSegmentationWeights = Resolve-ProjectPath `
    "utils\angleDetect\pointerSeg\resultSeg\best.pt"

if (-not (Test-Path -LiteralPath $script:PythonPath -PathType Leaf)) {
    throw "Python environment not found: $script:PythonPath"
}
if (-not (Test-Path -LiteralPath $syncgRootPath -PathType Container)) {
    throw "SyncG is not extracted below: $syncgRootPath"
}
if (-not (Test-Path -LiteralPath $rpmImageRootPath -PathType Container)) {
    throw "RPM-10K image directory not found: $rpmImageRootPath"
}
if (-not (Test-Path -LiteralPath $rpmLabelsPath -PathType Leaf)) {
    throw "RPM-10K labels not found: $rpmLabelsPath"
}
if (-not (Test-Path -LiteralPath $releasedSegmentationWeights -PathType Leaf)) {
    throw "released segmentation weights not found: $releasedSegmentationWeights"
}

$manifestDir = Resolve-ProjectPath "artifacts\manifests"
$predictionDir = Resolve-ProjectPath "artifacts\predictions"
$runDir = Resolve-ProjectPath "artifacts\runs"
foreach ($directory in @($manifestDir, $predictionDir, $runDir)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

$syncgTrainManifest = Join-Path $manifestDir "syncg_train.jsonl"
$syncgTestManifest = Join-Path $manifestDir "syncg_test.jsonl"
$rpmTestManifest = Join-Path $manifestDir "rpm10k_single_pointer_test.jsonl"

Push-Location $projectRoot
try {
    Invoke-CheckedPython `
        "-m" "experiments.datasets" "syncg" `
        "--root" $syncgRootPath `
        "--split" "train" `
        "--output" $syncgTrainManifest
    Invoke-CheckedPython `
        "-m" "experiments.datasets" "syncg" `
        "--root" $syncgRootPath `
        "--split" "test" `
        "--output" $syncgTestManifest
    Invoke-CheckedPython `
        "-m" "experiments.datasets" "rpm10k" `
        "--root" $rpmImageRootPath `
        "--labels" $rpmLabelsPath `
        "--output" $rpmTestManifest

    $segmentationRun = Join-Path $runDir "syncg_segmentation"
    $segmentationBest = Join-Path $segmentationRun "best.pt"
    $releasedCalibrated = Join-Path $segmentationRun "released_calibrated.pt"
    $segmentationSummary = Join-Path $segmentationRun "summary.json"
    if (
        (Test-Path -LiteralPath $segmentationBest -PathType Leaf) -and
        (Test-Path -LiteralPath $releasedCalibrated -PathType Leaf) -and
        (Test-Path -LiteralPath $segmentationSummary -PathType Leaf)
    ) {
        Write-Host "verified existing completed segmentation run: $segmentationRun"
    }
    else {
        $trainArguments = @(
            "-m", "experiments.train_syncg_segmentation",
            "--root", $syncgRootPath,
            "--output-dir", $segmentationRun,
            "--device", $Device,
            "--epochs", "$SegmentationEpochs",
            "--batch-size", "$SegmentationBatchSize",
            "--workers", "$Workers",
            "--seed", "$Seed"
        )
        if (Test-Path -LiteralPath (Join-Path $segmentationRun "last.pt") -PathType Leaf) {
            $trainArguments += "--resume"
        }
        Invoke-CheckedPython @trainArguments
    }

    Invoke-CheckedPython `
        "-m" "experiments.verify_segmentation_run" `
        "--run-dir" $segmentationRun `
        "--initial-weights" $releasedSegmentationWeights `
        "--require-formal-syncg" `
        "--expected-epochs" "$SegmentationEpochs" `
        "--expected-batch-size" "$SegmentationBatchSize" `
        "--expected-seed" "$Seed"

    Invoke-CheckedPython `
        "-m" "experiments.evaluate_syncg_segmentation" `
        "--root" $syncgRootPath `
        "--checkpoint" "released=$releasedCalibrated" `
        "--checkpoint" "finetuned=$segmentationBest" `
        "--output" (Join-Path $segmentationRun "syncg_test_metrics.json") `
        "--device" $Device `
        "--batch-size" "$($SegmentationBatchSize * 2)" `
        "--workers" "$Workers"

    $syncgTrainPredictions = Join-Path $predictionDir "syncg_train.jsonl"
    $syncgTestPredictions = Join-Path $predictionDir "syncg_test.jsonl"
    $rpmTestPredictions = Join-Path $predictionDir "rpm10k_single_pointer_test.jsonl"
    $rpmReleasedPredictions = Join-Path $predictionDir `
        "rpm10k_single_pointer_test_released_segmentation.jsonl"
    Collect-Predictions $syncgTrainManifest $syncgTrainPredictions $segmentationBest
    Collect-Predictions $syncgTestManifest $syncgTestPredictions $segmentationBest
    Collect-Predictions $rpmTestManifest $rpmTestPredictions $segmentationBest
    Collect-Predictions `
        $rpmTestManifest `
        $rpmReleasedPredictions `
        $releasedCalibrated

    Invoke-CheckedPython `
        "-m" "experiments.make_failure_table" `
        "--dataset" "SyncG=$syncgTestPredictions" `
        "--dataset" "RPM-10K single-pointer=$rpmTestPredictions" `
        "--output" (Join-Path $runDir "failure_table.md")
    Invoke-CheckedPython `
        "-m" "experiments.make_frontend_transfer_table" `
        "--released" $rpmReleasedPredictions `
        "--finetuned" $rpmTestPredictions `
        "--output" (Join-Path $runDir "frontend_transfer_table.md") `
        "--seed" "$Seed"

    $fitDir = Join-Path $runDir "syncg_full"
    Invoke-CheckedPython `
        "-m" "experiments.selective_experiment" "fit" `
        "--train-predictions" $syncgTrainPredictions `
        "--output-dir" $fitDir `
        "--feature-set" "full" `
        "--folds" "$Folds" `
        "--seed" "$Seed"

    $calibrator = Join-Path $fitDir "calibrator.joblib"
    $syncgEvaluation = Join-Path $runDir "syncg_test"
    $rpmEvaluation = Join-Path $runDir "rpm10k_single_pointer_zero_shot"
    Invoke-CheckedPython `
        "-m" "experiments.selective_experiment" "evaluate" `
        "--predictions" $syncgTestPredictions `
        "--calibrator" $calibrator `
        "--output-dir" $syncgEvaluation `
        "--seed" "$Seed"
    Invoke-CheckedPython `
        "-m" "experiments.selective_experiment" "evaluate" `
        "--predictions" $rpmTestPredictions `
        "--calibrator" $calibrator `
        "--output-dir" $rpmEvaluation `
        "--seed" "$Seed"

    Invoke-CheckedPython `
        "-m" "experiments.plot_risk_coverage" `
        "--dataset" "SyncG=$(Join-Path $syncgEvaluation "risk_coverage.csv")" `
        "--dataset" "RPM-10K single-pointer=$(Join-Path $rpmEvaluation "risk_coverage.csv")" `
        "--output" (Join-Path $runDir "risk_coverage.png")

    Invoke-CheckedPython `
        "-m" "experiments.make_paper_table" `
        "--syncg" (Join-Path $syncgEvaluation "metrics.json") `
        "--external" (Join-Path $rpmEvaluation "metrics.json") `
        "--external-name" "RPM-10K single-pointer" `
        "--output" (Join-Path $runDir "main_table.md")

    if ($RunFeatureAblations) {
        $ablationTableArguments = @(
            "-m", "experiments.make_ablation_table",
            "--full", (Join-Path $syncgEvaluation "metrics.json"),
            "--output", (Join-Path $runDir "ablation_table.md")
        )
        foreach ($featureSet in @("geometry", "no_mask", "no_ellipse", "disagreement")) {
            $ablationFit = Join-Path $runDir "ablation_$featureSet"
            $ablationEvaluation = Join-Path $runDir "ablation_${featureSet}_syncg_test"
            Invoke-CheckedPython `
                "-m" "experiments.selective_experiment" "fit" `
                "--train-predictions" $syncgTrainPredictions `
                "--output-dir" $ablationFit `
                "--feature-set" $featureSet `
                "--folds" "$Folds" `
                "--seed" "$Seed"
            Invoke-CheckedPython `
                "-m" "experiments.selective_experiment" "evaluate" `
                "--predictions" $syncgTestPredictions `
                "--calibrator" (Join-Path $ablationFit "calibrator.joblib") `
                "--output-dir" $ablationEvaluation `
                "--seed" "$Seed"
            $ablationTableArguments += @(
                "--variant",
                "$featureSet=$(Join-Path $ablationEvaluation "metrics.json")"
            )
        }
        Invoke-CheckedPython @ablationTableArguments
    }

    Write-Host ""
    Write-Host "paper experiment pipeline complete"
    Write-Host (Join-Path $runDir "main_table.md")
}
finally {
    Pop-Location
}

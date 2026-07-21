[CmdletBinding()]
param(
    [string]$Manifest = "artifacts\manifests\syncg_test.jsonl",
    [string]$Calibrator = "artifacts\runs\syncg_full\calibrator.joblib",
    [string]$SegmentationWeights = "artifacts\runs\syncg_segmentation\best.pt",
    [string]$RpmMetrics = "artifacts\runs\rpm10k_single_pointer_zero_shot\metrics.json",
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [int]$Seed = 20260720,
    [ValidateRange(0, 1000000)]
    [int]$Limit = 0,
    [ValidateRange(0, 100000)]
    [int]$BootstrapIterations = 2000,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$conditions = @(
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe"
)

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

$script:PythonPath = Resolve-ProjectPath $Python
$manifestPath = Resolve-ProjectPath $Manifest
$calibratorPath = Resolve-ProjectPath $Calibrator
$segmentationPath = Resolve-ProjectPath $SegmentationWeights
$rpmMetricsPath = Resolve-ProjectPath $RpmMetrics
$predictionRoot = Resolve-ProjectPath "artifacts\predictions\robustness"
$runRoot = Resolve-ProjectPath "artifacts\runs\robustness"

foreach ($path in @($script:PythonPath, $manifestPath, $calibratorPath, $segmentationPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "required file not found: $path"
    }
}
New-Item -ItemType Directory -Force -Path $predictionRoot, $runRoot | Out-Null

Push-Location $projectRoot
try {
    $reportArguments = @("-m", "experiments.make_robustness_report")
    foreach ($condition in $conditions) {
        $predictionPath = Join-Path $predictionRoot "syncg_test_${condition}.jsonl"
        $evaluationDir = Join-Path $runRoot $condition
        $collectArguments = @(
            "-m", "experiments.collect_predictions",
            "--manifest", $manifestPath,
            "--output", $predictionPath,
            "--segmentation-weights", $segmentationPath,
            "--correction-mode", "off",
            "--degradation", $condition,
            "--degradation-seed", "$Seed",
            "--log-level", "WARNING",
            "--device", $Device
        )
        if ($Limit -gt 0) {
            $collectArguments += @("--limit", "$Limit")
        }
        if ($Overwrite) {
            $collectArguments += "--overwrite"
        }
        elseif (Test-Path -LiteralPath $predictionPath -PathType Leaf) {
            $collectArguments += "--resume"
        }
        Invoke-CheckedPython @collectArguments

        Invoke-CheckedPython `
            "-m" "experiments.selective_experiment" "evaluate" `
            "--predictions" $predictionPath `
            "--calibrator" $calibratorPath `
            "--output-dir" $evaluationDir `
            "--seed" "$Seed" `
            "--bootstrap-iterations" "$BootstrapIterations"

        $reportArguments += @(
            "--condition",
            "$condition=$(Join-Path $evaluationDir 'metrics.json')"
        )
    }

    if (Test-Path -LiteralPath $rpmMetricsPath -PathType Leaf) {
        $reportArguments += @("--rpm-metrics", $rpmMetricsPath)
    }
    $expectedSamples = if ($Limit -gt 0) { $Limit } else { 4000 }
    $reportArguments += @(
        "--expected-samples", "$expectedSamples",
        "--bootstrap-iterations", "$BootstrapIterations",
        "--output", (Join-Path $runRoot "robustness_table.md"),
        "--plot", (Join-Path $runRoot "robustness_curves.png")
    )
    Invoke-CheckedPython @reportArguments

    Write-Host ""
    Write-Host "robustness experiment pipeline complete"
    Write-Host (Join-Path $runRoot "robustness_table.md")
}
finally {
    Pop-Location
}

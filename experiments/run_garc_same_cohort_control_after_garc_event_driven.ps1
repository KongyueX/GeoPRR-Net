#requires -Version 7.0

[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = (Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"),
    [string]$Protocol = (Join-Path (Split-Path -Parent $PSScriptRoot) "experiments\garc_common_split_progress_protocol.json"),
    [string]$GarcOutputRoot = "C:\pointer_read\garc_full_auto_formal_v1",
    [string]$GarcSummary = "C:\pointer_read\garc_full_auto_formal_v1\summary.json",
    [string]$Preregistration = "C:\pointer_read\garc_same_cohort_control_launcher\preregistration.v2.json",
    [string]$PreflightOutput = "C:\pointer_read\garc_same_cohort_control_launcher\preflight.v2.json",
    [string]$ControlOutputRoot = "C:\pointer_read\garc_same_cohort_control_v1",
    [string]$PromotionOutputRoot = "C:\pointer_read\garc_common_split_promotion_v1",
    [ValidateRange(0, 2147483647)]
    [int]$WaitForPid = 0,
    [string]$ExpectedWaitProcessStartUtc = "",
    [ValidateRange(1, 16)]
    [int]$TorchCpuThreads = 4,
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Auditor = Join-Path $ProjectRoot "experiments\garc_same_cohort_control.py"
$Runner = Join-Path $ProjectRoot "experiments\garc_full_auto_public.py"
$Evaluator = Join-Path $ProjectRoot "experiments\evaluate_garc_full_auto_public.py"
$Promotion = Join-Path $ProjectRoot "experiments\garc_common_split_promotion.py"
$ParentEventChain = Join-Path `
    $ProjectRoot "experiments\run_garc_full_auto_public_event_driven.ps1"
$ExpectedProcessPattern = "run_garc_full_auto_public_event_driven\.ps1"
$CurrentStage = "static-preflight"

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $LiteralPath).Hash.ToLowerInvariant()
}

function Get-StringSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)
    $Algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $Bytes = [Text.Encoding]::UTF8.GetBytes($Value)
        return [Convert]::ToHexString(
            $Algorithm.ComputeHash($Bytes)
        ).ToLowerInvariant()
    } finally {
        $Algorithm.Dispose()
    }
}

function Assert-RequiredFile {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)
    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        throw "Required SAME-412 control dependency is absent: $LiteralPath"
    }
}

function Assert-SafePointerReadPath {
    param(
        [Parameter(Mandatory = $true)][string]$LiteralPath,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $Resolved = [IO.Path]::GetFullPath($LiteralPath)
    if (-not $Resolved.StartsWith(
        "C:\pointer_read\", [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Label escaped C:\pointer_read: $Resolved"
    }
    if ($Resolved.TrimEnd('\') -ieq "C:\pointer_read") {
        throw "Refusing broad C:\pointer_read target for $Label."
    }
}

function ConvertTo-UtcIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not $Value.EndsWith("Z", [StringComparison]::Ordinal)) {
        throw "$Label must be a UTC ISO-8601 value ending in Z."
    }
    return [DateTimeOffset]::Parse(
        $Value,
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::RoundtripKind
    ).UtcDateTime
}

function Get-AuthenticatedGarcProcess {
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [Parameter(Mandatory = $true)][string]$ExpectedStartUtc,
        [switch]$AllowAbsent
    )
    if ($ProcessId -le 0) {
        throw "A positive authenticated GARC PID is required."
    }
    $ExpectedStart = ConvertTo-UtcIdentity `
        -Value $ExpectedStartUtc -Label "ExpectedWaitProcessStartUtc"
    $Cim = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $Cim) {
        if ($AllowAbsent) {
            return $null
        }
        throw "Authenticated GARC PID $ProcessId is absent."
    }
    $CommandLine = [string]$Cim.CommandLine
    if ($CommandLine -notmatch $ExpectedProcessPattern) {
        throw "PID $ProcessId is not the GARC formal event chain."
    }
    $ObservedStart = ([datetime]$Cim.CreationDate).ToUniversalTime()
    if ([Math]::Abs(($ObservedStart - $ExpectedStart).TotalSeconds) -gt 1.0) {
        throw "GARC PID start-time identity drift."
    }
    $Process = [Diagnostics.Process]::GetProcessById($ProcessId)
    $ProcessStart = $Process.StartTime.ToUniversalTime()
    if ([Math]::Abs(($ProcessStart - $ExpectedStart).TotalSeconds) -gt 0.001) {
        throw "GARC process-object start-time identity drift."
    }
    if ([Math]::Abs(($ProcessStart - $ObservedStart).TotalSeconds) -gt 1.0) {
        throw "GARC CIM/process start-time identity drift."
    }
    return [pscustomobject]@{
        Cim = $Cim
        ProcessId = $ProcessId
        StartUtc = $ProcessStart
        CimStartUtc = $ObservedStart
        CommandLine = $CommandLine
        CommandLineSha256 = Get-StringSha256 -Value $CommandLine
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$LogPath = ""
    )
    $script:CurrentStage = $Stage
    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        & $Python @Arguments
    } else {
        & $Python @Arguments 2>&1 |
            Tee-Object -LiteralPath $LogPath -Append |
            Out-Host
    }
    if ($LASTEXITCODE -ne 0) {
        throw "SAME-412 control stage $Stage exited with code $LASTEXITCODE."
    }
}

foreach ($Required in @(
    $Python,
    $Protocol,
    $Auditor,
    $Runner,
    $Evaluator,
    $Promotion,
    $ParentEventChain
)) {
    Assert-RequiredFile -LiteralPath $Required
}
foreach ($Pair in @(
    @($GarcOutputRoot, "GARC output root"),
    @($Preregistration, "control preregistration"),
    @($PreflightOutput, "control preflight"),
    @($ControlOutputRoot, "control output root"),
    @($PromotionOutputRoot, "promotion output root")
)) {
    Assert-SafePointerReadPath -LiteralPath $Pair[0] -Label $Pair[1]
}

if ($PreflightOnly) {
    if ($WaitForPid -le 0 -or [string]::IsNullOrWhiteSpace(
        $ExpectedWaitProcessStartUtc
    )) {
        throw (
            "Preflight preregistration requires the live parent GARC PID and " +
            "its exact UTC start identity."
        )
    }
    $Identity = Get-AuthenticatedGarcProcess `
        -ProcessId $WaitForPid `
        -ExpectedStartUtc $ExpectedWaitProcessStartUtc
    if (-not (Test-Path -LiteralPath $Preregistration -PathType Leaf)) {
        Invoke-Checked `
            -Stage "freeze-control-preregistration" `
            -Arguments @(
                "-m", "experiments.garc_same_cohort_control",
                "freeze-preregistration",
                "--output", $Preregistration,
                "--garc-output-root", $GarcOutputRoot,
                "--garc-process-id", "$WaitForPid",
                "--garc-process-start-utc", (
                    $Identity.StartUtc.ToString("o").Replace("+00:00", "Z")
                ),
                "--garc-process-command-sha256", $Identity.CommandLineSha256
            )
    } else {
        $Frozen = Get-Content `
            -LiteralPath $Preregistration -Raw | ConvertFrom-Json
        if (
            [int]$Frozen.garc_process.pid -ne $WaitForPid -or
            [string]$Frozen.garc_process.command_line_sha256 -ne
                $Identity.CommandLineSha256
        ) {
            throw "Existing control preregistration belongs to another GARC process."
        }
        $FrozenStart = ConvertTo-UtcIdentity `
            -Value ([string]$Frozen.garc_process.start_utc) `
            -Label "Frozen GARC process start"
        if ([Math]::Abs(($FrozenStart - $Identity.StartUtc).TotalSeconds) -gt 1.0) {
            throw "Existing control preregistration start identity drift."
        }
    }
    Invoke-Checked `
        -Stage "control-zero-data-preflight" `
        -Arguments @(
            "-m", "experiments.garc_same_cohort_control", "preflight",
            "--preregistration", $Preregistration,
            "--output", $PreflightOutput
        )
    return
}

Assert-RequiredFile -LiteralPath $Preregistration

# One authenticated kernel wait, never timer polling.  If the process already
# ended, the immutable final summary is the completion condition.
if ($WaitForPid -gt 0) {
    if ([string]::IsNullOrWhiteSpace($ExpectedWaitProcessStartUtc)) {
        throw "ExpectedWaitProcessStartUtc is required for a nonzero GARC PID."
    }
    $Identity = Get-AuthenticatedGarcProcess `
        -ProcessId $WaitForPid `
        -ExpectedStartUtc $ExpectedWaitProcessStartUtc `
        -AllowAbsent
    if ($null -ne $Identity) {
        $Process = [Diagnostics.Process]::GetProcessById($WaitForPid)
        if ([Math]::Abs((
            $Process.StartTime.ToUniversalTime() - $Identity.StartUtc
        ).TotalSeconds) -gt 1.0) {
            throw "GARC CIM/process start identity drift."
        }
        $CurrentStage = "wait-authenticated-parent-garc"
        $Process.WaitForExit()
    }
}
if (-not (Test-Path -LiteralPath $GarcSummary -PathType Leaf)) {
    throw "Authenticated parent GARC completed without summary.json."
}

try {
    Invoke-Checked `
        -Stage "prepare-control-execution" `
        -Arguments @(
            "-m", "experiments.garc_same_cohort_control", "prepare",
            "--preregistration", $Preregistration,
            "--garc-summary", $GarcSummary,
            "--output-root", $ControlOutputRoot
        )
    $ManifestPath = Join-Path $ControlOutputRoot "execution_manifest.json"
    Assert-RequiredFile -LiteralPath $ManifestPath
    $LogRoot = Join-Path $ControlOutputRoot "logs"

    Invoke-Checked `
        -Stage "materialize-secondary-control-plans" `
        -Arguments @(
            "-m", "experiments.garc_same_cohort_control",
            "materialize-plans", "--manifest", $ManifestPath
        ) `
        -LogPath (Join-Path $LogRoot "materialize-secondary-control-plans.log")

    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $Plan20 = [string]$Manifest.plans.'20260720'.path
    $Plan21 = [string]$Manifest.plans.'20260721'.path
    $Plan22 = [string]$Manifest.plans.'20260722'.path
    foreach ($Plan in @($Plan20, $Plan21, $Plan22)) {
        Assert-RequiredFile -LiteralPath $Plan
        Invoke-Checked `
            -Stage ("validate-control-plan-" + [IO.Path]::GetFileName($Plan)) `
            -Arguments @($Runner, "validate-plan", "--plan", $Plan) `
            -LogPath (Join-Path $LogRoot "validate-control-plans.log")
    }

    $PrimaryRoot = [string]$Manifest.prediction_roots.'20260720'
    $Shard21Root = [string]$Manifest.prediction_roots.'20260721'
    $Shard22Root = [string]$Manifest.prediction_roots.'20260722'

    Invoke-Checked `
        -Stage "infer-control-primary-seed-20260720" `
        -Arguments @(
            $Runner, "infer", "--plan", $Plan20,
            "--partition", "independent_validation",
            "--output-root", $PrimaryRoot,
            "--torch-cpu-threads", "$TorchCpuThreads", "--formal"
        ) `
        -LogPath (Join-Path $LogRoot "infer-control-primary-seed-20260720.log")
    Invoke-Checked `
        -Stage "infer-control-joint-shard-seed-20260721" `
        -Arguments @(
            $Runner, "infer", "--plan", $Plan21,
            "--partition", "independent_validation",
            "--output-root", $Shard21Root,
            "--torch-cpu-threads", "$TorchCpuThreads", "--formal",
            "--joint-oof-seed", "20260721"
        ) `
        -LogPath (Join-Path $LogRoot "infer-control-joint-shard-seed-20260721.log")
    Invoke-Checked `
        -Stage "infer-control-joint-shard-seed-20260722" `
        -Arguments @(
            $Runner, "infer", "--plan", $Plan22,
            "--partition", "independent_validation",
            "--output-root", $Shard22Root,
            "--torch-cpu-threads", "$TorchCpuThreads", "--formal",
            "--joint-oof-seed", "20260722"
        ) `
        -LogPath (Join-Path $LogRoot "infer-control-joint-shard-seed-20260722.log")

    $ControlReport = [string]$Manifest.control_report
    Invoke-Checked `
        -Stage "score-formal-same-412-control" `
        -Arguments @(
            $Evaluator, "validate",
            "--plan", $Plan20,
            "--prediction-root", $PrimaryRoot,
            "--calibration", ([string]$Manifest.control_calibration.path),
            "--joint-run", "$Plan21::$Shard21Root",
            "--joint-run", "$Plan22::$Shard22Root",
            "--output", $ControlReport
        ) `
        -LogPath (Join-Path $LogRoot "score-formal-same-412-control.log")

    # Seal and independently verify the control before promotion is callable.
    Invoke-Checked `
        -Stage "seal-formal-same-412-control" `
        -Arguments @(
            "-m", "experiments.garc_same_cohort_control", "seal",
            "--manifest", $ManifestPath,
            "--control-report", $ControlReport
        ) `
        -LogPath (Join-Path $LogRoot "seal-formal-same-412-control.log")
    Invoke-Checked `
        -Stage "verify-sealed-control-before-promotion" `
        -Arguments @(
            "-m", "experiments.garc_same_cohort_control", "verify",
            "--root", $ControlOutputRoot
        ) `
        -LogPath (Join-Path $LogRoot "verify-sealed-control-before-promotion.log")
    Assert-RequiredFile -LiteralPath (Join-Path $ControlOutputRoot "summary.json")
    Assert-RequiredFile -LiteralPath (Join-Path $ControlOutputRoot "seal.json")

    # This CPU-only decision follows the verified control seal in the same
    # foreground process.  It authorizes at most; it never starts training.
    Invoke-Checked `
        -Stage "common-split-promotion-after-control-seal" `
        -Arguments @(
            "-m", "experiments.garc_common_split_promotion", "promote",
            "--protocol", $Protocol,
            "--garc-summary", $GarcSummary,
            "--control-report", $ControlReport,
            "--output-root", $PromotionOutputRoot
        ) `
        -LogPath (Join-Path $LogRoot "common-split-promotion-after-control-seal.log")

    $DecisionPath = Join-Path $PromotionOutputRoot "decision.json"
    Assert-RequiredFile -LiteralPath $DecisionPath
    $Decision = Get-Content -LiteralPath $DecisionPath -Raw | ConvertFrom-Json
    [ordered]@{
        protocol = "garc_same_412_control_to_promotion_event_chain_v1"
        status = "complete"
        control_summary = Join-Path $ControlOutputRoot "summary.json"
        control_summary_sha256 = Get-FileSha256 (
            Join-Path $ControlOutputRoot "summary.json"
        )
        control_seal = Join-Path $ControlOutputRoot "seal.json"
        promotion_decision = $DecisionPath
        promotion_decision_sha256 = Get-FileSha256 $DecisionPath
        training_allowed = [bool]$Decision.training_allowed
        training_started_by_this_chain = $false
        time_polling = $false
        restricted_namespace_images_opened = 0
    } | ConvertTo-Json -Depth 6 -Compress
} catch {
    if (Test-Path -LiteralPath $ControlOutputRoot -PathType Container) {
        $FailurePath = Join-Path $ControlOutputRoot "failure.json"
        if (-not (Test-Path -LiteralPath $FailurePath)) {
            [ordered]@{
                protocol = "garc_same_412_control_event_chain_failure_v1"
                status = "failed"
                stage = $CurrentStage
                message = $_.Exception.Message
                partial_artifacts_preserved = $true
                promotion_started = $CurrentStage -eq (
                    "common-split-promotion-after-control-seal"
                )
                training_started = $false
                restricted_namespace_images_opened = 0
            } | ConvertTo-Json -Depth 5 |
                Set-Content -LiteralPath $FailurePath -Encoding utf8
        }
    }
    throw
}

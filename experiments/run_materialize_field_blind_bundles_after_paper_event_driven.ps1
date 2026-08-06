#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ProjectRoot = "D:\project\PointerMeterReaderFastAPI",
    [string]$Python = "D:\project\PointerMeterReaderFastAPI\.venv\Scripts\python.exe",
    [ValidateRange(0, 2147483647)]
    [int]$PaperWaiterProcessId = 25424,
    [string]$ExpectedPaperWaiterStartUtc = "2026-08-06T18:58:48.7736440Z",
    [string]$Spec = "C:\pointer_read\blind_bundle_materialization_v1\spec.json",
    [string]$GarcSummary = "C:\pointer_read\garc_full_auto_formal_v1\summary.json",
    [string]$PaperSummary = "C:\pointer_read\paper_final_results_v2\summary.json",
    [string]$PaperSeal = "C:\pointer_read\paper_final_results_v2\seal.json",
    [string]$OutputRoot = "C:\pointer_read\blind_bundle_materialization_v1\frozen",
    [switch]$PreflightOnly,
    [switch]$StartMaterialization
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProgressPreference = "SilentlyContinue"

if (-not $PreflightOnly -and -not $StartMaterialization) {
    throw "Choose -PreflightOnly or -StartMaterialization explicitly"
}
if ($PreflightOnly -and $StartMaterialization) {
    throw "-PreflightOnly and -StartMaterialization are mutually exclusive"
}

$Runner = Join-Path $ProjectRoot "experiments\materialize_field_blind_bundles.py"
$ExpectedFragment = "experiments\run_assemble_paper_results_event_driven.ps1"

function ConvertTo-ExactUtc([string]$Value, [string]$Label) {
    if (
        [string]::IsNullOrWhiteSpace($Value) -or
        -not $Value.EndsWith("Z", [System.StringComparison]::Ordinal)
    ) {
        throw "$Label must be an exact round-trip UTC timestamp with Z suffix"
    }
    try {
        return [DateTimeOffset]::ParseExact(
            $Value,
            "o",
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Globalization.DateTimeStyles]::RoundtripKind
        ).UtcDateTime
    } catch {
        throw "$Label must use yyyy-MM-ddTHH:mm:ss.fffffffZ"
    }
}

function Get-AuthenticatedPaperWaiter {
    $ExpectedStart = ConvertTo-ExactUtc $ExpectedPaperWaiterStartUtc "paper waiter start"
    $Cim = Get-CimInstance Win32_Process -Filter "ProcessId = $PaperWaiterProcessId"
    if ($null -eq $Cim) {
        return $null
    }
    $CommandLine = [string]$Cim.CommandLine
    if (
        [string]::IsNullOrWhiteSpace($CommandLine) -or
        $CommandLine.IndexOf(
            $ExpectedFragment,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -lt 0
    ) {
        throw "paper waiter PID belongs to an unexpected command"
    }
    $ActualStart = $Cim.CreationDate.ToUniversalTime()
    if ([math]::Abs(($ActualStart - $ExpectedStart).TotalMilliseconds) -gt 1.0) {
        throw "paper waiter creation time differs from the frozen identity"
    }
    return $Cim
}

function Invoke-ScientificPreflight {
    $Output = & $Python -m experiments.materialize_field_blind_bundles `
        preflight `
        --spec $Spec `
        --garc-summary $GarcSummary `
        --paper-summary $PaperSummary `
        --paper-seal $PaperSeal 2>&1
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -notin @(0, 2)) {
        throw "bundle materialization preflight crashed with exit code $ExitCode"
    }
    $Text = ($Output -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($Text)) {
        throw "bundle materialization preflight produced no JSON"
    }
    try {
        return $Text | ConvertFrom-Json -Depth 50
    } catch {
        throw "bundle materialization preflight returned invalid JSON"
    }
}

if ($PreflightOnly) {
    $Waiter = Get-AuthenticatedPaperWaiter
    $Preflight = Invoke-ScientificPreflight
    [ordered]@{
        schema_version = 1
        protocol = "field_blind_bundle_materialization_event_wrapper_preflight_v1"
        status = "preflight_only_complete"
        paper_waiter = [ordered]@{
            pid = $PaperWaiterProcessId
            exact_start_utc = $ExpectedPaperWaiterStartUtc
            authenticated_and_running = ($null -ne $Waiter)
            completed_artifact_present = (Test-Path -LiteralPath $PaperSummary -PathType Leaf)
        }
        scientific_preflight = $Preflight
        formal_materialization_started = $false
        field_manifest_opened = $false
        field_images_opened = $false
        field_labels_opened = $false
        feishu_messages_sent = 0
    } | ConvertTo-Json -Depth 50 -Compress
    exit 0
}

# This wrapper only authenticates and materializes an already-frozen spec.  It
# deliberately has no authority to invent the five source bundles, factory
# configs, runtime-artifact catalog, or shared frontend after the parent paper
# run.  Refuse before waiting when that prerequisite is absent; otherwise a
# caller could believe the chain was attached successfully even though it is
# guaranteed to fail only after the (potentially long) paper process exits.
if (-not (Test-Path -LiteralPath $Spec -PathType Leaf)) {
    throw (
        "materialization spec is absent; freeze-spec and all public-only input " +
        "artifacts must be prepared before -StartMaterialization can attach: $Spec"
    )
}

$Waiter = Get-AuthenticatedPaperWaiter
if ($null -eq $Waiter -and -not (Test-Path -LiteralPath $PaperSummary -PathType Leaf)) {
    throw "paper waiter is absent and its completed summary is unavailable"
}
while ($null -ne $Waiter) {
    Start-Sleep -Seconds 3
    $Waiter = Get-AuthenticatedPaperWaiter
}

$Preflight = Invoke-ScientificPreflight
if ($Preflight.ready -ne $true -or $Preflight.status -ne "ready") {
    throw "bundle materialization dependencies are not ready; run -PreflightOnly for the exact list"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "immutable materialization output root already exists: $OutputRoot"
}

$Result = & $Python -m experiments.materialize_field_blind_bundles `
    materialize --spec $Spec --output-root $OutputRoot 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "bundle materialization failed: $($Result -join ' ')"
}
$Result -join "`n"

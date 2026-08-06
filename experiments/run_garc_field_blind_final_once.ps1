#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = (Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"),
    [Parameter(Mandatory = $true)]
    [string]$Guard,
    [string]$DatasetIdentity,
    [string]$Finalization,
    [string]$RunRoot,
    [switch]$FreezeAuthorization,
    [switch]$PreflightOnly,
    [switch]$StartInference,
    [switch]$StartScoring
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProgressPreference = "SilentlyContinue"

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "The final field-blind wrapper requires PowerShell 7."
}

$Actions = @(@(
        [bool]$FreezeAuthorization,
        [bool]$PreflightOnly,
        [bool]$StartInference,
        [bool]$StartScoring
    ) | Where-Object { $_ })
if ($Actions.Count -ne 1) {
    throw "Select exactly one action: -FreezeAuthorization, -PreflightOnly, -StartInference, or -StartScoring."
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$Python = [System.IO.Path]::GetFullPath($Python)
$Guard = [System.IO.Path]::GetFullPath($Guard)
$GuardModule = Join-Path $ProjectRoot "experiments\garc_field_blind_guard.py"

foreach ($Required in @($Python, $GuardModule)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required executable/source is absent: $Required"
    }
}

function Invoke-GuardCommand {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $Output = & $Python -m experiments.garc_field_blind_guard @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "GARC field-blind guard failed with exit code $LASTEXITCODE."
    }
    $Text = @($Output) -join [Environment]::NewLine
    try {
        return $Text | ConvertFrom-Json -Depth 100
    } catch {
        throw "GARC field-blind guard returned malformed JSON."
    }
}

Push-Location $ProjectRoot
try {
    if ($FreezeAuthorization) {
        foreach ($RequiredValue in @(
            @{ Name = "DatasetIdentity"; Value = $DatasetIdentity },
            @{ Name = "Finalization"; Value = $Finalization },
            @{ Name = "RunRoot"; Value = $RunRoot }
        )) {
            if ([string]::IsNullOrWhiteSpace([string]$RequiredValue.Value)) {
                throw "$($RequiredValue.Name) is required for -FreezeAuthorization."
            }
        }
        if (Test-Path -LiteralPath $Guard) {
            throw "The immutable field authorization already exists: $Guard"
        }
        $Frozen = Invoke-GuardCommand @(
            "freeze",
            "--dataset-identity", [System.IO.Path]::GetFullPath($DatasetIdentity),
            "--finalization", [System.IO.Path]::GetFullPath($Finalization),
            "--output", $Guard,
            "--run-root", [System.IO.Path]::GetFullPath($RunRoot)
        )
        if ($Frozen.status -ne "frozen_authorized_not_started") {
            throw "The field authorization did not reach its frozen state."
        }
        [ordered]@{
            schema_version = 1
            protocol = "garc_field_blind_final_wrapper_v1"
            status = "authorization_frozen_no_field_manifest_or_image_access"
            guard = $Guard
            field_manifest_opened = $false
            field_labels_opened = $false
            field_images_opened = $false
            inference_started = $false
            scoring_started = $false
            feishu_notification_sent = $false
        } | ConvertTo-Json -Depth 8 -Compress
        return
    }

    if (-not (Test-Path -LiteralPath $Guard -PathType Leaf)) {
        throw "Final field authorization is absent; GARC selection and public paper evidence must be completed and frozen first: $Guard"
    }
    $Preflight = Invoke-GuardCommand @("preflight", "--guard", $Guard)
    if (
        $Preflight.status -ne "validated_without_field_data_access" -or
        $Preflight.field_manifest_opened -ne $false -or
        $Preflight.field_labels_opened -ne $false -or
        $Preflight.field_images_opened -ne $false
    ) {
        throw "The final field authorization preflight is not clean."
    }

    if ($PreflightOnly) {
        [ordered]@{
            schema_version = 1
            protocol = "garc_field_blind_final_wrapper_preflight_v1"
            status = "validated_no_field_manifest_label_or_image_access"
            method_name = $Preflight.method_name
            declared_images = $Preflight.declared_images
            guard_sha256 = $Preflight.guard_sha256
            field_manifest_opened = $false
            field_labels_opened = $false
            field_images_opened = $false
            inference_started = $false
            scoring_started = $false
            feishu_notification_sent = $false
        } | ConvertTo-Json -Depth 8 -Compress
        return
    }

    if ($StartInference) {
        $Result = Invoke-GuardCommand @("run-once", "--guard", $Guard)
        if ($Result.status -ne "complete_predictions_sealed_before_labels") {
            throw "One-shot image-only inference did not seal complete predictions."
        }
        $Result | ConvertTo-Json -Depth 100
        return
    }

    $Score = Invoke-GuardCommand @("score-once", "--guard", $Guard)
    if ($Score.status -ne "complete_one_shot_blind_score") {
        throw "One-shot blind scoring did not complete."
    }
    $Score | ConvertTo-Json -Depth 100
}
finally {
    Pop-Location
}

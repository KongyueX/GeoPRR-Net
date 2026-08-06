#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ProjectRoot = "D:\project\PointerMeterReaderFastAPI",
    [string]$Python = "D:\project\PointerMeterReaderFastAPI\.venv\Scripts\python.exe",
    [Parameter(Mandatory = $true)]
    [string]$Protocol,
    [string]$DatasetIdentity,
    [string]$PaperResults = "C:\pointer_read\paper_final_results_v2\summary.json",
    [string]$PaperTablesRoot = "D:\project\PointerMeterReaderFastAPI\paper\submission_mdpi\official\generated_tables",
    [string]$FrontendPlan,
    [string]$MethodRoster,
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
    throw "The multi-method blind wrapper requires PowerShell 7."
}
$Actions = @(@(
        [bool]$FreezeAuthorization,
        [bool]$PreflightOnly,
        [bool]$StartInference,
        [bool]$StartScoring
    ) | Where-Object { $_ })
if ($Actions.Count -ne 1) {
    throw "Select exactly one explicit one-shot action."
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$Python = [System.IO.Path]::GetFullPath($Python)
$Protocol = [System.IO.Path]::GetFullPath($Protocol)
$Runner = Join-Path $ProjectRoot "experiments\field_blind_multimethod.py"
foreach ($Required in @($Python, $Runner)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required executable/source is absent: $Required"
    }
}

function Invoke-Runner {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $Output = & $Python -m experiments.field_blind_multimethod @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Full-scene multi-method guard failed with exit code $LASTEXITCODE."
    }
    try {
        return (@($Output) -join [Environment]::NewLine) | ConvertFrom-Json -Depth 100
    } catch {
        throw "Full-scene multi-method guard returned malformed JSON."
    }
}

Push-Location $ProjectRoot
try {
    if ($FreezeAuthorization) {
        foreach ($RequiredValue in @(
            @{ Name = "DatasetIdentity"; Value = $DatasetIdentity },
            @{ Name = "PaperResults"; Value = $PaperResults },
            @{ Name = "PaperTablesRoot"; Value = $PaperTablesRoot },
            @{ Name = "FrontendPlan"; Value = $FrontendPlan },
            @{ Name = "MethodRoster"; Value = $MethodRoster },
            @{ Name = "RunRoot"; Value = $RunRoot }
        )) {
            if ([string]::IsNullOrWhiteSpace([string]$RequiredValue.Value)) {
                throw "$($RequiredValue.Name) is required for -FreezeAuthorization."
            }
        }
        if (Test-Path -LiteralPath $Protocol) {
            throw "The immutable full-scene authorization already exists: $Protocol"
        }
        $Result = Invoke-Runner @(
            "freeze",
            "--dataset-identity", [System.IO.Path]::GetFullPath($DatasetIdentity),
            "--paper-results", [System.IO.Path]::GetFullPath($PaperResults),
            "--paper-tables-root", [System.IO.Path]::GetFullPath($PaperTablesRoot),
            "--frontend-plan", [System.IO.Path]::GetFullPath($FrontendPlan),
            "--method-roster", [System.IO.Path]::GetFullPath($MethodRoster),
            "--output", $Protocol,
            "--run-root", [System.IO.Path]::GetFullPath($RunRoot)
        )
        if ($Result.status -ne "frozen_authorized_not_started") {
            throw "The full-scene multi-method authorization did not freeze."
        }
        [ordered]@{
            schema_version = 1
            protocol = "field_blind_multimethod_final_wrapper_v1"
            status = "authorization_frozen_without_field_manifest_label_or_image_access"
            field_manifest_opened = $false
            field_labels_opened = $false
            field_images_opened = $false
            inference_started = $false
            scoring_started = $false
            feishu_notification_sent = $false
        } | ConvertTo-Json -Compress
        return
    }

    if (-not (Test-Path -LiteralPath $Protocol -PathType Leaf)) {
        throw "Final full-scene multi-method authorization is absent; all five complete bundles, GARC selection, and public paper evidence must be frozen first."
    }
    $Preflight = Invoke-Runner @("preflight", "--protocol", $Protocol)
    if (
        $Preflight.status -ne "validated_without_field_manifest_label_or_image_access" -or
        $Preflight.field_manifest_opened -ne $false -or
        $Preflight.field_labels_opened -ne $false -or
        $Preflight.field_images_opened -ne $false -or
        $Preflight.same_roi_for_all_methods -ne $true -or
        @($Preflight.method_roles).Count -ne 5
    ) {
        throw "The full-scene multi-method preflight is not clean and complete."
    }
    if ($PreflightOnly) {
        $Preflight | Add-Member -NotePropertyName inference_started -NotePropertyValue $false
        $Preflight | Add-Member -NotePropertyName scoring_started -NotePropertyValue $false
        $Preflight | Add-Member -NotePropertyName feishu_notification_sent -NotePropertyValue $false
        $Preflight | ConvertTo-Json -Depth 20 -Compress
        return
    }
    if ($StartInference) {
        $Result = Invoke-Runner @("run-once", "--protocol", $Protocol)
        if ($Result.status -ne "complete_all_predictions_sealed_before_labels") {
            throw "Joint one-shot inference did not seal every method."
        }
        $Result | ConvertTo-Json -Depth 100
        return
    }
    $Result = Invoke-Runner @("score-once", "--protocol", $Protocol)
    if ($Result.status -ne "complete_one_shot_multimethod_blind_score") {
        throw "Joint blind scoring did not complete."
    }
    $Result | ConvertTo-Json -Depth 100
}
finally {
    Pop-Location
}

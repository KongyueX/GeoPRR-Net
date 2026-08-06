#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = (Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"),
    [ValidateRange(0, 2147483647)]
    [int]$GarcProcessId = 26268,
    [string]$ExpectedGarcStartUtc = "2026-08-06T18:35:40.5767690Z",
    [string]$ExpectedGarcCommandLineSha256 = (
        "a2ffb4ae59c9828708b949929f6b4bcb7256facd834667d73e96499ddfc451c6"
    ),
    [string]$ExpectedGarcSummarySha256 = "",
    [string]$GarcSummary = "C:\pointer_read\garc_full_auto_formal_v1\summary.json",
    [string]$ExternalProtocol = (Join-Path (Split-Path -Parent $PSScriptRoot) "experiments\garc_external_progress_412_protocol.json"),
    [string]$ExternalRoot = "C:\pointer_read\garc_external_progress_412_formal_v2",
    [string]$ExternalComparison = (
        "C:\pointer_read\garc_external_progress_412_formal_v2\comparison.json"
    ),
    [string]$UnderPressure1080Score = (
        "C:\pointer_read\under_pressure_official_public_" +
        "independent_validation_v1\score.json"
    ),
    [string]$UnderPressure412Score = (
        "C:\pointer_read\under_pressure_official_public_" +
        "independent_validation_v1\score_joint_oof_412.json"
    ),
    [string]$ExpectedUnderPressure1080ScoreSha256 = (
        "d1022f63c577856b3bfca116d8b0300f0e7cd7c4e9edcc4dbfe7dc2685a507cd"
    ),
    [string]$ExpectedUnderPressure412ScoreSha256 = (
        "7be897a7a4fa5daa973aea11d3e0bde175fb5a331f6eb471d3fd35510b0650ed"
    ),
    [string]$ExpectedExternalProtocolSha256 = (
        "f118b94c617a2ab0fc4353446d09d28ca4de086a6abbf477368ba387f99180ae"
    ),
    [string]$ExpectedAssemblerSha256 = (
        "ec74dc9f1a97613cb4c9de94318617b1bc7c2408c5745dc2aa57aeb1933813f8"
    ),
    [string]$ExpectedExternalEvaluatorSha256 = (
        "2aa50dc95df7bfa07e9b73deedf7b42a7e738171ebdee9421c4b1b724578a3a1"
    ),
    [string]$ExpectedRendererSha256 = (
        "b6f18c8459ee05e505ff497654fb1a3a371ce5cc9bbdd919f80ff36051152eb0"
    ),
    [string]$V5OofSummary = "C:\pointer_read\cagh_v5_enhanced_oof\summary.json",
    [string]$OutputRoot = "C:\pointer_read\paper_final_results_v2",
    [string]$RenderedTablesRoot = "",
    [ValidateRange(100, 1000000)]
    [int]$BootstrapIterations = 10000,
    [int]$BootstrapSeed = 20260806,
    [string]$ExternalDevice = "cuda:0",
    [switch]$PreflightOnly,
    [switch]$StartFormal
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProgressPreference = "SilentlyContinue"

$Assembler = Join-Path $ProjectRoot "experiments\assemble_paper_results.py"
$ExternalEvaluator = Join-Path $ProjectRoot "experiments\garc_external_progress_412.py"
$Renderer = Join-Path $ProjectRoot "experiments\render_paper_result_tables.py"
$Reporter = Join-Path $ProjectRoot "experiments\send_feishu_progress.ps1"
$Log = "$OutputRoot.wrapper.log"
$GarcCommandFragment = "experiments\run_garc_full_auto_public_event_driven.ps1"
$VdnOutput = Join-Path $ExternalRoot "vdn_predictions"
$TransformerOutput = Join-Path $ExternalRoot "transformer_sensitivity_predictions"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256([string]$Value) {
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
    return [Convert]::ToHexString(
        [System.Security.Cryptography.SHA256]::HashData($Bytes)
    ).ToLowerInvariant()
}

function Assert-Sha256([string]$Value, [string]$Label, [bool]$AllowEmpty = $false) {
    if ($AllowEmpty -and [string]::IsNullOrWhiteSpace($Value)) {
        return
    }
    if ($Value -notmatch "^[0-9a-fA-F]{64}$") {
        throw "$Label must be a SHA-256 hex digest"
    }
}

function ConvertTo-ExactUtc([string]$Value, [string]$Label) {
    if (
        [string]::IsNullOrWhiteSpace($Value) -or
        -not $Value.EndsWith("Z", [System.StringComparison]::Ordinal)
    ) {
        throw "$Label must be an exact round-trip UTC timestamp with a Z suffix"
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

function Assert-PublicPath([string]$Path, [string]$Label) {
    $Full = [System.IO.Path]::GetFullPath($Path)
    if (-not $Full.StartsWith(
        "C:\pointer_read\",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Label escaped C:\pointer_read"
    }
    $Tokens = [regex]::Split(
        $Full.ToLowerInvariant().Replace("-", "_").Replace(".", "_"),
        "[\\/_ ]+"
    )
    foreach ($Restricted in @(
        "field", "test", "sealed", "confirmatory", "confirmation", "xiangmu1", "xiangmu2"
    )) {
        if ($Tokens -contains $Restricted) {
            throw "$Label enters a restricted namespace"
        }
    }
    return $Full
}

function Assert-RenderedTablesPath([string]$Path) {
    $Full = [System.IO.Path]::GetFullPath($Path)
    $AllowedPackageRoot = [System.IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot "paper\submission_mdpi\official\")
    )
    $InPackage = $Full.StartsWith(
        $AllowedPackageRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )
    $InArtifactRoot = $Full.StartsWith(
        "C:\pointer_read\",
        [System.StringComparison]::OrdinalIgnoreCase
    )
    if (-not $InPackage -and -not $InArtifactRoot) {
        throw "RenderedTablesRoot escaped the paper package and C:\pointer_read"
    }
    return $Full
}

function Get-AuthenticatedGarcProcess(
    [int]$ProcessId,
    [datetime]$ExpectedStartUtc,
    [string]$ExpectedCommandLineSha256
) {
    $Cim = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $Cim) {
        throw "GARC PID is absent before authentication"
    }
    $CommandLine = [string]$Cim.CommandLine
    if (
        [string]::IsNullOrWhiteSpace($CommandLine) -or
        $CommandLine.IndexOf(
            $GarcCommandFragment,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -lt 0
    ) {
        throw "GARC PID belongs to an unexpected command"
    }
    if ((Get-TextSha256 $CommandLine) -ne $ExpectedCommandLineSha256.ToLowerInvariant()) {
        throw "GARC exact command-line SHA-256 mismatch"
    }
    $CimStartUtc = $Cim.CreationDate.ToUniversalTime()
    if ([math]::Abs(($CimStartUtc - $ExpectedStartUtc).TotalMilliseconds) -gt 1.0) {
        throw "GARC CIM creation time differs from the frozen identity"
    }

    $Bound = $null
    try {
        $Bound = [System.Diagnostics.Process]::GetProcessById($ProcessId)
        $null = $Bound.Handle
        $ProcessStartUtc = $Bound.StartTime.ToUniversalTime()
        if ([math]::Abs(($ProcessStartUtc - $ExpectedStartUtc).TotalMilliseconds) -gt 1.0) {
            throw "GARC Process StartTime differs from the frozen identity"
        }
        if ([math]::Abs(($ProcessStartUtc - $CimStartUtc).TotalMilliseconds) -gt 1.0) {
            throw "GARC CIM and Process start times identify different processes"
        }
        return $Bound
    } catch {
        if ($null -ne $Bound) {
            $Bound.Dispose()
        }
        throw
    }
}

function Invoke-CheckedPython([string[]]$Arguments) {
    $Output = & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
    return @($Output) -join [Environment]::NewLine
}

function Send-Event([string]$EventKey, [string]$Message, [string]$Eta) {
    & "C:\Program Files\PowerShell\7\pwsh.exe" -NoLogo -NoProfile `
        -File $Reporter -EventKey $EventKey -Message $Message -Eta $Eta | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Feishu progress notification failed"
    }
}

function Format-Metric([object]$Value, [int]$Digits = 5) {
    return [math]::Round([double]$Value, $Digits).ToString()
}

foreach ($Digest in @(
    [ordered]@{ value = $ExpectedGarcCommandLineSha256; label = "ExpectedGarcCommandLineSha256"; allow_empty = $false },
    [ordered]@{ value = $ExpectedGarcSummarySha256; label = "ExpectedGarcSummarySha256"; allow_empty = $true },
    [ordered]@{ value = $ExpectedUnderPressure1080ScoreSha256; label = "ExpectedUnderPressure1080ScoreSha256"; allow_empty = $false },
    [ordered]@{ value = $ExpectedUnderPressure412ScoreSha256; label = "ExpectedUnderPressure412ScoreSha256"; allow_empty = $false },
    [ordered]@{ value = $ExpectedExternalProtocolSha256; label = "ExpectedExternalProtocolSha256"; allow_empty = $false },
    [ordered]@{ value = $ExpectedAssemblerSha256; label = "ExpectedAssemblerSha256"; allow_empty = $false },
    [ordered]@{ value = $ExpectedExternalEvaluatorSha256; label = "ExpectedExternalEvaluatorSha256"; allow_empty = $false },
    [ordered]@{ value = $ExpectedRendererSha256; label = "ExpectedRendererSha256"; allow_empty = $false }
)) {
    Assert-Sha256 $Digest.value $Digest.label $Digest.allow_empty
}
if ($GarcProcessId -eq 0 -and [string]::IsNullOrWhiteSpace($ExpectedGarcSummarySha256)) {
    throw "Bind either a live exact GARC process or a frozen GARC summary SHA-256"
}
$GarcStart = if ($GarcProcessId -gt 0) {
    ConvertTo-ExactUtc $ExpectedGarcStartUtc "ExpectedGarcStartUtc"
} else {
    $null
}
foreach ($Required in @($Python, $Assembler, $ExternalEvaluator, $ExternalProtocol, $Renderer, $Reporter)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required file is absent: $Required"
    }
}
if ((Get-Sha256 $ExternalProtocol) -ne $ExpectedExternalProtocolSha256.ToLowerInvariant()) {
    throw "External comparison protocol SHA-256 drift"
}
if ((Get-Sha256 $Assembler) -ne $ExpectedAssemblerSha256.ToLowerInvariant()) {
    throw "Paper result assembler SHA-256 drift"
}
if ((Get-Sha256 $ExternalEvaluator) -ne $ExpectedExternalEvaluatorSha256.ToLowerInvariant()) {
    throw "External comparison evaluator SHA-256 drift"
}
if ((Get-Sha256 $Renderer) -ne $ExpectedRendererSha256.ToLowerInvariant()) {
    throw "Paper table renderer SHA-256 drift"
}

$Evidence = [ordered]@{
    garc_summary = Assert-PublicPath $GarcSummary "garc_summary"
    external_comparison = Assert-PublicPath $ExternalComparison "external_comparison"
    under_pressure_1080_score = Assert-PublicPath $UnderPressure1080Score "under_pressure_1080_score"
    under_pressure_412_score = Assert-PublicPath $UnderPressure412Score "under_pressure_412_score"
    v5_oof_summary = Assert-PublicPath $V5OofSummary "v5_oof_summary"
}
$ExternalRoot = Assert-PublicPath $ExternalRoot "external output root"
$OutputRoot = Assert-PublicPath $OutputRoot "paper output root"
if ([string]::IsNullOrWhiteSpace($RenderedTablesRoot)) {
    $RenderedTablesRoot = Join-Path `
        $ProjectRoot "paper\submission_mdpi\official\generated_tables"
}
$RenderedTablesRoot = Assert-RenderedTablesPath $RenderedTablesRoot
if (
    [System.IO.Path]::GetFullPath($ExternalComparison) -ne
    [System.IO.Path]::GetFullPath((Join-Path $ExternalRoot "comparison.json"))
) {
    throw "ExternalComparison must be ExternalRoot\comparison.json"
}

if ($PreflightOnly) {
    [ordered]@{
        schema_version = 1
        protocol = "paper_results_after_garc_event_wrapper_preflight_v2"
        status = "validated_no_wait_no_evidence_open_no_external_inference_no_assembly_no_notification"
        code = [ordered]@{
            assembler = [ordered]@{ path = $Assembler; sha256 = Get-Sha256 $Assembler }
            external_evaluator = [ordered]@{ path = $ExternalEvaluator; sha256 = Get-Sha256 $ExternalEvaluator }
            external_protocol = [ordered]@{ path = $ExternalProtocol; sha256 = Get-Sha256 $ExternalProtocol }
            renderer = [ordered]@{ path = $Renderer; sha256 = Get-Sha256 $Renderer }
        }
        garc_identity = [ordered]@{
            pid = $GarcProcessId
            exact_start_utc = if ($null -eq $GarcStart) { $null } else { $GarcStart.ToString("o") }
            command_line_sha256 = $ExpectedGarcCommandLineSha256.ToLowerInvariant()
            completed_summary_sha256 = if ([string]::IsNullOrWhiteSpace($ExpectedGarcSummarySha256)) { $null } else { $ExpectedGarcSummarySha256.ToLowerInvariant() }
        }
        evidence_paths = $Evidence
        external_root = $ExternalRoot
        output_root = $OutputRoot
        rendered_tables_root = $RenderedTablesRoot
        wait_primitive = "one retained System.Diagnostics.Process object WaitForExit"
        process_lookup_performed = $false
        process_wait_started = $false
        evidence_summary_files_opened = 0
        linked_artifacts_opened = 0
        public_truth_opened = 0
        restricted_namespace_artifacts_opened = 0
        external_inference_started = $false
        assembly_started = $false
        output_directories_created = 0
        feishu_notification_sent = $false
    } | ConvertTo-Json -Depth 8 -Compress
    return
}

if (-not $StartFormal) {
    throw "Formal external comparison and paper assembly require -StartFormal"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to overwrite existing paper output root: $OutputRoot"
}
if (Test-Path -LiteralPath $RenderedTablesRoot) {
    throw "Refusing to overwrite existing rendered-table output: $RenderedTablesRoot"
}

$CurrentStage = "authenticate-completed-under-pressure"
$GarcProcess = $null
try {
    foreach ($Completed in @(
        [ordered]@{
            label = "Under Pressure 1080 score"
            path = $Evidence.under_pressure_1080_score
            sha256 = $ExpectedUnderPressure1080ScoreSha256.ToLowerInvariant()
            protocol = "under_pressure_official_public_score_v1"
            status = "formal_public_independent_validation_complete"
        },
        [ordered]@{
            label = "Under Pressure joint 412 score"
            path = $Evidence.under_pressure_412_score
            sha256 = $ExpectedUnderPressure412ScoreSha256.ToLowerInvariant()
            protocol = "under_pressure_official_joint_oof_score_v1"
            status = "formal_joint_oof_score_complete"
        }
    )) {
        if (-not (Test-Path -LiteralPath $Completed.path -PathType Leaf)) {
            throw "$($Completed.label) is absent"
        }
        if ((Get-Sha256 $Completed.path) -ne $Completed.sha256) {
            throw "$($Completed.label) SHA-256 differs from the completed evidence"
        }
        $CompletedSummary = Get-Content -LiteralPath $Completed.path -Raw |
            ConvertFrom-Json -Depth 100
        if (
            $CompletedSummary.protocol -ne $Completed.protocol -or
            $CompletedSummary.status -ne $Completed.status
        ) {
            throw "$($Completed.label) protocol/status authentication failed"
        }
    }

    if ($GarcProcessId -gt 0) {
        $CurrentStage = "authenticate-and-wait-garc"
        try {
            $GarcProcess = Get-AuthenticatedGarcProcess `
                $GarcProcessId $GarcStart $ExpectedGarcCommandLineSha256
            $GarcProcess.WaitForExit()
            if ($GarcProcess.ExitCode -ne 0) {
                throw "Authenticated GARC process exited with code $($GarcProcess.ExitCode)"
            }
        } finally {
            if ($null -ne $GarcProcess) {
                $GarcProcess.Dispose()
                $GarcProcess = $null
            }
        }
    }

    $CurrentStage = "authenticate-garc-summary-and-handoff"
    if (-not (Test-Path -LiteralPath $Evidence.garc_summary -PathType Leaf)) {
        throw "GARC completion summary is absent"
    }
    $GarcSummarySha = Get-Sha256 $Evidence.garc_summary
    if (
        -not [string]::IsNullOrWhiteSpace($ExpectedGarcSummarySha256) -and
        $GarcSummarySha -ne $ExpectedGarcSummarySha256.ToLowerInvariant()
    ) {
        throw "GARC completion summary SHA-256 mismatch"
    }
    $Garc = Get-Content -LiteralPath $Evidence.garc_summary -Raw |
        ConvertFrom-Json -Depth 100
    if (
        $Garc.protocol -ne "garc_full_auto_public_event_chain_summary_v1" -or
        $Garc.status -ne "complete" -or
        $Garc.validation.paper_claim_allowed.joint_412_all_component_oof_end_to_end -ne $true -or
        $Garc.validation.paper_claim_allowed.full_1080_all_component_unseen -ne $false -or
        [int]$Garc.audit.restricted_namespace_images_opened -ne 0 -or
        $Garc.external_comparison_handoff.protocol -ne "garc_external_progress_412_comparison_v1" -or
        [int]$Garc.external_comparison_handoff.handoff.samples -ne 412 -or
        [int]$Garc.external_comparison_handoff.handoff.groups -ne 19
    ) {
        throw "GARC completion summary claim boundary failed"
    }
    $HandoffRoot = Assert-PublicPath `
        ([string]$Garc.external_comparison_handoff.handoff.root) "GARC handoff root"
    $HandoffSummary = Join-Path $HandoffRoot "summary.json"
    $HandoffSeal = Join-Path $HandoffRoot "seal.json"
    $ExternalPreflight = [System.IO.Path]::GetFullPath(
        [string]$Garc.external_comparison_handoff.preflight.path
    )
    foreach ($Binding in @(
        [ordered]@{ path = $HandoffSummary; sha256 = [string]$Garc.external_comparison_handoff.handoff.summary_sha256; label = "GARC handoff summary" },
        [ordered]@{ path = $HandoffSeal; sha256 = [string]$Garc.external_comparison_handoff.handoff.seal_sha256; label = "GARC handoff seal" },
        [ordered]@{ path = $ExternalPreflight; sha256 = [string]$Garc.external_comparison_handoff.preflight.sha256; label = "GARC external preflight" }
    )) {
        $Binding.path = Assert-PublicPath $Binding.path $Binding.label
        if (-not (Test-Path -LiteralPath $Binding.path -PathType Leaf)) {
            throw "$($Binding.label) is absent"
        }
        Assert-Sha256 $Binding.sha256 "$($Binding.label) SHA-256"
        if ((Get-Sha256 $Binding.path) -ne $Binding.sha256.ToLowerInvariant()) {
            throw "$($Binding.label) SHA-256 drift"
        }
    }

    if (-not (Test-Path -LiteralPath $Evidence.external_comparison -PathType Leaf)) {
        $CurrentStage = "run-vdn-transformer-external-comparison"
        if (Test-Path -LiteralPath $ExternalRoot) {
            throw "External output root already exists without a complete comparison"
        }
        [void](New-Item -ItemType Directory -Path $ExternalRoot)
        $null = Invoke-CheckedPython @(
            $ExternalEvaluator, "--protocol", $ExternalProtocol, "infer",
            "--preflight", $ExternalPreflight, "--handoff-root", $HandoffRoot,
            "--method", "vdn_official200", "--output-root", $VdnOutput,
            "--device", $ExternalDevice
        )
        $null = Invoke-CheckedPython @(
            $ExternalEvaluator, "--protocol", $ExternalProtocol, "infer",
            "--preflight", $ExternalPreflight, "--handoff-root", $HandoffRoot,
            "--method", "original_transformer", "--output-root", $TransformerOutput,
            "--device", $ExternalDevice
        )
        $null = Invoke-CheckedPython @(
            $ExternalEvaluator, "--protocol", $ExternalProtocol, "score",
            "--preflight", $ExternalPreflight, "--handoff-root", $HandoffRoot,
            "--external", "vdn_official200=$VdnOutput",
            "--external", "original_transformer=$TransformerOutput",
            "--output", $Evidence.external_comparison
        )
    }

    $CurrentStage = "authenticate-five-summaries"
    $EvidenceArguments = @(
        "--garc-summary", $Evidence.garc_summary,
        "--external-comparison", $Evidence.external_comparison,
        "--under-pressure-1080-score", $Evidence.under_pressure_1080_score,
        "--under-pressure-412-score", $Evidence.under_pressure_412_score,
        "--v5-oof-summary", $Evidence.v5_oof_summary
    )
    $ReadinessText = Invoke-CheckedPython (@($Assembler, "preflight") + $EvidenceArguments)
    $Readiness = $ReadinessText | ConvertFrom-Json -Depth 20
    if (
        $Readiness.status -ne "ready" -or
        $Readiness.ready -ne $true -or
        @($Readiness.missing).Count -ne 0 -or
        @($Readiness.incomplete).Count -ne 0 -or
        [int]$Readiness.linked_artifacts_opened -ne 0 -or
        [int]$Readiness.public_truth_opened -ne 0 -or
        [int]$Readiness.restricted_namespace_artifacts_opened -ne 0
    ) {
        throw "Five-summary readiness authentication failed"
    }

    $CurrentStage = "assemble-tables-statistics-and-manuscript-data"
    $AssemblyText = Invoke-CheckedPython (
        @($Assembler, "assemble") + $EvidenceArguments + @(
            "--output-root", $OutputRoot,
            "--bootstrap-iterations", "$BootstrapIterations",
            "--bootstrap-seed", "$BootstrapSeed"
        )
    )
    $AssemblyResult = $AssemblyText | ConvertFrom-Json -Depth 20
    if ($AssemblyResult.status -ne "complete") {
        throw "Paper result assembler did not report completion"
    }
    $SummaryPath = Join-Path $OutputRoot "summary.json"
    $SealPath = Join-Path $OutputRoot "seal.json"
    foreach ($Required in @($SummaryPath, $SealPath)) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            throw "Paper result artifact is absent: $Required"
        }
    }
    if ((Get-Sha256 $SummaryPath) -ne [string]$AssemblyResult.sha256) {
        throw "Paper result summary differs from assembler completion output"
    }
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json -Depth 100
    $Seal = Get-Content -LiteralPath $SealPath -Raw | ConvertFrom-Json -Depth 100
    if (
        $Summary.protocol -ne "pointer_meter_paper_result_assembly_v1" -or
        $Summary.status -ne "complete" -or
        [int]$Summary.cohort.samples -ne 412 -or
        [int]$Summary.cohort.groups -ne 19 -or
        $Seal.protocol -ne "pointer_meter_paper_result_assembly_v1" -or
        $Seal.status -ne "sealed" -or
        [string]$Seal.artifacts.summary.sha256 -ne (Get-Sha256 $SummaryPath) -or
        [string]$Summary.code.assembler.sha256 -ne (Get-Sha256 $Assembler) -or
        $Summary.audit.all_prediction_and_score_seals_verified_before_public_truth_opened -ne $true -or
        $Summary.audit.public_truth_used_for_scoring_only -ne $true -or
        [int]$Summary.audit.restricted_namespace_artifacts_opened -ne 0
    ) {
        throw "Assembled paper summary/seal authentication failed"
    }
    $ExpectedSourceNames = @($Evidence.Keys | Sort-Object)
    $ObservedSourceNames = @($Summary.sources.PSObject.Properties.Name | Sort-Object)
    if (($ExpectedSourceNames -join "|") -ne ($ObservedSourceNames -join "|")) {
        throw "Assembled five-summary source roster drift"
    }
    foreach ($Name in $Evidence.Keys) {
        $Binding = $Summary.sources.$Name
        if (
            [System.IO.Path]::GetFullPath([string]$Binding.path) -ne
                [System.IO.Path]::GetFullPath([string]$Evidence.$Name) -or
            [string]$Binding.sha256 -ne (Get-Sha256 $Evidence.$Name)
        ) {
            throw "Assembled source binding drift: $Name"
        }
    }
    $Rows = @($Summary.main_table)
    if ($Rows.Count -ne 3) {
        throw "Strict main table must contain exactly GARC, VDN, and Under Pressure"
    }
    $GarcRow = @($Rows | Where-Object method -eq "garc")
    $VdnRow = @($Rows | Where-Object method -eq "vdn_official200")
    $UnderPressureRow = @($Rows | Where-Object method -eq "under_pressure_official")
    if ($GarcRow.Count -ne 1 -or $VdnRow.Count -ne 1 -or $UnderPressureRow.Count -ne 1) {
        throw "Strict main-table method roster drift"
    }
    $TransformerRows = @(
        $Summary.sensitivity_table | Where-Object method -eq "original_transformer"
    )
    if (
        $TransformerRows.Count -ne 1 -or
        $TransformerRows[0].strict_main_table_eligible -ne $false
    ) {
        throw "Original Transformer escaped its sensitivity-only role"
    }

    $CurrentStage = "render-and-authenticate-four-submission-tables"
    $RenderText = Invoke-CheckedPython @(
        $Renderer,
        "--source-root", $OutputRoot,
        "--output-root", $RenderedTablesRoot
    )
    $RenderResult = $RenderText | ConvertFrom-Json -Depth 20
    $RenderManifestPath = Join-Path $RenderedTablesRoot "manifest.json"
    $RenderSealPath = Join-Path $RenderedTablesRoot "seal.json"
    if (
        $RenderResult.status -ne "complete" -or
        [System.IO.Path]::GetFullPath([string]$RenderResult.manifest) -ne
            [System.IO.Path]::GetFullPath($RenderManifestPath)
    ) {
        throw "Paper table renderer did not report the expected completion artifact"
    }
    foreach ($Required in @($RenderManifestPath, $RenderSealPath)) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            throw "Rendered table authentication artifact is absent: $Required"
        }
    }
    $RenderManifest = Get-Content -LiteralPath $RenderManifestPath -Raw |
        ConvertFrom-Json -Depth 100
    $RenderSeal = Get-Content -LiteralPath $RenderSealPath -Raw |
        ConvertFrom-Json -Depth 100
    $ExpectedRenderedNames = @(
        "component_table", "main_table", "manifest",
        "paired_group_bootstrap", "sensitivity_table"
    ) | Sort-Object
    $ObservedRenderedNames = @(
        $RenderSeal.artifacts.PSObject.Properties.Name | Sort-Object
    )
    if (
        $RenderManifest.protocol -ne "paper_result_latex_tables_v1" -or
        $RenderManifest.status -ne "complete" -or
        $RenderSeal.protocol -ne "paper_result_latex_tables_v1" -or
        $RenderSeal.status -ne "sealed" -or
        [System.IO.Path]::GetFullPath([string]$RenderManifest.source.path) -ne
            [System.IO.Path]::GetFullPath($OutputRoot) -or
        [string]$RenderManifest.source.summary_sha256 -ne (Get-Sha256 $SummaryPath) -or
        [string]$RenderManifest.source.seal_sha256 -ne (Get-Sha256 $SealPath) -or
        ($ExpectedRenderedNames -join "|") -ne ($ObservedRenderedNames -join "|") -or
        [string]$RenderSeal.artifacts.manifest.sha256 -ne
            (Get-Sha256 $RenderManifestPath)
    ) {
        throw "Rendered table manifest/seal/source binding authentication failed"
    }
} catch {
    $AssemblyError = $_
    try {
        Send-Event `
            -EventKey "paper-public-results-after-garc-v2-anomaly" `
            -Message (
                "论文公开结果链异常：stage=$CurrentStage，" +
                "type=$($AssemblyError.Exception.GetType().Name)。" +
                "既有密封制品已保留，未访问现场盲测。"
            ) `
            -Eta "预计15–45分钟完成进程身份、制品哈希或阶段输出核验后恢复。"
    } catch {
        $LogParent = Split-Path -Parent $Log
        if (Test-Path -LiteralPath $LogParent -PathType Container) {
            Add-Content -LiteralPath $Log -Value (
                "Feishu paper-chain anomaly report failed: $($_.Exception.Message)"
            )
        }
        Write-Warning "Paper chain failed and its Feishu anomaly notification also failed."
    }
    throw $AssemblyError
}

# Notification transport is deliberately outside the scientific failure block.
try {
    Send-Event `
        -EventKey "paper-public-results-after-garc-v2-complete" `
        -Message (
            "论文公开结果主表与统计已生成并密封（412张/19组）：" +
            "GARC NMAE=$(Format-Metric $GarcRow[0].full_denominator_nmae_failure_penalty_1)，" +
            "VDN=$(Format-Metric $VdnRow[0].full_denominator_nmae_failure_penalty_1)，" +
            "Under Pressure=$(Format-Metric $UnderPressureRow[0].full_denominator_nmae_failure_penalty_1)；" +
            "Original Transformer仅作为固定权重敏感性；" +
            "四张投稿表已绑定结果seal。结果=$OutputRoot。"
        ) `
        -Eta "预计1–2小时将密封表格和统计区间写入终稿；现场盲测仍等待最终模型冻结。"
} catch {
    Add-Content -LiteralPath $Log -Value (
        "Feishu paper-chain completion report failed: $($_.Exception.Message)"
    )
    Write-Warning "Paper results are complete, but completion notification failed."
}

Write-Output $SummaryPath

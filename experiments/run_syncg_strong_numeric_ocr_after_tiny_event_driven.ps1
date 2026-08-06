#requires -Version 7.0

[CmdletBinding()]
param(
    [ValidateRange(1, 2147483647)]
    [int]$WaitForPid = 25852,

    [Parameter(Mandatory = $true)]
    [string]$WaitForStartedAtUtc,

    [ValidateRange(1, 2147483647)]
    [int]$StrongSeed = 20260818,

    [ValidateRange(1, 2147483647)]
    [int]$TinySeed = 20260817,

    [string]$TinySummaryPath =
        "C:\pointer_read\syncg_numeric_ocr_garc_aligned_runs\seed_20260817\summary.json",

    [string]$TinyRecognizerPath =
        "C:\pointer_read\syncg_numeric_ocr_garc_aligned_runs\seed_20260817\recognizer.pt",

    [string]$Corpus = "C:\pointer_read\syncg_numeric_ocr_garc_aligned_v1",

    [string]$GateDecisionPath =
        "C:\pointer_read\syncg_strong_numeric_ocr_gate_v2\decision.json",

    [string]$CalibrationReportPath =
        "C:\pointer_read\syncg_garc_calibration_tiny_ocr_seed_20260817_v1\summary.json",

    [string]$StrongCalibrationReportPath =
        "C:\pointer_read\syncg_garc_calibration_strong_ocr_seed_20260818_v1\summary.json",

    [string]$QualificationPath =
        "C:\pointer_read\syncg_strong_numeric_ocr_candidate_qualification_v1\decision.json",

    [string]$OutputRoot =
        "C:\pointer_read\syncg_strong_numeric_ocr_garc_aligned_runs",

    [string]$TinyWaitCommandPattern =
        "run_syncg_garc_aligned_numeric_ocr_after_v5_event_driven\.ps1",

    [switch]$Resume,

    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "Strong SyncG numeric OCR event chain requires PowerShell 7 or newer."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$GateScript = Join-Path $PSScriptRoot "decide_syncg_strong_numeric_ocr_upgrade.py"
$CalibrationEvaluator = Join-Path $PSScriptRoot `
    "evaluate_syncg_ocr_garc_calibration.py"
$UpgradeProtocol = Join-Path $PSScriptRoot `
    "syncg_strong_numeric_ocr_upgrade_protocol.json"
$Qualifier = Join-Path $PSScriptRoot `
    "qualify_syncg_strong_numeric_ocr_candidate.py"
$GarcSelector = Join-Path $PSScriptRoot "evaluate_garc_full_auto_public.py"
$Trainer = Join-Path $PSScriptRoot "train_syncg_strong_numeric_ocr.py"
$StrongImplementation = Join-Path $PSScriptRoot "syncg_strong_numeric_ocr.py"
$TinyTrainerEvaluator = Join-Path $PSScriptRoot "train_syncg_numeric_ocr.py"
$ResumeUtility = Join-Path $PSScriptRoot "ocr_training_resume.py"
$CorpusBuilder = Join-Path $PSScriptRoot "build_garc_aligned_syncg_numeric_ocr_public.py"
$EvidenceVerifier = Join-Path $PSScriptRoot "verify_garc_ocr_training_evidence.py"
$Reporter = Join-Path $PSScriptRoot "send_feishu_progress.ps1"
$PretrainedBackbone =
    "C:\pointer_read\strong_numeric_ocr\weights\mobilenet_v3_small-047dcff4.pth"
$RunRoot = Join-Path $OutputRoot "seed_$StrongSeed"
$SummaryPath = Join-Path $RunRoot "summary.json"
$LogPath = Join-Path $OutputRoot "seed_$StrongSeed.log"
$LockPath = Join-Path $OutputRoot "seed_$StrongSeed.event-driven.lock.json"
$EvidencePath = Join-Path $RunRoot "garc_aligned_training_evidence.json"
$TerminalPath = Join-Path (Split-Path -Parent $GateDecisionPath) `
    "strong_followon_terminal.json"
$ConservativeImageBatchSize = 8
$Workers = 0
$Epochs = 12
$FrozenEpochs = 2
$LearningRate = 2.5e-4
$LockOwned = $false
$CorpusSummaryPath = Join-Path $Corpus "summary.json"
$CorpusSealPath = Join-Path $Corpus "seal.json"
$ExpectedSchedulerPattern =
    "run_syncg_garc_aligned_numeric_ocr_after_v5_event_driven\.ps1"
$SchedulerSource = Join-Path $PSScriptRoot `
    "run_syncg_garc_aligned_numeric_ocr_after_v5_event_driven.ps1"

function Assert-RequiredFile {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)
    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        throw "Required file is absent: $LiteralPath"
    }
}

function Get-VerifiedAlignedCorpus {
    $VerificationJson = (& $Python -m `
        experiments.build_garc_aligned_syncg_numeric_ocr_public `
        verify --corpus $Corpus 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Aligned OCR corpus verifier returned exit code $LASTEXITCODE"
    }
    try {
        $Verification = $VerificationJson | ConvertFrom-Json
    } catch {
        throw "Aligned OCR corpus verifier did not return one JSON object"
    }
    if (
        $Verification.status -ne "verified" -or
        $Verification.protocol -ne "syncg_public_numeric_ocr_garc_aligned_v1" -or
        -not [bool]$Verification.algorithm_fit_exact_coverage -or
        -not [bool]$Verification.outer_group_overlap_zero -or
        -not [bool]$Verification.outer_sample_overlap_zero -or
        [int]$Verification.samples -ne 12176 -or
        [int]$Verification.groups -ne 551
    ) {
        throw "Aligned OCR corpus lacks exact 12176/551 fit-only coverage or zero outer overlap"
    }
    return $Verification
}

function Assert-FrozenSourcesUnchanged {
    if (
        (Get-FileHash -Algorithm SHA256 -LiteralPath $WrapperSource).Hash.ToLowerInvariant() -ne
            $WrapperSha256 -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $SchedulerSource).Hash.ToLowerInvariant() -ne
            $SchedulerSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $GateScript).Hash.ToLowerInvariant() -ne
            $GateSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $CalibrationEvaluator).Hash.ToLowerInvariant() -ne
            $CalibrationEvaluatorSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $UpgradeProtocol).Hash.ToLowerInvariant() -ne
            $UpgradeProtocolSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $Qualifier).Hash.ToLowerInvariant() -ne
            $QualifierSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $GarcSelector).Hash.ToLowerInvariant() -ne
            $GarcSelectorSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $Trainer).Hash.ToLowerInvariant() -ne
            $TrainerSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $StrongImplementation).Hash.ToLowerInvariant() -ne
            $StrongImplementationSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $TinyTrainerEvaluator).Hash.ToLowerInvariant() -ne
            $TinyTrainerEvaluatorSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $ResumeUtility).Hash.ToLowerInvariant() -ne
            $ResumeUtilitySha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $CorpusBuilder).Hash.ToLowerInvariant() -ne
            $CorpusBuilderSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $EvidenceVerifier).Hash.ToLowerInvariant() -ne
            $EvidenceVerifierSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $CorpusSummaryPath).Hash.ToLowerInvariant() -ne
            $CorpusSummarySha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $CorpusSealPath).Hash.ToLowerInvariant() -ne
            $CorpusSealSha256AtStart -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $PretrainedBackbone).Hash.ToLowerInvariant() -ne
            $PretrainedBackboneSha256AtStart
    ) {
        throw "Strong wrapper, scheduler, gate, trainer, verifier, or frozen corpus changed after supervisor start"
    }
}

function Write-NewJsonArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$LiteralPath,
        [Parameter(Mandatory = $true)]$Value
    )
    $Resolved = [System.IO.Path]::GetFullPath($LiteralPath)
    if (-not $Resolved.StartsWith(
        "C:\pointer_read\",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "JSON artifact escaped C:\pointer_read"
    }
    [void](New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Resolved))
    $Json = $Value | ConvertTo-Json -Depth 12
    $Stream = [System.IO.File]::Open(
        $Resolved,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Json + "`n")
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    } finally {
        $Stream.Dispose()
    }
}

function Write-StrongFollowonTerminal {
    param(
        [Parameter(Mandatory = $true)][bool]$StrongTrained,
        [string]$StrongSummary = "",
        [string]$StrongEvidence = "",
        [string]$StrongCalibrationReport = "",
        [string]$StrongQualification = ""
    )
    $StrongRecord = if ($StrongTrained) {
        $QualificationRecord = Get-Content `
            -LiteralPath $StrongQualification -Raw | ConvertFrom-Json
        [ordered]@{
            training_status = "formal_training_complete"
            seed = $StrongSeed
            summary = [System.IO.Path]::GetFullPath($StrongSummary)
            summary_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $StrongSummary
            ).Hash.ToLowerInvariant()
            evidence = [System.IO.Path]::GetFullPath($StrongEvidence)
            evidence_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $StrongEvidence
            ).Hash.ToLowerInvariant()
            calibration_component_report = [ordered]@{
                path = [System.IO.Path]::GetFullPath($StrongCalibrationReport)
                sha256 = (
                    Get-FileHash -Algorithm SHA256 `
                        -LiteralPath $StrongCalibrationReport
                ).Hash.ToLowerInvariant()
                protocol = "syncg_garc_calibration_recognizer_component_v1"
            }
            candidate_qualification = [ordered]@{
                status = [string]$QualificationRecord.status
                eligible_for_garc_calibration =
                    [bool]$QualificationRecord.strong_candidate_eligible_for_garc_calibration
                artifact = [System.IO.Path]::GetFullPath($StrongQualification)
                artifact_sha256 = (
                    Get-FileHash -Algorithm SHA256 -LiteralPath $StrongQualification
                ).Hash.ToLowerInvariant()
                protocol = "syncg_strong_numeric_ocr_candidate_qualification_v1"
            }
        }
    } else {
        [ordered]@{
            training_status = "not_started_by_frozen_calibration_gate"
            seed = $StrongSeed
            summary = $null
            summary_sha256 = $null
            evidence = $null
            evidence_sha256 = $null
            calibration_component_report = $null
            candidate_qualification = [ordered]@{
                status = "not_applicable_strong_not_activated"
                eligible_for_garc_calibration = $false
                artifact = $null
                artifact_sha256 = $null
                protocol = $null
            }
        }
    }
    $Terminal = [ordered]@{
        schema_version = 1
        protocol = "syncg_strong_numeric_ocr_followon_terminal_v1"
        status = "complete"
        terminal_role = "candidate_availability_for_downstream_garc_calibration_selection"
        selection_partition = "garc_outer_calibration"
        activation_only_not_retention_selection = $true
        downstream_recognizer_selection = if (
            $StrongTrained -and
            [bool]$StrongRecord.candidate_qualification.eligible_for_garc_calibration
        ) {
            "pending_garc_same_calibration_selector_with_qualified_strong_candidate"
        } elseif ($StrongTrained) {
            "tiny_only_strong_candidate_rejected_by_component_qualification"
        } else {
            "tiny_only_strong_not_activated"
        }
        activation_decision = [ordered]@{
            path = [System.IO.Path]::GetFullPath($GateDecisionPath)
            sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $GateDecisionPath
            ).Hash.ToLowerInvariant()
            protocol = "syncg_strong_numeric_ocr_gate_decision_v2"
            train_strong_recognizer = [bool]$Decision.train_strong_recognizer
        }
        tiny = [ordered]@{
            seed = $TinySeed
            summary = [System.IO.Path]::GetFullPath($TinySummaryPath)
            summary_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $TinySummaryPath
            ).Hash.ToLowerInvariant()
            checkpoint = [System.IO.Path]::GetFullPath($TinyRecognizerPath)
            checkpoint_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $TinyRecognizerPath
            ).Hash.ToLowerInvariant()
        }
        calibration_component_report = [ordered]@{
            path = [System.IO.Path]::GetFullPath($CalibrationReportPath)
            sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $CalibrationReportPath
            ).Hash.ToLowerInvariant()
            samples = 2224
            groups = 100
        }
        strong = $StrongRecord
        access = [ordered]@{
            unique_calibration_samples = 2224
            calibration_component_evaluation_passes = if ($StrongTrained) { 2 } else { 1 }
            calibration_image_and_annotation_opens = if ($StrongTrained) { 4448 } else { 2224 }
            independent_validation_opened = 0
            development_excluded_opened = 0
            joint_oof_412_19_opened = 0
            field_test_sealed_confirmatory_opened = 0
        }
        wrapper = [ordered]@{
            path = $WrapperSource
            sha256 = $WrapperSha256
        }
        completed_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    Write-NewJsonArtifact -LiteralPath $TerminalPath -Value $Terminal
}

function Send-ProgressEvent {
    param(
        [Parameter(Mandatory = $true)][string]$EventKey,
        [Parameter(Mandatory = $true)][string]$Message,
        [Parameter(Mandatory = $true)][string]$Eta
    )
    try {
        & $Reporter -EventKey $EventKey -Message $Message -Eta $Eta
        if ($LASTEXITCODE -ne 0) {
            throw "Progress reporter returned exit code $LASTEXITCODE"
        }
    } catch {
        Write-Warning "Progress notification failed for event '$EventKey'; training state is unchanged."
    }
}

function Get-ProcessStartUtc {
    param([Parameter(Mandatory = $true)][int]$ProcessId)
    $Process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $Process) {
        return $null
    }
    return ([datetime]$Process.CreationDate).ToUniversalTime()
}

function ConvertTo-ExactUtc {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Label
    )
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

function Get-AuthenticatedProcess {
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [Parameter(Mandatory = $true)][datetime]$ExpectedStartUtc,
        [Parameter(Mandatory = $true)][string]$ExpectedCommandPattern,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $Cim = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $Cim) {
        return $null
    }
    if ([string]$Cim.CommandLine -notmatch $ExpectedCommandPattern) {
        throw "$Label PID belongs to an unexpected process"
    }
    $CimStartUtc = ([datetime]$Cim.CreationDate).ToUniversalTime()
    if (
        [math]::Abs(($CimStartUtc - $ExpectedStartUtc).TotalMilliseconds) -gt 1.0
    ) {
        throw "$Label PID start-time identity mismatch"
    }
    $Bound = $null
    try {
        $Bound = [System.Diagnostics.Process]::GetProcessById($ProcessId)
        $null = $Bound.Handle
        $ProcessStartUtc = $Bound.StartTime.ToUniversalTime()
        if (
            [math]::Abs(($ProcessStartUtc - $ExpectedStartUtc).TotalMilliseconds) -gt 1.0 -or
            [math]::Abs(($ProcessStartUtc - $CimStartUtc).TotalMilliseconds) -gt 1.0
        ) {
            throw "$Label CIM/process start identity drift"
        }
        return $Bound
    } catch [System.ArgumentException] {
        if ($null -ne $Bound) {
            $Bound.Dispose()
        }
        return $null
    } catch {
        if ($null -ne $Bound) {
            $Bound.Dispose()
        }
        throw
    }
}

function Wait-ForAuthenticatedProcess {
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [Parameter(Mandatory = $true)][datetime]$ExpectedStartUtc,
        [Parameter(Mandatory = $true)][string]$ExpectedCommandPattern,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$CompletionArtifact
    )
    $Bound = Get-AuthenticatedProcess `
        -ProcessId $ProcessId `
        -ExpectedStartUtc $ExpectedStartUtc `
        -ExpectedCommandPattern $ExpectedCommandPattern `
        -Label $Label
    if ($null -eq $Bound) {
        if (-not (Test-Path -LiteralPath $CompletionArtifact -PathType Leaf)) {
            throw "$Label PID is absent and no completed artifact is available"
        }
        return
    }
    try {
        # Retain the already-authenticated native process handle through wait;
        # never perform a second PID-only lookup that could observe PID reuse.
        $Bound.WaitForExit()
        if ($Bound.ExitCode -ne 0) {
            throw "$Label exited with code $($Bound.ExitCode)"
        }
    } finally {
        $Bound.Dispose()
    }
}

function Remove-AuthenticatedStaleLock {
    if (-not (Test-Path -LiteralPath $LockPath -PathType Leaf)) {
        return
    }
    if (-not $Resume) {
        throw "Strong OCR event-chain lock exists; explicit -Resume is required"
    }
    $Existing = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
    if (
        $Existing.protocol -ne "syncg_strong_numeric_ocr_event_chain_lock_v4" -or
        [int]$Existing.owner_pid -lt 1 -or
        [string]::IsNullOrWhiteSpace([string]$Existing.owner_started_at_utc)
    ) {
        throw "Strong OCR lock lacks an authenticated owner PID/start identity"
    }
    $ExpectedStart = [datetime]::Parse(
        [string]$Existing.owner_started_at_utc,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [System.Globalization.DateTimeStyles]::AssumeUniversal
    ).ToUniversalTime()
    $ObservedStart = Get-ProcessStartUtc -ProcessId ([int]$Existing.owner_pid)
    if (
        $null -ne $ObservedStart -and
        [math]::Abs(($ObservedStart - $ExpectedStart).TotalSeconds) -le 1.0
    ) {
        throw "Strong OCR lock is owned by live PID $($Existing.owner_pid)"
    }
    Remove-Item -LiteralPath $LockPath -Force
}

function New-OwnedLock {
    $OwnerStart = Get-ProcessStartUtc -ProcessId $PID
    if ($null -eq $OwnerStart) {
        throw "Cannot authenticate current Strong OCR wrapper process"
    }
    $LockValue = [ordered]@{
        protocol = "syncg_strong_numeric_ocr_event_chain_lock_v4"
        owner_pid = $PID
        owner_started_at_utc = $OwnerStart.ToString("o")
        wait_for_pid = $WaitForPid
        wait_for_started_at_utc = $WaitForStartedAtUtc
        tiny_seed = $TinySeed
        tiny_summary = $TinySummaryPath
        tiny_recognizer = $TinyRecognizerPath
        tiny_wait_command_pattern = $TinyWaitCommandPattern
        corpus = $Corpus
        strong_seed = $StrongSeed
        resume = [bool]$Resume
        wrapper_sha256 = $WrapperSha256
        scheduler_sha256_at_wrapper_start = $SchedulerSha256AtStart
        gate_sha256_at_wrapper_start = $GateSha256AtStart
        calibration_evaluator_sha256_at_wrapper_start =
            $CalibrationEvaluatorSha256AtStart
        upgrade_protocol_sha256_at_wrapper_start = $UpgradeProtocolSha256AtStart
        qualifier_sha256_at_wrapper_start = $QualifierSha256AtStart
        garc_selector_sha256_at_wrapper_start = $GarcSelectorSha256AtStart
        trainer_sha256_at_wrapper_start = $TrainerSha256AtStart
        strong_implementation_sha256_at_wrapper_start =
            $StrongImplementationSha256AtStart
        tiny_trainer_evaluator_sha256_at_wrapper_start =
            $TinyTrainerEvaluatorSha256AtStart
        resume_utility_sha256_at_wrapper_start = $ResumeUtilitySha256AtStart
        pretrained_backbone_sha256_at_wrapper_start =
            $PretrainedBackboneSha256AtStart
        corpus_builder_sha256_at_wrapper_start = $CorpusBuilderSha256AtStart
        evidence_verifier_sha256_at_wrapper_start = $EvidenceVerifierSha256AtStart
        corpus_summary_sha256_at_wrapper_start = $CorpusSummarySha256AtStart
        corpus_seal_sha256_at_wrapper_start = $CorpusSealSha256AtStart
        created_at_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Compress
    $LockStream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $LockBytes = [System.Text.Encoding]::UTF8.GetBytes($LockValue + "`n")
        $LockStream.Write($LockBytes, 0, $LockBytes.Length)
    } finally {
        $LockStream.Dispose()
    }
    $script:LockOwned = $true
}

function Remove-OwnedLock {
    if (-not $LockOwned -or -not (Test-Path -LiteralPath $LockPath -PathType Leaf)) {
        return
    }
    $Existing = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
    $CurrentStart = Get-ProcessStartUtc -ProcessId $PID
    if (
        [int]$Existing.owner_pid -ne $PID -or
        $null -eq $CurrentStart -or
        [math]::Abs((
            [datetime]::Parse([string]$Existing.owner_started_at_utc).ToUniversalTime() -
            $CurrentStart
        ).TotalSeconds) -gt 1.0
    ) {
        throw "Refusing to remove a Strong OCR lock not owned by this PID/start identity"
    }
    Remove-Item -LiteralPath $LockPath -Force
    $script:LockOwned = $false
}

foreach ($Required in @(
    $Python,
    $GateScript,
    $CalibrationEvaluator,
    $UpgradeProtocol,
    $Qualifier,
    $GarcSelector,
    $Trainer,
    $StrongImplementation,
    $TinyTrainerEvaluator,
    $ResumeUtility,
    $CorpusBuilder,
    $EvidenceVerifier,
    $Reporter,
    $PretrainedBackbone,
    $CorpusSummaryPath,
    $CorpusSealPath,
    $SchedulerSource
)) {
    Assert-RequiredFile -LiteralPath $Required
}
$WrapperSource = $MyInvocation.MyCommand.Path
$WrapperSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $WrapperSource
).Hash.ToLowerInvariant()
$TrainerSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $Trainer
).Hash.ToLowerInvariant()
$StrongImplementationSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $StrongImplementation
).Hash.ToLowerInvariant()
$TinyTrainerEvaluatorSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $TinyTrainerEvaluator
).Hash.ToLowerInvariant()
$ResumeUtilitySha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $ResumeUtility
).Hash.ToLowerInvariant()
$SchedulerSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $SchedulerSource
).Hash.ToLowerInvariant()
$GateSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $GateScript
).Hash.ToLowerInvariant()
$CalibrationEvaluatorSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $CalibrationEvaluator
).Hash.ToLowerInvariant()
$UpgradeProtocolSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $UpgradeProtocol
).Hash.ToLowerInvariant()
$QualifierSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $Qualifier
).Hash.ToLowerInvariant()
$GarcSelectorSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $GarcSelector
).Hash.ToLowerInvariant()
$CorpusBuilderSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $CorpusBuilder
).Hash.ToLowerInvariant()
$EvidenceVerifierSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $EvidenceVerifier
).Hash.ToLowerInvariant()
$CorpusSummarySha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $CorpusSummaryPath
).Hash.ToLowerInvariant()
$CorpusSealSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $CorpusSealPath
).Hash.ToLowerInvariant()
$PretrainedBackboneSha256AtStart = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $PretrainedBackbone
).Hash.ToLowerInvariant()

$ResolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not $ResolvedOutput.StartsWith(
    "C:\pointer_read\",
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Strong OCR output root escaped C:\pointer_read: $ResolvedOutput"
}
$ExpectedOutputRoot = [System.IO.Path]::GetFullPath(
    "C:\pointer_read\syncg_strong_numeric_ocr_garc_aligned_runs"
)
if (-not $ResolvedOutput.Equals(
    $ExpectedOutputRoot,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Formal Strong OCR output root differs from the frozen aligned root"
}
if ($TinySeed -ne 20260817 -or $StrongSeed -ne 20260818) {
    throw "Formal Tiny/Strong seeds differ from the frozen 20260817/20260818 pair"
}
$ExpectedCorpusRoot = [System.IO.Path]::GetFullPath(
    "C:\pointer_read\syncg_numeric_ocr_garc_aligned_v1"
)
$ResolvedCorpus = [System.IO.Path]::GetFullPath($Corpus)
if (-not $ResolvedCorpus.Equals(
    $ExpectedCorpusRoot,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Formal Strong OCR requires the frozen GARC-aligned corpus root"
}
$ExpectedTinyRunRoot = [System.IO.Path]::GetFullPath(
    "C:\pointer_read\syncg_numeric_ocr_garc_aligned_runs\seed_$TinySeed"
)
if (
    -not [System.IO.Path]::GetFullPath($TinySummaryPath).Equals(
        (Join-Path $ExpectedTinyRunRoot "summary.json"),
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or
    -not [System.IO.Path]::GetFullPath($TinyRecognizerPath).Equals(
        (Join-Path $ExpectedTinyRunRoot "recognizer.pt"),
        [System.StringComparison]::OrdinalIgnoreCase
    )
) {
    throw "Tiny summary/recognizer paths do not match the frozen aligned seed run"
}
$ExpectedCalibrationReport = [System.IO.Path]::GetFullPath(
    "C:\pointer_read\syncg_garc_calibration_tiny_ocr_seed_20260817_v1\summary.json"
)
$ExpectedGateDecision = [System.IO.Path]::GetFullPath(
    "C:\pointer_read\syncg_strong_numeric_ocr_gate_v2\decision.json"
)
$ExpectedStrongCalibrationReport = [System.IO.Path]::GetFullPath(
    "C:\pointer_read\syncg_garc_calibration_strong_ocr_seed_20260818_v1\summary.json"
)
$ExpectedQualification = [System.IO.Path]::GetFullPath(
    "C:\pointer_read\syncg_strong_numeric_ocr_candidate_qualification_v1\decision.json"
)
if (
    -not [System.IO.Path]::GetFullPath($CalibrationReportPath).Equals(
        $ExpectedCalibrationReport,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or
    -not [System.IO.Path]::GetFullPath($GateDecisionPath).Equals(
        $ExpectedGateDecision,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or
    -not [System.IO.Path]::GetFullPath($StrongCalibrationReportPath).Equals(
        $ExpectedStrongCalibrationReport,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or
    -not [System.IO.Path]::GetFullPath($QualificationPath).Equals(
        $ExpectedQualification,
        [System.StringComparison]::OrdinalIgnoreCase
    )
) {
    throw "Calibration report or activation decision path differs from the frozen v2 protocol"
}
$UpgradeContract = Get-Content -LiteralPath $UpgradeProtocol -Raw | ConvertFrom-Json
if (
    $UpgradeContract.protocol -ne "syncg_strong_numeric_ocr_upgrade_v2" -or
    $UpgradeContract.status -ne "frozen_before_outer_calibration_metrics" -or
    $UpgradeContract.frozen_activation_gate.selection_partition -ne
        "garc_outer_calibration" -or
    $UpgradeContract.frozen_activation_gate.independent_validation_role -ne
        "unopened_until_final_model_freeze" -or
    $UpgradeContract.implementation_bindings.activation_decider.sha256 -ne
        $GateSha256AtStart -or
    $UpgradeContract.implementation_bindings.calibration_evaluator.sha256 -ne
        $CalibrationEvaluatorSha256AtStart -or
    $UpgradeContract.implementation_bindings.strong_candidate_qualifier.sha256 -ne
        $QualifierSha256AtStart -or
    $UpgradeContract.implementation_bindings.final_garc_selector.sha256 -ne
        $GarcSelectorSha256AtStart
) {
    throw "Frozen outer-calibration upgrade protocol or implementation binding drift"
}
$CorpusVerification = Get-VerifiedAlignedCorpus

$SchedulerStartUtc = ConvertTo-ExactUtc `
    -Value $WaitForStartedAtUtc -Label "WaitForStartedAtUtc"
if ($TinyWaitCommandPattern -ne $ExpectedSchedulerPattern) {
    throw "Tiny wait command pattern may not override the frozen aligned scheduler"
}
$AuthenticatedSchedulerPattern = (
    "(?is)" + [regex]::Escape($SchedulerSource) +
    ".*\s-Seed\s+" + $TinySeed + "(?:\s|$)"
)

if ($PreflightOnly) {
    $SchedulerBound = Get-AuthenticatedProcess `
        -ProcessId $WaitForPid -ExpectedStartUtc $SchedulerStartUtc `
        -ExpectedCommandPattern $AuthenticatedSchedulerPattern `
        -Label "V5-to-aligned-Tiny scheduler"
    $Plan = [ordered]@{
        protocol = "syncg_strong_numeric_ocr_event_chain_preflight_v2"
        status = "validated_no_wait_no_training_no_notification"
        powershell = $PSVersionTable.PSVersion.ToString()
        wait_for_pid = $WaitForPid
        wait_for_started_at_utc = $SchedulerStartUtc.ToString("o")
        authenticated_processes_present = [ordered]@{
            v5_to_aligned_tiny_scheduler = $null -ne $SchedulerBound
        }
        wait_primitive = "retained System.Diagnostics.Process.WaitForExit"
        tiny_summary = $TinySummaryPath
        calibration_report = $CalibrationReportPath
        strong_calibration_report = $StrongCalibrationReportPath
        candidate_qualification = $QualificationPath
        activation_decision = $GateDecisionPath
        followon_terminal = $TerminalPath
        frozen_gate_script = $GateScript
        wrapper_sha256 = $WrapperSha256
        scheduler_sha256_at_wrapper_start = $SchedulerSha256AtStart
        gate_sha256_at_wrapper_start = $GateSha256AtStart
        calibration_evaluator_sha256_at_wrapper_start =
            $CalibrationEvaluatorSha256AtStart
        upgrade_protocol_sha256_at_wrapper_start = $UpgradeProtocolSha256AtStart
        qualifier_sha256_at_wrapper_start = $QualifierSha256AtStart
        garc_selector_sha256_at_wrapper_start = $GarcSelectorSha256AtStart
        trainer_sha256_at_wrapper_start = $TrainerSha256AtStart
        strong_implementation_sha256_at_wrapper_start =
            $StrongImplementationSha256AtStart
        tiny_trainer_evaluator_sha256_at_wrapper_start =
            $TinyTrainerEvaluatorSha256AtStart
        resume_utility_sha256_at_wrapper_start = $ResumeUtilitySha256AtStart
        pretrained_backbone_sha256_at_wrapper_start =
            $PretrainedBackboneSha256AtStart
        corpus_builder_sha256_at_wrapper_start = $CorpusBuilderSha256AtStart
        evidence_verifier_sha256_at_wrapper_start = $EvidenceVerifierSha256AtStart
        corpus_summary_sha256_at_wrapper_start = $CorpusSummarySha256AtStart
        corpus_seal_sha256_at_wrapper_start = $CorpusSealSha256AtStart
        corpus_verification = $CorpusVerification
        strong_seed = $StrongSeed
        tiny_seed = $TinySeed
        corpus = $Corpus
        device = "cuda"
        image_batch_size = $ConservativeImageBatchSize
        workers = $Workers
        epochs = $Epochs
        output_root = $OutputRoot
        output_under_c_pointer_read = $true
        formal_summary_already_exists = (
            Test-Path -LiteralPath $SummaryPath -PathType Leaf
        )
        resume_requested = [bool]$Resume
        dbnet_plus_plus_started = $false
        gpu_scheduling = "wait for authenticated V5-to-Tiny scheduler, then run Strong alone only if the frozen calibration gate selects it"
        field_test_sealed_access = $false
        outer_calibration_images_opened = 0
        independent_validation_images_opened = 0
        development_excluded_images_opened = 0
        joint_oof_412_19_samples_opened = 0
        calibration_evaluation_started = $false
        strong_training_started = $false
        strong_calibration_evaluation_started = $false
        strong_candidate_qualification_started = $false
        followon_terminal_written = $false
        gpu_work_started = $false
        feishu_message_sent = $false
    }
    if ($null -ne $SchedulerBound) {
        $SchedulerBound.Dispose()
    }
    Write-Output ($Plan | ConvertTo-Json -Compress)
    exit 0
}

try {
    Wait-ForAuthenticatedProcess `
        -ProcessId $WaitForPid -ExpectedStartUtc $SchedulerStartUtc `
        -ExpectedCommandPattern $AuthenticatedSchedulerPattern `
        -Label "V5-to-aligned-Tiny scheduler" `
        -CompletionArtifact $TinySummaryPath

    foreach ($Required in @($TinySummaryPath, $TinyRecognizerPath)) {
        Assert-RequiredFile -LiteralPath $Required
    }
    $TinySummary = Get-Content -LiteralPath $TinySummaryPath -Raw | ConvertFrom-Json
    if (
        $TinySummary.status -ne "complete" -or
        $TinySummary.mode -ne "formal" -or
        [int]$TinySummary.seed -ne $TinySeed
    ) {
        throw "Tiny OCR wrapper ended without a valid formal summary"
    }
    if (
        $TinySummary.components.recognizer.validation.tokens -lt 1 -or
        $TinySummary.components.detector.validation.samples -lt 1
    ) {
        throw "Tiny OCR summary has incomplete component validation metrics"
    }
    Assert-FrozenSourcesUnchanged

    if (-not (Test-Path -LiteralPath $CalibrationReportPath -PathType Leaf)) {
        & $Python -m experiments.evaluate_syncg_ocr_garc_calibration `
            --corpus $Corpus `
            --recognizer-kind tiny `
            --recognizer-summary $TinySummaryPath `
            --output $CalibrationReportPath `
            --device cuda `
            --batch-size 16 `
            --workers 0 `
            --run-formal | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "GARC outer-calibration Tiny component evaluation returned exit code $LASTEXITCODE"
        }
    }
    Assert-RequiredFile -LiteralPath $CalibrationReportPath
    Assert-FrozenSourcesUnchanged

    if (-not (Test-Path -LiteralPath $GateDecisionPath -PathType Leaf)) {
        & $Python $GateScript `
            --corpus $Corpus `
            --tiny-summary $TinySummaryPath `
            --calibration-report $CalibrationReportPath `
            --protocol $UpgradeProtocol `
            --output $GateDecisionPath | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Frozen outer-calibration Strong activation gate returned exit code $LASTEXITCODE"
        }
    }
    Assert-RequiredFile -LiteralPath $GateDecisionPath
    $Decision = Get-Content -LiteralPath $GateDecisionPath -Raw | ConvertFrom-Json
    $ExpectedCheckNames = @(
        "garc_calibration_exact_accuracy_below",
        "garc_calibration_character_accuracy_below",
        "garc_calibration_parseable_fraction_below"
    )
    $CheckProperties = @($Decision.component_checks.PSObject.Properties)
    $ObservedCheckNames = @($CheckProperties | ForEach-Object { $_.Name })
    $ObservedTriggered = @(
        $CheckProperties |
            Where-Object { [bool]$_.Value } |
            ForEach-Object { $_.Name } |
            Sort-Object
    )
    $DeclaredTriggered = @(
        $Decision.triggered_component_checks | Sort-Object
    )
    $ExpectedDecisionStatus = if ([bool]$Decision.train_strong_recognizer) {
        "strong_recognizer_required"
    } else {
        "tiny_component_gate_pass"
    }
    if (
        $Decision.protocol -ne "syncg_strong_numeric_ocr_gate_decision_v2" -or
        $Decision.status -ne $ExpectedDecisionStatus -or
        $Decision.selection_partition -ne "garc_outer_calibration" -or
        @(Compare-Object $ExpectedCheckNames $ObservedCheckNames).Count -ne 0 -or
        @(Compare-Object $ObservedTriggered $DeclaredTriggered).Count -ne 0 -or
        ([bool]$Decision.train_strong_recognizer) -ne
            ($ObservedTriggered.Count -gt 0) -or
        -not ([string]$Decision.upgrade_protocol.path).Equals(
            [System.IO.Path]::GetFullPath($UpgradeProtocol),
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        $Decision.upgrade_protocol.sha256 -ne $UpgradeProtocolSha256AtStart -or
        -not ([string]$Decision.training_corpus.root).Equals(
            $ResolvedCorpus,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        $Decision.training_corpus.summary_sha256 -ne $CorpusSummarySha256AtStart -or
        $Decision.training_corpus.seal_sha256 -ne $CorpusSealSha256AtStart -or
        [int]$Decision.training_corpus.samples -ne 12176 -or
        [int]$Decision.training_corpus.groups -ne 551 -or
        [int]$Decision.tiny_summary.seed -ne $TinySeed -or
        -not ([string]$Decision.tiny_summary.path).Equals(
            [System.IO.Path]::GetFullPath($TinySummaryPath),
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        $Decision.tiny_summary.sha256 -ne (
            Get-FileHash -Algorithm SHA256 -LiteralPath $TinySummaryPath
        ).Hash.ToLowerInvariant() -or
        -not ([string]$Decision.tiny_checkpoint.path).Equals(
            [System.IO.Path]::GetFullPath($TinyRecognizerPath),
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        $Decision.tiny_checkpoint.sha256 -ne (
            Get-FileHash -Algorithm SHA256 -LiteralPath $TinyRecognizerPath
        ).Hash.ToLowerInvariant() -or
        $Decision.calibration_component_report.protocol -ne
            "syncg_garc_calibration_recognizer_component_v1" -or
        -not ([string]$Decision.calibration_component_report.path).Equals(
            [System.IO.Path]::GetFullPath($CalibrationReportPath),
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        $Decision.calibration_component_report.sha256 -ne (
            Get-FileHash -Algorithm SHA256 -LiteralPath $CalibrationReportPath
        ).Hash.ToLowerInvariant() -or
        -not ([string]$Decision.calibration_roster.path).Equals(
            [System.IO.Path]::GetFullPath(
                "C:\pointer_read\automatic_numeric_range_public_protocol_20260806_v1\manifests\calibration.label_free.jsonl"
            ),
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        $Decision.calibration_roster.sha256 -ne
            "a1dd6e06b18555f9a2d1547e2c418e4dfb989ee3677e00bc088f3faba105ecbc" -or
        [int]$Decision.calibration_roster.samples -ne 2224 -or
        [int]$Decision.calibration_roster.groups -ne 100 -or
        $Decision.calibration_roster.sample_ids_sha256 -ne
            "e610210b7381592a347b39a1d2875f9fec91ff49f741c031ddc8465ecfc40302" -or
        $Decision.calibration_roster.group_ids_sha256 -ne
            "d6e1ae0648590d1b5e61db492196aabf0f32d296062363fb01f348314ee9082a" -or
        $Decision.algorithm_fit_inner_validation.role -ne
            "diagnostic_only_not_used_by_activation_checks" -or
        [bool]$Decision.gpu_work_started -or
        [int]$Decision.data_access_audit.images_opened_by_decision -ne 0 -or
        [int]$Decision.data_access_audit.annotations_opened_by_decision -ne 0 -or
        [int]$Decision.data_access_audit.independent_validation_opened_by_decision -ne 0 -or
        [int]$Decision.data_access_audit.development_excluded_opened_by_decision -ne 0 -or
        [int]$Decision.data_access_audit.joint_oof_412_19_opened_by_decision -ne 0 -or
        [int]$Decision.data_access_audit.field_test_sealed_confirmatory_opened_by_decision -ne 0
    ) {
        throw "Strong-recognizer outer-calibration gate identity or restricted-access binding drift"
    }

    $TinyMetrics = $Decision.garc_outer_calibration
    if (-not [bool]$Decision.train_strong_recognizer) {
        if (Test-Path -LiteralPath $TerminalPath) {
            throw "Strong follow-on terminal already exists; refusing overwrite"
        }
        Write-StrongFollowonTerminal -StrongTrained $false
        Send-ProgressEvent `
            -EventKey "syncg-garc-calibration-tiny-gate-pass-v2" `
            -Message (
                "Tiny 仪表数字 OCR 已通过冻结 GARC calibration 组件门槛：" +
                "100组 calibration exact=$([math]::Round(100 * [double]$TinyMetrics.exact_accuracy, 2))%，" +
                "字符准确率=$([math]::Round(100 * [double]$TinyMetrics.character_accuracy, 2))%。" +
                "不启动 Strong recognizer；independent/development/joint/现场数据均未读取。"
            ) `
            -Eta "随后预计 2–4 小时完成公开 GARC calibration 端到端检测器门槛并冻结最终配置。"
        exit 0
    }

    $StrongTrainingAlreadyComplete = $false
    if (Test-Path -LiteralPath $SummaryPath -PathType Leaf) {
        $Existing = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
        if ($Existing.status -ne "complete" -or $Existing.mode -ne "formal") {
            throw "Strong OCR summary exists but is not a complete formal run"
        }
        $StrongTrainingAlreadyComplete = $true
    }
    if (-not $StrongTrainingAlreadyComplete) {
        if ($Resume) {
            if (-not (Test-Path -LiteralPath $RunRoot -PathType Container)) {
                throw "-Resume requires an existing incomplete Strong OCR run"
            }
        } else {
            if (Test-Path -LiteralPath $RunRoot) {
                throw "Strong OCR run directory already exists without a complete summary"
            }
            if (Test-Path -LiteralPath $LogPath -PathType Leaf) {
                throw "Strong OCR formal log already exists"
            }
        }

        [void](New-Item -ItemType Directory -Force -Path $OutputRoot)
        Remove-AuthenticatedStaleLock
        New-OwnedLock
        Assert-FrozenSourcesUnchanged

        Send-ProgressEvent `
            -EventKey "syncg-numeric-ocr-tiny-to-strong-seed-$StrongSeed-v2" `
            -Message (
                "仪表数字 OCR 已按冻结 GARC calibration activation 门切换到 Strong 候选训练：" +
                "100组 calibration exact=$([math]::Round(100 * [double]$TinyMetrics.exact_accuracy, 2))%，" +
                "字符准确率=$([math]::Round(100 * [double]$TinyMetrics.character_accuracy, 2))%；" +
                "启动 MobileNetV3+SVTR-style CTC，seed=$StrongSeed，" +
                "batch=$ConservativeImageBatchSize，独占 RTX4060。" +
                "Strong 仅作为候选，不会被默认选用；IV/development/joint/现场均未读取。"
            ) `
            -Eta "RTX4060 8GB 单任务预计 2.5–5 小时；完成后由 GARC 在同一 calibration 成对选择，再冻结后开 IV。"

        $TrainerArguments = @(
            $Trainer,
            "--corpus", $Corpus,
            "--output-dir", $OutputRoot,
            "--pretrained-backbone", $PretrainedBackbone,
            "--device", "cuda",
            "--seed", "$StrongSeed",
            "--workers", "$Workers",
            "--epochs", "$Epochs",
            "--frozen-epochs", "$FrozenEpochs",
            "--image-batch-size", "$ConservativeImageBatchSize",
            "--learning-rate", "$LearningRate",
            "--run-formal"
        )
        if ($Resume) {
            $TrainerArguments += "--resume"
            & $Python @TrainerArguments 2>&1 |
                Tee-Object -LiteralPath $LogPath -Append
        } else {
            & $Python @TrainerArguments 2>&1 |
                Tee-Object -LiteralPath $LogPath
        }
        $TrainerExitCode = $LASTEXITCODE
        if ($TrainerExitCode -ne 0) {
            throw "Strong recognizer trainer exited with code $TrainerExitCode"
        }
    }
    Assert-RequiredFile -LiteralPath $SummaryPath
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    if (
        $Summary.status -ne "complete" -or
        $Summary.mode -ne "formal" -or
        [int]$Summary.seed -ne $StrongSeed -or
        -not [bool]$Summary.gpu_work_started -or
        [string]$Summary.code.trainer_sha256 -ne $TrainerSha256AtStart -or
        [string]$Summary.code.implementation_sha256 -ne
            $StrongImplementationSha256AtStart -or
        [string]$Summary.code.evaluator_sha256 -ne
            $TinyTrainerEvaluatorSha256AtStart -or
        [string]$Summary.initialization.sha256 -ne
            $PretrainedBackboneSha256AtStart -or
        [string]$Summary.corpus.summary_sha256 -ne $CorpusSummarySha256AtStart
    ) {
        throw "Strong recognizer summary failed formal completion checks"
    }
    Assert-FrozenSourcesUnchanged
    if (-not (Test-Path -LiteralPath $EvidencePath -PathType Leaf)) {
        & $Python $EvidenceVerifier `
            --corpus $Corpus `
            --tiny-summary $TinySummaryPath `
            --strong-summary $SummaryPath `
            --output $EvidencePath | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "GARC-aligned Tiny/Strong evidence verifier returned exit code $LASTEXITCODE"
        }
    }
    Assert-RequiredFile -LiteralPath $EvidencePath
    $Evidence = Get-Content -LiteralPath $EvidencePath -Raw | ConvertFrom-Json
    if (
        $Evidence.protocol -ne "garc_aligned_ocr_training_evidence_v1" -or
        $Evidence.status -ne "verified_garc_aligned_ocr_training_evidence" -or
        [int]$Evidence.tiny.seed -ne $TinySeed -or
        [int]$Evidence.strong.seed -ne $StrongSeed -or
        $Evidence.tiny.summary.sha256 -ne (
            Get-FileHash -Algorithm SHA256 -LiteralPath $TinySummaryPath
        ).Hash.ToLowerInvariant() -or
        $Evidence.strong.summary.sha256 -ne (
            Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath
        ).Hash.ToLowerInvariant() -or
        -not [bool]$Evidence.audit.all_checkpoints_bind_same_corpus -or
        [int]$Evidence.audit.public_images_opened -ne 0 -or
        [int]$Evidence.audit.public_annotations_opened -ne 0 -or
        [int]$Evidence.audit.outer_values_opened -ne 0 -or
        [int]$Evidence.audit.restricted_namespace_images_opened -ne 0
    ) {
        throw "Strong OCR evidence identity or restricted-access audit drift"
    }
    if (-not (Test-Path -LiteralPath $StrongCalibrationReportPath -PathType Leaf)) {
        & $Python -m experiments.evaluate_syncg_ocr_garc_calibration `
            --corpus $Corpus `
            --recognizer-kind strong `
            --recognizer-summary $SummaryPath `
            --output $StrongCalibrationReportPath `
            --device cuda `
            --batch-size 16 `
            --workers 0 `
            --run-formal | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "GARC outer-calibration Strong component evaluation returned exit code $LASTEXITCODE"
        }
    }
    Assert-RequiredFile -LiteralPath $StrongCalibrationReportPath
    Assert-FrozenSourcesUnchanged
    if (-not (Test-Path -LiteralPath $QualificationPath -PathType Leaf)) {
        & $Python -m experiments.qualify_syncg_strong_numeric_ocr_candidate `
            --tiny-report $CalibrationReportPath `
            --strong-report $StrongCalibrationReportPath `
            --activation $GateDecisionPath `
            --protocol $UpgradeProtocol `
            --output $QualificationPath | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Strong candidate qualifier returned exit code $LASTEXITCODE"
        }
    }
    Assert-RequiredFile -LiteralPath $QualificationPath
    $Qualification = Get-Content -LiteralPath $QualificationPath -Raw |
        ConvertFrom-Json
    $QualificationChecks = @($Qualification.checks.PSObject.Properties)
    $AllQualificationChecksPassed = @(
        $QualificationChecks | Where-Object { -not [bool]$_.Value }
    ).Count -eq 0
    $ExpectedQualificationStatus = if (
        [bool]$Qualification.strong_candidate_eligible_for_garc_calibration
    ) {
        "strong_candidate_qualified"
    } else {
        "strong_candidate_rejected"
    }
    if (
        $Qualification.protocol -ne
            "syncg_strong_numeric_ocr_candidate_qualification_v1" -or
        $Qualification.status -ne $ExpectedQualificationStatus -or
        $Qualification.selection_role -ne
            "candidate_eligibility_only_not_final_selection" -or
        ([bool]$Qualification.strong_candidate_eligible_for_garc_calibration) -ne
            $AllQualificationChecksPassed -or
        $Qualification.upgrade_protocol.sha256 -ne $UpgradeProtocolSha256AtStart -or
        $Qualification.activation_decision.sha256 -ne (
            Get-FileHash -Algorithm SHA256 -LiteralPath $GateDecisionPath
        ).Hash.ToLowerInvariant() -or
        $Qualification.tiny_report.sha256 -ne (
            Get-FileHash -Algorithm SHA256 -LiteralPath $CalibrationReportPath
        ).Hash.ToLowerInvariant() -or
        $Qualification.strong_report.sha256 -ne (
            Get-FileHash -Algorithm SHA256 -LiteralPath $StrongCalibrationReportPath
        ).Hash.ToLowerInvariant() -or
        $Qualification.training_corpus.summary_sha256 -ne
            $CorpusSummarySha256AtStart -or
        $Qualification.training_corpus.seal_sha256 -ne $CorpusSealSha256AtStart -or
        [int]$Qualification.calibration_roster.samples -ne 2224 -or
        [int]$Qualification.calibration_roster.groups -ne 100 -or
        $Qualification.calibration_roster.sample_ids_sha256 -ne
            "e610210b7381592a347b39a1d2875f9fec91ff49f741c031ddc8465ecfc40302" -or
        $Qualification.calibration_roster.group_ids_sha256 -ne
            "d6e1ae0648590d1b5e61db492196aabf0f32d296062363fb01f348314ee9082a" -or
        $Qualification.downstream_contract.final_authority -ne
            "garc_numeric_recognizer_selection_v2" -or
        -not [bool]$Qualification.downstream_contract.strong_is_never_selected_by_this_artifact -or
        [int]$Qualification.data_access_audit.images_opened_by_qualifier -ne 0 -or
        [int]$Qualification.data_access_audit.annotations_opened_by_qualifier -ne 0 -or
        [int]$Qualification.data_access_audit.independent_validation_opened -ne 0 -or
        [int]$Qualification.data_access_audit.development_excluded_opened -ne 0 -or
        [int]$Qualification.data_access_audit.joint_oof_412_19_opened -ne 0 -or
        [int]$Qualification.data_access_audit.field_test_sealed_confirmatory_opened -ne 0 -or
        $Qualification.code.sha256 -ne $QualifierSha256AtStart
    ) {
        throw "Strong candidate qualification identity, semantics, or restricted-access audit drift"
    }
    if (Test-Path -LiteralPath $TerminalPath) {
        throw "Strong follow-on terminal already exists; refusing overwrite"
    }
    Write-StrongFollowonTerminal `
        -StrongTrained $true `
        -StrongSummary $SummaryPath `
        -StrongEvidence $EvidencePath `
        -StrongCalibrationReport $StrongCalibrationReportPath `
        -StrongQualification $QualificationPath
    $Metrics = $Summary.metrics.validation
    $StrongCalibrationMetrics = $Qualification.strong_metrics
    $QualificationLabel = if (
        [bool]$Qualification.strong_candidate_eligible_for_garc_calibration
    ) {
        "qualified，可进入 GARC 最终 calibration selector"
    } else {
        "rejected，GARC 必须使用 Tiny-only"
    }
    Send-ProgressEvent `
        -EventKey "syncg-strong-numeric-ocr-seed-$StrongSeed-qualified-complete-v2" `
        -Message (
            "Strong 仪表数字 OCR 正式训练及同队列资格判定完成：seed=$StrongSeed，" +
            "algorithm-fit inner validation exact=$([math]::Round(100 * [double]$Metrics.exact_accuracy, 2))%，" +
            "outer-calibration exact=$([math]::Round(100 * [double]$StrongCalibrationMetrics.exact_accuracy, 2))%，" +
            "相对 Tiny exact 增益=$([math]::Round(100 * [double]$Qualification.metric_deltas.exact_accuracy_gain, 2))个百分点；" +
            "qualification=$QualificationLabel；" +
            "耗时=$([math]::Round([double]$Summary.elapsed_seconds / 3600, 2)) 小时。" +
            "该结果仅确认候选训练完成，不代表 Strong 已胜出；" +
            "GARC 将在同一 outer calibration 成对选择后才冻结并打开 IV。"
        ) `
        -Eta "随后预计 2–4 小时完成同一 calibration 的候选选择和检测器门槛；之后一次性打开 IV。"
} catch {
    Send-ProgressEvent `
        -EventKey $(if ($Resume) { "syncg-strong-numeric-ocr-seed-$StrongSeed-resume-anomaly-v2" } else { "syncg-strong-numeric-ocr-seed-$StrongSeed-anomaly-v2" }) `
        -Message (
            "Strong 仪表数字 OCR 事件链异常停止：seed=$StrongSeed，" +
            "$($_.Exception.GetType().Name)。日志与已有制品已保留；" +
            "未启动 DBNet++；independent/development/joint/现场数据均未读取。"
        ) `
        -Eta "预计 10–30 分钟核对 Tiny summary、冻结 gate、锁或训练日志后恢复。"
    throw
} finally {
    try {
        Remove-OwnedLock
    } catch {
        Write-Warning "Strong OCR lock cleanup failed; lock preserved for authenticated recovery."
    }
}

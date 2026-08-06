#requires -Version 7.0

[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)]
    [int]$WaitForPid = 0,

    [string]$WaitForStartedAtUtc = "",

    [ValidateRange(0, 2147483647)]
    [int]$WaitForV5Pid = 0,

    [string]$WaitForV5StartedAtUtc = "",

    [string]$Protocol =
        "C:\pointer_read\automatic_numeric_range_public_protocol_20260806_v1\protocol.json",

    [string]$JointOofSummary =
        "C:\pointer_read\garc_full_auto_public_v1\manifests\summary.json",

    [string]$TinySummary =
        "C:\pointer_read\syncg_numeric_ocr_garc_aligned_runs\seed_20260817\summary.json",

    [string]$StrongSummary =
        "C:\pointer_read\syncg_strong_numeric_ocr_garc_aligned_runs\seed_20260818\summary.json",

    [ValidateRange(1, 2147483647)]
    [int]$ExpectedTinySeed = 20260817,

    [ValidateRange(1, 2147483647)]
    [int]$ExpectedStrongSeed = 20260818,

    [string]$OcrCorpus =
        "C:\pointer_read\syncg_numeric_ocr_garc_aligned_v1",

    [string]$OcrActivationDecision =
        "C:\pointer_read\syncg_strong_numeric_ocr_gate_v2\decision.json",

    [string]$StrongFollowonTerminal =
        "C:\pointer_read\syncg_strong_numeric_ocr_gate_v2\strong_followon_terminal.json",

    [string]$StrongCandidateQualification =
        "C:\pointer_read\syncg_strong_numeric_ocr_candidate_qualification_v1\decision.json",

    [string]$V5OofSummary =
        "C:\pointer_read\cagh_v5_enhanced_oof\summary.json",

    [string]$V5GateSummary =
        "C:\pointer_read\cagh_v5_before_ocr_gate_v1\summary.json",

    [string]$OutputRoot = "C:\pointer_read\garc_full_auto_formal_v1",

    [ValidateRange(1, 8)]
    [int]$TorchCpuThreads = 4,

    [ValidateRange(1, 100)]
    [int]$FusionScreenGroups = 16,

    [ValidateRange(1, 8)]
    [int]$FusionScreenSamplesPerGroup = 2,

    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "The GARC formal event chain requires PowerShell 7 or newer."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "garc_full_auto_public.py"
$Evaluator = Join-Path $PSScriptRoot "evaluate_garc_full_auto_public.py"
$ProgressFactory = Join-Path $PSScriptRoot "garc_pepd_progress_factory.py"
$OcrEvidenceVerifier = Join-Path $PSScriptRoot "verify_garc_ocr_training_evidence.py"
$ExternalComparator = Join-Path $PSScriptRoot "garc_external_progress_412.py"
$ExternalProtocol = Join-Path $PSScriptRoot "garc_external_progress_412_protocol.json"
$Reporter = Join-Path $PSScriptRoot "send_feishu_progress.ps1"
$ReferenceDetector = Join-Path (
    $ProjectRoot
) "utils\angleDetect\yoloDetection\result\yolo_pointbest.pt"
$ExpectedJointSummarySha256 =
    "5fb85223204a52ce168b8aed10835054238aef0513bade2fec6047ece323dccb"
$ExpectedCohortSha256 =
    "680cfb8f77fb20517903f62db52ab0931bf46f8b3b8db1e43d8040c0f7d03363"
$ExpectedMappingSha256 =
    "9d88314f339d3b173d9d04b3ea4aa3ecac15bb0affca8166bbf82bf0425a9de9"
$Seeds = @(20260720, 20260721, 20260722)
$ExpectedPepdCheckpointSha256 = @{
    20260720 = "6c3560f5c29f430db33721beb753ec4f5582580792fa098c4a6f1057746a01d9"
    20260721 = "9b7afd6378dc7024a50c6dfbf6bfb9e091c4c2502aa84a49c95671d6e07e98a3"
    20260722 = "d952db3746528c4ebc06919f39e49ff86017887fb468dc4358256609f7fbdd21"
}
$CurrentStage = "dependency-wait"
$RunRootCreated = $false

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $LiteralPath).Hash.ToLowerInvariant()
}

function Assert-RequiredFile {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)
    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        throw "Required file is absent: $LiteralPath"
    }
}

function Assert-OutputRoot {
    $Resolved = [System.IO.Path]::GetFullPath($OutputRoot)
    if (-not $Resolved.StartsWith(
        "C:\pointer_read\",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "GARC output escaped C:\pointer_read: $Resolved"
    }
    if ($Resolved.TrimEnd('\') -ieq "C:\pointer_read") {
        throw "Refusing broad C:\pointer_read output root"
    }
}

function Send-ProgressEvent {
    param(
        [Parameter(Mandatory = $true)][string]$EventKey,
        [Parameter(Mandatory = $true)][string]$Message,
        [Parameter(Mandatory = $true)][string]$Eta
    )
    try {
        & $Reporter -EventKey $EventKey -Message $Message -Eta $Eta | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Progress reporter returned exit code $LASTEXITCODE"
        }
    } catch {
        Write-Warning "Feishu notification failed for stable event $EventKey."
    }
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $script:CurrentStage = $Stage
    $LogPath = Join-Path $script:LogRoot "$Stage.log"
    & $Python @Arguments 2>&1 | Tee-Object -LiteralPath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Python stage $Stage exited with code $ExitCode"
    }
}

function Resolve-ArtifactPath {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $Candidate = if ([System.IO.Path]::IsPathRooted($Value)) {
        $Value
    } else {
        Join-Path $ProjectRoot $Value
    }
    $Resolved = [System.IO.Path]::GetFullPath($Candidate)
    Assert-RequiredFile -LiteralPath $Resolved
    return $Resolved
}

function Read-And-Validate-Upstream {
    param([Parameter(Mandatory = $true)][string]$OcrEvidencePath)
    foreach ($Path in @($TinySummary, $V5OofSummary, $V5GateSummary)) {
        Assert-RequiredFile -LiteralPath $Path
    }
    $CorpusSummaryPath = Join-Path $OcrCorpus "summary.json"
    Assert-RequiredFile -LiteralPath $CorpusSummaryPath
    $Corpus = Get-Content -LiteralPath $CorpusSummaryPath -Raw | ConvertFrom-Json
    if (
        $Corpus.protocol -ne "syncg_public_numeric_ocr_v1" -or
        $Corpus.alignment_protocol -ne "syncg_public_numeric_ocr_garc_aligned_v1" -or
        [int]$Corpus.garc_partition_bindings.algorithm_fit.samples -ne 12176 -or
        [int]$Corpus.garc_partition_bindings.algorithm_fit.groups -ne 551 -or
        -not [bool]$Corpus.alignment_audit.all_outer_group_overlap_zero -or
        -not [bool]$Corpus.alignment_audit.all_outer_sample_overlap_zero
    ) {
        throw "OCR corpus is not the frozen GARC-aligned algorithm-fit corpus"
    }
    $ExpectedCorpusBinding = [ordered]@{
        root = [System.IO.Path]::GetFullPath($OcrCorpus)
        summary_sha256 = Get-FileSha256 $CorpusSummaryPath
        samples_sha256 = [string]$Corpus.artifacts.samples_sha256
        tokens_sha256 = [string]$Corpus.artifacts.tokens_sha256
    }
    Assert-RequiredFile -LiteralPath $OcrEvidencePath
    Assert-RequiredFile -LiteralPath $OcrActivationDecision
    Assert-RequiredFile -LiteralPath $StrongFollowonTerminal
    $OcrEvidence = Get-Content -LiteralPath $OcrEvidencePath -Raw | ConvertFrom-Json
    if (
        $OcrEvidence.protocol -ne "garc_aligned_ocr_training_evidence_v1" -or
        $OcrEvidence.status -ne "verified_garc_aligned_ocr_training_evidence" -or
        -not [bool]$OcrEvidence.corpus.algorithm_fit_exact_coverage -or
        -not [bool]$OcrEvidence.corpus.outer_group_overlap_zero -or
        -not [bool]$OcrEvidence.corpus.outer_sample_overlap_zero -or
        [int]$OcrEvidence.audit.restricted_namespace_images_opened -ne 0
    ) {
        throw "Aligned OCR training evidence verifier did not pass"
    }
    $Activation = Get-Content `
        -LiteralPath $OcrActivationDecision -Raw | ConvertFrom-Json
    $Followon = Get-Content `
        -LiteralPath $StrongFollowonTerminal -Raw | ConvertFrom-Json
    $ParentProtocolForActivation = Get-Content `
        -LiteralPath $Protocol -Raw | ConvertFrom-Json
    $ExpectedOuterCalibration = $ParentProtocolForActivation.partitions.calibration
    $AllowedActivationStatus = @(
        "strong_recognizer_required",
        "tiny_component_gate_pass"
    )
    if (
        [int]$Activation.schema_version -ne 2 -or
        $Activation.protocol -ne "syncg_strong_numeric_ocr_gate_decision_v2" -or
        $AllowedActivationStatus -notcontains [string]$Activation.status -or
        $Activation.selection_partition -ne "garc_outer_calibration" -or
        [bool]$Activation.train_strong_recognizer -ne
            ($Activation.status -eq "strong_recognizer_required") -or
        [bool]$Activation.gpu_work_started -or
        [int]$Activation.calibration_roster.samples -ne
            [int]$ExpectedOuterCalibration.samples -or
        [int]$Activation.calibration_roster.groups -ne
            [int]$ExpectedOuterCalibration.groups -or
        [string]$Activation.calibration_roster.sha256 -ne
            [string]$ExpectedOuterCalibration.sha256 -or
        [string]$Activation.calibration_roster.sample_ids_sha256 -ne
            [string]$ExpectedOuterCalibration.sample_ids_sha256 -or
        [string]$Activation.calibration_roster.group_ids_sha256 -ne
            [string]$ExpectedOuterCalibration.group_ids_sha256
    ) {
        throw "Outer-calibration OCR activation decision is invalid"
    }
    foreach ($Name in @(
        "images_opened_by_decision",
        "annotations_opened_by_decision",
        "independent_validation_opened_by_decision",
        "development_excluded_opened_by_decision",
        "joint_oof_412_19_opened_by_decision",
        "field_test_sealed_confirmatory_opened_by_decision"
    )) {
        if ([int]$Activation.data_access_audit.$Name -ne 0) {
            throw "OCR activation decision crossed a forbidden data boundary: $Name"
        }
    }
    foreach ($Name in @("root", "summary_sha256", "samples_sha256", "tokens_sha256")) {
        if ([string]$Activation.training_corpus.$Name -ne [string]$ExpectedCorpusBinding[$Name]) {
            throw "OCR activation decision aligned-corpus binding drift"
        }
    }
    if (
        [int]$Followon.schema_version -ne 1 -or
        $Followon.protocol -ne "syncg_strong_numeric_ocr_followon_terminal_v1" -or
        $Followon.status -ne "complete" -or
        $Followon.terminal_role -ne
            "candidate_availability_for_downstream_garc_calibration_selection" -or
        $Followon.selection_partition -ne "garc_outer_calibration" -or
        -not [bool]$Followon.activation_only_not_retention_selection -or
        [string]$Followon.activation_decision.sha256 -ne
            (Get-FileSha256 $OcrActivationDecision) -or
        [string]$Followon.calibration_component_report.sha256 -ne
            [string]$Activation.calibration_component_report.sha256 -or
        [bool]$Followon.activation_decision.train_strong_recognizer -ne
            [bool]$Activation.train_strong_recognizer
    ) {
        throw "Strong follow-on terminal does not bind the activation decision"
    }
    foreach ($Name in @(
        "independent_validation_opened",
        "development_excluded_opened",
        "joint_oof_412_19_opened",
        "field_test_sealed_confirmatory_opened"
    )) {
        if ([int]$Followon.access.$Name -ne 0) {
            throw "Strong follow-on terminal crossed a forbidden data boundary: $Name"
        }
    }
    foreach ($Binding in @(
        $Activation.upgrade_protocol,
        $Activation.calibration_component_report
    )) {
        Assert-RequiredFile -LiteralPath ([string]$Binding.path)
        if ((Get-FileSha256 ([string]$Binding.path)) -ne [string]$Binding.sha256) {
            throw "OCR activation decision input hash drift"
        }
    }
    $Tiny = Get-Content -LiteralPath $TinySummary -Raw | ConvertFrom-Json
    if (
        $Tiny.status -ne "complete" -or
        $Tiny.mode -ne "formal" -or
        [int]$Tiny.seed -ne $ExpectedTinySeed -or
        $Tiny.component_selection -ne "both"
    ) {
        throw "Tiny detector/recognizer summary is not complete formal evidence"
    }
    if (
        [string]$OcrEvidence.tiny.summary.sha256 -ne (Get-FileSha256 $TinySummary) -or
        [int]$OcrEvidence.tiny.seed -ne $ExpectedTinySeed
    ) {
        throw "Aligned OCR verifier/Tiny summary binding drift"
    }
    if (
        [string]$Activation.tiny_summary.sha256 -ne (Get-FileSha256 $TinySummary) -or
        [int]$Activation.tiny_summary.seed -ne $ExpectedTinySeed
    ) {
        throw "OCR activation decision/Tiny summary binding drift"
    }
    if (
        [string]$Followon.tiny.summary_sha256 -ne (Get-FileSha256 $TinySummary) -or
        [int]$Followon.tiny.seed -ne $ExpectedTinySeed
    ) {
        throw "Strong follow-on terminal/Tiny summary binding drift"
    }
    foreach ($Name in @("root", "summary_sha256", "samples_sha256", "tokens_sha256")) {
        if ([string]$Tiny.corpus.$Name -ne [string]$ExpectedCorpusBinding[$Name]) {
            throw "Tiny OCR summary is not bound to the frozen aligned corpus"
        }
    }
    $TinyDetector = Resolve-ArtifactPath `
        -Value ([string]$Tiny.artifacts.detector) `
        -Label "Tiny detector"
    $TinyRecognizer = Resolve-ArtifactPath `
        -Value ([string]$Tiny.artifacts.recognizer) `
        -Label "Tiny recognizer"
    if (
        (Get-FileSha256 $TinyDetector) -ne [string]$Tiny.artifact_sha256.detector -or
        (Get-FileSha256 $TinyRecognizer) -ne [string]$Tiny.artifact_sha256.recognizer
    ) {
        throw "Tiny checkpoint hash drift"
    }
    if ([string]$Activation.tiny_checkpoint.sha256 -ne (Get-FileSha256 $TinyRecognizer)) {
        throw "OCR activation decision/Tiny recognizer binding drift"
    }
    if ([string]$Followon.tiny.checkpoint_sha256 -ne (Get-FileSha256 $TinyRecognizer)) {
        throw "Strong follow-on terminal/Tiny recognizer binding drift"
    }
    if (
        [int]$Tiny.components.detector.validation.samples -lt 1 -or
        [int]$Tiny.components.recognizer.validation.tokens -lt 1
    ) {
        throw "Tiny component validation inventory is incomplete"
    }

    $Strong = $null
    $StrongRecognizer = $null
    if (Test-Path -LiteralPath $StrongSummary -PathType Leaf) {
        $Strong = Get-Content -LiteralPath $StrongSummary -Raw | ConvertFrom-Json
        if (
            $Strong.status -ne "complete" -or
            $Strong.mode -ne "formal" -or
            [int]$Strong.seed -ne $ExpectedStrongSeed
        ) {
            throw "Strong recognizer summary is not complete formal evidence"
        }
        foreach ($Name in @("root", "summary_sha256", "samples_sha256", "tokens_sha256")) {
            if ([string]$Strong.corpus.$Name -ne [string]$ExpectedCorpusBinding[$Name]) {
                throw "Strong OCR summary is not bound to the frozen aligned corpus"
            }
        }
        $StrongRecognizer = Resolve-ArtifactPath `
            -Value ([string]$Strong.artifact) `
            -Label "Strong recognizer"
        if ((Get-FileSha256 $StrongRecognizer) -ne [string]$Strong.artifact_sha256) {
            throw "Strong recognizer checkpoint hash drift"
        }
    }
    if (($null -ne $Strong) -ne [bool]$OcrEvidence.strong.available) {
        throw "Aligned OCR verifier optional-Strong availability drift"
    }
    if (($null -ne $Strong) -ne [bool]$Activation.train_strong_recognizer) {
        throw "Optional Strong evidence does not match the outer-calibration activation decision"
    }
    $ExpectedFollowonStrongStatus = if ($null -ne $Strong) {
        "formal_training_complete"
    } else {
        "not_started_by_frozen_calibration_gate"
    }
    if ([string]$Followon.strong.training_status -ne $ExpectedFollowonStrongStatus) {
        throw "Strong follow-on terminal candidate availability drift"
    }
    $StrongCandidateEligible = $false
    $Qualification = $null
    if ($null -ne $Strong) {
        Assert-RequiredFile -LiteralPath $StrongCandidateQualification
        $Qualification = Get-Content `
            -LiteralPath $StrongCandidateQualification -Raw | ConvertFrom-Json
        if (
            $Qualification.protocol -ne
                "syncg_strong_numeric_ocr_candidate_qualification_v1" -or
            $Qualification.status -notin @(
                "strong_candidate_qualified",
                "strong_candidate_rejected"
            ) -or
            $Qualification.selection_role -ne
                "candidate_eligibility_only_not_final_selection" -or
            [string]$Qualification.activation_decision.sha256 -ne
                (Get-FileSha256 $OcrActivationDecision)
        ) {
            throw "Strong candidate qualification artifact is invalid"
        }
        $StrongCandidateEligible = [bool]`
            $Qualification.strong_candidate_eligible_for_garc_calibration
        if (
            $StrongCandidateEligible -ne
                ($Qualification.status -eq "strong_candidate_qualified")
        ) {
            throw "Strong candidate qualification status/eligibility drift"
        }
        if (
            [string]$Followon.strong.candidate_qualification.status -ne
                [string]$Qualification.status -or
            [bool]$Followon.strong.candidate_qualification.eligible_for_garc_calibration -ne
                $StrongCandidateEligible -or
            [string]$Followon.strong.candidate_qualification.protocol -ne
                "syncg_strong_numeric_ocr_candidate_qualification_v1" -or
            [System.IO.Path]::GetFullPath(
                [string]$Followon.strong.candidate_qualification.artifact
            ) -ne [System.IO.Path]::GetFullPath($StrongCandidateQualification) -or
            [string]$Followon.strong.candidate_qualification.artifact_sha256 -ne
                (Get-FileSha256 $StrongCandidateQualification)
        ) {
            throw "Strong follow-on terminal/qualification binding drift"
        }
        foreach ($Name in @(
            "images_opened_by_qualifier",
            "annotations_opened_by_qualifier",
            "independent_validation_opened",
            "development_excluded_opened",
            "joint_oof_412_19_opened",
            "field_test_sealed_confirmatory_opened"
        )) {
            if ([int]$Qualification.data_access_audit.$Name -ne 0) {
                throw "Strong qualification crossed a forbidden data boundary: $Name"
            }
        }
    } elseif (Test-Path -LiteralPath $StrongCandidateQualification -PathType Leaf) {
        throw "Non-activated Strong unexpectedly has a qualification artifact"
    } elseif (
        [string]$Followon.strong.candidate_qualification.status -ne
            "not_applicable_strong_not_activated" -or
        [bool]$Followon.strong.candidate_qualification.eligible_for_garc_calibration
    ) {
        throw "Non-activated Strong terminal qualification state drift"
    }
    if (
        $null -ne $Strong -and
        (
            [string]$OcrEvidence.strong.summary.sha256 -ne
                (Get-FileSha256 $StrongSummary) -or
            [int]$OcrEvidence.strong.seed -ne $ExpectedStrongSeed
        )
    ) {
        throw "Aligned OCR verifier/Strong summary binding drift"
    }
    if (
        $null -ne $Strong -and
        (
            [System.IO.Path]::GetFullPath([string]$Followon.strong.summary) -ne
                [System.IO.Path]::GetFullPath($StrongSummary) -or
            [string]$Followon.strong.summary_sha256 -ne
                (Get-FileSha256 $StrongSummary) -or
            [int]$Followon.strong.seed -ne $ExpectedStrongSeed -or
            -not (Test-Path -LiteralPath ([string]$Followon.strong.evidence) -PathType Leaf) -or
            [string]$Followon.strong.evidence_sha256 -ne
                (Get-FileSha256 ([string]$Followon.strong.evidence))
        )
    ) {
        throw "Strong follow-on terminal/formal Strong evidence binding drift"
    }
    $ExpectedDownstreamSelection = if ($StrongCandidateEligible) {
        "pending_garc_same_calibration_selector_with_qualified_strong_candidate"
    } elseif ($null -ne $Strong) {
        "tiny_only_strong_candidate_rejected_by_component_qualification"
    } else {
        "tiny_only_strong_not_activated"
    }
    if ([string]$Followon.downstream_recognizer_selection -ne $ExpectedDownstreamSelection) {
        throw "Strong follow-on terminal downstream selection state drift"
    }
    if (
        $null -ne $Qualification -and
        (
            [string]$Followon.strong.calibration_component_report.sha256 -ne
                [string]$Qualification.strong_report.sha256 -or
            [string]$Followon.strong.calibration_component_report.protocol -ne
                "syncg_garc_calibration_recognizer_component_v1"
        )
    ) {
        throw "Strong follow-on terminal/calibration report binding drift"
    }

    $V5 = Get-Content -LiteralPath $V5OofSummary -Raw | ConvertFrom-Json
    $V5Gate = Get-Content -LiteralPath $V5GateSummary -Raw | ConvertFrom-Json
    if (
        $V5Gate.protocol -ne "cagh_v5_before_ocr_public_gate_v1" -or
        $V5Gate.status -ne "complete" -or
        $V5Gate.decision -ne "pass" -or
        [string]$V5Gate.runtime_inputs.aggregate_chain.aggregate_summary_sha256 -ne
            (Get-FileSha256 $V5OofSummary)
    ) {
        throw "V5-before-OCR gate is absent, failed, or bound to another aggregate"
    }
    foreach ($Name in @(
        "images_read",
        "annotations_read",
        "prediction_rows_read",
        "public_test_samples_read",
        "field_samples_read",
        "sealed_samples_read",
        "confirmatory_samples_read"
    )) {
        if ([int]$V5Gate.data_access_audit.$Name -ne 0) {
            throw "V5-before-OCR gate crossed a forbidden data boundary: $Name"
        }
    }
    if (
        $V5.status -ne "complete" -or
        $V5.mode -ne "formal" -or
        $V5.protocol -ne "cagh_v5_enhanced_authoritative_pepd_oof_v1" -or
        -not [bool]$V5.strict_oof.overlap_and_assignment_audit.all_rows_jointly_unseen_by_pepd_and_head -or
        [int]$V5.strict_oof.overlap_and_assignment_audit.eligible_union_samples -ne 4380 -or
        [int]$V5.strict_oof.overlap_and_assignment_audit.eligible_union_groups -ne 197
    ) {
        throw "Enhanced V5 OOF final evidence is incomplete or ineligible"
    }
    $GeometryHeadBySeed = @{}
    $GeometryBackboneBySeed = @{}
    $GeometryFoldSummaryBySeed = @{}
    foreach ($Seed in $Seeds) {
        $Entries = @(
            $V5.strict_oof.folds | Where-Object { [int]$_.pepd_seed -eq $Seed }
        )
        if ($Entries.Count -ne 1) {
            throw "Enhanced V5 OOF fold $Seed is absent or duplicated"
        }
        $FoldSummaryPath = Resolve-ArtifactPath `
            -Value ([string]$Entries[0].summary) `
            -Label "Enhanced V5 OOF fold $Seed summary"
        if ((Get-FileSha256 $FoldSummaryPath) -ne [string]$Entries[0].summary_sha256) {
            throw "Enhanced V5 OOF fold $Seed summary hash drift"
        }
        $Fold = Get-Content -LiteralPath $FoldSummaryPath -Raw | ConvertFrom-Json
        if (
            $Fold.status -ne "complete" -or
            $Fold.protocol -ne "cagh_v5_enhanced_authoritative_pepd_oof_v1" -or
            [int]$Fold.pepd_seed -ne $Seed -or
            -not [bool]$Fold.jointly_unseen_contract -or
            [int]$Fold.split_identity.group_overlap -ne 0
        ) {
            throw "Enhanced V5 OOF fold $Seed proof drift"
        }
        $Head = Resolve-ArtifactPath `
            -Value ([string]$Fold.artifacts.checkpoint) `
            -Label "Enhanced V5 OOF fold $Seed head"
        $Backbone = Resolve-ArtifactPath `
            -Value ([string]$Fold.pepd_checkpoint.path) `
            -Label "Enhanced V5 OOF fold $Seed PEPD backbone"
        if (
            (Get-FileSha256 $Head) -ne [string]$Fold.artifacts.checkpoint_sha256 -or
            (Get-FileSha256 $Backbone) -ne [string]$Fold.pepd_checkpoint.sha256 -or
            (Get-FileSha256 $Backbone) -ne $ExpectedPepdCheckpointSha256[$Seed]
        ) {
            throw "Enhanced V5 OOF fold $Seed checkpoint hash drift"
        }
        $GeometryHeadBySeed[$Seed] = $Head
        $GeometryBackboneBySeed[$Seed] = $Backbone
        $GeometryFoldSummaryBySeed[$Seed] = $FoldSummaryPath
    }
    return [ordered]@{
        OcrEvidencePath = $OcrEvidencePath
        OcrEvidenceSha256 = Get-FileSha256 $OcrEvidencePath
        OcrActivationDecision = $Activation
        OcrActivationDecisionPath = $OcrActivationDecision
        OcrActivationDecisionSha256 = Get-FileSha256 $OcrActivationDecision
        StrongFollowonTerminalPath = $StrongFollowonTerminal
        StrongFollowonTerminalSha256 = Get-FileSha256 $StrongFollowonTerminal
        StrongCandidateEligible = $StrongCandidateEligible
        StrongCandidateQualificationPath = if ($null -ne $Qualification) {
            $StrongCandidateQualification
        } else {
            $null
        }
        StrongCandidateQualificationSha256 = if ($null -ne $Qualification) {
            Get-FileSha256 $StrongCandidateQualification
        } else {
            $null
        }
        Tiny = $Tiny
        OcrCorpus = $ExpectedCorpusBinding
        TinyDetector = $TinyDetector
        TinyRecognizer = $TinyRecognizer
        StrongTrained = $null -ne $Strong
        StrongAvailable = $StrongCandidateEligible
        Strong = $Strong
        StrongRecognizer = $StrongRecognizer
        V5 = $V5
        V5GateSummaryPath = $V5GateSummary
        V5GateSummarySha256 = Get-FileSha256 $V5GateSummary
        GeometryHeadBySeed = $GeometryHeadBySeed
        GeometryBackboneBySeed = $GeometryBackboneBySeed
        GeometryFoldSummaryBySeed = $GeometryFoldSummaryBySeed
    }
}

function Assert-JointCohort {
    Assert-RequiredFile -LiteralPath $JointOofSummary
    if ((Get-FileSha256 $JointOofSummary) -ne $ExpectedJointSummarySha256) {
        throw "Frozen joint OOF summary SHA-256 drift"
    }
    $Joint = Get-Content -LiteralPath $JointOofSummary -Raw | ConvertFrom-Json
    if (
        [int]$Joint.input_size -ne 768 -or
        [int]$Joint.joint_oof.samples -ne 412 -or
        [int]$Joint.joint_oof.groups -ne 19 -or
        [int]$Joint.joint_oof.samples_by_pepd_seed.'20260720' -ne 168 -or
        [int]$Joint.joint_oof.samples_by_pepd_seed.'20260721' -ne 116 -or
        [int]$Joint.joint_oof.samples_by_pepd_seed.'20260722' -ne 128 -or
        [string]$Joint.artifacts.cohort.sha256 -ne $ExpectedCohortSha256 -or
        [string]$Joint.artifacts.mapping.sha256 -ne $ExpectedMappingSha256
    ) {
        throw "Frozen joint OOF cohort identity drift"
    }
}

function Wait-ForAuthenticatedProcess {
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [Parameter(Mandatory = $true)][string]$StartedAtUtc,
        [Parameter(Mandatory = $true)][string]$ExpectedCommand,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ($ProcessId -eq 0) {
        return
    }
    if ([string]::IsNullOrWhiteSpace($StartedAtUtc)) {
        throw "$Label PID requires an explicit StartedAtUtc identity"
    }
    $Cim = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $Cim) {
        return
    }
    if ([string]$Cim.CommandLine -notmatch $ExpectedCommand) {
        throw "$Label PID belongs to an unexpected process"
    }
    $ObservedStart = ([datetime]$Cim.CreationDate).ToUniversalTime()
    if (-not [string]::IsNullOrWhiteSpace($StartedAtUtc)) {
        $ExpectedStart = [datetime]::Parse(
            $StartedAtUtc,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Globalization.DateTimeStyles]::AssumeUniversal
        ).ToUniversalTime()
        if ([math]::Abs(($ObservedStart - $ExpectedStart).TotalSeconds) -gt 1.0) {
            throw "$Label PID start identity drift"
        }
    }
    try {
        $Process = [System.Diagnostics.Process]::GetProcessById($ProcessId)
    } catch [System.ArgumentException] {
        # Expected completion may race the CIM read by a few milliseconds.
        # The authenticated summary/hash checks immediately after this wait
        # remain the authoritative completion condition.
        return
    }
    if ([math]::Abs(($Process.StartTime.ToUniversalTime() - $ObservedStart).TotalSeconds) -gt 1.0) {
        throw "$Label PID CIM/process start identity drift"
    }
    # One kernel wait on the already authenticated process object; no timer or polling.
    $Process.WaitForExit()
}

function Wait-For-UpstreamProcesses {
    Wait-ForAuthenticatedProcess `
        -ProcessId $WaitForPid `
        -StartedAtUtc $WaitForStartedAtUtc `
        -ExpectedCommand (
            "run_syncg_strong_numeric_ocr_after_tiny_event_driven.ps1|" +
            "train_syncg_strong_numeric_ocr.py"
        ) `
        -Label "Strong OCR"
    Wait-ForAuthenticatedProcess `
        -ProcessId $WaitForV5Pid `
        -StartedAtUtc $WaitForV5StartedAtUtc `
        -ExpectedCommand "run_cagh_v5_enhanced_oof.py" `
        -Label "enhanced-V5 OOF"
}

function New-GarcPlan {
    param(
        [Parameter(Mandatory = $true)][string]$Tag,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][ValidateSet("tiny", "strong")]
        [string]$RecognizerKind,
        [Parameter(Mandatory = $true)][ValidateSet("top1", "topk")]
        [string]$Consensus,
        [Parameter(Mandatory = $true)]
        [ValidateSet("v5", "v5_pepd_fusion", "v5_pepd_base_fusion")]
        [string]$GeometryMode,
        [Parameter(Mandatory = $true)][string]$RecognizerPath
    )
    $Plan = Join-Path $PlanRoot "$Tag.seed_$Seed.plan.json"
    $Arguments = @(
        $Runner, "freeze-plan",
        "--protocol", $Protocol,
        "--output", $Plan,
        "--progress-binding", $BindingBySeed[$Seed],
        "--progress-factory", $ProgressFactory,
        "--detector-checkpoint", $Upstream.TinyDetector,
        "--recognizer-checkpoint", $RecognizerPath,
        "--geometry-checkpoint", $Upstream.GeometryHeadBySeed[$Seed],
        "--geometry-provider", "enhanced_v5_oof_fold",
        "--geometry-backbone-checkpoint", $Upstream.GeometryBackboneBySeed[$Seed],
        "--geometry-oof-summary", $V5OofSummary,
        "--geometry-oof-seed", "$Seed",
        "--recognizer", $RecognizerKind,
        "--consensus", $Consensus,
        "--geometry-mode", $GeometryMode,
        "--method-name", "GARC-$Tag-fold$Seed+auto-ref",
        "--reference-mode", "auto",
        "--reference-detector-sha256", $ReferenceDetectorSha256,
        "--device", "cuda:0",
        "--input-size", "768",
        "--detector-threshold", "0.40",
        "--posterior-top-k", "5",
        "--joint-oof-summary", $JointOofSummary,
        "--formal-plan"
    )
    Invoke-CheckedPython -Stage "freeze-$Tag-seed-$Seed" -Arguments $Arguments |
        Out-Host
    Invoke-CheckedPython `
        -Stage "validate-$Tag-seed-$Seed" `
        -Arguments @($Runner, "validate-plan", "--plan", $Plan) |
        Out-Host
    return $Plan
}

function Invoke-GarcInference {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string]$Plan,
        [Parameter(Mandatory = $true)][string]$Partition,
        [Parameter(Mandatory = $true)][string]$Output,
        [switch]$Formal,
        [int]$JointSeed = 0,
        [int]$Groups = 2,
        [int]$SamplesPerGroup = 1
    )
    $Arguments = @(
        $Runner, "infer", "--plan", $Plan,
        "--partition", $Partition,
        "--output-root", $Output,
        "--torch-cpu-threads", "$TorchCpuThreads"
    )
    if ($Formal) {
        $Arguments += "--formal"
        if ($JointSeed -ne 0) {
            $Arguments += @("--joint-oof-seed", "$JointSeed")
        }
    } else {
        $Arguments += @(
            "--smoke", "--smoke-groups", "$Groups",
            "--smoke-samples-per-group", "$SamplesPerGroup"
        )
    }
    Invoke-CheckedPython -Stage $Stage -Arguments $Arguments | Out-Host
}

function Invoke-Calibration {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string]$Plan,
        [Parameter(Mandatory = $true)][string]$Predictions,
        [Parameter(Mandatory = $true)][string]$Output,
        [switch]$AllowSmoke
    )
    $Arguments = @(
        $Evaluator, "calibrate", "--plan", $Plan,
        "--prediction-root", $Predictions, "--output", $Output
    )
    if ($AllowSmoke) {
        $Arguments += "--allow-smoke"
    }
    Invoke-CheckedPython -Stage $Stage -Arguments $Arguments | Out-Host
}

foreach ($Required in @(
    $Python,
    $Runner,
    $Evaluator,
    $ProgressFactory,
    $OcrEvidenceVerifier,
    $ExternalComparator,
    $ExternalProtocol,
    $Reporter,
    $Protocol,
    $JointOofSummary,
    $ReferenceDetector
)) {
    Assert-RequiredFile -LiteralPath $Required
}
Assert-OutputRoot
Assert-JointCohort
$ReferenceDetectorSha256 = Get-FileSha256 $ReferenceDetector

if ($PreflightOnly) {
    if (
        ($WaitForPid -ne 0 -and [string]::IsNullOrWhiteSpace($WaitForStartedAtUtc)) -or
        ($WaitForV5Pid -ne 0 -and [string]::IsNullOrWhiteSpace($WaitForV5StartedAtUtc))
    ) {
        throw "Every nonzero upstream PID requires an explicit StartedAtUtc identity"
    }
    $Cim = if ($WaitForPid -eq 0) {
        $null
    } else {
        Get-CimInstance Win32_Process -Filter "ProcessId = $WaitForPid"
    }
    if (
        $null -ne $Cim -and
        [string]$Cim.CommandLine -notmatch (
            "run_syncg_strong_numeric_ocr_after_tiny_event_driven.ps1|" +
            "train_syncg_strong_numeric_ocr.py"
        )
    ) {
        throw "WaitForPid belongs to an unexpected process"
    }
    $V5Cim = if ($WaitForV5Pid -eq 0) {
        $null
    } else {
        Get-CimInstance Win32_Process -Filter "ProcessId = $WaitForV5Pid"
    }
    if (
        $null -ne $V5Cim -and
        [string]$V5Cim.CommandLine -notmatch "run_cagh_v5_enhanced_oof.py"
    ) {
        throw "WaitForV5Pid belongs to an unexpected process"
    }
    $Preflight = [ordered]@{
        protocol = "garc_full_auto_public_event_chain_preflight_v1"
        status = "validated_no_wait_no_inference_no_notification"
        powershell = $PSVersionTable.PSVersion.ToString()
        wait_for_pid = $WaitForPid
        wait_process_present = $null -ne $Cim
        wait_for_v5_pid = $WaitForV5Pid
        wait_v5_process_present = $null -ne $V5Cim
        observed_start_utc = if ($null -eq $Cim) {
            $null
        } else {
            ([datetime]$Cim.CreationDate).ToUniversalTime().ToString("o")
        }
        wait_primitive = "System.Diagnostics.Process.WaitForExit"
        explicit_pid_and_start_time_recommended = $true
        dependency_summaries_present = [ordered]@{
            tiny = Test-Path -LiteralPath $TinySummary -PathType Leaf
            strong = Test-Path -LiteralPath $StrongSummary -PathType Leaf
            strong_followon_terminal = Test-Path `
                -LiteralPath $StrongFollowonTerminal -PathType Leaf
            strong_candidate_qualification = Test-Path `
                -LiteralPath $StrongCandidateQualification -PathType Leaf
            v5_oof = Test-Path -LiteralPath $V5OofSummary -PathType Leaf
            v5_gate = Test-Path -LiteralPath $V5GateSummary -PathType Leaf
        }
        ocr = [ordered]@{
            corpus = [System.IO.Path]::GetFullPath($OcrCorpus)
            corpus_summary_present = Test-Path `
                -LiteralPath (Join-Path $OcrCorpus "summary.json") -PathType Leaf
            expected_tiny_seed = $ExpectedTinySeed
            expected_strong_seed = $ExpectedStrongSeed
            strong_is_optional_candidate = $true
            selection_partition = "calibration"
            forbidden_selection_partitions = @(
                "development_excluded",
                "independent_validation",
                "joint_oof_412_19"
            )
            evidence_verifier_sha256 = Get-FileSha256 $OcrEvidenceVerifier
            evidence_verification_started = $false
        }
        joint_oof = [ordered]@{
            summary_sha256 = $ExpectedJointSummarySha256
            samples = 412
            groups = 19
            seed_counts = [ordered]@{
                '20260720' = 168
                '20260721' = 116
                '20260722' = 128
            }
        }
        calibration_variants = @("tiny_top1", "tiny_topk", "strong_topk_if_available")
        geometry_screen_candidates = @(
            "v5_pepd_fusion",
            "v5_pepd_base_fusion"
        )
        external_comparison = [ordered]@{
            protocol_sha256 = Get-FileSha256 $ExternalProtocol
            comparator_source_sha256 = Get-FileSha256 $ExternalComparator
            preflight_only_before_garc_completion = $true
            inference_started = $false
            transformer_claim_scope = "fixed-checkpoint same-input sensitivity only"
        }
        independent_validation_runs =
            "one 1080 fixed-fold sensitivity run plus two authenticated OOF seed shards"
        output_root = [System.IO.Path]::GetFullPath($OutputRoot)
        output_root_exists = Test-Path -LiteralPath $OutputRoot
        time_polling = $false
        gpu_work_started = $false
        public_images_opened = 0
        restricted_namespace_images_opened = 0
        dbnet_plus_plus_started = $false
        feishu_message_sent = $false
    }
    Write-Output ($Preflight | ConvertTo-Json -Depth 8 -Compress)
    exit 0
}

try {
    Wait-For-UpstreamProcesses
    Assert-JointCohort
    if (Test-Path -LiteralPath $OutputRoot) {
        throw "Refusing to overwrite GARC formal output: $OutputRoot"
    }
    [void](New-Item -ItemType Directory -Path $OutputRoot)
    $RunRootCreated = $true
    $LogRoot = Join-Path $OutputRoot "logs"
    $PlanRoot = Join-Path $OutputRoot "plans"
    $BindingRoot = Join-Path $OutputRoot "progress_bindings"
    $CalibrationRoot = Join-Path $OutputRoot "calibration"
    $ValidationRoot = Join-Path $OutputRoot "independent_validation"
    foreach ($Directory in @(
        $LogRoot,
        $PlanRoot,
        $BindingRoot,
        $CalibrationRoot,
        $ValidationRoot
    )) {
        [void](New-Item -ItemType Directory -Path $Directory)
    }
    $OcrEvidencePath = Join-Path $OutputRoot "ocr_training_evidence.json"
    $OcrEvidenceArguments = @(
        $OcrEvidenceVerifier,
        "--corpus", $OcrCorpus,
        "--tiny-summary", $TinySummary,
        "--output", $OcrEvidencePath
    )
    if (Test-Path -LiteralPath $StrongSummary -PathType Leaf) {
        $OcrEvidenceArguments += @("--strong-summary", $StrongSummary)
    }
    Invoke-CheckedPython `
        -Stage "verify-garc-aligned-ocr-training-evidence" `
        -Arguments $OcrEvidenceArguments | Out-Host
    $Upstream = Read-And-Validate-Upstream -OcrEvidencePath $OcrEvidencePath
    $LockPath = Join-Path $OutputRoot "event_chain.lock.json"
    $Lock = [ordered]@{
        protocol = "garc_full_auto_public_event_chain_lock_v1"
        owner_pid = $PID
        wait_for_pid = $WaitForPid
        wait_for_started_at_utc = $WaitForStartedAtUtc
        wait_for_v5_pid = $WaitForV5Pid
        wait_for_v5_started_at_utc = $WaitForV5StartedAtUtc
        ocr_training_evidence_sha256 = $Upstream.OcrEvidenceSha256
        strong_followon_terminal_sha256 = $Upstream.StrongFollowonTerminalSha256
        v5_before_ocr_gate_summary_sha256 = $Upstream.V5GateSummarySha256
        aligned_ocr_corpus_summary_sha256 = $Upstream.OcrCorpus.summary_sha256
        optional_strong_available = [bool]$Upstream.StrongAvailable
        strong_trained = [bool]$Upstream.StrongTrained
        joint_oof_summary_sha256 = $ExpectedJointSummarySha256
        created_utc = [datetime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Compress
    $Stream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Lock + "`n")
        $Stream.Write($Bytes, 0, $Bytes.Length)
    } finally {
        $Stream.Dispose()
    }

    $RecognizerCandidateMessage = if ($Upstream.StrongAvailable) {
        "先比较 Tiny-top1、Tiny-topK、Strong-topK"
    } else {
        "Strong 未作为已认证候选提供，先比较 Tiny-top1、Tiny-topK"
    }
    Send-ProgressEvent `
        -EventKey "garc-full-auto-public-v1-upstream-to-calibration-v1" `
        -Message (
            "GARC 正式公开实验已进入 calibration 阶段：aligned Tiny、" +
            $(if ($Upstream.StrongAvailable) { "可选 Strong、" } else { "" }) +
            "V5 OOF 与 412/19 联合 OOF 清单均通过哈希检查；" +
            "$RecognizerCandidateMessage；selection 不读取 independent/development/412。"
        ) `
        -Eta "预计 2.5–5 小时完成三候选 calibration、冻结识别器与置信门槛。"

    $BindingBySeed = @{}
    foreach ($Seed in $Seeds) {
        $Binding = Join-Path $BindingRoot "pepd_seed_$Seed.binding.json"
        Invoke-CheckedPython `
            -Stage "freeze-progress-binding-$Seed" `
            -Arguments @(
                $ProgressFactory, "freeze-binding", "--seed", "$Seed",
                "--output", $Binding, "--runtime-device", "cuda:0"
            ) | Out-Host
        $BindingBySeed[$Seed] = $Binding
    }

    $TinyTop1Plan = New-GarcPlan `
        -Tag "v5-tiny-top1" -Seed 20260720 `
        -RecognizerKind "tiny" -Consensus "top1" -GeometryMode "v5" `
        -RecognizerPath $Upstream.TinyRecognizer
    $TinyTopKPlan = New-GarcPlan `
        -Tag "v5-tiny-topk" -Seed 20260720 `
        -RecognizerKind "tiny" -Consensus "topk" -GeometryMode "v5" `
        -RecognizerPath $Upstream.TinyRecognizer
    $BaseCandidates = @(
        [ordered]@{ Name="tiny_top1"; Plan=$TinyTop1Plan },
        [ordered]@{ Name="tiny_topk"; Plan=$TinyTopKPlan }
    )
    $StrongTopKPlan = $null
    if ($Upstream.StrongAvailable) {
        $StrongTopKPlan = New-GarcPlan `
            -Tag "v5-strong-topk" -Seed 20260720 `
            -RecognizerKind "strong" -Consensus "topk" -GeometryMode "v5" `
            -RecognizerPath $Upstream.StrongRecognizer
        $BaseCandidates += [ordered]@{
            Name = "strong_topk"
            Plan = $StrongTopKPlan
        }
    }
    foreach ($Candidate in $BaseCandidates) {
        $Preflight = Join-Path $OutputRoot "preflight-$($Candidate.Name)"
        Invoke-GarcInference `
            -Stage "preflight-$($Candidate.Name)" `
            -Plan $Candidate.Plan -Partition "calibration" -Output $Preflight
        $Predictions = Join-Path $CalibrationRoot "$($Candidate.Name)-predictions"
        $Calibration = Join-Path $CalibrationRoot "$($Candidate.Name).json"
        Invoke-GarcInference `
            -Stage "infer-calibration-$($Candidate.Name)" `
            -Plan $Candidate.Plan -Partition "calibration" `
            -Output $Predictions -Formal
        Invoke-Calibration `
            -Stage "seal-calibration-$($Candidate.Name)" `
            -Plan $Candidate.Plan -Predictions $Predictions -Output $Calibration
        $Candidate["Calibration"] = $Calibration
    }

    $RecognizerDecisionPath = Join-Path $OutputRoot "recognizer_selection.json"
    $TinyTop1Candidate = @($BaseCandidates | Where-Object Name -eq "tiny_top1")[0]
    $TinyTopKCandidate = @($BaseCandidates | Where-Object Name -eq "tiny_topk")[0]
    $RecognizerSelectionArguments = @(
        $Evaluator, "select-recognizer",
        "--tiny-top1", "$TinyTop1Plan::$($TinyTop1Candidate.Calibration)",
        "--tiny-topk", "$TinyTopKPlan::$($TinyTopKCandidate.Calibration)",
        "--training-evidence", $Upstream.OcrEvidencePath,
        "--activation-decision", $Upstream.OcrActivationDecisionPath
    )
    if ($Upstream.StrongAvailable) {
        $StrongTopKCandidate = @(
            $BaseCandidates | Where-Object Name -eq "strong_topk"
        )[0]
        $RecognizerSelectionArguments += @(
            "--strong-topk", "$StrongTopKPlan::$($StrongTopKCandidate.Calibration)"
        )
    }
    if ($Upstream.StrongTrained) {
        $RecognizerSelectionArguments += @(
            "--strong-qualification",
            $Upstream.StrongCandidateQualificationPath
        )
    }
    $RecognizerSelectionArguments += @("--output", $RecognizerDecisionPath)
    Invoke-CheckedPython `
        -Stage "select-recognizer-calibration-only" `
        -Arguments $RecognizerSelectionArguments | Out-Host
    $RecognizerDecision = Get-Content `
        -LiteralPath $RecognizerDecisionPath -Raw | ConvertFrom-Json
    $RecognizerDecisionSha256 = Get-FileSha256 $RecognizerDecisionPath
    if (
        $RecognizerDecision.protocol -ne "garc_numeric_recognizer_selection_v2" -or
        $RecognizerDecision.status -ne "frozen_calibration_only_selection" -or
        $RecognizerDecision.audit.selection_data_partition -ne "calibration" -or
        [int]$RecognizerDecision.audit.independent_validation_artifacts_opened -ne 0 -or
        [int]$RecognizerDecision.audit.restricted_namespace_images_opened -ne 0 -or
        -not [bool]$RecognizerDecision.audit.formal_calibration_results_only -or
        [string]$RecognizerDecision.ocr_training_evidence.sha256 -ne
            [string]$Upstream.OcrEvidenceSha256 -or
        [string]$RecognizerDecision.outer_calibration_activation_decision.sha256 -ne
            [string]$Upstream.OcrActivationDecisionSha256 -or
        (
            $Upstream.StrongTrained -and
            [string]$RecognizerDecision.strong_candidate_qualification.sha256 -ne
                [string]$Upstream.StrongCandidateQualificationSha256
        ) -or
        (
            -not $Upstream.StrongTrained -and
            $null -ne $RecognizerDecision.strong_candidate_qualification
        ) -or
        ($null -eq $RecognizerDecision.strong_topk) -ne (-not $Upstream.StrongAvailable)
    ) {
        throw "Recognizer selection escaped the frozen outer-calibration boundary"
    }
    $ParentProtocol = Get-Content -LiteralPath $Protocol -Raw | ConvertFrom-Json
    $ExpectedCalibration = $ParentProtocol.partitions.calibration
    foreach ($Candidate in @(
        $RecognizerDecision.tiny_top1,
        $RecognizerDecision.tiny_topk,
        $RecognizerDecision.strong_topk
    )) {
        if ($null -eq $Candidate) {
            continue
        }
        if (
            [int]$Candidate.samples -ne [int]$ExpectedCalibration.samples -or
            [int]$Candidate.groups -ne [int]$ExpectedCalibration.groups -or
            [string]$Candidate.sample_ids_sha256 -ne
                [string]$ExpectedCalibration.sample_ids_sha256
        ) {
            throw "Recognizer candidate is not the frozen 2224/100 outer calibration cohort"
        }
    }
    $SelectedBasePlan = [string]$RecognizerDecision.selected_plan
    $SelectedBaseCalibration = [string]$RecognizerDecision.selected_calibration
    $SelectedRecognizer = [string]$RecognizerDecision.selected_recognizer
    $SelectedConsensus = [string]$RecognizerDecision.selected_consensus
    $SelectedRecognizerPath = if ($SelectedRecognizer -eq "strong") {
        if (-not $Upstream.StrongAvailable) {
            throw "Recognizer selection chose unavailable Strong evidence"
        }
        $Upstream.StrongRecognizer
    } else {
        $Upstream.TinyRecognizer
    }
    $SelectedCheckpointSha256 = if ($SelectedRecognizer -eq "strong") {
        Get-FileSha256 $Upstream.StrongRecognizer
    } else {
        Get-FileSha256 $Upstream.TinyRecognizer
    }
    $SelectedDecisionCandidate = if ($SelectedRecognizer -eq "strong") {
        $RecognizerDecision.strong_topk
    } elseif ($SelectedConsensus -eq "topk") {
        $RecognizerDecision.tiny_topk
    } else {
        $RecognizerDecision.tiny_top1
    }
    if (
        [string]$SelectedDecisionCandidate.checkpoints.recognizer.sha256 -ne
            $SelectedCheckpointSha256 -or
        [string]$SelectedDecisionCandidate.checkpoints.detector.sha256 -ne
            (Get-FileSha256 $Upstream.TinyDetector)
    ) {
        throw "Recognizer selection artifact/checkpoint binding drift"
    }

    Send-ProgressEvent `
        -EventKey "garc-full-auto-public-v1-recognizer-selected-v1" `
        -Message (
            "GARC calibration-only 识别器/解码选择完成：" +
            "recognizer=$SelectedRecognizer，consensus=$SelectedConsensus；" +
            "置信门槛已由公开 calibration 冻结，尚未读取 independent-validation。"
        ) `
        -Eta "预计 0.5–2 小时完成低成本 geometry 融合筛选；仅过门槛候选进入正式 calibration。"

    $FusionPlan = New-GarcPlan `
        -Tag "v5-pepd-fusion-$SelectedRecognizer-$SelectedConsensus" `
        -Seed 20260720 -RecognizerKind $SelectedRecognizer `
        -Consensus $SelectedConsensus -GeometryMode "v5_pepd_fusion" `
        -RecognizerPath $SelectedRecognizerPath
    $BaseScreenPredictions = Join-Path $CalibrationRoot "geometry-base-screen-predictions"
    $BaseScreenCalibration = Join-Path $CalibrationRoot "geometry-base-screen.json"
    Invoke-GarcInference `
        -Stage "geometry-base-screen-infer" -Plan $SelectedBasePlan `
        -Partition "calibration" -Output $BaseScreenPredictions `
        -Groups $FusionScreenGroups `
        -SamplesPerGroup $FusionScreenSamplesPerGroup
    Invoke-Calibration `
        -Stage "geometry-base-screen-calibrate" -Plan $SelectedBasePlan `
        -Predictions $BaseScreenPredictions -Output $BaseScreenCalibration `
        -AllowSmoke
    $FusionScreenPredictions = Join-Path $CalibrationRoot "geometry-pepd-screen-predictions"
    $FusionScreenCalibration = Join-Path $CalibrationRoot "geometry-pepd-screen.json"
    Invoke-GarcInference `
        -Stage "geometry-pepd-screen-infer" -Plan $FusionPlan `
        -Partition "calibration" -Output $FusionScreenPredictions `
        -Groups $FusionScreenGroups `
        -SamplesPerGroup $FusionScreenSamplesPerGroup
    Invoke-Calibration `
        -Stage "geometry-pepd-screen-calibrate" -Plan $FusionPlan `
        -Predictions $FusionScreenPredictions -Output $FusionScreenCalibration `
        -AllowSmoke

    # The real Base Mask--Geometry candidate is enabled only when the runner
    # exposes the frozen mode.  It is never represented by v5_pepd_fusion.
    $BaseFusionPlan = $null
    $BaseFusionScreenCalibration = $null
    $FreezeHelp = (& $Python $Runner "freeze-plan" "--help" 2>&1) -join "`n"
    if ($FreezeHelp -match "v5_pepd_base_fusion") {
        $BaseFusionPlan = New-GarcPlan `
            -Tag "v5-pepd-base-fusion-$SelectedRecognizer-$SelectedConsensus" `
            -Seed 20260720 -RecognizerKind $SelectedRecognizer `
            -Consensus $SelectedConsensus `
            -GeometryMode "v5_pepd_base_fusion" `
            -RecognizerPath $SelectedRecognizerPath
        $BaseFusionScreenPredictions = Join-Path `
            $CalibrationRoot "geometry-pepd-base-screen-predictions"
        $BaseFusionScreenCalibration = Join-Path `
            $CalibrationRoot "geometry-pepd-base-screen.json"
        Invoke-GarcInference `
            -Stage "geometry-pepd-base-screen-infer" -Plan $BaseFusionPlan `
            -Partition "calibration" -Output $BaseFusionScreenPredictions `
            -Groups $FusionScreenGroups `
            -SamplesPerGroup $FusionScreenSamplesPerGroup
        Invoke-Calibration `
            -Stage "geometry-pepd-base-screen-calibrate" -Plan $BaseFusionPlan `
            -Predictions $BaseFusionScreenPredictions `
            -Output $BaseFusionScreenCalibration -AllowSmoke
    }

    $GeometryScreenDecisionPath = Join-Path $OutputRoot "geometry_screen_selection.json"
    $GeometryScreenArguments = @(
        $Evaluator, "select-geometry", "--stage", "screen",
        "--base", "$SelectedBasePlan::$BaseScreenCalibration",
        "--fusion", "$FusionPlan::$FusionScreenCalibration"
    )
    if ($null -ne $BaseFusionPlan) {
        $GeometryScreenArguments += @(
            "--fusion", "$BaseFusionPlan::$BaseFusionScreenCalibration"
        )
    }
    $GeometryScreenArguments += @("--output", $GeometryScreenDecisionPath)
    Invoke-CheckedPython `
        -Stage "select-geometry-screen" `
        -Arguments $GeometryScreenArguments | Out-Host
    $GeometryScreenDecision = Get-Content `
        -LiteralPath $GeometryScreenDecisionPath -Raw | ConvertFrom-Json

    $FormalFusionPairs = @()
    foreach ($Candidate in @($GeometryScreenDecision.fusion_candidates)) {
        if (-not [bool]$Candidate.retention_pass) {
            continue
        }
        $CandidatePlan = [string]$Candidate.plan
        $CandidateMode = [string]$Candidate.geometry_mode
        $CandidatePredictions = Join-Path `
            $CalibrationRoot "geometry-$CandidateMode-formal-predictions"
        $CandidateCalibration = Join-Path `
            $CalibrationRoot "geometry-$CandidateMode-formal.json"
        Invoke-GarcInference `
            -Stage "geometry-$CandidateMode-formal-infer" `
            -Plan $CandidatePlan -Partition "calibration" `
            -Output $CandidatePredictions -Formal
        Invoke-Calibration `
            -Stage "geometry-$CandidateMode-formal-calibrate" `
            -Plan $CandidatePlan -Predictions $CandidatePredictions `
            -Output $CandidateCalibration
        $FormalFusionPairs += "$CandidatePlan::$CandidateCalibration"
    }

    $SelectedPlan = $SelectedBasePlan
    $SelectedCalibration = $SelectedBaseCalibration
    $SelectedGeometry = "v5"
    $GeometryFinalDecisionPath = $GeometryScreenDecisionPath
    if ($FormalFusionPairs.Count -gt 0) {
        $GeometryFinalDecisionPath = Join-Path $OutputRoot "geometry_final_selection.json"
        $GeometryFinalArguments = @(
            $Evaluator, "select-geometry", "--stage", "final",
            "--base", "$SelectedBasePlan::$SelectedBaseCalibration"
        )
        foreach ($Pair in $FormalFusionPairs) {
            $GeometryFinalArguments += @("--fusion", $Pair)
        }
        $GeometryFinalArguments += @("--output", $GeometryFinalDecisionPath)
        Invoke-CheckedPython `
            -Stage "select-geometry-final" `
            -Arguments $GeometryFinalArguments | Out-Host
        $GeometryFinalDecision = Get-Content `
            -LiteralPath $GeometryFinalDecisionPath -Raw | ConvertFrom-Json
        $SelectedPlan = [string]$GeometryFinalDecision.selected_plan
        $SelectedCalibration = [string]$GeometryFinalDecision.selected_calibration
        $SelectedGeometry = [string]$GeometryFinalDecision.selected_geometry
    }

    Send-ProgressEvent `
        -EventKey "garc-full-auto-public-v1-geometry-selected-v1" `
        -Message (
            "GARC geometry 的 calibration-only 冻结选择完成：" +
            "selected=$SelectedGeometry；未通过门槛的融合不进入验证，" +
            "independent-validation 仍保持未读。"
        ) `
        -Eta "预计 1–2.5 小时完成1080/50固定折敏感性表与三折412/19共同未见主E2E。"

    # The recognizer decision was atomically written from outer calibration.
    # Re-authenticate its exact bytes immediately before the first independent-
    # validation inference so no later stage can silently substitute a model.
    if (
        (Get-FileSha256 $RecognizerDecisionPath) -ne $RecognizerDecisionSha256 -or
        [string]$RecognizerDecision.code.sha256 -ne (Get-FileSha256 $Evaluator) -or
        [string]$RecognizerDecision.selected_plan -ne $SelectedBasePlan -or
        [string]$RecognizerDecision.selected_calibration -ne $SelectedBaseCalibration -or
        [string]$RecognizerDecision.selected_recognizer -ne $SelectedRecognizer -or
        [string]$RecognizerDecision.selected_consensus -ne $SelectedConsensus
    ) {
        throw "Recognizer selection artifact changed before independent validation"
    }

    $SelectedPlanObject = Get-Content -LiteralPath $SelectedPlan -Raw | ConvertFrom-Json
    $ValidationPlans = @{
        20260720 = $SelectedPlan
    }
    foreach ($Seed in @(20260721, 20260722)) {
        $ValidationPlans[$Seed] = New-GarcPlan `
            -Tag "selected-$SelectedGeometry-$SelectedRecognizer-$SelectedConsensus" `
            -Seed $Seed -RecognizerKind $SelectedRecognizer `
            -Consensus $SelectedConsensus -GeometryMode $SelectedGeometry `
            -RecognizerPath $SelectedRecognizerPath
        $FoldPreflight = Join-Path $OutputRoot "preflight-selected-seed-$Seed"
        Invoke-GarcInference `
            -Stage "preflight-selected-seed-$Seed" `
            -Plan $ValidationPlans[$Seed] -Partition "calibration" `
            -Output $FoldPreflight
    }

    Send-ProgressEvent `
        -EventKey "garc-full-auto-public-v1-calibration-to-validation-v1" `
        -Message (
            "GARC 已锁定最终 calibration 方案并进入一次性公开评估：" +
            "1080张/50组仅作seed20固定折敏感性；正式主E2E必须将progress、" +
            "enhanced-V5 head和geometry backbone逐折路由并精确合并412张/19组。"
        ) `
        -Eta "预计 1–2.5 小时完成；完成、异常或关键门槛判定时再汇报。"

    $PrimaryValidation = Join-Path $ValidationRoot "primary-seed-20260720"
    Invoke-GarcInference `
        -Stage "validation-primary-1080" -Plan $ValidationPlans[20260720] `
        -Partition "independent_validation" -Output $PrimaryValidation -Formal
    $Shard21 = Join-Path $ValidationRoot "joint-shard-seed-20260721"
    Invoke-GarcInference `
        -Stage "validation-joint-shard-20260721" `
        -Plan $ValidationPlans[20260721] `
        -Partition "independent_validation" -Output $Shard21 `
        -Formal -JointSeed 20260721
    $Shard22 = Join-Path $ValidationRoot "joint-shard-seed-20260722"
    Invoke-GarcInference `
        -Stage "validation-joint-shard-20260722" `
        -Plan $ValidationPlans[20260722] `
        -Partition "independent_validation" -Output $Shard22 `
        -Formal -JointSeed 20260722

    $ValidationReport = Join-Path $OutputRoot "validation.json"
    Invoke-CheckedPython `
        -Stage "validate-sealed-independent" `
        -Arguments @(
            $Evaluator, "validate",
            "--plan", $ValidationPlans[20260720],
            "--prediction-root", $PrimaryValidation,
            "--calibration", $SelectedCalibration,
            "--joint-run", "$($ValidationPlans[20260721])::$Shard21",
            "--joint-run", "$($ValidationPlans[20260722])::$Shard22",
            "--output", $ValidationReport
        ) | Out-Host
    $Validation = Get-Content -LiteralPath $ValidationReport -Raw | ConvertFrom-Json
    if (
        $Validation.mode -ne "formal" -or
        [bool]$Validation.evidence_eligibility.range_component_claim -or
        -not [bool]$Validation.evidence_eligibility.ocr_and_fixed_fold_geometry_sensitivity -or
        -not [bool]$Validation.evidence_eligibility.joint_oof_end_to_end_claim -or
        [int]$Validation.overlap_audit.jointly_unseen_samples -ne 412 -or
        [int]$Validation.overlap_audit.jointly_unseen_groups -ne 19 -or
        [int]$Validation.overlap_audit.primary_fixed_fold_sensitivity.geometry_unseen_samples -ne 168 -or
        [int]$Validation.overlap_audit.primary_fixed_fold_sensitivity.geometry_unseen_groups -ne 8 -or
        [int]$Validation.overlap_audit.primary_fixed_fold_sensitivity.geometry_fit_overlap_samples -ne 912 -or
        [int]$Validation.overlap_audit.primary_fixed_fold_sensitivity.geometry_fit_overlap_groups -ne 42
    ) {
        throw "Final GARC report failed fixed-fold sensitivity or 412/19 all-component OOF checks"
    }
    $Range = $Validation.metrics.range_component_frozen_acceptance
    $JointRange = $Validation.metrics.joint_oof_range_frozen_acceptance
    $JointMetric = $Validation.metrics.joint_oof_end_to_end_frozen_acceptance
    $DetectorGate = $Validation.metrics.detector_independent_validation.dbnet_plus_plus_gate

    # Freeze the exact same-ROI/same-range packet for external progress
    # comparators.  This stage authenticates metadata and existing GARC seals
    # only; it does not run VDN or Transformer inference.
    $ExternalRoot = Join-Path $OutputRoot "external_progress_comparison"
    $ExternalPreflightPath = Join-Path $ExternalRoot "preflight.json"
    Invoke-CheckedPython `
        -Stage "external-progress-preflight" `
        -Arguments @(
            "-m", "experiments.garc_external_progress_412",
            "--protocol", $ExternalProtocol,
            "preflight", "--output", $ExternalPreflightPath
        ) | Out-Host
    $ExternalPreflight = Get-Content `
        -LiteralPath $ExternalPreflightPath -Raw | ConvertFrom-Json
    if (
        $ExternalPreflight.status -ne "passed_with_transformer_sensitivity_only" -or
        -not [bool]$ExternalPreflight.methods.vdn_official200.strict_412_progress_oof_eligible -or
        [bool]$ExternalPreflight.methods.original_transformer.strict_412_progress_oof_eligible -or
        [bool]$ExternalPreflight.audit.external_inference_started -or
        [int]$ExternalPreflight.audit.restricted_namespace_images_opened -ne 0
    ) {
        throw "External comparison preflight claim scope drift"
    }
    $ExternalHandoffRoot = Join-Path $ExternalRoot "handoff"
    Invoke-CheckedPython `
        -Stage "external-progress-build-handoff" `
        -Arguments @(
            "-m", "experiments.garc_external_progress_412",
            "--protocol", $ExternalProtocol,
            "build-handoff",
            "--preflight", $ExternalPreflightPath,
            "--primary-plan", $ValidationPlans[20260720],
            "--primary-prediction-root", $PrimaryValidation,
            "--calibration", $SelectedCalibration,
            "--joint-run", "$($ValidationPlans[20260721])::$Shard21",
            "--joint-run", "$($ValidationPlans[20260722])::$Shard22",
            "--output-root", $ExternalHandoffRoot
        ) | Out-Host
    $ExternalHandoffSummaryPath = Join-Path $ExternalHandoffRoot "summary.json"
    $ExternalHandoffSealPath = Join-Path $ExternalHandoffRoot "seal.json"
    Assert-RequiredFile -LiteralPath $ExternalHandoffSummaryPath
    Assert-RequiredFile -LiteralPath $ExternalHandoffSealPath
    $ExternalHandoff = Get-Content `
        -LiteralPath $ExternalHandoffSummaryPath -Raw | ConvertFrom-Json
    if (
        $ExternalHandoff.status -ne "label_free_handoff_sealed" -or
        [int]$ExternalHandoff.cohort.samples -ne 412 -or
        [int]$ExternalHandoff.cohort.physical_groups -ne 19 -or
        -not [bool]$ExternalHandoff.garc.all_412_progress_geometry_head_and_backbone_group_unseen -or
        [bool]$ExternalHandoff.audit.external_inference_started -or
        [int]$ExternalHandoff.audit.restricted_namespace_images_opened -ne 0
    ) {
        throw "External comparison handoff inventory or claim scope drift"
    }
    $ExternalVdnRoot = Join-Path $ExternalRoot "vdn_predictions"
    $ExternalTransformerRoot = Join-Path $ExternalRoot "transformer_predictions"
    $ExternalScorePath = Join-Path $ExternalRoot "score.json"
    $ExternalCommandsPath = Join-Path $ExternalRoot "launch_commands.json"
    $ExternalCommands = [ordered]@{
        schema_version = 1
        protocol = "garc_external_progress_412_launch_handoff_v1"
        status = "frozen_not_started"
        working_directory = $ProjectRoot
        python = $Python
        comparator = [ordered]@{
            path = $ExternalComparator
            sha256 = Get-FileSha256 $ExternalComparator
        }
        frozen_protocol = [ordered]@{
            path = $ExternalProtocol
            sha256 = Get-FileSha256 $ExternalProtocol
        }
        preflight = [ordered]@{
            path = $ExternalPreflightPath
            sha256 = Get-FileSha256 $ExternalPreflightPath
        }
        handoff = [ordered]@{
            root = $ExternalHandoffRoot
            summary_sha256 = Get-FileSha256 $ExternalHandoffSummaryPath
            seal_sha256 = Get-FileSha256 $ExternalHandoffSealPath
        }
        stages = @(
            [ordered]@{
                name = "vdn_official200_strict_oof_infer"
                claim_scope = "strict grouped-OOF external progress comparator"
                argv = @(
                    "-m", "experiments.garc_external_progress_412",
                    "--protocol", $ExternalProtocol, "infer",
                    "--preflight", $ExternalPreflightPath,
                    "--handoff-root", $ExternalHandoffRoot,
                    "--method", "vdn_official200",
                    "--output-root", $ExternalVdnRoot,
                    "--device", "cuda:0"
                )
            },
            [ordered]@{
                name = "original_transformer_same_input_sensitivity_infer"
                claim_scope = "fixed-checkpoint sensitivity only; not strict OOF"
                argv = @(
                    "-m", "experiments.garc_external_progress_412",
                    "--protocol", $ExternalProtocol, "infer",
                    "--preflight", $ExternalPreflightPath,
                    "--handoff-root", $ExternalHandoffRoot,
                    "--method", "original_transformer",
                    "--output-root", $ExternalTransformerRoot,
                    "--device", "cuda:0"
                )
            },
            [ordered]@{
                name = "paired_412_score_after_both_inference_runs"
                claim_scope = "VDN strict table; Transformer sensitivity table"
                argv = @(
                    "-m", "experiments.garc_external_progress_412",
                    "--protocol", $ExternalProtocol, "score",
                    "--preflight", $ExternalPreflightPath,
                    "--handoff-root", $ExternalHandoffRoot,
                    "--external", "vdn_official200=$ExternalVdnRoot",
                    "--external", "original_transformer=$ExternalTransformerRoot",
                    "--output", $ExternalScorePath
                )
            }
        )
        audit = [ordered]@{
            external_inference_started = $false
            public_images_opened_by_handoff_stage = 0
            restricted_namespace_images_opened = 0
        }
    }
    $ExternalCommands | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $ExternalCommandsPath -Encoding utf8

    $FinalSummary = [ordered]@{
        schema_version = 1
        protocol = "garc_full_auto_public_event_chain_summary_v1"
        status = "complete"
        selected = [ordered]@{
            recognizer = $SelectedRecognizer
            consensus = $SelectedConsensus
            geometry = $SelectedGeometry
            plan = $ValidationPlans[20260720]
            plan_sha256 = Get-FileSha256 $ValidationPlans[20260720]
            calibration = $SelectedCalibration
            calibration_sha256 = Get-FileSha256 $SelectedCalibration
        }
        aligned_ocr_training_evidence = [ordered]@{
            path = $Upstream.OcrEvidencePath
            sha256 = $Upstream.OcrEvidenceSha256
            corpus_summary_sha256 = $Upstream.OcrCorpus.summary_sha256
            tiny_seed = $ExpectedTinySeed
            strong_candidate_available = [bool]$Upstream.StrongAvailable
            strong_trained = [bool]$Upstream.StrongTrained
            qualification = if ($Upstream.StrongTrained) {
                [ordered]@{
                    path = $Upstream.StrongCandidateQualificationPath
                    sha256 = $Upstream.StrongCandidateQualificationSha256
                    eligible = [bool]$Upstream.StrongCandidateEligible
                }
            } else {
                $null
            }
            strong_seed = if ($Upstream.StrongAvailable) {
                $ExpectedStrongSeed
            } else {
                $null
            }
            selection_partition = "calibration"
            independent_validation_opened_before_selection = 0
        }
        strong_followon_terminal = [ordered]@{
            path = $Upstream.StrongFollowonTerminalPath
            sha256 = $Upstream.StrongFollowonTerminalSha256
            strong_candidate_available = [bool]$Upstream.StrongAvailable
            retention_selection_authority = $RecognizerDecisionPath
        }
        v5_before_ocr_gate = [ordered]@{
            path = $Upstream.V5GateSummaryPath
            sha256 = $Upstream.V5GateSummarySha256
            decision = "pass"
        }
        recognizer_selection = [ordered]@{
            path = $RecognizerDecisionPath
            sha256 = $RecognizerDecisionSha256
            protocol = "garc_numeric_recognizer_selection_v2"
            selected_recognizer = $SelectedRecognizer
            selected_consensus = $SelectedConsensus
            selected_checkpoint_sha256 = $SelectedCheckpointSha256
            selection_partition = "calibration"
            independent_validation_artifacts_opened = 0
        }
        validation = [ordered]@{
            path = $ValidationReport
            sha256 = Get-FileSha256 $ValidationReport
            fixed_fold_sensitivity_samples = 1080
            fixed_fold_sensitivity_groups = 50
            fixed_seed_20260720_geometry_unseen_samples = 168
            fixed_seed_20260720_geometry_unseen_groups = 8
            fixed_seed_20260720_geometry_fit_overlap_samples = 912
            fixed_seed_20260720_geometry_fit_overlap_groups = 42
            joint_samples = 412
            joint_groups = 19
            fixed_fold_sensitivity_range_coverage = [double]$Range.coverage
            fixed_fold_sensitivity_pair_exact_full_denominator =
                [double]$Range.pair_rounded_exact_full_denominator
            joint_range_coverage = [double]$JointRange.coverage
            joint_range_pair_exact_full_denominator =
                [double]$JointRange.pair_rounded_exact_full_denominator
            joint_reading_nmae_full_denominator =
                [double]$JointMetric.reading_nmae_full_denominator_failure_penalty_1
            paper_claim_allowed = [ordered]@{
                full_1080_all_component_unseen = $false
                fixed_fold_1080_sensitivity = $true
                joint_412_all_component_oof_end_to_end = $true
            }
        }
        detector_gate = [ordered]@{
            recall_iou_0_5 = [double]$DetectorGate.observed_detector_box_recall_iou_0_5
            threshold = [double]$DetectorGate.threshold
            dbnet_plus_plus_triggered = [bool]$DetectorGate.triggered
            dbnet_plus_plus_started = $false
        }
        external_comparison_handoff = [ordered]@{
            protocol = "garc_external_progress_412_comparison_v1"
            preflight = [ordered]@{
                path = $ExternalPreflightPath
                sha256 = Get-FileSha256 $ExternalPreflightPath
            }
            handoff = [ordered]@{
                root = $ExternalHandoffRoot
                summary_sha256 = Get-FileSha256 $ExternalHandoffSummaryPath
                seal_sha256 = Get-FileSha256 $ExternalHandoffSealPath
                samples = 412
                groups = 19
            }
            launch_commands = [ordered]@{
                path = $ExternalCommandsPath
                sha256 = Get-FileSha256 $ExternalCommandsPath
            }
            vdn_inference_started = $false
            transformer_inference_started = $false
            transformer_claim_scope = "fixed-checkpoint same-input sensitivity only"
            under_pressure_412_scoring_owned_by_parent_chain = $true
        }
        audit = [ordered]@{
            selection_partition = "calibration"
            independent_validation_reports = 1
            time_polling = $false
            restricted_namespace_images_opened = 0
            dbnet_plus_plus_started = $false
        }
    }
    $FinalSummaryPath = Join-Path $OutputRoot "summary.json"
    $FinalSummary | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $FinalSummaryPath -Encoding utf8

    Send-ProgressEvent `
        -EventKey "garc-full-auto-public-v1-complete-v1" `
        -Message (
            "GARC 正式公开实验完成：selected=$SelectedRecognizer/" +
            "$SelectedConsensus/$SelectedGeometry；1080/50仅为固定seed20敏感性" +
            "（geometry未见168/8、训练重叠912/42），coverage=" +
            "$([math]::Round(100 * [double]$Range.coverage, 2))%，" +
            "pair-exact(full)=" +
            "$([math]::Round(100 * [double]$Range.pair_rounded_exact_full_denominator, 2))%；" +
            "正式all-component joint OOF=412/19，range coverage=" +
            "$([math]::Round(100 * [double]$JointRange.coverage, 2))%，" +
            "full-denominator NMAE=" +
            "$([math]::Round([double]$JointMetric.reading_nmae_full_denominator_failure_penalty_1, 5))；" +
            "DBNet++ gate triggered=$([bool]$DetectorGate.triggered)，但未启动 DBNet++。" +
            "VDN/Transformer 的同ROI同量程412交接包已冻结（外部推理尚未启动）。"
        ) `
        -Eta "主 GARC 结果已完成；后续 VDN/Transformer 同量程后端对比预计 1–3 小时。"
} catch {
    if ($RunRootCreated) {
        $Failure = [ordered]@{
            protocol = "garc_full_auto_public_event_chain_failure_v1"
            status = "failed"
            stage = $CurrentStage
            exception_type = $_.Exception.GetType().Name
            message = $_.Exception.Message
            existing_artifacts_preserved = $true
            dbnet_plus_plus_started = $false
            restricted_namespace_images_opened = 0
        }
        $Failure | ConvertTo-Json -Depth 5 |
            Set-Content -LiteralPath (
                Join-Path $OutputRoot "failure.json"
            ) -Encoding utf8
    }
    Send-ProgressEvent `
        -EventKey "garc-full-auto-public-v1-anomaly-$CurrentStage-v1" `
        -Message (
            "GARC 正式公开事件链异常停止：stage=$CurrentStage，" +
            "type=$($_.Exception.GetType().Name)。已保留全部已封存制品；" +
            "未启动 DBNet++，未读取现场数据。"
        ) `
        -Eta "预计 10–30 分钟核验对应 stage 日志、冻结制品与哈希后恢复。"
    throw
}

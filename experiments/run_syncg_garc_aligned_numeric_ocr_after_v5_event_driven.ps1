#requires -Version 7.0

[CmdletBinding()]
param(
    [ValidateRange(1, 2147483647)]
    [int]$WaitForV5ChainPid = 2348,

    [Parameter(Mandatory = $true)]
    [string]$WaitForV5ChainStartedAtUtc,

    [ValidateRange(1, 2147483647)]
    [int]$WaitForV5Pid = 24720,

    [Parameter(Mandatory = $true)]
    [string]$WaitForV5StartedAtUtc,

    [ValidateRange(1, 2147483647)]
    [int]$Seed = 20260817,

    [string]$Corpus =
        "C:\pointer_read\syncg_numeric_ocr_garc_aligned_v1",

    [string]$OutputDir =
        "C:\pointer_read\syncg_numeric_ocr_garc_aligned_runs",

    [string]$V5Summary =
        "C:\pointer_read\cagh_v5_enhanced_oof\summary.json",

    [string]$GateOutputRoot =
        "C:\pointer_read\cagh_v5_before_ocr_gate_v1",

    [switch]$Resume,

    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "The aligned OCR scheduler requires PowerShell 7 or newer."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PowerShell7 = "C:\Program Files\PowerShell\7\pwsh.exe"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainerWrapper = Join-Path $PSScriptRoot "run_syncg_numeric_ocr_training.ps1"
$CorpusBuilder = Join-Path $PSScriptRoot "build_garc_aligned_syncg_numeric_ocr_public.py"
$GateEvaluator = Join-Path $PSScriptRoot "evaluate_cagh_v5_before_ocr_gate.py"
$GateProtocol = Join-Path $PSScriptRoot "cagh_v5_before_ocr_gate_protocol.json"
$Reporter = Join-Path $PSScriptRoot "send_feishu_progress.ps1"
$CorpusSummary = Join-Path $Corpus "summary.json"
$CorpusSeal = Join-Path $Corpus "seal.json"
$ExpectedSummary = Join-Path (Join-Path $OutputDir "seed_$Seed") "summary.json"
$GateSummary = Join-Path $GateOutputRoot "summary.json"
$ExpectedAlignmentProtocol = "syncg_public_numeric_ocr_garc_aligned_v1"
$ExpectedOcrProtocol = "syncg_public_numeric_ocr_v1"
$ExpectedV5Protocol = "cagh_v5_enhanced_authoritative_pepd_oof_v1"
$ExpectedGateProtocol = "cagh_v5_before_ocr_public_gate_v1"
$GateFailNotified = $false
$Phase = "preflight"

function Assert-RequiredFile {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)
    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        throw "Required file is absent: $LiteralPath"
    }
}

function ConvertTo-ExactUtc {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not $Value.EndsWith("Z", [System.StringComparison]::Ordinal)) {
        throw "$Label must be an exact UTC timestamp with a Z suffix"
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
    $ObservedStartUtc = ([datetime]$Cim.CreationDate).ToUniversalTime()
    if (
        [math]::Abs(($ObservedStartUtc - $ExpectedStartUtc).TotalMilliseconds) -gt
        1.0
    ) {
        throw "$Label PID start-time identity mismatch"
    }
    $Bound = $null
    try {
        $Bound = [System.Diagnostics.Process]::GetProcessById($ProcessId)
        $null = $Bound.Handle
        if (
            [math]::Abs((
                $Bound.StartTime.ToUniversalTime() - $ObservedStartUtc
            ).TotalMilliseconds) -gt 1.0
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
        [Parameter(Mandatory = $true)][string]$Label
    )
    $Bound = Get-AuthenticatedProcess `
        -ProcessId $ProcessId `
        -ExpectedStartUtc $ExpectedStartUtc `
        -ExpectedCommandPattern $ExpectedCommandPattern `
        -Label $Label
    if ($null -eq $Bound) {
        if (-not (Test-Path -LiteralPath $V5Summary -PathType Leaf)) {
            throw "$Label PID is absent and no V5 completion artifact exists"
        }
        return
    }
    try {
        $Bound.WaitForExit()
    } finally {
        $Bound.Dispose()
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
            throw "progress reporter returned exit code $LASTEXITCODE"
        }
    } catch {
        Write-Warning "Feishu notification failed for stable event $EventKey."
    }
}

function Format-Metric {
    param([Parameter(Mandatory = $true)][double]$Value)
    return $Value.ToString(
        "0.000000",
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

foreach ($Required in @(
    $PowerShell7,
    $Python,
    $TrainerWrapper,
    $CorpusBuilder,
    $GateEvaluator,
    $GateProtocol,
    $Reporter,
    $CorpusSummary,
    $CorpusSeal
)) {
    Assert-RequiredFile -LiteralPath $Required
}

$ResolvedOutput = [System.IO.Path]::GetFullPath($OutputDir)
if (-not $ResolvedOutput.StartsWith(
    "C:\pointer_read\",
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Aligned OCR output escaped C:\pointer_read: $ResolvedOutput"
}
$ResolvedGateOutput = [System.IO.Path]::GetFullPath($GateOutputRoot)
if (-not $ResolvedGateOutput.StartsWith(
    "C:\pointer_read\",
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "V5 gate output escaped C:\pointer_read: $ResolvedGateOutput"
}
$ResolvedV5Summary = [System.IO.Path]::GetFullPath($V5Summary)
$ResolvedV5Root = [System.IO.Path]::GetFullPath(
    (Split-Path -Parent $ResolvedV5Summary)
)
$GateContract = Get-Content -LiteralPath $GateProtocol -Raw |
    ConvertFrom-Json
if (
    $GateContract.protocol -ne $ExpectedGateProtocol -or
    $GateContract.status -ne "frozen_public_only"
) {
    throw "V5-before-OCR gate protocol is not frozen"
}
$DeclaredV5Root = [System.IO.Path]::GetFullPath(
    [string]$GateContract.v5_artifacts.output_root
)
if (-not $ResolvedV5Root.Equals(
    $DeclaredV5Root,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "V5 summary root differs from the frozen gate protocol"
}
$ExpectedV5SummaryPath = [System.IO.Path]::GetFullPath(
    (Join-Path $DeclaredV5Root "summary.json")
)
if (-not $ResolvedV5Summary.Equals(
    $ExpectedV5SummaryPath,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "V5 completion summary is not the frozen formal summary"
}
$DeclaredGateOutput = [System.IO.Path]::GetFullPath(
    [string]$GateContract.output.default_root
)
if (-not $ResolvedGateOutput.Equals(
    $DeclaredGateOutput,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Gate output root differs from the frozen formal output root"
}

$V5ChainStartUtc = ConvertTo-ExactUtc `
    -Value $WaitForV5ChainStartedAtUtc -Label "WaitForV5ChainStartedAtUtc"
$V5StartUtc = ConvertTo-ExactUtc `
    -Value $WaitForV5StartedAtUtc -Label "WaitForV5StartedAtUtc"
$WrapperSource = $MyInvocation.MyCommand.Path
$WrapperSha256 = (
    Get-FileHash -LiteralPath $WrapperSource -Algorithm SHA256
).Hash.ToLowerInvariant()
$TrainerWrapperSha256 = (
    Get-FileHash -LiteralPath $TrainerWrapper -Algorithm SHA256
).Hash.ToLowerInvariant()
$CorpusBuilderSha256 = (
    Get-FileHash -LiteralPath $CorpusBuilder -Algorithm SHA256
).Hash.ToLowerInvariant()
$GateEvaluatorSha256 = (
    Get-FileHash -LiteralPath $GateEvaluator -Algorithm SHA256
).Hash.ToLowerInvariant()
$GateProtocolSha256 = (
    Get-FileHash -LiteralPath $GateProtocol -Algorithm SHA256
).Hash.ToLowerInvariant()
$CorpusSummarySha256 = (
    Get-FileHash -LiteralPath $CorpusSummary -Algorithm SHA256
).Hash.ToLowerInvariant()
$CorpusSealSha256 = (
    Get-FileHash -LiteralPath $CorpusSeal -Algorithm SHA256
).Hash.ToLowerInvariant()

$V5ChainBound = Get-AuthenticatedProcess `
    -ProcessId $WaitForV5ChainPid `
    -ExpectedStartUtc $V5ChainStartUtc `
    -ExpectedCommandPattern "chain_cagh_v5_oof_after_enhanced_screen\.ps1" `
    -Label "V5 OOF chain"
$V5Bound = Get-AuthenticatedProcess `
    -ProcessId $WaitForV5Pid `
    -ExpectedStartUtc $V5StartUtc `
    -ExpectedCommandPattern "run_cagh_v5_enhanced_oof\.py" `
    -Label "V5 OOF runner"

if ($PreflightOnly) {
    $Plan = [ordered]@{
        protocol = "syncg_garc_aligned_numeric_ocr_after_v5_preflight_v1"
        status = "validated_no_wait_no_training_no_notification"
        wait_for_v5_chain_pid = $WaitForV5ChainPid
        wait_for_v5_chain_started_at_utc = $V5ChainStartUtc.ToString("o")
        wait_for_v5_pid = $WaitForV5Pid
        wait_for_v5_started_at_utc = $V5StartUtc.ToString("o")
        authenticated_processes_present = [ordered]@{
            v5_chain = $null -ne $V5ChainBound
            v5_runner = $null -ne $V5Bound
        }
        wait_primitive = "retained System.Diagnostics.Process.WaitForExit"
        seed = $Seed
        corpus = $Corpus
        output_dir = $OutputDir
        expected_summary = $ExpectedSummary
        v5_summary = $ResolvedV5Summary
        gate_protocol = $GateProtocol
        gate_output_root = $GateOutputRoot
        gate_summary = $GateSummary
        wrapper_sha256 = $WrapperSha256
        trainer_wrapper_sha256 = $TrainerWrapperSha256
        corpus_builder_sha256 = $CorpusBuilderSha256
        gate_evaluator_sha256 = $GateEvaluatorSha256
        gate_protocol_sha256 = $GateProtocolSha256
        corpus_summary_sha256 = $CorpusSummarySha256
        corpus_seal_sha256 = $CorpusSealSha256
        gpu_work_started = $false
        feishu_message_sent = $false
        public_images_opened = 0
        outer_labels_opened = 0
        restricted_namespace_images_opened = 0
    }
    foreach ($Bound in @($V5ChainBound, $V5Bound)) {
        if ($null -ne $Bound) {
            $Bound.Dispose()
        }
    }
    Write-Output ($Plan | ConvertTo-Json -Compress)
    exit 0
}

foreach ($Bound in @($V5ChainBound, $V5Bound)) {
    if ($null -ne $Bound) {
        $Bound.Dispose()
    }
}

try {
    $Phase = "v5_wait"
    Wait-ForAuthenticatedProcess `
        -ProcessId $WaitForV5ChainPid `
        -ExpectedStartUtc $V5ChainStartUtc `
        -ExpectedCommandPattern "chain_cagh_v5_oof_after_enhanced_screen\.ps1" `
        -Label "V5 OOF chain"
    Wait-ForAuthenticatedProcess `
        -ProcessId $WaitForV5Pid `
        -ExpectedStartUtc $V5StartUtc `
        -ExpectedCommandPattern "run_cagh_v5_enhanced_oof\.py" `
        -Label "V5 OOF runner"

    Assert-RequiredFile -LiteralPath $V5Summary
    $V5 = Get-Content -LiteralPath $V5Summary -Raw | ConvertFrom-Json
    $StrictOof = $V5.strict_oof
    if (
        $V5.status -ne "complete" -or
        $V5.protocol -ne $ExpectedV5Protocol -or
        $null -eq $StrictOof -or
        $StrictOof.status -ne "complete" -or
        $StrictOof.protocol -ne $ExpectedV5Protocol -or
        @($StrictOof.folds).Count -ne 3 -or
        [int]$StrictOof.field_samples_read -ne 0 -or
        [int]$StrictOof.public_test_samples_read -ne 0
    ) {
        throw "V5 OOF completion artifact failed protocol checks"
    }
    $StrictSummaryPath = [System.IO.Path]::GetFullPath(
        [string]$V5.artifacts.strict_oof_summary
    )
    $ExpectedStrictSummaryPath = [System.IO.Path]::GetFullPath(
        (Join-Path $DeclaredV5Root "strict_oof_summary.json")
    )
    if (-not $StrictSummaryPath.Equals(
        $ExpectedStrictSummaryPath,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "V5 strict-OOF summary escaped the frozen formal root"
    }
    Assert-RequiredFile -LiteralPath $StrictSummaryPath
    if (
        (Get-FileHash -LiteralPath $StrictSummaryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
        ([string]$V5.artifacts.strict_oof_summary_sha256).ToLowerInvariant()
    ) {
        throw "V5 strict-OOF summary hash differs from the aggregate completion artifact"
    }
    $ExpectedFoldSeeds = @(
        $GateContract.folds | ForEach-Object { [int]$_.pepd_seed }
    )
    $ObservedFoldSeeds = @(
        $StrictOof.folds | ForEach-Object { [int]$_.pepd_seed }
    )
    if (
        @($ObservedFoldSeeds | Select-Object -Unique).Count -ne 3 -or
        @(Compare-Object $ExpectedFoldSeeds $ObservedFoldSeeds).Count -ne 0
    ) {
        throw "V5 aggregate fold seeds differ from the frozen gate folds"
    }
    foreach ($Fold in @($StrictOof.folds)) {
        $FoldSeed = [int]$Fold.pepd_seed
        $ExpectedFoldPath = [System.IO.Path]::GetFullPath(
            (Join-Path $DeclaredV5Root "folds\pepd_seed_$FoldSeed\summary.json")
        )
        $ObservedFoldPath = [System.IO.Path]::GetFullPath([string]$Fold.summary)
        if (-not $ObservedFoldPath.Equals(
            $ExpectedFoldPath,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "V5 fold $FoldSeed summary path differs from the formal root"
        }
        Assert-RequiredFile -LiteralPath $ObservedFoldPath
        if (
            (Get-FileHash -LiteralPath $ObservedFoldPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            ([string]$Fold.summary_sha256).ToLowerInvariant()
        ) {
            throw "V5 fold $FoldSeed summary hash differs from the aggregate completion artifact"
        }
    }
    $V5SummarySha256 = (
        Get-FileHash -LiteralPath $ResolvedV5Summary -Algorithm SHA256
    ).Hash.ToLowerInvariant()

    if (
        (Get-FileHash -LiteralPath $WrapperSource -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $WrapperSha256 -or
        (Get-FileHash -LiteralPath $TrainerWrapper -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $TrainerWrapperSha256 -or
        (Get-FileHash -LiteralPath $CorpusBuilder -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $CorpusBuilderSha256 -or
        (Get-FileHash -LiteralPath $GateEvaluator -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $GateEvaluatorSha256 -or
        (Get-FileHash -LiteralPath $GateProtocol -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $GateProtocolSha256 -or
        (Get-FileHash -LiteralPath $CorpusSummary -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $CorpusSummarySha256 -or
        (Get-FileHash -LiteralPath $CorpusSeal -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            $CorpusSealSha256
    ) {
        throw "Aligned OCR scheduler source or frozen corpus changed while waiting"
    }

    $Phase = "gate_evaluation"
    $GateJson = (& $Python $GateEvaluator `
        --protocol $GateProtocol `
        --v5-output-root $ResolvedV5Root `
        --output-root $GateOutputRoot 2>&1) -join "`n"
    $GateExitCode = $LASTEXITCODE
    if ($GateExitCode -notin @(0, 2)) {
        throw "V5-before-OCR evaluator failed with exit code $GateExitCode"
    }
    Assert-RequiredFile -LiteralPath $GateSummary
    $Gate = Get-Content -LiteralPath $GateSummary -Raw | ConvertFrom-Json
    $GateAggregate = $Gate.runtime_inputs.aggregate_chain
    if (
        $Gate.status -ne "complete" -or
        $Gate.protocol -ne $ExpectedGateProtocol -or
        $Gate.protocol_file.sha256 -ne $GateProtocolSha256 -or
        $Gate.frozen_inputs.evaluator_source.sha256 -ne $GateEvaluatorSha256 -or
        -not ([string]$Gate.runtime_inputs.v5_output_root).Equals(
            $DeclaredV5Root,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        -not ([string]$GateAggregate.aggregate_summary).Equals(
            $ResolvedV5Summary,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        $GateAggregate.aggregate_summary_sha256 -ne $V5SummarySha256 -or
        [int]$Gate.data_access_audit.images_read -ne 0 -or
        [int]$Gate.data_access_audit.annotations_read -ne 0 -or
        [int]$Gate.data_access_audit.public_test_samples_read -ne 0 -or
        [int]$Gate.data_access_audit.field_samples_read -ne 0 -or
        [int]$Gate.data_access_audit.sealed_samples_read -ne 0 -or
        [int]$Gate.data_access_audit.confirmatory_samples_read -ne 0
    ) {
        throw "V5-before-OCR result failed protocol or data-access checks"
    }
    $FailedChecks = @($Gate.checks | Where-Object { -not [bool]$_.passed })
    if ($Gate.decision -eq "pass") {
        if ($GateExitCode -ne 0 -or $FailedChecks.Count -ne 0) {
            throw "V5 gate pass conflicts with evaluator exit code or checks"
        }
    } elseif ($Gate.decision -eq "fail") {
        if ($GateExitCode -ne 2 -or $FailedChecks.Count -eq 0) {
            throw "V5 gate fail conflicts with evaluator exit code or checks"
        }
        $GateFailNotified = $true
        Send-ProgressEvent `
            -EventKey "cagh-v5-before-ocr-gate-fail-v1" `
            -Message (
                "V5 公共三折门控未通过，TinyOCR 未启动：" +
                "同队列 V5 NMAE=$(Format-Metric ([double]$Gate.same_cohort_comparison.v5_nmae))，" +
                "相对旧 V4 改善=$(Format-Metric ([double]$Gate.same_cohort_comparison.relative_nmae_improvement))，" +
                "未通过检查=$($FailedChecks.Count)。未读取现场或冻结测试数据。"
            ) `
            -Eta "OCR 保持闭锁；预计 20–40 分钟核查仅公开数据的失败项后决定 V5 调整。"
        throw "V5-before-OCR gate decision is fail; OCR launch is forbidden"
    } else {
        throw "V5-before-OCR decision is neither pass nor fail"
    }

    $Phase = "ocr_prelaunch"
    $VerifyJson = (& $Python -m `
        experiments.build_garc_aligned_syncg_numeric_ocr_public `
        verify --corpus $Corpus 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Aligned OCR corpus verification failed: $VerifyJson"
    }
    $Verify = $VerifyJson | ConvertFrom-Json
    if (
        $Verify.status -ne "verified" -or
        -not [bool]$Verify.algorithm_fit_exact_coverage -or
        -not [bool]$Verify.outer_group_overlap_zero -or
        -not [bool]$Verify.outer_sample_overlap_zero -or
        [int]$Verify.samples -ne 12176 -or
        [int]$Verify.groups -ne 551
    ) {
        throw "Aligned OCR corpus verification did not prove exact fit-only coverage"
    }
    $CorpusRecord = Get-Content -LiteralPath $CorpusSummary -Raw | ConvertFrom-Json
    if (
        $CorpusRecord.protocol -ne $ExpectedOcrProtocol -or
        $CorpusRecord.alignment_protocol -ne $ExpectedAlignmentProtocol -or
        -not [bool]$CorpusRecord.alignment_audit.all_outer_group_overlap_zero -or
        -not [bool]$CorpusRecord.alignment_audit.all_outer_sample_overlap_zero
    ) {
        throw "Aligned OCR corpus identity drift"
    }

    Send-ProgressEvent `
        -EventKey "cagh-v5-gate-pass-to-garc-aligned-tiny-seed-$Seed-v1" `
        -Message (
            "V5 公共三折门控已通过并切换到修正划分的 TinyOCR：" +
            "同队列 V5 NMAE=$(Format-Metric ([double]$Gate.same_cohort_comparison.v5_nmae))，" +
            "相对旧 V4 改善=$(Format-Metric ([double]$Gate.same_cohort_comparison.relative_nmae_improvement))；" +
            "训练语料仅含公开 algorithm-fit 12176张/551组，" +
            "与外层 calibration、development、independent-validation 样本和实体组均零重叠；" +
            "seed=$Seed，未读取现场数据。"
        ) `
        -Eta "预计 5–10 小时完成 TinyOCR；随后按冻结门槛决定 Strong OCR，再进入 GARC。"

    $Arguments = @(
        "-NoLogo",
        "-NoProfile",
        "-File", $TrainerWrapper,
        "-Seed", "$Seed",
        "-Workers", "2",
        "-Corpus", $Corpus,
        "-OutputDir", $OutputDir
    )
    if ($Resume) {
        $Arguments += "-Resume"
    }
    $Phase = "ocr_training"
    & $PowerShell7 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Aligned TinyOCR training wrapper exited with code $LASTEXITCODE"
    }
    Assert-RequiredFile -LiteralPath $ExpectedSummary
} catch {
    if (-not $GateFailNotified) {
        $AnomalyEvent = if ($Phase -eq "ocr_training") {
            "garc-aligned-tiny-ocr-seed-$Seed-anomaly-v1"
        } elseif ($Phase -eq "ocr_prelaunch") {
            "garc-aligned-tiny-ocr-prelaunch-anomaly-v1"
        } else {
            "cagh-v5-before-ocr-gate-anomaly-v1"
        }
        $AnomalyMessage = if ($Phase -eq "ocr_training") {
            "修正划分的 TinyOCR 独立训练异常；已停止本事件链，未读取现场或冻结测试数据。"
        } elseif ($Phase -eq "ocr_prelaunch") {
            "V5 门控通过后的 TinyOCR 启动前校验异常；OCR 未启动，未读取现场或冻结测试数据。"
        } else {
            "V5 等待、完成身份或公共性能门控异常；TinyOCR 未启动，未读取现场或冻结测试数据。"
        }
        Send-ProgressEvent `
            -EventKey $AnomalyEvent `
            -Message $AnomalyMessage `
            -Eta "预计 10–30 分钟完成身份、协议、清单、锁或日志核对后恢复。"
    }
    throw
}

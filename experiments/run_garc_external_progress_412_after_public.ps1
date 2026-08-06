[CmdletBinding()]
param(
    [string]$ProjectRoot = "D:\project\PointerMeterReaderFastAPI",
    [string]$Python = "D:\project\PointerMeterReaderFastAPI\.venv\Scripts\python.exe",
    [string]$Protocol = "D:\project\PointerMeterReaderFastAPI\experiments\garc_external_progress_412_protocol.json",
    [string]$GarcSummary,
    [int]$GarcProcessId = 0,
    [string]$ExpectedGarcProcessStartUtc,
    [string]$ExpectedGarcSummarySha256,
    [string]$OutputRoot = "C:\pointer_read\garc_external_progress_412_formal_v1",
    [bool]$RunTransformerSensitivity = $true,
    [switch]$PreflightOnly,
    [switch]$StartFormal
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Evaluator = Join-Path $ProjectRoot "experiments\garc_external_progress_412.py"
$Feishu = Join-Path $ProjectRoot "experiments\send_feishu_progress.ps1"
$Preflight = Join-Path $OutputRoot "preflight.json"
$WrapperPreflight = Join-Path $OutputRoot "wrapper_preflight.json"
$VdnOutput = Join-Path $OutputRoot "vdn_predictions"
$TransformerOutput = Join-Path $OutputRoot "transformer_sensitivity_predictions"
$ScoreOutput = Join-Path $OutputRoot "comparison.json"
$ExpectedGarcCommandPattern =
    "run_garc_full_auto_public_event_driven\.ps1"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Invoke-CheckedPython([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Send-Event([string]$EventKey, [string]$Message, [string]$Eta) {
    & "C:\Program Files\PowerShell\7\pwsh.exe" -NoLogo -NoProfile -File $Feishu `
        -EventKey $EventKey -Message $Message -Eta $Eta | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Feishu progress notification failed"
    }
}

function Resolve-Handoff([object]$Summary) {
    $Value = $Summary.external_comparison_handoff
    if ($null -eq $Value) {
        throw "GARC summary has no external_comparison_handoff"
    }
    if ($Value -is [string]) {
        return [string]$Value
    }
    if (
        $Value.PSObject.Properties.Name -contains "handoff" -and
        $null -ne $Value.handoff -and
        $Value.handoff.PSObject.Properties.Name -contains "root"
    ) {
        $Candidate = [string]$Value.handoff.root
        if (-not [string]::IsNullOrWhiteSpace($Candidate)) {
            return $Candidate
        }
    }
    foreach ($Name in @("path", "handoff_path", "vdn_transformer")) {
        if ($Value.PSObject.Properties.Name -contains $Name) {
            $Candidate = [string]$Value.$Name
            if (-not [string]::IsNullOrWhiteSpace($Candidate)) {
                return $Candidate
            }
        }
    }
    throw "GARC summary external handoff path is absent"
}

function Wait-ForAuthenticatedGarcProcess(
    [int]$ProcessId,
    [string]$ExpectedStartUtc
) {
    if ($ProcessId -le 0) {
        return
    }
    if ([string]::IsNullOrWhiteSpace($ExpectedStartUtc)) {
        throw "Live GARC PID requires -ExpectedGarcProcessStartUtc"
    }

    $Cim = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $Cim) {
        throw "Live GARC PID is absent before authentication"
    }
    if ([string]$Cim.CommandLine -notmatch $ExpectedGarcCommandPattern) {
        throw "GARC PID belongs to an unexpected process"
    }

    $ExpectedStart = [datetime]::Parse(
        $ExpectedStartUtc,
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::AssumeUniversal
    ).ToUniversalTime()
    $CimStart = ([datetime]$Cim.CreationDate).ToUniversalTime()
    if ([math]::Abs(($CimStart - $ExpectedStart).TotalSeconds) -gt 1.0) {
        throw "GARC PID start-time identity mismatch"
    }

    try {
        $Process = [System.Diagnostics.Process]::GetProcessById($ProcessId)
    } catch [System.ArgumentException] {
        # Completion may race the transition from CIM authentication to the
        # kernel process handle. The summary checks below remain authoritative.
        return
    }
    if (
        [math]::Abs(
            ($Process.StartTime.ToUniversalTime() - $CimStart).TotalSeconds
        ) -gt 1.0
    ) {
        throw "GARC PID CIM/process start identity drift"
    }
    # Wait on the exact authenticated kernel process object. Looking the PID up
    # again here would allow a completed PID to be reused between check and wait.
    $Process.WaitForExit()
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python runtime is absent: $Python"
}
if (-not (Test-Path -LiteralPath $Evaluator -PathType Leaf)) {
    throw "External evaluator is absent: $Evaluator"
}
if (-not (Test-Path -LiteralPath $Protocol -PathType Leaf)) {
    throw "External protocol is absent: $Protocol"
}

if ($PreflightOnly) {
    if (-not (Test-Path -LiteralPath $OutputRoot)) {
        New-Item -ItemType Directory -Path $OutputRoot | Out-Null
    }
    if (Test-Path -LiteralPath $Preflight -PathType Leaf) {
        Invoke-CheckedPython @(
            $Evaluator, "--protocol", $Protocol, "preflight",
            "--output", $Preflight, "--verify-only"
        )
    } else {
        Invoke-CheckedPython @(
            $Evaluator, "--protocol", $Protocol, "preflight",
            "--output", $Preflight
        )
    }
    $Payload = [ordered]@{
        schema_version = 1
        protocol = "garc_external_progress_412_event_wrapper_preflight_v1"
        status = "validated_no_wait_no_inference_no_notification"
        evaluator_sha256 = Get-Sha256 $Evaluator
        protocol_sha256 = Get-Sha256 $Protocol
        preflight_sha256 = Get-Sha256 $Preflight
        process_wait_started = $false
        external_inference_started = $false
        feishu_notification_sent = $false
        restricted_namespace_images_opened = 0
    }
    $Payload | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $WrapperPreflight -Encoding utf8
    Write-Output $WrapperPreflight
    exit 0
}

if (-not $StartFormal) {
    throw "Formal execution requires -StartFormal"
}
if ([string]::IsNullOrWhiteSpace($GarcSummary)) {
    throw "-GarcSummary is required"
}
if ($GarcProcessId -le 0 -and [string]::IsNullOrWhiteSpace($ExpectedGarcSummarySha256)) {
    throw "Bind either a live GARC process identity or an expected summary SHA256"
}

$CurrentStage = "wait-garc"
try {
    if ($GarcProcessId -gt 0) {
        Wait-ForAuthenticatedGarcProcess `
            -ProcessId $GarcProcessId `
            -ExpectedStartUtc $ExpectedGarcProcessStartUtc
    }

    $CurrentStage = "authenticate-garc-summary"
    if (-not (Test-Path -LiteralPath $GarcSummary -PathType Leaf)) {
        throw "GARC completion summary is absent"
    }
    $GarcHash = Get-Sha256 $GarcSummary
    if (
        -not [string]::IsNullOrWhiteSpace($ExpectedGarcSummarySha256) -and
        $GarcHash -ne $ExpectedGarcSummarySha256.ToLowerInvariant()
    ) {
        throw "GARC completion summary hash mismatch"
    }
    $Garc = Get-Content -LiteralPath $GarcSummary -Raw | ConvertFrom-Json
    if ($Garc.status -ne "complete") {
        throw "GARC completion summary status is not complete"
    }
    $Handoff = Resolve-Handoff $Garc
    if (-not (Test-Path -LiteralPath (Join-Path $Handoff "seal.json") -PathType Leaf)) {
        throw "GARC external handoff is not sealed"
    }
    if (-not (Test-Path -LiteralPath $Preflight -PathType Leaf)) {
        throw "External preflight is absent: $Preflight"
    }

    $CurrentStage = "vdn-infer"
    Invoke-CheckedPython @(
        $Evaluator, "--protocol", $Protocol, "infer",
        "--preflight", $Preflight, "--handoff-root", $Handoff,
        "--method", "vdn_official200", "--output-root", $VdnOutput,
        "--device", "cuda:0"
    )

    $External = @("vdn_official200=$VdnOutput")
    if ($RunTransformerSensitivity) {
        $CurrentStage = "transformer-sensitivity-infer"
        Invoke-CheckedPython @(
            $Evaluator, "--protocol", $Protocol, "infer",
            "--preflight", $Preflight, "--handoff-root", $Handoff,
            "--method", "original_transformer", "--output-root", $TransformerOutput,
            "--device", "cuda:0"
        )
        $External += "original_transformer=$TransformerOutput"
    }

    $CurrentStage = "score"
    $ScoreArguments = @(
        $Evaluator, "--protocol", $Protocol, "score",
        "--preflight", $Preflight, "--handoff-root", $Handoff,
        "--output", $ScoreOutput
    )
    foreach ($Binding in $External) {
        $ScoreArguments += @("--external", $Binding)
    }
    Invoke-CheckedPython $ScoreArguments
    $Score = Get-Content -LiteralPath $ScoreOutput -Raw | ConvertFrom-Json
    if (
        $Score.status -ne "complete" -or
        $Score.claim_eligibility.original_transformer_strict_oof_claim -ne $false
    ) {
        throw "External score claim boundary failed"
    }
    $Vdn = $Score.metrics.vdn_official200
    Send-Event `
        -EventKey "garc-vdn-external-412-complete-v1" `
        -Message (
            "GARC 同412/19外部对照完成：VDN strict-OOF full-denominator NMAE=" +
            "$([math]::Round([double]$Vdn.full_denominator_nmae_failure_penalty_1, 5))，" +
            "coverage=$([math]::Round(100 * [double]$Vdn.coverage, 2))%；" +
            "Original Transformer仅标记为paper_strict=false的固定权重敏感性。"
        ) `
        -Eta "外部412对照已完成；预计30–60分钟合并论文主表、统计区间与限制说明。"
} catch {
    Send-Event `
        -EventKey "garc-vdn-external-412-anomaly-v1" `
        -Message (
            "GARC 412外部对照异常停止：stage=$CurrentStage，" +
            "type=$($_.Exception.GetType().Name)。现有密封制品已保留。"
        ) `
        -Eta "预计15–45分钟核验进程身份、summary/handoff/prediction seal后恢复。"
    throw
}

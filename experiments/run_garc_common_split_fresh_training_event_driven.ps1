#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = (Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"),
    [string]$Protocol = (Join-Path (Split-Path -Parent $PSScriptRoot) "experiments\garc_common_split_progress_protocol.json"),
    [string]$ExecutionSpec = (Join-Path (Split-Path -Parent $PSScriptRoot) "experiments\garc_common_split_fresh_execution_spec.json"),
    [string]$PromotionRoot = "C:\pointer_read\garc_common_split_promotion_v1",
    [string]$OutputRoot = "C:\pointer_read\garc_common_split_fresh_training_v1",
    [string]$PreflightOutput = "C:\pointer_read\garc_common_split_fresh_training_launcher\preflight.json",
    [ValidateRange(0, 2147483647)]
    [int]$WaitForPromotionPid = 0,
    [string]$ExpectedPromotionStartUtc = "",
    [string]$ExpectedPromotionCommandLineSha256 = "",
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Trainer = Join-Path $ProjectRoot "experiments\garc_common_split_fresh_training.py"
$Reporter = Join-Path $ProjectRoot "experiments\send_feishu_progress.ps1"
$Authorization = Join-Path $PromotionRoot "training_authorization.json"
$PromotionSeal = Join-Path $PromotionRoot "seal.json"
$RunManifest = Join-Path $OutputRoot "run_manifest.json"
$ValidatedOutput = Join-Path $OutputRoot "validate_only.json"
$ExpectedPromotionPattern = "run_garc_common_split_promotion_event_driven\.ps1"
$Seeds = @(20260816, 20260817, 20260818)

function Get-StringSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)
    return [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData(
            [Text.Encoding]::UTF8.GetBytes($Value)
        )
    ).ToLowerInvariant()
}

function Assert-OutputRoot {
    $Resolved = [IO.Path]::GetFullPath($OutputRoot)
    if (-not $Resolved.StartsWith(
        "C:\pointer_read\", [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Fresh-training output escaped C:\pointer_read."
    }
    if ($Resolved.TrimEnd('\') -ieq "C:\pointer_read") {
        throw "Refusing broad C:\pointer_read output."
    }
}

function Wait-ForAuthenticatedPromotion {
    if ($WaitForPromotionPid -le 0) {
        throw "Promotion artifacts are absent and no authenticated wait PID was supplied."
    }
    if (
        [string]::IsNullOrWhiteSpace($ExpectedPromotionStartUtc) -or
        [string]::IsNullOrWhiteSpace($ExpectedPromotionCommandLineSha256)
    ) {
        throw "Promotion start time and command-line SHA-256 are required."
    }
    if (-not $ExpectedPromotionStartUtc.EndsWith("Z", [StringComparison]::Ordinal)) {
        throw "ExpectedPromotionStartUtc must be UTC with a Z suffix."
    }
    $ExpectedStart = [DateTimeOffset]::ParseExact(
        $ExpectedPromotionStartUtc,
        "o",
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::RoundtripKind
    ).UtcDateTime
    $Cim = Get-CimInstance Win32_Process -Filter (
        "ProcessId = $WaitForPromotionPid"
    )
    if ($null -eq $Cim) {
        throw "Promotion wait process $WaitForPromotionPid does not exist."
    }
    $CommandLine = [string]$Cim.CommandLine
    if ($CommandLine -notmatch $ExpectedPromotionPattern) {
        throw "PID $WaitForPromotionPid is not the promotion event chain."
    }
    if ((Get-StringSha256 $CommandLine) -ne $ExpectedPromotionCommandLineSha256) {
        throw "Promotion process command-line identity drift."
    }
    $CimStart = ([datetime]$Cim.CreationDate).ToUniversalTime()
    if ([Math]::Abs(($CimStart - $ExpectedStart).TotalSeconds) -gt 1.0) {
        throw "Promotion process start-time identity drift."
    }
    $Process = [Diagnostics.Process]::GetProcessById($WaitForPromotionPid)
    if ([Math]::Abs(($Process.StartTime.ToUniversalTime() - $ExpectedStart).TotalSeconds) -gt 1.0) {
        throw "Native promotion process identity drift."
    }
    # Native process waiting is event-driven; no timer or polling loop exists.
    $Process.WaitForExit()
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE."
    }
}

function Send-ProgressEvent {
    param(
        [Parameter(Mandatory = $true)][string]$EventKey,
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$Eta = ""
    )
    try {
        & $Reporter -EventKey $EventKey -Message $Message -Eta $Eta | Out-Host
    } catch {
        Write-Warning "Feishu notification failed after scientific state was saved: $($_.Exception.Message)"
    }
}

foreach ($Required in @($Python, $Protocol, $Trainer)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required fresh-training dependency is absent: $Required"
    }
}
Assert-OutputRoot

if ($PreflightOnly) {
    Invoke-CheckedPython -Stage "fresh-training static preflight" -Arguments @(
        "-m", "experiments.garc_common_split_fresh_training", "preflight",
        "--protocol", $Protocol,
        "--output", $PreflightOutput
    )
    return
}

if (-not (Test-Path -LiteralPath $ExecutionSpec -PathType Leaf)) {
    throw (
        "Frozen execution spec is absent. The common protocol does not pin " +
        "PEPD/V5 schedules or the progress-fusion grid; training remains blocked: " +
        $ExecutionSpec
    )
}

if (
    -not (Test-Path -LiteralPath $Authorization -PathType Leaf) -or
    -not (Test-Path -LiteralPath $PromotionSeal -PathType Leaf)
) {
    Wait-ForAuthenticatedPromotion
}
if (
    -not (Test-Path -LiteralPath $Authorization -PathType Leaf) -or
    -not (Test-Path -LiteralPath $PromotionSeal -PathType Leaf)
) {
    throw "Authenticated promotion exited without authorization and seal artifacts."
}

try {
    Invoke-CheckedPython -Stage "authenticated fresh-training preflight" -Arguments @(
        "-m", "experiments.garc_common_split_fresh_training", "preflight",
        "--protocol", $Protocol,
        "--execution-spec", $ExecutionSpec,
        "--authorization", $Authorization,
        "--promotion-seal", $PromotionSeal,
        "--output", $PreflightOutput
    )
    $Preflight = Get-Content -LiteralPath $PreflightOutput -Raw | ConvertFrom-Json
    if (-not [bool]$Preflight.training_allowed) {
        throw "Authenticated fresh-training preflight did not authorize execution."
    }
    Send-ProgressEvent `
        -EventKey "garc-common-split-fresh-training-phase-start-v1" `
        -Message (
            "GARC 已由通过的412/19 pilot gate切换到 common-split fresh training：" +
            "将严格按551组algorithm_fit训练、100组calibration选择，" +
            "三个新种子均禁止复用旧PEPD/V5权重；1080/50验证仍未读取。"
        ) `
        -Eta "预计13.5–24小时完成三种子训练、matched enhanced-V5与calibration冻结。"

    foreach ($Seed in $Seeds) {
        $env:PYTHONHASHSEED = [string]$Seed
        $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
        Invoke-CheckedPython -Stage "fresh common-split seed $Seed" -Arguments @(
            "-m", "experiments.garc_common_split_fresh_training", "train-seed",
            "--protocol", $Protocol,
            "--execution-spec", $ExecutionSpec,
            "--authorization", $Authorization,
            "--promotion-seal", $PromotionSeal,
            "--output-root", $OutputRoot,
            "--seed", [string]$Seed,
            "--device", "cuda:0",
            "--resume"
        )
        $Summary = Join-Path $OutputRoot "seeds\seed_$Seed\summary.json"
        if (-not (Test-Path -LiteralPath $Summary -PathType Leaf)) {
            throw "Seed $Seed exited without an authenticated summary."
        }
        $Result = Get-Content -LiteralPath $Summary -Raw | ConvertFrom-Json
        Send-ProgressEvent `
            -EventKey "garc-common-split-fresh-seed-$Seed-complete-v1" `
            -Message (
                "Common-split fresh seed $Seed 已完成：" +
                "PEPD selected epoch=$($Result.pepd.selected_epoch)，" +
                "enhanced-V5 selected geometry epoch=$($Result.v5_enhanced.selected_stage_epoch)，" +
                "calibration NMAE=$([double]$Result.progress_fusion.selected.nmae_failure_penalty_1)。"
            ) `
            -Eta "其余种子将按冻结顺序自动继续；全部完成后生成等权ensemble并验证清单。"
    }

    Invoke-CheckedPython -Stage "fresh common-split finalize" -Arguments @(
        "-m", "experiments.garc_common_split_fresh_training", "finalize",
        "--protocol", $Protocol,
        "--execution-spec", $ExecutionSpec,
        "--authorization", $Authorization,
        "--promotion-seal", $PromotionSeal,
        "--output-root", $OutputRoot
    )
    Invoke-CheckedPython -Stage "fresh common-split validate-only" -Arguments @(
        "-m", "experiments.garc_common_split_progress", "validate-only",
        "--protocol", $Protocol,
        "--run-manifest", $RunManifest,
        "--output", $ValidatedOutput
    )
    $Validated = Get-Content -LiteralPath $ValidatedOutput -Raw | ConvertFrom-Json
    Send-ProgressEvent `
        -EventKey "garc-common-split-fresh-three-seed-complete-v1" `
        -Message (
            "GARC common-split fresh三种子训练、matched enhanced-V5、" +
            "progress-fusion calibration与等权ensemble已全部冻结并通过清单验证；" +
            "状态=$($Validated.status)，1080/50一次性验证仍未读取。"
        ) `
        -Eta "下一阶段预计1.5–3小时完成一次性1080/50推理、封存和正式评分。"
} catch {
    Send-ProgressEvent `
        -EventKey "garc-common-split-fresh-training-anomaly-v1" `
        -Message ("GARC common-split fresh训练链异常：" + $_.Exception.Message) `
        -Eta "已保留原子epoch journal；修复后可从完整epoch恢复，不需要从头训练。"
    throw
}

#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ProjectRoot = "D:\project\PointerMeterReaderFastAPI",
    [string]$Python = "D:\project\PointerMeterReaderFastAPI\.venv\Scripts\python.exe",
    [ValidateRange(0, 2147483647)]
    [int]$WaitForPid = 0,
    [string]$WaitForStartedAtUtc = "",
    [string]$WaitForCommandFragment =
        "experiments\run_syncg_meter_detector_after_paper_event_driven.ps1",
    [string]$GarcSummary =
        "C:\pointer_read\garc_full_auto_formal_v1\summary.json",
    [string]$PaperSummary =
        "C:\pointer_read\paper_final_results_v2\summary.json",
    [string]$PaperSeal =
        "C:\pointer_read\paper_final_results_v2\seal.json",
    [string]$FrontendPlan =
        "C:\pointer_read\syncg_meter_detector_frontend_v1\frontend_plan.json",
    [string]$BuilderOutputRoot =
        "C:\pointer_read\blind_bundle_materialization_v1",
    [string]$MaterializedOutputRoot =
        "C:\pointer_read\blind_bundle_materialization_v1\frozen",
    [switch]$PreflightOnly,
    [switch]$StartFormal
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProgressPreference = "SilentlyContinue"
$PSNativeCommandUseErrorActionPreference = $false
$env:PYTHONDONTWRITEBYTECODE = "1"

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "PowerShell 7 or newer is required."
}
if ($PreflightOnly -eq $StartFormal) {
    throw "Choose exactly one of -PreflightOnly or -StartFormal."
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$Python = [System.IO.Path]::GetFullPath($Python)
$Builder = Join-Path $ProjectRoot `
    "experiments\build_field_blind_bundle_inputs.py"
$Materializer = Join-Path $ProjectRoot `
    "experiments\materialize_field_blind_bundles.py"
$FrontendVerifier = Join-Path $ProjectRoot `
    "experiments\syncg_meter_detector_frontend.py"
$RuntimeFactory = Join-Path $ProjectRoot `
    "experiments\field_blind_runtime_factory.py"
$Reporter = Join-Path $ProjectRoot "experiments\send_feishu_progress.ps1"
$DetectorWrapper = Join-Path $ProjectRoot `
    "experiments\run_syncg_meter_detector_after_paper_event_driven.ps1"
$Spec = Join-Path $BuilderOutputRoot "spec.json"
$CurrentStage = "initialize"

foreach ($Required in @(
    $Python,
    $Builder,
    $Materializer,
    $FrontendVerifier,
    $RuntimeFactory,
    $Reporter,
    $DetectorWrapper
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required event-chain artifact is absent: $Required"
    }
}

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

function Assert-PointerReadOutput([string]$Value, [string]$Label) {
    $Root = [System.IO.Path]::GetFullPath("C:\pointer_read")
    $Full = [System.IO.Path]::GetFullPath($Value)
    $Prefix = $Root.TrimEnd("\") + "\"
    if (
        -not $Full.Equals($Root, [System.StringComparison]::OrdinalIgnoreCase) -and
        -not $Full.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "$Label must stay under C:\pointer_read: $Full"
    }
    $Tokens = @(
        $Full.ToLowerInvariant() -split '[\\/_ .-]+' |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $Forbidden = @(
        "field",
        "test",
        "sealed",
        "confirmatory",
        "confirmation",
        "xiangmu1",
        "xiangmu2"
    )
    $Observed = @($Tokens | Where-Object { $_ -in $Forbidden })
    if ($Observed.Count -gt 0) {
        throw "$Label enters a restricted namespace: $($Observed -join ',')"
    }
    return $Full
}

function Get-AuthenticatedDetectorProcess {
    param([switch]$RequirePresent)

    if ($WaitForPid -eq 0) {
        if ($RequirePresent) {
            throw "-WaitForPid must be supplied for the formal event chain"
        }
        return $null
    }
    if ([string]::IsNullOrWhiteSpace($WaitForStartedAtUtc)) {
        throw "-WaitForStartedAtUtc is required whenever -WaitForPid is nonzero"
    }
    if ([string]::IsNullOrWhiteSpace($WaitForCommandFragment)) {
        throw "-WaitForCommandFragment must identify the authenticated detector wrapper"
    }
    $Cim = Get-CimInstance Win32_Process `
        -Filter "ProcessId = $WaitForPid"
    if ($null -eq $Cim) {
        if ($RequirePresent) {
            throw "Authenticated public detector PID is no longer present"
        }
        return $null
    }
    $CommandLine = [string]$Cim.CommandLine
    if (
        [string]::IsNullOrWhiteSpace($CommandLine) -or
        $CommandLine.IndexOf(
            $WaitForCommandFragment,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -lt 0
    ) {
        throw "Detector PID belongs to an unexpected command"
    }
    $ExpectedStart = ConvertTo-ExactUtc `
        $WaitForStartedAtUtc "detector wrapper start"
    $ActualStart = $Cim.CreationDate.ToUniversalTime()
    if ([math]::Abs(($ActualStart - $ExpectedStart).TotalMilliseconds) -gt 1.0) {
        throw "Detector PID creation time differs from the frozen identity"
    }
    $Process = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($null -eq $Process) {
        throw "Authenticated detector process disappeared before a wait handle was acquired"
    }
    return [pscustomobject]@{
        Observed = $true
        Pid = $WaitForPid
        ExactStartUtc = $WaitForStartedAtUtc
        CommandFragment = $WaitForCommandFragment
        Process = $Process
    }
}

function Invoke-PythonJson {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int[]]$AllowedExitCodes = @(0)
    )
    $Output = & $Python @Arguments 2>&1
    $ExitCode = $LASTEXITCODE
    $Text = ($Output | ForEach-Object { [string]$_ }) -join "`n"
    $Text = $Text.Trim()
    if ($ExitCode -notin $AllowedExitCodes) {
        throw (
            "Python command failed with exit code ${ExitCode}: " +
            (($Arguments -join " ") + "`n" + $Text).Trim()
        )
    }
    if ([string]::IsNullOrWhiteSpace($Text)) {
        throw "Python JSON command produced no output: $($Arguments -join ' ')"
    }
    try {
        $Value = $Text | ConvertFrom-Json -Depth 100
    } catch {
        throw "Python command returned invalid JSON: $Text"
    }
    return [pscustomobject]@{
        ExitCode = $ExitCode
        Value = $Value
    }
}

function Test-PythonSources {
    $Code = @'
import ast
import pathlib
import sys
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
'@
    $Output = & $Python -c $Code $Builder $Materializer $FrontendVerifier 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python source syntax preflight failed: $($Output -join ' ')"
    }
}

function Get-BuilderPreflight {
    return Invoke-PythonJson -AllowedExitCodes @(0, 2) -Arguments @(
        "-m", "experiments.build_field_blind_bundle_inputs", "preflight",
        "--garc-summary", $GarcSummary,
        "--paper-summary", $PaperSummary,
        "--paper-seal", $PaperSeal,
        "--frontend-plan", $FrontendPlan,
        "--output-root", $BuilderOutputRoot
    )
}

function Get-MaterializerPreflight {
    return Invoke-PythonJson -AllowedExitCodes @(0, 2) -Arguments @(
        "-m", "experiments.materialize_field_blind_bundles", "preflight",
        "--spec", $Spec,
        "--garc-summary", $GarcSummary,
        "--paper-summary", $PaperSummary,
        "--paper-seal", $PaperSeal
    )
}

function Test-FrontendAuthority {
    $Result = Invoke-PythonJson -Arguments @(
        "-m", "experiments.syncg_meter_detector_frontend",
        "--verify", "--frontend-plan", $FrontendPlan
    )
    if ($Result.Value.status -ne "verified") {
        throw "Public meter frontend did not verify"
    }
    return $Result.Value
}

function Send-StageEvent(
    [string]$EventKey,
    [string]$Message,
    [string]$Eta
) {
    & $Reporter -EventKey $EventKey -Message $Message -Eta $Eta | Out-Host
}

if ($PreflightOnly) {
    Test-PythonSources
    [void][scriptblock]::Create((Get-Content -LiteralPath $PSCommandPath -Raw))
    $Detector = Get-AuthenticatedDetectorProcess
    $Inputs = [ordered]@{
        garc_summary = Test-Path -LiteralPath $GarcSummary -PathType Leaf
        paper_summary = Test-Path -LiteralPath $PaperSummary -PathType Leaf
        paper_seal = Test-Path -LiteralPath $PaperSeal -PathType Leaf
        frontend_plan = Test-Path -LiteralPath $FrontendPlan -PathType Leaf
    }
    $AllInputsPresent = -not ($Inputs.Values -contains $false)
    $Scientific = $null
    if ($AllInputsPresent) {
        if (Test-Path -LiteralPath $Spec -PathType Leaf) {
            $Scientific = (Get-MaterializerPreflight).Value
        } elseif (-not (Test-Path -LiteralPath $BuilderOutputRoot)) {
            $Scientific = (Get-BuilderPreflight).Value
        } else {
            $Scientific = [ordered]@{
                status = "not_ready"
                ready = $false
                reason = "builder_output_exists_without_spec"
            }
        }
    } else {
        $Scientific = [ordered]@{
            status = "not_ready"
            ready = $false
            missing = @(
                $Inputs.GetEnumerator() |
                    Where-Object { -not $_.Value } |
                    ForEach-Object { $_.Key }
            )
        }
    }
    [ordered]@{
        schema_version = 1
        protocol = "field_bundle_after_detector_event_chain_preflight_v1"
        status = "preflight_only_complete"
        detector_process = [ordered]@{
            pid = $WaitForPid
            expected_start_utc = $WaitForStartedAtUtc
            command_fragment = $WaitForCommandFragment
            authenticated_and_running = ($null -ne $Detector)
            frontend_artifact_present = [bool]$Inputs.frontend_plan
        }
        exact_inputs = $Inputs
        scientific_preflight = $Scientific
        intended_outputs = [ordered]@{
            builder_root = $BuilderOutputRoot
            spec = $Spec
            materialized_root = $MaterializedOutputRoot
        }
        audit = [ordered]@{
            waited_for_process = $false
            writes = 0
            formal_chain_started = $false
            field_manifest_opened = $false
            field_images_opened = $false
            field_labels_opened = $false
            blind_inference_started = $false
            feishu_messages_sent = 0
        }
    } | ConvertTo-Json -Depth 100 -Compress
    return
}

$BuilderOutputRoot = Assert-PointerReadOutput `
    $BuilderOutputRoot "builder output root"
$MaterializedOutputRoot = Assert-PointerReadOutput `
    $MaterializedOutputRoot "materialized output root"
$Spec = Join-Path $BuilderOutputRoot "spec.json"
$Detector = $null

try {
    $CurrentStage = "authenticate-detector-process"
    $Detector = Get-AuthenticatedDetectorProcess -RequirePresent

    $CurrentStage = "wait-public-detector"
    try {
        $Detector.Process.WaitForExit()
        $DetectorExitCode = $Detector.Process.ExitCode
    } catch {
        throw "Could not observe authenticated detector completion: $($_.Exception.Message)"
    }
    if ($DetectorExitCode -ne 0) {
        throw "Public detector wrapper exited with code $DetectorExitCode"
    }

    $CurrentStage = "authenticate-public-authorities"
    $null = Test-FrontendAuthority
    foreach ($RequiredAuthority in @(
        $GarcSummary,
        $PaperSummary,
        $PaperSeal,
        $FrontendPlan
    )) {
        if (-not (Test-Path -LiteralPath $RequiredAuthority -PathType Leaf)) {
            throw "Required completed public authority is absent: $RequiredAuthority"
        }
    }

    if (Test-Path -LiteralPath $BuilderOutputRoot) {
        if (-not (Test-Path -LiteralPath $Spec -PathType Leaf)) {
            throw "Builder output exists without an authenticated spec: $BuilderOutputRoot"
        }
        $ExistingPreflight = Get-MaterializerPreflight
        if (
            $ExistingPreflight.ExitCode -ne 0 -or
            $ExistingPreflight.Value.ready -ne $true -or
            $ExistingPreflight.Value.status -ne "ready"
        ) {
            throw "Existing materialization spec failed public-authority validation"
        }
        $BuildDisposition = "reused_authenticated_spec"
    } else {
        $BuilderPreflight = Get-BuilderPreflight
        if (
            $BuilderPreflight.ExitCode -ne 0 -or
            $BuilderPreflight.Value.ready -ne $true -or
            $BuilderPreflight.Value.status -ne "ready"
        ) {
            throw (
                "GARC/paper/frontend are not jointly ready for bundle preparation: " +
                ($BuilderPreflight.Value | ConvertTo-Json -Depth 20 -Compress)
            )
        }
        $BuildDisposition = "new_public_input_freeze"
    }

    Send-StageEvent `
        -EventKey "syncg-public-frontend-to-five-method-bundle-materialization-v1" `
        -Message (
            "公开共享仪表检测前端已完成认证；GARC、论文结果和frontend lineage均已就绪，" +
            "现切换到五方法source/config/catalog冻结与metadata物化。当前不读取现场照片，" +
            "不启动盲测。"
        ) `
        -Eta "预计5–20分钟完成五方法bundle冻结、物化和独立校验。"

    if ($BuildDisposition -eq "new_public_input_freeze") {
        $CurrentStage = "freeze-five-method-public-inputs"
        $Build = Invoke-PythonJson -Arguments @(
            "-m", "experiments.build_field_blind_bundle_inputs", "build",
            "--garc-summary", $GarcSummary,
            "--paper-summary", $PaperSummary,
            "--paper-seal", $PaperSeal,
            "--frontend-plan", $FrontendPlan,
            "--output-root", $BuilderOutputRoot
        )
        if (
            $Build.Value.status -ne "complete" -or
            [string]$Build.Value.spec.path -ne $Spec -or
            -not (Test-Path -LiteralPath $Spec -PathType Leaf)
        ) {
            throw "Five-method input builder did not produce the frozen spec"
        }
    }

    $CurrentStage = "verify-frozen-spec"
    $MaterializerPreflight = Get-MaterializerPreflight
    if (
        $MaterializerPreflight.ExitCode -ne 0 -or
        $MaterializerPreflight.Value.ready -ne $true -or
        $MaterializerPreflight.Value.status -ne "ready"
    ) {
        throw "Frozen five-method spec failed materializer preflight"
    }

    $CurrentStage = "materialize-five-method-bundles"
    if (Test-Path -LiteralPath $MaterializedOutputRoot) {
        $Materialized = Invoke-PythonJson -Arguments @(
            "-m", "experiments.materialize_field_blind_bundles", "verify",
            "--root", $MaterializedOutputRoot
        )
    } else {
        $Materialized = Invoke-PythonJson -Arguments @(
            "-m", "experiments.materialize_field_blind_bundles", "materialize",
            "--spec", $Spec,
            "--output-root", $MaterializedOutputRoot
        )
    }
    if ($Materialized.Value.verified -ne $true) {
        throw "Five-method materialization verification failed"
    }
    $Verified = Invoke-PythonJson -Arguments @(
        "-m", "experiments.materialize_field_blind_bundles", "verify",
        "--root", $MaterializedOutputRoot
    )
    if (
        $Verified.Value.status -ne "verified" -or
        $Verified.Value.verified -ne $true -or
        @($Verified.Value.method_roles).Count -ne 5
    ) {
        throw "Independent five-method roster verification failed"
    }

    $CurrentStage = "complete"
    Send-StageEvent `
        -EventKey "five-method-public-bundles-materialized-complete-v1" `
        -Message (
            "五方法公开输入冻结与metadata物化已完成：GARC-final、V5-complete、PEPD、" +
            "VDN和Original Transformer共5/5 bundle均通过哈希、provider identity、" +
            "共享range及frontend校验。现场盲测尚未启动。"
        ) `
        -Eta "下一步仅在明确授权后执行1200+原图一次性盲测，预计3–8小时。"

    [ordered]@{
        schema_version = 1
        protocol = "field_bundle_after_detector_event_chain_v1"
        status = "complete"
        detector = [ordered]@{
            pid = $WaitForPid
            exact_start_utc = $WaitForStartedAtUtc
            authenticated_before_wait = $true
        }
        builder_disposition = $BuildDisposition
        spec = $Spec
        materialized_root = $MaterializedOutputRoot
        method_roles = @($Verified.Value.method_roles)
        audit = [ordered]@{
            field_manifest_opened = $false
            field_images_opened = $false
            field_labels_opened = $false
            blind_inference_started = $false
        }
    } | ConvertTo-Json -Depth 20 -Compress
} catch {
    $OriginalError = $_
    try {
        Send-StageEvent `
            -EventKey "five-method-bundle-$CurrentStage-unexpected-stop-v1" `
            -Message (
                "五方法bundle准备链在阶段 '$CurrentStage' 异常停止：" +
                $OriginalError.Exception.Message +
                "。未启动现场盲测，已有不可变制品保持原状。"
            ) `
            -Eta "预计15–60分钟完成定位；修复后可从已认证frontend或spec继续。"
    } catch {
        Write-Error "Bundle-chain notification failed: $($_.Exception.Message)"
    }
    throw $OriginalError
}

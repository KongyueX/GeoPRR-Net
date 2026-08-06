#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = (Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"),
    [string]$PowerShell7 = "C:\Program Files\PowerShell\7\pwsh.exe",
    [ValidateRange(0, 2147483647)]
    [int]$WaitForPid = 0,
    [string]$WaitForStartedAtUtc = "",
    [string]$WaitForCommandFragment = "",
    [string]$WaitForCommandLineSha256 = "",
    [string]$MaterializedRoot = "",
    [string]$MethodRoster = "",
    [string]$FrontendPlan = "",
    [string]$DatasetIdentity = "",
    [string]$FrozenManifest = "",
    [string]$FrozenLabels = "",
    [string]$AuthorizationProtocol = "",
    [string]$RunRoot = "",
    [string]$PaperResults = "",
    [string]$PaperTablesRoot = "",
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
$PowerShell7 = [System.IO.Path]::GetFullPath($PowerShell7)
$ChainPreflight = Join-Path $ProjectRoot `
    "experiments\field_blind_final_chain_preflight.py"
$BlindWrapper = Join-Path $ProjectRoot `
    "experiments\run_field_blind_multimethod_final_once.ps1"
$BlindRunner = Join-Path $ProjectRoot `
    "experiments\field_blind_multimethod.py"
$Materializer = Join-Path $ProjectRoot `
    "experiments\materialize_field_blind_bundles.py"
$MaterializationWrapper = Join-Path $ProjectRoot `
    "experiments\run_field_bundle_materialization_after_detector_event_driven.ps1"
$Reporter = Join-Path $ProjectRoot "experiments\send_feishu_progress.ps1"
$CurrentStage = "initialize"
$script:FeishuAttempts = 0
$script:FeishuFailures = 0

foreach ($RequiredSource in @(
    $Python,
    $PowerShell7,
    $ChainPreflight,
    $BlindWrapper,
    $BlindRunner,
    $Materializer,
    $MaterializationWrapper,
    $Reporter
)) {
    if (-not (Test-Path -LiteralPath $RequiredSource -PathType Leaf)) {
        throw "Required final-chain source is absent: $RequiredSource"
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

function Get-Sha256Text([string]$Value) {
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
    return [Convert]::ToHexString(
        [System.Security.Cryptography.SHA256]::HashData($Bytes)
    ).ToLowerInvariant()
}

function Get-Sha256File([string]$Path) {
    $Stream = [System.IO.File]::OpenRead($Path)
    try {
        return [Convert]::ToHexString(
            [System.Security.Cryptography.SHA256]::HashData($Stream)
        ).ToLowerInvariant()
    } finally {
        $Stream.Dispose()
    }
}

function Assert-ExplicitAbsolutePath([string]$Value, [string]$Label) {
    if (-not [System.IO.Path]::IsPathFullyQualified($Value)) {
        throw "-$Label must be supplied as an explicit absolute path"
    }
}

function Test-IsSameOrDescendant([string]$Candidate, [string]$Root) {
    $FullCandidate = [System.IO.Path]::GetFullPath($Candidate).TrimEnd("\")
    $FullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd("\")
    return (
        $FullCandidate.Equals(
            $FullRoot, [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        $FullCandidate.StartsWith(
            $FullRoot + "\", [System.StringComparison]::OrdinalIgnoreCase
        )
    )
}

function Assert-FormalOutputIsolation {
    foreach ($Output in @(
        @{ Name = "AuthorizationProtocol"; Value = $AuthorizationProtocol },
        @{ Name = "RunRoot"; Value = $RunRoot }
    )) {
        foreach ($FrozenRoot in @(
            @{ Name = "MaterializedRoot"; Value = $MaterializedRoot },
            @{ Name = "PaperTablesRoot"; Value = $PaperTablesRoot }
        )) {
            if (Test-IsSameOrDescendant $Output.Value $FrozenRoot.Value) {
                throw (
                    "-$($Output.Name) must not write inside frozen " +
                    "-$($FrozenRoot.Name)"
                )
            }
        }
        foreach ($FrozenFile in @(
            $DatasetIdentity,
            $FrozenManifest,
            $FrozenLabels,
            $MethodRoster,
            $FrontendPlan,
            $PaperResults
        )) {
            if (
                $Output.Value.Equals(
                    $FrozenFile, [System.StringComparison]::OrdinalIgnoreCase
                )
            ) {
                throw "-$($Output.Name) must be distinct from every frozen input"
            }
        }
    }
    if (
        (Test-IsSameOrDescendant $AuthorizationProtocol $RunRoot) -or
        $AuthorizationProtocol.Equals(
            $RunRoot, [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "AuthorizationProtocol must remain outside RunRoot"
    }
}

function Get-AuthenticatedMaterializerProcess {
    param([switch]$RequirePresent)

    if ($WaitForPid -eq 0) {
        if ($RequirePresent) {
            throw "-WaitForPid must identify the future five-method materialization process"
        }
        return $null
    }
    if ([string]::IsNullOrWhiteSpace($WaitForStartedAtUtc)) {
        throw "-WaitForStartedAtUtc is required whenever -WaitForPid is nonzero"
    }
    if ([string]::IsNullOrWhiteSpace($WaitForCommandFragment)) {
        throw "-WaitForCommandFragment is required whenever -WaitForPid is nonzero"
    }
    if (
        $WaitForCommandLineSha256 -notmatch '^[0-9a-fA-F]{64}$'
    ) {
        throw "-WaitForCommandLineSha256 must be the exact 64-hex command-line digest"
    }
    $Cim = Get-CimInstance Win32_Process -Filter "ProcessId = $WaitForPid"
    if ($null -eq $Cim) {
        if ($RequirePresent) {
            throw "Authenticated five-method materialization PID is no longer present"
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
        throw "Materialization PID belongs to an unexpected command"
    }
    if (
        $CommandLine.IndexOf(
            $MaterializationWrapper,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -lt 0
    ) {
        throw "PID is not the canonical five-method materialization wrapper"
    }
    $ActualCommandHash = Get-Sha256Text $CommandLine
    if (
        $ActualCommandHash -ne $WaitForCommandLineSha256.ToLowerInvariant()
    ) {
        throw "Materialization PID command line differs from the frozen identity"
    }
    $ExpectedStart = ConvertTo-ExactUtc `
        $WaitForStartedAtUtc "materialization process start"
    $ActualStart = $Cim.CreationDate.ToUniversalTime()
    if ([math]::Abs(($ActualStart - $ExpectedStart).TotalMilliseconds) -gt 1.0) {
        throw "Materialization PID creation time differs from the frozen identity"
    }
    $Process = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($null -eq $Process) {
        throw "Authenticated materialization process disappeared before wait handle acquisition"
    }
    $HandleStart = $Process.StartTime.ToUniversalTime()
    if ([math]::Abs(($HandleStart - $ExpectedStart).TotalMilliseconds) -gt 1.0) {
        throw "Materialization PID was reused before wait handle acquisition"
    }
    return [pscustomobject]@{
        Process = $Process
        Pid = $WaitForPid
        ExactStartUtc = $WaitForStartedAtUtc
        CommandLineSha256 = $ActualCommandHash
        CommandFragment = $WaitForCommandFragment
    }
}

function Invoke-PythonJson {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $Output = & $Python @Arguments 2>&1
    $ExitCode = $LASTEXITCODE
    $Lines = @(
        $Output |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($ExitCode -ne 0) {
        throw (
            "Python gate failed with exit code ${ExitCode}: " +
            (($Lines -join "`n").Trim())
        )
    }
    if ($Lines.Count -eq 0) {
        throw "Python gate produced no JSON"
    }
    try {
        return $Lines[-1] | ConvertFrom-Json -Depth 100
    } catch {
        throw "Python gate returned malformed terminal JSON: $($Lines[-1])"
    }
}

function Invoke-BlindWrapper {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $Output = & $PowerShell7 -NoLogo -NoProfile -File $BlindWrapper `
        -ProjectRoot $ProjectRoot -Python $Python @Arguments 2>&1
    $ExitCode = $LASTEXITCODE
    $Text = (@($Output | ForEach-Object { [string]$_ }) -join "`n").Trim()
    if ($ExitCode -ne 0) {
        throw "Blind one-shot wrapper failed with exit code ${ExitCode}: $Text"
    }
    if ([string]::IsNullOrWhiteSpace($Text)) {
        throw "Blind one-shot wrapper produced no JSON"
    }
    try {
        return $Text | ConvertFrom-Json -Depth 100
    } catch {
        throw "Blind one-shot wrapper returned malformed JSON"
    }
}

function Get-VerifiedJsonBinding {
    param(
        [Parameter(Mandatory = $true)]$Binding,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $PathText = [string]$Binding.path
    $Digest = ([string]$Binding.sha256).ToLowerInvariant()
    if (-not [System.IO.Path]::IsPathFullyQualified($PathText)) {
        throw "$Label path is not absolute"
    }
    if ($Digest -notmatch '^[0-9a-f]{64}$') {
        throw "$Label digest is invalid"
    }
    $Path = [System.IO.Path]::GetFullPath($PathText)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is absent"
    }
    if ((Get-Sha256File $Path) -ne $Digest) {
        throw "$Label hash drift"
    }
    try {
        $Value = Get-Content -LiteralPath $Path -Raw |
            ConvertFrom-Json -Depth 100
    } catch {
        throw "$Label is not valid JSON"
    }
    return [pscustomobject]@{
        Path = $Path
        Sha256 = $Digest
        Value = $Value
    }
}

function Send-StageEvent(
    [string]$EventKey,
    [string]$Message,
    [string]$Eta
) {
    $script:FeishuAttempts += 1
    try {
        $ReportOutput = & $PowerShell7 -NoLogo -NoProfile -File $Reporter `
            -EventKey $EventKey -Message $Message -Eta $Eta 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "notification process exited with code $LASTEXITCODE"
        }
        Write-Verbose (($ReportOutput | ForEach-Object { [string]$_ }) -join " ")
    } catch {
        $script:FeishuFailures += 1
        Write-Warning "Feishu event '$EventKey' was not delivered: $($_.Exception.Message)"
    }
}

function Assert-FormalParameters {
    $Required = @(
        @{ Name = "MaterializedRoot"; Value = $MaterializedRoot },
        @{ Name = "MethodRoster"; Value = $MethodRoster },
        @{ Name = "FrontendPlan"; Value = $FrontendPlan },
        @{ Name = "DatasetIdentity"; Value = $DatasetIdentity },
        @{ Name = "FrozenManifest"; Value = $FrozenManifest },
        @{ Name = "FrozenLabels"; Value = $FrozenLabels },
        @{ Name = "AuthorizationProtocol"; Value = $AuthorizationProtocol },
        @{ Name = "RunRoot"; Value = $RunRoot },
        @{ Name = "PaperResults"; Value = $PaperResults },
        @{ Name = "PaperTablesRoot"; Value = $PaperTablesRoot }
    )
    foreach ($Item in $Required) {
        if ([string]::IsNullOrWhiteSpace([string]$Item.Value)) {
            throw "-$($Item.Name) must be supplied explicitly for -StartFormal"
        }
    }
}

function Get-SuppliedParameterAudit {
    return [ordered]@{
        materialized_root = -not [string]::IsNullOrWhiteSpace($MaterializedRoot)
        method_roster = -not [string]::IsNullOrWhiteSpace($MethodRoster)
        frontend_plan = -not [string]::IsNullOrWhiteSpace($FrontendPlan)
        dataset_identity = -not [string]::IsNullOrWhiteSpace($DatasetIdentity)
        frozen_manifest = -not [string]::IsNullOrWhiteSpace($FrozenManifest)
        frozen_labels = -not [string]::IsNullOrWhiteSpace($FrozenLabels)
        authorization_protocol = -not [string]::IsNullOrWhiteSpace($AuthorizationProtocol)
        run_root = -not [string]::IsNullOrWhiteSpace($RunRoot)
        paper_results = -not [string]::IsNullOrWhiteSpace($PaperResults)
        paper_tables_root = -not [string]::IsNullOrWhiteSpace($PaperTablesRoot)
    }
}

if ($PreflightOnly) {
    [void][scriptblock]::Create((Get-Content -LiteralPath $PSCommandPath -Raw))
    $MaterializerProcess = Get-AuthenticatedMaterializerProcess
    [ordered]@{
        schema_version = 1
        protocol = "field_blind_multimethod_after_bundles_event_chain_preflight_v1"
        status = "preflight_only_complete"
        materializer_process = [ordered]@{
            pid = $WaitForPid
            expected_start_utc = $WaitForStartedAtUtc
            command_fragment_supplied = -not [string]::IsNullOrWhiteSpace($WaitForCommandFragment)
            command_sha256_supplied = $WaitForCommandLineSha256 -match '^[0-9a-fA-F]{64}$'
            authenticated_and_running = ($null -ne $MaterializerProcess)
        }
        explicit_parameters = Get-SuppliedParameterAudit
        audit = [ordered]@{
            writes = 0
            waited_for_process = $false
            public_materialization_opened = $false
            dataset_identity_opened = $false
            field_manifest_opened = $false
            field_manifest_hashed = $false
            field_images_opened = $false
            field_labels_opened = $false
            field_labels_hashed = $false
            inference_started = $false
            scoring_started = $false
            feishu_messages_sent = 0
        }
    } | ConvertTo-Json -Depth 20 -Compress
    return
}

Assert-FormalParameters
foreach ($FormalPath in @(
    @{ Name = "MaterializedRoot"; Value = $MaterializedRoot },
    @{ Name = "MethodRoster"; Value = $MethodRoster },
    @{ Name = "FrontendPlan"; Value = $FrontendPlan },
    @{ Name = "DatasetIdentity"; Value = $DatasetIdentity },
    @{ Name = "FrozenManifest"; Value = $FrozenManifest },
    @{ Name = "FrozenLabels"; Value = $FrozenLabels },
    @{ Name = "AuthorizationProtocol"; Value = $AuthorizationProtocol },
    @{ Name = "RunRoot"; Value = $RunRoot },
    @{ Name = "PaperResults"; Value = $PaperResults },
    @{ Name = "PaperTablesRoot"; Value = $PaperTablesRoot }
)) {
    Assert-ExplicitAbsolutePath $FormalPath.Value $FormalPath.Name
}
$MaterializedRoot = [System.IO.Path]::GetFullPath($MaterializedRoot)
$MethodRoster = [System.IO.Path]::GetFullPath($MethodRoster)
$FrontendPlan = [System.IO.Path]::GetFullPath($FrontendPlan)
$DatasetIdentity = [System.IO.Path]::GetFullPath($DatasetIdentity)
$FrozenManifest = [System.IO.Path]::GetFullPath($FrozenManifest)
$FrozenLabels = [System.IO.Path]::GetFullPath($FrozenLabels)
$AuthorizationProtocol = [System.IO.Path]::GetFullPath($AuthorizationProtocol)
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
$PaperResults = [System.IO.Path]::GetFullPath($PaperResults)
$PaperTablesRoot = [System.IO.Path]::GetFullPath($PaperTablesRoot)
Assert-FormalOutputIsolation

if (
    $AuthorizationProtocol -in @(
        $DatasetIdentity,
        $FrozenManifest,
        $FrozenLabels,
        $MethodRoster,
        $FrontendPlan,
        $PaperResults
    )
) {
    throw "AuthorizationProtocol must be a new path distinct from every frozen input"
}
if ($RunRoot -in @($DatasetIdentity, $FrozenManifest, $FrozenLabels)) {
    throw "RunRoot must be distinct from every owner-frozen input"
}

$MaterializerProcess = $null
try {
    # No owner identity, manifest, labels, or image is touched before this
    # exact PID/start/command identity exits successfully.
    $CurrentStage = "authenticate-five-method-materialization"
    $MaterializerProcess = Get-AuthenticatedMaterializerProcess -RequirePresent

    $CurrentStage = "wait-five-method-materialization"
    $MaterializerProcess.Process.WaitForExit()
    $MaterializerExitCode = $MaterializerProcess.Process.ExitCode
    if ($MaterializerExitCode -ne 0) {
        throw "Authenticated five-method materialization exited with code $MaterializerExitCode"
    }

    $CurrentStage = "verify-five-method-materialization"
    $PublicGate = Invoke-PythonJson @(
        "-m", "experiments.field_blind_final_chain_preflight",
        "public-materialization",
        "--materialized-root", $MaterializedRoot,
        "--method-roster", $MethodRoster,
        "--frontend-plan", $FrontendPlan
    )
    if (
        $PublicGate.status -ne "public_materialization_verified" -or
        @($PublicGate.method_roles).Count -ne 5 -or
        $PublicGate.sealed_bundle_count -ne 5 -or
        $PublicGate.field_manifest_opened -ne $false -or
        $PublicGate.field_images_opened -ne $false -or
        $PublicGate.field_labels_opened -ne $false
    ) {
        throw "Completed materialization did not pass the independent public gate"
    }

    $CurrentStage = "instantiate-all-public-runtimes"
    $RuntimeGate = Invoke-PythonJson @(
        "-m", "experiments.field_blind_final_chain_preflight",
        "runtime",
        "--method-roster", $MethodRoster,
        "--frontend-plan", $FrontendPlan
    )
    if (
        $RuntimeGate.status -ne "all_public_runtimes_instantiated_without_dataset_access" -or
        @($RuntimeGate.method_roles).Count -ne 5 -or
        $RuntimeGate.shared_detector_instantiated -ne $true
    ) {
        throw "All-model runtime preflight did not complete"
    }

    # Only after successful materialization and model-instantiation gates may
    # the explicit owner metadata be opened.  Manifest/labels themselves are
    # still neither opened nor hashed here.
    $CurrentStage = "bind-explicit-owner-metadata"
    $DatasetGate = Invoke-PythonJson @(
        "-m", "experiments.field_blind_final_chain_preflight",
        "dataset-bindings",
        "--dataset-identity", $DatasetIdentity,
        "--manifest", $FrozenManifest,
        "--labels", $FrozenLabels
    )
    if (
        $DatasetGate.status -ne "explicit_owner_bindings_verified_without_data_access" -or
        $DatasetGate.manifest.opened -ne $false -or
        $DatasetGate.manifest.hashed -ne $false -or
        $DatasetGate.labels.opened -ne $false -or
        $DatasetGate.labels.hashed -ne $false
    ) {
        throw "Explicit owner-frozen dataset bindings did not verify cleanly"
    }

    $CurrentStage = "freeze-one-shot-authorization"
    $Frozen = Invoke-BlindWrapper @(
        "-Protocol", $AuthorizationProtocol,
        "-DatasetIdentity", $DatasetIdentity,
        "-PaperResults", $PaperResults,
        "-PaperTablesRoot", $PaperTablesRoot,
        "-FrontendPlan", $FrontendPlan,
        "-MethodRoster", $MethodRoster,
        "-RunRoot", $RunRoot,
        "-FreezeAuthorization"
    )
    if (
        $Frozen.status -ne "authorization_frozen_without_field_manifest_label_or_image_access" -or
        $Frozen.field_manifest_opened -ne $false -or
        $Frozen.field_labels_opened -ne $false -or
        $Frozen.field_images_opened -ne $false
    ) {
        throw "One-shot authorization did not freeze without restricted access"
    }
    $AuthorizationPreflight = Invoke-BlindWrapper @(
        "-Protocol", $AuthorizationProtocol,
        "-PreflightOnly"
    )
    if (
        $AuthorizationPreflight.status -ne "validated_without_field_manifest_label_or_image_access" -or
        @($AuthorizationPreflight.method_roles).Count -ne 5
    ) {
        throw "Frozen one-shot authorization failed final preflight"
    }

    Send-StageEvent `
        -EventKey "five-method-bundles-to-final-blind-inference-v1" `
        -Message (
            "五方法bundle、共享公开仪表检测器和全部运行时已完成独立认证；" +
            "显式冻结清单/标签仅完成路径绑定且尚未读取。现开始一次性五方法现场盲测推理。"
        ) `
        -Eta "预计3–8小时完成全图推理并共同封存五方法预测，随后自动评分。"

    $CurrentStage = "one-shot-five-method-inference"
    $Inference = Invoke-BlindWrapper @(
        "-Protocol", $AuthorizationProtocol,
        "-StartInference"
    )
    if (
        $Inference.status -ne "complete_all_predictions_sealed_before_labels" -or
        $Inference.field_labels_opened -ne $false -or
        [int]$Inference.rows -ne [int]$DatasetGate.declared_images
    ) {
        throw "Five-method inference did not jointly seal predictions before labels"
    }
    $PredictionSeal = Get-VerifiedJsonBinding `
        -Binding $Inference.prediction_seal `
        -Label "joint prediction seal"
    if (
        $PredictionSeal.Value.status -ne "all_method_predictions_sealed_together_before_labels" -or
        $PredictionSeal.Value.field_labels_opened -ne $false -or
        $PredictionSeal.Value.same_shared_roi_for_all_methods -ne $true -or
        [int]$PredictionSeal.Value.predictions.rows -ne [int]$DatasetGate.declared_images -or
        ([string]$PredictionSeal.Value.predictions.sha256).ToLowerInvariant() -ne
            ([string]$Inference.predictions.sha256).ToLowerInvariant()
    ) {
        throw "Joint prediction seal content is incomplete or inconsistent"
    }

    Send-StageEvent `
        -EventKey "five-method-final-blind-inference-complete-to-scoring-v1" `
        -Message (
            "一次性现场盲测推理已完成，五种方法的全部预测已在读取标签前共同封存；" +
            "现在切换到一次性评分。"
        ) `
        -Eta "预计5–20分钟完成全样本NMAE、覆盖率、组宏平均和P95汇总。"

    $CurrentStage = "one-shot-five-method-scoring"
    $Score = Invoke-BlindWrapper @(
        "-Protocol", $AuthorizationProtocol,
        "-StartScoring"
    )
    if (
        $Score.status -ne "complete_one_shot_multimethod_blind_score" -or
        $Score.prediction_seal_verified_before_label_access -ne $true -or
        $Score.no_post_result_tuning -ne $true
    ) {
        throw "Five-method blind scoring did not complete under the frozen seal"
    }
    $SummaryBinding = Get-VerifiedJsonBinding `
        -Binding $Score.summary `
        -Label "completed blind score summary"
    $ScoreSeal = Get-VerifiedJsonBinding `
        -Binding $Score.seal `
        -Label "completed blind score seal"
    $Summary = $SummaryBinding.Value
    if (
        $Summary.status -ne "complete_one_shot_multimethod_blind_score" -or
        $Summary.prediction_seal_verified_before_label_access -ne $true -or
        $Summary.method_or_threshold_selection_after_result -ne $false -or
        $ScoreSeal.Value.status -ne "sealed" -or
        ([string]$ScoreSeal.Value.summary_sha256).ToLowerInvariant() -ne
            $SummaryBinding.Sha256
    ) {
        throw "Blind score summary status drift"
    }
    $ExpectedRoles = @(
        "garc_final",
        "v5_complete",
        "pepd_shared_range",
        "vdn_shared_range",
        "transformer_shared_range"
    )
    $ObservedRoles = @($Summary.methods.PSObject.Properties.Name)
    if (
        $ObservedRoles.Count -ne $ExpectedRoles.Count -or
        @(Compare-Object $ExpectedRoles $ObservedRoles).Count -ne 0
    ) {
        throw "Blind score summary does not contain exactly five method roles"
    }
    $MetricText = @(
        foreach ($Role in $ExpectedRoles) {
            $Metric = $Summary.methods.$Role
            if ($null -eq $Metric) {
                throw "Blind score summary lacks role: $Role"
            }
            if (
                [int]$Metric.samples -ne [int]$DatasetGate.declared_images -or
                ([int]$Metric.successful + [int]$Metric.failures) -ne
                    [int]$DatasetGate.declared_images
            ) {
                throw "Blind score summary denominator drift for role: $Role"
            }
            $Nmae = ([double]$Metric.full_denominator_nmae).ToString(
                "0.00000", [System.Globalization.CultureInfo]::InvariantCulture
            )
            $Coverage = (100.0 * [double]$Metric.coverage).ToString(
                "0.00", [System.Globalization.CultureInfo]::InvariantCulture
            )
            "${Role}: NMAE=${Nmae}, Coverage=${Coverage}%"
        }
    ) -join "；"

    $CurrentStage = "complete"
    Send-StageEvent `
        -EventKey "five-method-final-blind-score-complete-v1" `
        -Message (
            "冻结的1200+全图一次性五方法盲测与评分均已完成，预测封存先于标签读取，" +
            "且未做结果后调参。" + $MetricText
        ) `
        -Eta "现场盲测阶段已完成；下一步将把冻结结果写入论文表格与结论。"

    [ordered]@{
        schema_version = 1
        protocol = "field_blind_multimethod_after_bundles_event_chain_v1"
        status = "complete"
        materializer_identity = [ordered]@{
            pid = $MaterializerProcess.Pid
            exact_start_utc = $MaterializerProcess.ExactStartUtc
            command_line_sha256 = $MaterializerProcess.CommandLineSha256
            exit_code = $MaterializerExitCode
        }
        public_gate = $PublicGate
        runtime_gate = [ordered]@{
            status = $RuntimeGate.status
            method_roles = @($RuntimeGate.method_roles)
            shared_detector_instantiated = $RuntimeGate.shared_detector_instantiated
        }
        dataset_gate = $DatasetGate
        authorization_protocol = $AuthorizationProtocol
        inference = $Inference
        score = $Score
        metrics = $Summary.methods
        feishu = [ordered]@{
            attempts = $script:FeishuAttempts
            failures = $script:FeishuFailures
        }
    } | ConvertTo-Json -Depth 100
} catch {
    $OriginalError = $_
    Send-StageEvent `
        -EventKey "five-method-final-blind-$CurrentStage-unexpected-stop-v1" `
        -Message (
            "最终五方法盲测链在阶段 '$CurrentStage' 异常停止；具体诊断仅保留在本地终端，" +
            "避免把冻结数据路径发送到外部。监督器不会重试或绕过已有一次性声明/封存。"
        ) `
        -Eta "预计15–60分钟完成审计；是否可恢复取决于一次性声明是否已经创建。"
    throw $OriginalError
}

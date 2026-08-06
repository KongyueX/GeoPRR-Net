#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$ProjectRoot = "D:\project\PointerMeterReaderFastAPI",
    [string]$PaperSummary = "C:\pointer_read\paper_final_results_v2\summary.json",
    [string]$Corpus = "C:\pointer_read\syncg_meter_detector_public_v1",
    [string]$OutputDir = "C:\pointer_read\syncg_meter_detector_runs",
    [string]$FrontendOutput = "C:\pointer_read\syncg_meter_detector_frontend_v1",
    [string]$Pretrained = "C:\pointer_read\public_pretrained\yolo11n-ultralytics-assets-v8.3.0.pt",
    [int]$Seed = 20260819,
    [int]$Workers = 4,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProgressPreference = "SilentlyContinue"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "PowerShell 7 or newer is required."
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Builder = Join-Path $ProjectRoot "experiments\build_syncg_meter_detector_public.py"
$Fetcher = Join-Path $ProjectRoot "experiments\fetch_syncg_meter_detector_pretrained.py"
$Trainer = Join-Path $ProjectRoot "experiments\train_syncg_meter_detector.py"
$TrainingWrapper = Join-Path $ProjectRoot "experiments\run_syncg_meter_detector_training.ps1"
$FrontendModule = Join-Path $ProjectRoot "experiments\syncg_meter_detector_frontend.py"
$Reporter = Join-Path $ProjectRoot "experiments\send_feishu_progress.ps1"
$RunSummary = Join-Path (Join-Path $OutputDir "seed_$Seed") "summary.json"
$FrontendPlan = Join-Path $FrontendOutput "frontend_plan.json"
$PaperSeal = Join-Path (Split-Path -Parent $PaperSummary) "seal.json"
$ChainLock = Join-Path $OutputDir "seed_$Seed.event_chain.lock.json"
$ChainLockOwned = $false
foreach ($Required in @($Python, $Builder, $Fetcher, $Trainer, $TrainingWrapper, $FrontendModule, $Reporter)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required detector-chain artifact is absent: $Required"
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-PaperGateState {
    if (
        -not (Test-Path -LiteralPath $PaperSummary -PathType Leaf) -or
        -not (Test-Path -LiteralPath $PaperSeal -PathType Leaf)
    ) {
        return [pscustomobject]@{ State = "waiting"; Reason = "paper summary/seal not atomically published" }
    }
    try {
        $Summary = Get-Content -LiteralPath $PaperSummary -Raw | ConvertFrom-Json -Depth 100
        $Seal = Get-Content -LiteralPath $PaperSeal -Raw | ConvertFrom-Json -Depth 100
    } catch {
        return [pscustomobject]@{ State = "terminal_invalid"; Reason = "paper JSON parse failure: $($_.Exception.Message)" }
    }
    if (
        $Summary.protocol -ne "pointer_meter_paper_result_assembly_v1" -or
        $Summary.status -ne "complete" -or
        $Seal.protocol -ne "pointer_meter_paper_result_assembly_v1" -or
        $Seal.status -ne "sealed"
    ) {
        return [pscustomobject]@{ State = "terminal_invalid"; Reason = "paper protocol/status drift" }
    }
    $Observed = Get-Sha256 $PaperSummary
    if ([string]$Seal.artifacts.summary.sha256 -ne $Observed) {
        return [pscustomobject]@{ State = "terminal_invalid"; Reason = "paper summary/seal hash drift" }
    }
    return [pscustomobject]@{ State = "ready"; Reason = "authenticated paper result assembly" }
}

function Wait-PaperGate {
    $Initial = Get-PaperGateState
    if ($Initial.State -eq "ready") {
        return
    }
    if ($Initial.State -eq "terminal_invalid") {
        throw $Initial.Reason
    }
    $TargetRoot = Split-Path -Parent $PaperSummary
    $WatchRoot = Split-Path -Parent $TargetRoot
    if (-not (Test-Path -LiteralPath $WatchRoot -PathType Container)) {
        throw "Event watch root is absent: $WatchRoot"
    }
    $Watchers = @(
        [System.IO.FileSystemWatcher]::new($WatchRoot, (Split-Path -Leaf $TargetRoot)),
        [System.IO.FileSystemWatcher]::new($WatchRoot, (Split-Path -Leaf $PaperSummary)),
        [System.IO.FileSystemWatcher]::new($WatchRoot, (Split-Path -Leaf $PaperSeal))
    )
    $Watchers[0].IncludeSubdirectories = $false
    $Watchers[0].NotifyFilter = [System.IO.NotifyFilters]::DirectoryName -bor [System.IO.NotifyFilters]::FileName
    foreach ($Watcher in $Watchers[1..2]) {
        $Watcher.IncludeSubdirectories = $true
        $Watcher.NotifyFilter = [System.IO.NotifyFilters]::FileName -bor [System.IO.NotifyFilters]::LastWrite -bor [System.IO.NotifyFilters]::Size
    }
    $Registrations = [System.Collections.Generic.List[object]]::new()
    $SourceIds = [System.Collections.Generic.List[string]]::new()
    for ($Index = 0; $Index -lt $Watchers.Count; $Index++) {
        foreach ($EventName in @("Changed", "Created", "Renamed", "Error")) {
            $SourceId = "syncg-meter-detector-paper-$Index-$EventName-$PID"
            $SourceIds.Add($SourceId)
            $Registrations.Add(
                (Register-ObjectEvent -InputObject $Watchers[$Index] -EventName $EventName -SourceIdentifier $SourceId)
            )
        }
        $Watchers[$Index].EnableRaisingEvents = $true
    }
    try {
        while ($true) {
            $State = Get-PaperGateState
            if ($State.State -eq "ready") {
                return
            }
            if ($State.State -eq "terminal_invalid") {
                throw $State.Reason
            }
            $Event = Wait-Event
            if ($null -ne $Event) {
                $IsOwnEvent = $SourceIds.Contains([string]$Event.SourceIdentifier)
                $IsWatcherError = $IsOwnEvent -and [string]$Event.SourceIdentifier -match "-Error-"
                Remove-Event -EventIdentifier $Event.EventIdentifier
                if ($IsWatcherError) {
                    $AfterError = Get-PaperGateState
                    if ($AfterError.State -ne "ready") {
                        throw "Paper gate FileSystemWatcher reported an overflow/error before a valid terminal artifact was available"
                    }
                }
            }
        }
    } finally {
        foreach ($SourceId in $SourceIds) {
            Unregister-Event -SourceIdentifier $SourceId -ErrorAction SilentlyContinue
        }
        $Registrations | Remove-Job -Force -ErrorAction SilentlyContinue
        foreach ($Watcher in $Watchers) {
            $Watcher.Dispose()
        }
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

function Enter-ChainLock {
    [void](New-Item -ItemType Directory -Force -Path $OutputDir)
    if (Test-Path -LiteralPath $ChainLock -PathType Leaf) {
        $Existing = Get-Content -LiteralPath $ChainLock -Raw | ConvertFrom-Json
        if (
            $Existing.protocol -ne "syncg_public_meter_detector_event_chain_lock_v1" -or
            [int]$Existing.owner_pid -lt 1 -or
            [string]::IsNullOrWhiteSpace([string]$Existing.owner_started_at_utc)
        ) {
            throw "Detector event-chain lock is malformed: $ChainLock"
        }
        $Expected = [datetime]::Parse([string]$Existing.owner_started_at_utc).ToUniversalTime()
        $Observed = Get-ProcessStartUtc -ProcessId ([int]$Existing.owner_pid)
        if ($null -ne $Observed -and [math]::Abs(($Observed - $Expected).TotalSeconds) -le 1.0) {
            return $false
        }
        Remove-Item -LiteralPath $ChainLock -Force
    }
    $Started = Get-ProcessStartUtc -ProcessId $PID
    if ($null -eq $Started) {
        throw "Cannot authenticate current detector event-chain process"
    }
    $Value = [ordered]@{
        protocol = "syncg_public_meter_detector_event_chain_lock_v1"
        owner_pid = $PID
        owner_started_at_utc = $Started.ToString("o")
        seed = $Seed
        paper_summary = [System.IO.Path]::GetFullPath($PaperSummary)
        output_dir = [System.IO.Path]::GetFullPath($OutputDir)
        created_at_utc = [datetime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Compress
    $Stream = [System.IO.File]::Open(
        $ChainLock,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Value + "`n")
        $Stream.Write($Bytes, 0, $Bytes.Length)
    } finally {
        $Stream.Dispose()
    }
    $script:ChainLockOwned = $true
    return $true
}

function Exit-ChainLock {
    if (-not $ChainLockOwned -or -not (Test-Path -LiteralPath $ChainLock -PathType Leaf)) {
        return
    }
    $Existing = Get-Content -LiteralPath $ChainLock -Raw | ConvertFrom-Json
    $Observed = Get-ProcessStartUtc -ProcessId $PID
    if (
        [int]$Existing.owner_pid -ne $PID -or
        $null -eq $Observed -or
        [math]::Abs((([datetime]::Parse([string]$Existing.owner_started_at_utc)).ToUniversalTime() - $Observed).TotalSeconds) -gt 1.0
    ) {
        throw "Refusing to remove a detector event-chain lock not owned by this PID/start identity"
    }
    Remove-Item -LiteralPath $ChainLock -Force
    $script:ChainLockOwned = $false
}

function Send-ProgressBestEffort {
    param(
        [Parameter(Mandatory = $true)][string]$EventKey,
        [Parameter(Mandatory = $true)][string]$Message,
        [Parameter(Mandatory = $true)][string]$Eta
    )
    try {
        & $Reporter -EventKey $EventKey -Message $Message -Eta $Eta
    } catch {
        Write-Warning "Progress notification is pending/retryable and did not alter detector state: $($_.Exception.Message)"
    }
}

function Get-DetectorTerminalState {
    $RunSeal = Join-Path (Split-Path -Parent $RunSummary) "seal.json"
    $HasSummary = Test-Path -LiteralPath $RunSummary -PathType Leaf
    $HasSeal = Test-Path -LiteralPath $RunSeal -PathType Leaf
    if (-not $HasSummary -and -not $HasSeal) {
        return [pscustomobject]@{ State = "absent_or_incomplete"; Summary = $null; Reason = "no terminal summary/seal" }
    }
    if ($HasSummary -ne $HasSeal) {
        return [pscustomobject]@{ State = "terminal_invalid"; Summary = $null; Reason = "partial detector terminal publication" }
    }
    try {
        $Summary = Get-Content -LiteralPath $RunSummary -Raw | ConvertFrom-Json -Depth 100
        $Seal = Get-Content -LiteralPath $RunSeal -Raw | ConvertFrom-Json -Depth 100
    } catch {
        return [pscustomobject]@{ State = "terminal_invalid"; Summary = $null; Reason = "detector terminal JSON parse failure" }
    }
    if (
        $Summary.protocol -ne "syncg_public_meter_detector_training_v1" -or
        [int]$Summary.run.seed -ne $Seed -or
        $Seal.protocol -ne "syncg_public_meter_detector_training_seal_v1" -or
        [string]$Seal.summary_sha256 -ne (Get-Sha256 $RunSummary)
    ) {
        return [pscustomobject]@{ State = "terminal_invalid"; Summary = $Summary; Reason = "detector terminal identity/hash drift" }
    }
    if (
        $Summary.status -eq "complete" -and
        $Seal.status -eq "sealed" -and
        [bool]$Summary.validation.gate_pass -and
        [bool]$Seal.validation_gate_pass
    ) {
        return [pscustomobject]@{ State = "passed"; Summary = $Summary; Reason = "authenticated pass" }
    }
    if (
        $Summary.status -eq "complete_gate_failed" -and
        $Seal.status -eq "sealed_gate_failed" -and
        -not [bool]$Summary.validation.gate_pass -and
        -not [bool]$Seal.validation_gate_pass
    ) {
        return [pscustomobject]@{ State = "gate_failed"; Summary = $Summary; Reason = "authenticated frozen gate failure" }
    }
    return [pscustomobject]@{ State = "terminal_invalid"; Summary = $Summary; Reason = "detector terminal status mismatch" }
}

$CoreFailed = $false
Push-Location $ProjectRoot
try {
    if ($PreflightOnly) {
        & $Python -m py_compile $Builder $Fetcher $Trainer $FrontendModule
        if ($LASTEXITCODE -ne 0) {
            throw "Detector-chain Python preflight failed"
        }
        [void][scriptblock]::Create((Get-Content -LiteralPath $TrainingWrapper -Raw))
        [void][scriptblock]::Create((Get-Content -LiteralPath $PSCommandPath -Raw))
        Write-Output "syncg public meter detector event chain preflight passed"
        return
    }

    if (-not (Enter-ChainLock)) {
        Write-Output "An authenticated detector event chain is already running; no duplicate was started."
        return
    }
    Wait-PaperGate
    if (-not (Test-Path -LiteralPath $Pretrained -PathType Leaf)) {
        & $Python $Fetcher --output $Pretrained | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Public YOLO11n checkpoint fetch/authentication failed with code $LASTEXITCODE"
        }
    } else {
        & $Python $Fetcher --output $Pretrained | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Existing public YOLO11n checkpoint authentication failed with code $LASTEXITCODE"
        }
    }
    if (-not (Test-Path -LiteralPath $Corpus -PathType Container)) {
        & $Python $Builder --output-dir $Corpus
        if ($LASTEXITCODE -ne 0) {
            throw "Public SyncG detector corpus build failed with code $LASTEXITCODE"
        }
    } else {
        & $Python $Builder --output-dir $Corpus --verify-only | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Existing public detector corpus verification failed with code $LASTEXITCODE"
        }
    }

    $Terminal = Get-DetectorTerminalState
    if ($Terminal.State -eq "terminal_invalid") {
        throw $Terminal.Reason
    }
    if ($Terminal.State -eq "gate_failed") {
        $Metrics = $Terminal.Summary.validation.selected_box_metrics
        Send-ProgressBestEffort `
            -EventKey "syncg-public-meter-detector-seed-$Seed-gate-failed-v1" `
            -Message "共享仪表检测器公开独立验证已完成但未通过冻结门槛（seed=$Seed）：IoU50召回=$([math]::Round(100 * $Metrics.selected_box_iou50_recall, 2))%，框选择精度=$([math]::Round(100 * $Metrics.selected_box_precision, 2))%，mean IoU=$([math]::Round($Metrics.mean_selected_iou, 4))；该权重不会进入现场盲测。" `
            -Eta "这是冻结的终态结果；需另立公开数据方案后才能重训。"
        return
    }
    if ($Terminal.State -ne "passed") {
        $RunDir = Split-Path -Parent $RunSummary
        $Intent = Join-Path $OutputDir "seed_$Seed.run_intent.json"
        $SelectionClaim = Join-Path $RunDir "selection_claim.json"
        $LastCheckpoint = Join-Path $RunDir "weights\last.pt"
        $Resume = Test-Path -LiteralPath $RunDir -PathType Container
        if ($Resume -and (
            -not (Test-Path -LiteralPath $Intent -PathType Leaf) -or
            (
                -not (Test-Path -LiteralPath $SelectionClaim -PathType Leaf) -and
                -not (Test-Path -LiteralPath $LastCheckpoint -PathType Leaf)
            )
        )) {
            throw "Incomplete detector run lacks an authenticated run intent and resumable last.pt/selection claim; refusing ambiguous resume"
        }
        Send-ProgressBestEffort `
            -EventKey "paper-to-syncg-public-meter-detector-seed-$Seed-stage-v1" `
            -Message "论文公开结果链已完成，现切换到共享仪表检测器训练（仅 SyncG/train，seed=$Seed）；训练完成后将冻结前端再进入最终原图盲测。" `
            -Eta "RTX 4060 预计2.5–5小时（含公开校准与独立验证）。"
        $TrainingArguments = @{
            Seed = $Seed
            Workers = $Workers
            Corpus = $Corpus
            OutputDir = $OutputDir
            Pretrained = $Pretrained
            SuppressNotifications = $true
        }
        if ($Resume) {
            $TrainingArguments.Resume = $true
        }
        & $TrainingWrapper @TrainingArguments
        $Terminal = Get-DetectorTerminalState
        if ($Terminal.State -eq "gate_failed") {
            $Metrics = $Terminal.Summary.validation.selected_box_metrics
            Send-ProgressBestEffort `
                -EventKey "syncg-public-meter-detector-seed-$Seed-gate-failed-v1" `
                -Message "共享仪表检测器公开独立验证已完成但未通过冻结门槛（seed=$Seed）：IoU50召回=$([math]::Round(100 * $Metrics.selected_box_iou50_recall, 2))%，框选择精度=$([math]::Round(100 * $Metrics.selected_box_precision, 2))%，mean IoU=$([math]::Round($Metrics.mean_selected_iou, 4))；该权重不会进入现场盲测。" `
                -Eta "这是冻结的终态结果；需另立公开数据方案后才能重训。"
            return
        }
        if ($Terminal.State -ne "passed") {
            throw "Detector training wrapper returned without an authenticated passing terminal state: $($Terminal.Reason)"
        }
    }
    if (-not (Test-Path -LiteralPath $FrontendPlan -PathType Leaf)) {
        & $Python $FrontendModule `
            --freeze `
            --training-summary $RunSummary `
            --output-dir $FrontendOutput | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Public shared frontend freeze failed with code $LASTEXITCODE"
        }
    } else {
        & $Python $FrontendModule --verify --frontend-plan $FrontendPlan | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Existing public shared frontend verification failed with code $LASTEXITCODE"
        }
        $FrozenPlan = Get-Content -LiteralPath $FrontendPlan -Raw | ConvertFrom-Json -Depth 100
        if (
            [System.IO.Path]::GetFullPath([string]$FrozenPlan.selection_audit.training_summary.path) -ne
                [System.IO.Path]::GetFullPath($RunSummary) -or
            [string]$FrozenPlan.selection_audit.training_summary.sha256 -ne (Get-Sha256 $RunSummary)
        ) {
            throw "Existing frontend plan is valid for a different detector run/seed"
        }
    }
    $Summary = Get-Content -LiteralPath $RunSummary -Raw | ConvertFrom-Json -Depth 100
    $Metrics = $Summary.validation.selected_box_metrics
    Send-ProgressBestEffort `
        -EventKey "syncg-public-meter-frontend-seed-$Seed-frozen-complete-v1" `
        -Message "共享仪表检测前端已完成并冻结（仅 SyncG/train）：IoU50召回=$([math]::Round(100 * $Metrics.selected_box_iou50_recall, 2))%，框选择精度=$([math]::Round(100 * $Metrics.selected_box_precision, 2))%，mean IoU=$([math]::Round($Metrics.mean_selected_iou, 4))。下一阶段可按同一ROI启动五方法最终原图盲测。" `
        -Eta "五方法1200+原图一次性推理与封存预计3–8小时，随后评分与论文终表约1–2小时。"
} catch {
    $CoreFailed = $true
    $CoreError = $_
    Send-ProgressBestEffort `
        -EventKey "syncg-public-meter-detector-chain-seed-$Seed-unexpected-stop-v1" `
        -Message "共享仪表检测器训练/冻结链异常停止（seed=$Seed）：$($CoreError.Exception.Message)" `
        -Eta "先验证 run intent、last.pt/selection claim 与终态 seal，再决定安全恢复。"
    throw $CoreError
} finally {
    try {
        Exit-ChainLock
    } catch {
        if ($CoreFailed) {
            Write-Warning "Detector event-chain lock cleanup also failed: $($_.Exception.Message)"
        } else {
            Pop-Location
            throw
        }
    }
    Pop-Location
}

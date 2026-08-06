#requires -Version 7.0
[CmdletBinding()]
param(
    [int]$Seed = 20260819,
    [int]$Workers = 4,
    [string]$Corpus = "C:\pointer_read\syncg_meter_detector_public_v1",
    [string]$OutputDir = "C:\pointer_read\syncg_meter_detector_runs",
    [string]$Pretrained = "C:\pointer_read\public_pretrained\yolo11n-ultralytics-assets-v8.3.0.pt",
    [switch]$Resume,
    [switch]$PreflightOnly,
    [switch]$SuppressNotifications
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProgressPreference = "SilentlyContinue"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "PowerShell 7 or newer is required."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Trainer = Join-Path $PSScriptRoot "train_syncg_meter_detector.py"
$Reporter = Join-Path $PSScriptRoot "send_feishu_progress.ps1"
$RunDir = Join-Path $OutputDir "seed_$Seed"
$SummaryPath = Join-Path $RunDir "summary.json"
$SealPath = Join-Path $RunDir "seal.json"
$LogPath = Join-Path $OutputDir "seed_$Seed.log"
$LockPath = Join-Path $OutputDir "seed_$Seed.training.lock.json"
$LockOwned = $false

$RequiredArtifacts = @($Python, $Trainer, $Pretrained)
if (-not $SuppressNotifications) {
    $RequiredArtifacts += $Reporter
}
foreach ($Required in $RequiredArtifacts) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required detector artifact is absent: $Required"
    }
}
if (-not (Test-Path -LiteralPath $Corpus -PathType Container)) {
    throw "Public detector corpus is absent: $Corpus"
}

function Get-ProcessStartUtc {
    param([Parameter(Mandatory = $true)][int]$ProcessId)
    $Process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $Process) {
        return $null
    }
    return ([datetime]$Process.CreationDate).ToUniversalTime()
}

function Get-CommandLineSha256 {
    param([Parameter(Mandatory = $true)][string]$CommandLine)
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($CommandLine)
    $Digest = [System.Security.Cryptography.SHA256]::HashData($Bytes)
    return [Convert]::ToHexString($Digest).ToLowerInvariant()
}

function Remove-AuthenticatedStaleLock {
    if (-not (Test-Path -LiteralPath $LockPath -PathType Leaf)) {
        return
    }
    $Existing = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
    if (
        $Existing.protocol -ne "syncg_public_meter_detector_training_lock_v1" -or
        [int]$Existing.owner_pid -lt 1 -or
        [string]::IsNullOrWhiteSpace([string]$Existing.owner_started_at_utc)
    ) {
        throw "Detector writer lock lacks an authenticated owner identity"
    }
    $Expected = [datetime]::Parse([string]$Existing.owner_started_at_utc).ToUniversalTime()
    $Observed = Get-ProcessStartUtc -ProcessId ([int]$Existing.owner_pid)
    if ($null -ne $Observed -and [math]::Abs(($Observed - $Expected).TotalSeconds) -le 1.0) {
        throw "Detector writer lock is owned by live PID $($Existing.owner_pid)"
    }
    $HasChildPending = $Existing.PSObject.Properties.Name -contains "child_pending"
    $HasChildPid = $Existing.PSObject.Properties.Name -contains "child_pid"
    if ($HasChildPending -and [bool]$Existing.child_pending -and (-not $HasChildPid -or $null -eq $Existing.child_pid)) {
        throw "Detector wrapper died while child creation was pending; refusing automatic recovery until orphan-process audit"
    }
    if ($HasChildPid -and $null -ne $Existing.child_pid) {
        if (
            -not ($Existing.PSObject.Properties.Name -contains "child_started_at_utc") -or
            -not ($Existing.PSObject.Properties.Name -contains "child_command_sha256") -or
            [string]::IsNullOrWhiteSpace([string]$Existing.child_started_at_utc) -or
            [string]::IsNullOrWhiteSpace([string]$Existing.child_command_sha256)
        ) {
            throw "Detector lock contains a child PID without an authenticated child identity"
        }
        $ChildObserved = Get-ProcessStartUtc -ProcessId ([int]$Existing.child_pid)
        if ($null -ne $ChildObserved) {
            $ChildExpected = [datetime]::Parse([string]$Existing.child_started_at_utc).ToUniversalTime()
            $ChildProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$Existing.child_pid)"
            $ChildHash = Get-CommandLineSha256 -CommandLine ([string]$ChildProcess.CommandLine)
            if (
                [math]::Abs(($ChildObserved - $ChildExpected).TotalSeconds) -le 1.0 -and
                $ChildHash -eq [string]$Existing.child_command_sha256
            ) {
                throw "Authenticated detector Python child PID $($Existing.child_pid) is still running; refusing duplicate writer"
            }
        }
    }
    Remove-Item -LiteralPath $LockPath -Force
}

function New-OwnedLock {
    $Started = Get-ProcessStartUtc -ProcessId $PID
    if ($null -eq $Started) {
        throw "Cannot authenticate current detector wrapper process"
    }
    $Value = [ordered]@{
        protocol = "syncg_public_meter_detector_training_lock_v1"
        owner_pid = $PID
        owner_started_at_utc = $Started.ToString("o")
        seed = $Seed
        resume = [bool]$Resume
        child_pending = $false
        child_pid = $null
        child_started_at_utc = $null
        child_command_sha256 = $null
        created_at_utc = [datetime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Compress
    $Stream = [System.IO.File]::Open(
        $LockPath,
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
    $script:LockOwned = $true
}

function Update-OwnedLock {
    param(
        [Parameter(Mandatory = $true)][bool]$ChildPending,
        [System.Diagnostics.Process]$ChildProcess = $null
    )
    $Existing = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
    $OwnerObserved = Get-ProcessStartUtc -ProcessId $PID
    if (
        [int]$Existing.owner_pid -ne $PID -or
        $null -eq $OwnerObserved -or
        [math]::Abs((([datetime]::Parse([string]$Existing.owner_started_at_utc)).ToUniversalTime() - $OwnerObserved).TotalSeconds) -gt 1.0
    ) {
        throw "Refusing to update a detector lock not owned by this PID/start identity"
    }
    $Existing.child_pending = $ChildPending
    if ($null -ne $ChildProcess) {
        $ChildIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $($ChildProcess.Id)"
        if ($null -eq $ChildIdentity) {
            throw "Cannot authenticate newly started detector Python child"
        }
        $Existing.child_pid = $ChildProcess.Id
        $Existing.child_started_at_utc = ([datetime]$ChildIdentity.CreationDate).ToUniversalTime().ToString("o")
        $Existing.child_command_sha256 = Get-CommandLineSha256 -CommandLine ([string]$ChildIdentity.CommandLine)
    }
    $Payload = $Existing | ConvertTo-Json -Compress
    $Temporary = "$LockPath.tmp.$PID"
    [System.IO.File]::WriteAllText($Temporary, $Payload + "`n", [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $Temporary -Destination $LockPath -Force
}

function Remove-OwnedLock {
    if (-not $LockOwned -or -not (Test-Path -LiteralPath $LockPath -PathType Leaf)) {
        return
    }
    $Existing = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
    $Observed = Get-ProcessStartUtc -ProcessId $PID
    if (
        [int]$Existing.owner_pid -ne $PID -or
        $null -eq $Observed -or
        [math]::Abs((([datetime]::Parse([string]$Existing.owner_started_at_utc)).ToUniversalTime() - $Observed).TotalSeconds) -gt 1.0
    ) {
        throw "Refusing to remove a detector lock not owned by this PID/start identity"
    }
    Remove-Item -LiteralPath $LockPath -Force
    $script:LockOwned = $false
}

function Invoke-LoggedTrainer {
    param([Parameter(Mandatory = $true)][string[]]$ArgumentList)
    Update-OwnedLock -ChildPending $true
    $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $Python
    $StartInfo.WorkingDirectory = $ProjectRoot
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    foreach ($Argument in $ArgumentList) {
        $StartInfo.ArgumentList.Add([string]$Argument)
    }
    $Child = [System.Diagnostics.Process]::new()
    $Child.StartInfo = $StartInfo
    if (-not $Child.Start()) {
        throw "Failed to start detector Python trainer"
    }
    Update-OwnedLock -ChildPending $false -ChildProcess $Child
    $StandardOutput = $Child.StandardOutput.ReadToEndAsync()
    $StandardError = $Child.StandardError.ReadToEndAsync()
    $Child.WaitForExit()
    $OutputText = $StandardOutput.GetAwaiter().GetResult()
    $ErrorText = $StandardError.GetAwaiter().GetResult()
    foreach ($Text in @($OutputText, $ErrorText)) {
        if (-not [string]::IsNullOrWhiteSpace($Text)) {
            Write-Host $Text.TrimEnd()
            Add-Content -LiteralPath $LogPath -Value $Text.TrimEnd()
        }
    }
    return [int]$Child.ExitCode
}

function Send-ProgressBestEffort {
    param(
        [Parameter(Mandatory = $true)][string]$EventKey,
        [Parameter(Mandatory = $true)][string]$Message,
        [Parameter(Mandatory = $true)][string]$Eta
    )
    if ($SuppressNotifications) {
        return
    }
    try {
        & $Reporter -EventKey $EventKey -Message $Message -Eta $Eta
    } catch {
        Write-Warning "Detector notification is pending/retryable and did not alter scientific state: $($_.Exception.Message)"
    }
}

$CoreFailed = $false
Push-Location $ProjectRoot
try {
    if ($PreflightOnly) {
        & $Python $Trainer `
            --corpus $Corpus `
            --output-dir $OutputDir `
            --pretrained $Pretrained `
            --seed $Seed `
            --device 0 `
            --workers $Workers `
            --validate-only
        if ($LASTEXITCODE -ne 0) {
            throw "Detector preflight exited with code $LASTEXITCODE"
        }
        return
    }
    if (
        (Test-Path -LiteralPath $SummaryPath -PathType Leaf) -or
        (Test-Path -LiteralPath $SealPath -PathType Leaf)
    ) {
        throw "Terminal detector summary/seal exists; the supervisor must classify it before any new invocation"
    }
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $RunDir -PathType Container)) {
            throw "-Resume requires an existing incomplete detector run: $RunDir"
        }
    } elseif (Test-Path -LiteralPath $RunDir) {
        throw "Refusing to overwrite detector run without -Resume: $RunDir"
    }
    [void](New-Item -ItemType Directory -Force -Path $OutputDir)
    Remove-AuthenticatedStaleLock
    New-OwnedLock
    $Arguments = @(
        $Trainer,
        "--corpus", $Corpus,
        "--output-dir", $OutputDir,
        "--pretrained", $Pretrained,
        "--seed", "$Seed",
        "--device", "0",
        "--workers", "$Workers",
        "--run-formal"
    )
    if ($Resume) {
        $Arguments += "--resume"
    }
    $ExitCode = Invoke-LoggedTrainer -ArgumentList $Arguments
    if ($ExitCode -ne 0) {
        throw "Public meter detector training exited with code $ExitCode"
    }
    if (
        -not (Test-Path -LiteralPath $SummaryPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $SealPath -PathType Leaf)
    ) {
        throw "Detector training ended without an atomic summary/seal terminal pair"
    }
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json -Depth 100
    $Seal = Get-Content -LiteralPath $SealPath -Raw | ConvertFrom-Json -Depth 100
    $SummarySha = (Get-FileHash -LiteralPath $SummaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (
        $Summary.protocol -ne "syncg_public_meter_detector_training_v1" -or
        [int]$Summary.run.seed -ne $Seed -or
        $Seal.protocol -ne "syncg_public_meter_detector_training_seal_v1" -or
        [string]$Seal.summary_sha256 -ne $SummarySha
    ) {
        throw "Detector terminal summary/seal identity drift"
    }
    if (
        $Summary.status -eq "complete_gate_failed" -and
        $Seal.status -eq "sealed_gate_failed" -and
        -not [bool]$Summary.validation.gate_pass -and
        -not [bool]$Seal.validation_gate_pass
    ) {
        $Metrics = $Summary.validation.selected_box_metrics
        Send-ProgressBestEffort `
            -EventKey "syncg-public-meter-detector-seed-$Seed-gate-failed-v1" `
            -Message "共享仪表检测器公开独立验证已完成但未通过冻结门槛（seed=$Seed）：IoU50召回=$([math]::Round(100 * $Metrics.selected_box_iou50_recall, 2))%，框选择精度=$([math]::Round(100 * $Metrics.selected_box_precision, 2))%，mean IoU=$([math]::Round($Metrics.mean_selected_iou, 4))；该权重不会进入现场盲测。" `
            -Eta "这是冻结的终态结果；需另立公开数据方案后才能重训。"
        return
    }
    if (
        $Summary.status -ne "complete" -or
        $Seal.status -ne "sealed" -or
        -not [bool]$Summary.validation.gate_pass -or
        -not [bool]$Seal.validation_gate_pass
    ) {
        throw "Detector terminal status is neither authenticated pass nor authenticated gate failure"
    }
    $Metrics = $Summary.validation.selected_box_metrics
    Send-ProgressBestEffort `
        -EventKey "syncg-public-meter-detector-seed-$Seed-complete-v1" `
        -Message "共享仪表检测器训练与独立验证完成（seed=$Seed）：IoU50召回=$([math]::Round(100 * $Metrics.selected_box_iou50_recall, 2))%，框选择精度=$([math]::Round(100 * $Metrics.selected_box_precision, 2))%，mean IoU=$([math]::Round($Metrics.mean_selected_iou, 4))；已通过公开数据门槛。" `
        -Eta "冻结共享前端约5分钟，之后可进入最终原图盲测。"
} catch {
    $CoreFailed = $true
    $CoreError = $_
    $IsDuplicateWriter = $CoreError.Exception.Message -match "owned by live PID|still running; refusing duplicate writer"
    if (-not $PreflightOnly -and -not $IsDuplicateWriter) {
        Send-ProgressBestEffort `
            -EventKey "syncg-public-meter-detector-seed-$Seed-unexpected-stop-v1" `
            -Message "共享仪表检测器训练或验证异常停止（seed=$Seed）：$($CoreError.Exception.Message)" `
            -Eta "先验证 run intent、last.pt/selection claim 与终态 seal，再决定安全恢复。"
    }
    throw $CoreError
} finally {
    try {
        Remove-OwnedLock
    } catch {
        if ($CoreFailed) {
            Write-Warning "Detector lock cleanup also failed: $($_.Exception.Message)"
        } else {
            Pop-Location
            throw
        }
    }
    Pop-Location
}

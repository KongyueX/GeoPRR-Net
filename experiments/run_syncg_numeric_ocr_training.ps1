[CmdletBinding()]
param(
    [int]$Seed = 20260806,
    [int]$Workers = 2,
    [string]$Corpus = "C:\pointer_read\syncg_numeric_ocr_public_v1",
    [string]$OutputDir = "C:\pointer_read\syncg_numeric_ocr_runs",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "PowerShell 7 or newer is required."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Trainer = Join-Path $PSScriptRoot "train_syncg_numeric_ocr.py"
$Reporter = Join-Path $PSScriptRoot "send_feishu_progress.ps1"
$RunDir = Join-Path $OutputDir "seed_$Seed"
$LogPath = Join-Path $OutputDir "seed_$Seed.log"
$SummaryPath = Join-Path $RunDir "summary.json"
$LockPath = Join-Path $OutputDir "seed_$Seed.training.lock.json"
$LockOwned = $false

function Get-ProcessStartUtc {
    param([Parameter(Mandatory = $true)][int]$ProcessId)
    $Process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $Process) {
        return $null
    }
    return ([datetime]$Process.CreationDate).ToUniversalTime()
}

function Remove-AuthenticatedStaleLock {
    if (-not (Test-Path -LiteralPath $LockPath -PathType Leaf)) {
        return
    }
    if (-not $Resume) {
        throw "OCR writer lock exists; explicit -Resume is required: $LockPath"
    }
    $Existing = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
    if (
        $Existing.protocol -ne "syncg_numeric_ocr_training_lock_v1" -or
        [int]$Existing.owner_pid -lt 1 -or
        [string]::IsNullOrWhiteSpace([string]$Existing.owner_started_at_utc)
    ) {
        throw "OCR writer lock lacks an authenticated owner PID/start identity"
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
        throw "OCR writer lock is owned by live PID $($Existing.owner_pid)"
    }
    # The PID is absent or has been reused with a different start time.  Only
    # this authenticated stale-lock case is recoverable.
    Remove-Item -LiteralPath $LockPath -Force
}

function New-OwnedLock {
    $OwnerStart = Get-ProcessStartUtc -ProcessId $PID
    if ($null -eq $OwnerStart) {
        throw "Cannot authenticate current OCR wrapper process"
    }
    $Value = [ordered]@{
        protocol = "syncg_numeric_ocr_training_lock_v1"
        owner_pid = $PID
        owner_started_at_utc = $OwnerStart.ToString("o")
        seed = $Seed
        resume = [bool]$Resume
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
        throw "Refusing to remove an OCR lock not owned by this PID/start identity"
    }
    Remove-Item -LiteralPath $LockPath -Force
    $script:LockOwned = $false
}

foreach ($Required in @($Python, $Trainer, $Reporter)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required file is absent: $Required"
    }
}
if (-not (Test-Path -LiteralPath $Corpus -PathType Container)) {
    throw "Frozen OCR corpus is absent: $Corpus"
}
[void](New-Item -ItemType Directory -Force -Path $OutputDir)

try {
    if (Test-Path -LiteralPath $SummaryPath -PathType Leaf) {
        throw "Completed OCR summary exists; refusing overwrite: $SummaryPath"
    }
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $RunDir -PathType Container)) {
            throw "-Resume requires an existing incomplete OCR run: $RunDir"
        }
    } else {
        if (Test-Path -LiteralPath $RunDir) {
            throw "Refusing to overwrite OCR training run: $RunDir"
        }
        if (Test-Path -LiteralPath $LogPath) {
            throw "Refusing to overwrite OCR training log: $LogPath"
        }
    }
    Remove-AuthenticatedStaleLock
    New-OwnedLock

    $TrainerArguments = @(
        $Trainer,
        "--corpus", $Corpus,
        "--output-dir", $OutputDir,
        "--device", "cuda",
        "--component", "both",
        "--seed", "$Seed",
        "--workers", "$Workers",
        "--run-formal"
    )
    if ($Resume) {
        $TrainerArguments += "--resume"
        & $Python @TrainerArguments 2>&1 |
            Tee-Object -FilePath $LogPath -Append
    } else {
        & $Python @TrainerArguments 2>&1 |
            Tee-Object -FilePath $LogPath
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Numeric OCR training exited with code $LASTEXITCODE"
    }

    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Numeric OCR training ended without summary.json"
    }
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $Recognizer = $Summary.components.recognizer.validation
    $Detector = $Summary.components.detector.validation
    $Message = @(
        "专用仪表数字 OCR 首种子完成（seed=$Seed）。",
        "识别器独立验证 exact=$([math]::Round(100 * $Recognizer.exact_accuracy, 2))%，字符准确率=$([math]::Round(100 * $Recognizer.character_accuracy, 2))%；",
        "检测器独立验证 pixel Dice=$([math]::Round(100 * $Detector.pixel_dice_at_0_40, 2))%；",
        "耗时=$([math]::Round($Summary.elapsed_seconds / 3600, 2)) 小时。",
        "下一步将接入 GARC 完整量程恢复并按冻结门槛决定升级 DBNet++/PGNet 或 SVTR/PP-OCRv3。"
    ) -join " "
    & $Reporter `
        -EventKey "syncg-numeric-ocr-seed-$Seed-complete" `
        -Message $Message `
        -Eta "完整量程接入与独立验证约2–4小时；若需升级成熟OCR骨干，额外约6–18小时。"
} catch {
    $ErrorText = $_.Exception.Message
    try {
        & $Reporter `
            -EventKey $(if ($Resume) { "syncg-numeric-ocr-seed-$Seed-resume-error-v1" } else { "syncg-numeric-ocr-seed-$Seed-error" }) `
            -Message "专用仪表数字 OCR 训练异常停止（seed=$Seed）：$ErrorText" `
            -Eta "定位异常后重新评估。"
    } catch {
        Add-Content -LiteralPath $LogPath -Value "Feishu exception report failed: $($_.Exception.Message)"
    }
    throw
} finally {
    try {
        Remove-OwnedLock
    } catch {
        Add-Content -LiteralPath $LogPath -Value "OCR lock cleanup failed: $($_.Exception.Message)"
    }
}

param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 200,

    [string[]]$Seeds = @("20262020", "20262021", "20262022"),

    [string]$Python = ".\.venv\Scripts\python.exe",

    [string]$Manifest = ".\artifacts\manifests\syncg_train.jsonl",

    [Parameter(Mandatory)]
    [string]$OuterSplit,

    [string]$VdnSource = ".\artifacts\vendor\VectorDetectionNetwork",

    [Parameter(Mandatory)]
    [string]$RoiManifest,

    [Parameter(Mandatory)]
    [string]$PixelReference,

    [string]$RunRoot = ".\artifacts\runs\geoprr_vdn_matched",

    [ValidateRange(1, 168)]
    [int]$PerSeedTimeoutHours = 18,

    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

function Resolve-ExistingPath {
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [Parameter(Mandatory)]
        [string]$Label
    )
    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    }
    else {
        Join-Path $ProjectRoot $Path
    }
    if (-not (Test-Path -LiteralPath $candidate)) {
        throw "$Label does not exist: $candidate"
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Resolve-OutputPath {
    param([Parameter(Mandatory)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Path))
}

function Invoke-TrainingWithTimeout {
    param(
        [Parameter(Mandatory)][string]$PythonPath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$StdoutPath,
        [Parameter(Mandatory)][string]$StderrPath,
        [Parameter(Mandatory)][int]$TimeoutSeconds,
        [Parameter(Mandatory)][int]$Seed
    )
    $process = Start-Process `
        -FilePath $PythonPath `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -WindowStyle Hidden `
        -PassThru
    $started = [DateTimeOffset]::UtcNow
    $nextHeartbeat = $started
    while (-not $process.HasExited) {
        $now = [DateTimeOffset]::UtcNow
        $elapsed = ($now - $started).TotalSeconds
        if ($elapsed -ge $TimeoutSeconds) {
            Stop-Process -Id $process.Id -Force
            $process.WaitForExit()
            throw "VDN seed $Seed exceeded the fixed $TimeoutSeconds-second timeout"
        }
        if ($now -ge $nextHeartbeat) {
            Write-Host (
                "VDN seed={0} alive pid={1} elapsed_minutes={2:N1}" -f `
                    $Seed, $process.Id, ($elapsed / 60.0)
            )
            $nextHeartbeat = $now.AddMinutes(1)
        }
        Start-Sleep -Seconds 30
        $process.Refresh()
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        Write-Host "VDN seed $Seed failed; stderr tail follows"
        if (Test-Path -LiteralPath $StderrPath) {
            Get-Content -LiteralPath $StderrPath -Tail 80
        }
        throw "VDN seed $Seed exited with code $($process.ExitCode); no retry was attempted"
    }
}

$PythonPath = Resolve-ExistingPath -Path $Python -Label "Python"
$ManifestPath = Resolve-ExistingPath -Path $Manifest -Label "SyncG manifest"
$OuterSplitPath = Resolve-ExistingPath -Path $OuterSplit -Label "outer split"
$VdnSourcePath = Resolve-ExistingPath -Path $VdnSource -Label "VDN source"
$RoiManifestPath = Resolve-ExistingPath -Path $RoiManifest -Label "ROI manifest"
$PixelReferencePath = Resolve-ExistingPath -Path $PixelReference -Label "pixel reference"
$RunRootPath = Resolve-OutputPath -Path $RunRoot

$SeedValues = @(
    $Seeds |
        ForEach-Object { $_ -split "," } |
        ForEach-Object { [int]$_.Trim() }
)
if ($SeedValues.Count -ne 3 -or @($SeedValues | Sort-Object -Unique).Count -ne 3) {
    throw "The formal comparison requires exactly three distinct seeds"
}
New-Item -ItemType Directory -Path $RunRootPath -Force | Out-Null

$PredictionPaths = @()
foreach ($Seed in $SeedValues) {
    $SeedDirectory = Join-Path $RunRootPath "seed_$Seed"
    $SeedDirectoryExists = Test-Path -LiteralPath $SeedDirectory
    if ($SeedDirectoryExists -and -not $Resume) {
        throw "Fresh formal seed directory already exists: $SeedDirectory"
    }
    if (-not $SeedDirectoryExists) {
        New-Item -ItemType Directory -Path $SeedDirectory | Out-Null
    }
    $ResumeThisSeed = $Resume -and $SeedDirectoryExists
    $LogStem = "train"
    if ($ResumeThisSeed) {
        $Attempt = 1
        while (Test-Path -LiteralPath (Join-Path $SeedDirectory "train.resume_$Attempt.stdout.log")) {
            $Attempt += 1
        }
        $LogStem = "train.resume_$Attempt"
    }
    $StdoutPath = Join-Path $SeedDirectory "$LogStem.stdout.log"
    $StderrPath = Join-Path $SeedDirectory "$LogStem.stderr.log"
    $TrainingArguments = @(
        "-u",
        "-m", "experiments.train_vdn_syncg",
        "--manifest", $ManifestPath,
        "--outer-split", $OuterSplitPath,
        "--matched-geoprr-split",
        "--vdn-source", $VdnSourcePath,
        "--output-dir", $SeedDirectory,
        "--device", "cuda:0",
        "--epochs", [string]$Epochs,
        "--batch-size", "8",
        "--workers", "4",
        "--image-size", "384",
        "--learning-rate", "0.001",
        "--seed", [string]$Seed
    )
    if ($ResumeThisSeed) {
        $TrainingArguments += "--resume"
    }
    Invoke-TrainingWithTimeout `
        -PythonPath $PythonPath `
        -Arguments $TrainingArguments `
        -WorkingDirectory $ProjectRoot `
        -StdoutPath $StdoutPath `
        -StderrPath $StderrPath `
        -TimeoutSeconds ($PerSeedTimeoutHours * 3600) `
        -Seed $Seed

    $CheckpointPath = Join-Path $SeedDirectory "last.pt"
    if (-not (Test-Path -LiteralPath $CheckpointPath)) {
        throw "VDN terminal checkpoint is missing after successful training: $CheckpointPath"
    }
    $PredictionPath = Join-Path $SeedDirectory "syncg_predictions.jsonl"
    if (Test-Path -LiteralPath $PredictionPath) {
        if (-not $Resume) {
            throw "VDN matched prediction already exists: $PredictionPath"
        }
        Write-Host "Reusing completed VDN prediction: $PredictionPath"
        $PredictionPaths += $PredictionPath
        continue
    }
    & $PythonPath -u -m experiments.evaluate_geoprr_vdn_matched `
        --checkpoint $CheckpointPath `
        --vdn-source $VdnSourcePath `
        --roi-manifest $RoiManifestPath `
        --syncg-manifest $ManifestPath `
        --output $PredictionPath `
        --seed $Seed `
        --expected-epochs $Epochs `
        --device cuda:0 `
        --batch-size 32
    if ($LASTEXITCODE -ne 0) {
        throw "VDN matched evaluation failed for seed $Seed; no retry was attempted"
    }
    $PredictionPaths += $PredictionPath
}

$GeoPRRPaths = @(
    (Join-Path $ProjectRoot "artifacts\runs\unified_pointer_reader\seed_20262020\full\syncg.json"),
    (Join-Path $ProjectRoot "artifacts\runs\unified_pointer_reader\seed_20262021\full\syncg.json"),
    (Join-Path $ProjectRoot "artifacts\runs\unified_pointer_reader\seed_20262022\full\syncg.json")
)
foreach ($Path in $GeoPRRPaths) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "GeoPRR formal evaluation is missing: $Path"
    }
}

$SummaryPath = Join-Path $RunRootPath "summary.json"
if (Test-Path -LiteralPath $SummaryPath) {
    if ($Resume) {
        Write-Host "VDN matched experiment is already complete: $SummaryPath"
        exit 0
    }
    throw "VDN matched summary already exists: $SummaryPath"
}
$SummaryArguments = @(
    "-u",
    "-m", "experiments.summarize_geoprr_vdn_matched"
)
foreach ($Path in $PredictionPaths) {
    $SummaryArguments += @("--vdn", $Path)
}
foreach ($Path in $GeoPRRPaths) {
    $SummaryArguments += @("--geoprr", $Path)
}
$SummaryArguments += @(
    "--pixel-reference", $PixelReferencePath,
    "--output", $SummaryPath,
    "--bootstrap-replicates", "20000",
    "--bootstrap-seed", "20262020"
)
& $PythonPath @SummaryArguments
if ($LASTEXITCODE -ne 0) {
    throw "VDN matched summary failed; no retry was attempted"
}
Write-Host "VDN matched experiment complete: $SummaryPath"

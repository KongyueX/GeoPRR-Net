param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$SourceRoot = "artifacts\vendor\Detect-and-read-meters",
    [string]$Proxy = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repository = "https://github.com/shuyansy/Detect-and-read-meters.git"
$Commit = "e5e16803de2c06b3dfb248df16ee91c05879cd61"
$CheckpointUrl = "https://drive.google.com/file/d/1sHmEEf9E0_kvL0LW1S5Y5jjFgjx_O5Dj/view"
$CheckpointSha256 = "6F5BCFD5F57C535DBC4DA827BA7538E1C305F33D3DA84D43215B125500A4300A"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = (Resolve-Path (Join-Path $ProjectRoot $Python)).Path
$SourcePath = [System.IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot $SourceRoot)
)
$CheckpointPath = Join-Path $SourcePath "model\meter_data\textgraph_vgg_100.pth"

function Invoke-Git {
    param([string[]]$Arguments)

    if ($Proxy) {
        & git -c "http.proxy=$Proxy" @Arguments
    }
    else {
        & git @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "git failed with exit code $LASTEXITCODE"
    }
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path $SourcePath)) {
        New-Item -ItemType Directory -Force -Path (
            Split-Path -Parent $SourcePath
        ) | Out-Null
        Invoke-Git -Arguments @(
            "clone",
            "--branch", "v2",
            "--single-branch",
            $Repository,
            $SourcePath
        )
    }
    if (-not (Test-Path (Join-Path $SourcePath ".git"))) {
        throw "$SourcePath exists but is not a Git checkout"
    }
    Invoke-Git -Arguments @("-C", $SourcePath, "checkout", "--detach", $Commit)
    $ActualCommit = (& git -C $SourcePath rev-parse HEAD).Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $Commit) {
        throw "HARR checkout is $ActualCommit, expected $Commit"
    }
    $TrackedChanges = & git -C $SourcePath status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or $TrackedChanges) {
        throw "HARR checkout contains tracked modifications"
    }

    New-Item -ItemType Directory -Force -Path (
        Split-Path -Parent $CheckpointPath
    ) | Out-Null
    $NeedsDownload = $true
    if (Test-Path $CheckpointPath) {
        $ExistingHash = (
            Get-FileHash $CheckpointPath -Algorithm SHA256
        ).Hash.ToUpperInvariant()
        $NeedsDownload = $ExistingHash -ne $CheckpointSha256
    }
    if ($NeedsDownload) {
        if ($Proxy) {
            $env:HTTP_PROXY = $Proxy
            $env:HTTPS_PROXY = $Proxy
        }
        & $PythonPath -m gdown --fuzzy $CheckpointUrl -O $CheckpointPath
        if ($LASTEXITCODE -ne 0) {
            throw "gdown failed with exit code $LASTEXITCODE"
        }
    }
    $ActualHash = (
        Get-FileHash $CheckpointPath -Algorithm SHA256
    ).Hash.ToUpperInvariant()
    if ($ActualHash -ne $CheckpointSha256) {
        throw "HARR checkpoint SHA-256 is $ActualHash, expected $CheckpointSha256"
    }

    Write-Host "HARR source: $SourcePath"
    Write-Host "HARR commit: $ActualCommit"
    Write-Host "HARR checkpoint: $CheckpointPath"
    Write-Host "HARR checkpoint SHA-256: $ActualHash"
}
finally {
    Pop-Location
}

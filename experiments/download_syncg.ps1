[CmdletBinding()]
param(
    [string]$OutputDir = "datasets\SyncG_archive",
    [string]$ExtractDir = "datasets\SyncG",
    [string]$SocksProxy = "socks5h://127.0.0.1:7890",
    [string]$DataProxy = "http://127.0.0.1:7890",
    [ValidateRange(1, 16)]
    [int]$Connections = 1,
    [ValidateRange(1, 100)]
    [int]$MaxUrlRefreshes = 24,
    [string[]]$IncludeFiles = @(),
    [switch]$Extract
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo = "YihengDeng/syncG"
$commit = "14204c3f5b35d160fafa39ad195cd5a63e6e9c12"
$files = @(
    @{
        Name = "syncG.zip"
        Size = [int64]3490141123
        Sha256 = "cdf0e09757813ca2599e857f248fdb33c9519336a7d43641b0ca6e232b8e0b28"
    },
    @{
        Name = "syncG.z01"
        Size = [int64]4294967296
        Sha256 = "5caf646d4537137d2334d2100c5222381ac0ebc127edb3b0fd26ce17da70c83a"
    },
    @{
        Name = "syncG.z02"
        Size = [int64]4294967296
        Sha256 = "958f86879b3cc09e10ef3b558c73b417b51fd163f9a7cb695cefd6154cf5f464"
    },
    @{
        Name = "syncG.z03"
        Size = [int64]4294967296
        Sha256 = "89a4496deaf0bacbd81a0267c8f14d204a01c85e0b204365fd43aa515dfa1e82"
    },
    @{
        Name = "syncG.z04"
        Size = [int64]4294967296
        Sha256 = "389ec29213349f5e83507b341144f7f5bca636e9d20b68531f91d3139b162d1a"
    }
)

if ($IncludeFiles.Count -gt 0) {
    $knownNames = @($files | ForEach-Object { $_.Name })
    $unknownNames = @($IncludeFiles | Where-Object { $_ -notin $knownNames })
    if ($unknownNames.Count -gt 0) {
        throw (
            "Unknown SyncG archive part(s): " + ($unknownNames -join ", ") +
            ". Expected one or more of: " + ($knownNames -join ", ")
        )
    }
    $files = @($files | Where-Object { $_.Name -in $IncludeFiles })
}

function Find-Executable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [string[]]$Fallbacks = @()
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    foreach ($candidate in $Fallbacks) {
        $matches = Get-ChildItem -Path $candidate -Filter $Name -File -Recurse `
            -ErrorAction SilentlyContinue
        if ($matches) {
            return ($matches | Select-Object -First 1 -ExpandProperty FullName)
        }
    }
    throw "$Name was not found. Install it before running this script."
}

function Invoke-Curl {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $script:CurlPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed with exit code $LASTEXITCODE"
    }
}

function Get-SignedUrl {
    param([Parameter(Mandatory = $true)][string]$Name)

    # Xet CDN signatures are short-lived. A cache-busting value guarantees a
    # fresh Location header when a multi-GB transfer outlives one signature.
    $refresh = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $resolveUrl = "https://huggingface.co/datasets/$repo/resolve/$commit/${Name}?download=true&refresh=$refresh"
    $curlArgs = @(
        "-fsSI",
        "--connect-timeout", "20",
        "--max-time", "90",
        "--retry", "5",
        "--retry-all-errors"
    )
    if ($SocksProxy) {
        $curlArgs += @("--proxy", $SocksProxy)
    }
    $headers = & $script:CurlPath @curlArgs $resolveUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve the signed CDN URL for $Name"
    }
    $location = $headers -split "`r?`n" |
        Where-Object { $_ -match "^Location:" } |
        Select-Object -First 1
    if (-not $location) {
        throw "Hugging Face returned no signed CDN URL for $Name"
    }
    return ($location -replace "^Location:\s*", "")
}

function Test-File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int64]$Size,
        [Parameter(Mandatory = $true)][string]$Sha256
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    if ((Get-Item -LiteralPath $Path).Length -ne $Size) {
        return $false
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    return $actual -eq $Sha256
}

$script:CurlPath = Find-Executable -Name "curl.exe"
$aria2Path = Find-Executable -Name "aria2c.exe" -Fallbacks @(
    "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$outputRoot = (Resolve-Path -LiteralPath $OutputDir).Path

foreach ($metadataName in @("README.md", "train_id.txt", "test_id.txt")) {
    $destination = Join-Path $outputRoot $metadataName
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        continue
    }
    $url = "https://huggingface.co/datasets/$repo/resolve/$commit/$metadataName"
    $curlArgs = @(
        "-fL",
        "--connect-timeout", "20",
        "--max-time", "180",
        "--retry", "5",
        "--retry-all-errors",
        "--output", $destination
    )
    if ($SocksProxy) {
        $curlArgs += @("--proxy", $SocksProxy)
    }
    Invoke-Curl @curlArgs $url
}

foreach ($file in $files) {
    $destination = Join-Path $outputRoot $file.Name
    if (Test-File -Path $destination -Size $file.Size -Sha256 $file.Sha256) {
        Write-Host "verified, skipping $($file.Name)"
        continue
    }

    $verified = $false
    for ($attempt = 1; $attempt -le $MaxUrlRefreshes; $attempt++) {
        $signedUrl = Get-SignedUrl -Name $file.Name
        Write-Host "downloading $($file.Name) ($($file.Size) bytes), signed URL $attempt/$MaxUrlRefreshes"
        $ariaArgs = @(
            "--dir=$outputRoot",
            "--out=$($file.Name)",
            "--continue=true",
            "--split=$Connections",
            "--max-connection-per-server=$Connections",
            "--min-split-size=16M",
            "--file-allocation=none",
            "--disable-ipv6=true",
            "--check-integrity=true",
            "--checksum=sha-256=$($file.Sha256)",
            "--max-tries=5",
            "--retry-wait=5",
            "--timeout=30"
        )
        if ($DataProxy) {
            $ariaArgs += "--all-proxy=$DataProxy"
        }
        & $aria2Path @ariaArgs $signedUrl
        if (Test-File -Path $destination -Size $file.Size -Sha256 $file.Sha256) {
            $verified = $true
            break
        }
        Write-Warning (
            "transfer for $($file.Name) stopped before verification " +
            "(aria2 exit $LASTEXITCODE); refreshing the signed URL"
        )
    }
    if (-not $verified) {
        throw (
            "size or SHA-256 verification failed for $($file.Name) after " +
            "$MaxUrlRefreshes signed URLs"
        )
    }
}

if ($Extract) {
    $sevenZip = Find-Executable -Name "7z.exe" -Fallbacks @(
        "$env:ProgramFiles\7-Zip",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    )
    New-Item -ItemType Directory -Force -Path $ExtractDir | Out-Null
    $extractRoot = (Resolve-Path -LiteralPath $ExtractDir).Path
    & $sevenZip x (Join-Path $outputRoot "syncG.zip") "-o$extractRoot" -y
    if ($LASTEXITCODE -ne 0) {
        throw "7-Zip extraction failed with exit code $LASTEXITCODE"
    }
}

Write-Host "SyncG files verified below $outputRoot"

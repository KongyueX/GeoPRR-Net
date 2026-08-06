[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Message,

    [ValidateNotNullOrEmpty()]
    [string]$EventKey = "project-progress",

    [string]$Eta = "",

    [switch]$Force
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "Feishu progress reporting requires PowerShell 7 or newer."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SecretPath = Join-Path $ProjectRoot "artifacts\private\feishu_webhook.dpapi"
$SigningSecretPath = Join-Path (
    $ProjectRoot
) "artifacts\private\feishu_signing_secret.dpapi"
$StatePath = Join-Path $ProjectRoot "artifacts\supervisors\feishu_progress_state.json"

if (-not (Test-Path -LiteralPath $SecretPath -PathType Leaf)) {
    throw "Encrypted Feishu webhook is not configured: $SecretPath"
}

$Encrypted = (Get-Content -LiteralPath $SecretPath -Raw).Trim()
$ProtectedHook = ConvertTo-SecureString $Encrypted
$Credential = [System.Management.Automation.PSCredential]::new(
    "feishu-webhook",
    $ProtectedHook
)
$Webhook = $Credential.GetNetworkCredential().Password

$BaseMessage = $Message
$DeliveredMessage = if ([string]::IsNullOrWhiteSpace($Eta)) {
    $BaseMessage
} else {
    "$BaseMessage`n`n预计时间：$($Eta.Trim())"
}
$MessageBytes = [System.Text.Encoding]::UTF8.GetBytes($DeliveredMessage)
$MessageHash = [Convert]::ToHexString(
    [System.Security.Cryptography.SHA256]::HashData($MessageBytes)
).ToLowerInvariant()

$StateDirectory = Split-Path -Parent $StatePath
[void](New-Item -ItemType Directory -Force -Path $StateDirectory)
$StatePathHash = [Convert]::ToHexString(
    [System.Security.Cryptography.SHA256]::HashData(
        [System.Text.Encoding]::UTF8.GetBytes(
            [System.IO.Path]::GetFullPath($StatePath).ToLowerInvariant()
        )
    )
).Substring(0, 24)
$Mutex = [System.Threading.Mutex]::new(
    $false,
    "Local\PointerMeterFeishuProgress-$StatePathHash"
)
$LockAcquired = $false
try {
    try {
        $LockAcquired = $Mutex.WaitOne([TimeSpan]::FromSeconds(30))
    } catch [System.Threading.AbandonedMutexException] {
        $LockAcquired = $true
    }
    if (-not $LockAcquired) {
        throw "Timed out waiting for the Feishu progress-state lock."
    }

    $State = @{}
    if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        try {
            $Loaded = Get-Content -LiteralPath $StatePath -Raw |
                ConvertFrom-Json -AsHashtable
            if ($null -ne $Loaded) {
                $State = $Loaded
            }
        } catch {
            Write-Warning (
                "Ignoring unreadable Feishu progress state; the next " +
                "successful delivery will replace it."
            )
        }
    }
    if (-not $Force -and $State[$EventKey] -eq $MessageHash) {
        Write-Output "Feishu progress event unchanged; skipped: $EventKey"
        return
    }

    $PayloadObject = @{
        msg_type = "text"
        content = @{
            text = $DeliveredMessage
        }
    }
    if (Test-Path -LiteralPath $SigningSecretPath -PathType Leaf) {
        $EncryptedSigningSecret = (
            Get-Content -LiteralPath $SigningSecretPath -Raw
        ).Trim()
        $ProtectedSignature = ConvertTo-SecureString $EncryptedSigningSecret
        $SigningCredential = [System.Management.Automation.PSCredential]::new(
            "feishu-signing-secret",
            $ProtectedSignature
        )
        $SigningSecret = $SigningCredential.GetNetworkCredential().Password
        $Timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds().ToString()
        $StringToSign = "$Timestamp`n$SigningSecret"
        $Hmac = [System.Security.Cryptography.HMACSHA256]::new(
            [System.Text.Encoding]::UTF8.GetBytes($StringToSign)
        )
        try {
            $SignatureBytes = $Hmac.ComputeHash([byte[]]::new(0))
        } finally {
            $Hmac.Dispose()
        }
        $PayloadObject.timestamp = $Timestamp
        $PayloadObject.sign = [Convert]::ToBase64String($SignatureBytes)
    }
    $Payload = $PayloadObject | ConvertTo-Json -Depth 4 -Compress
    $Response = Invoke-RestMethod `
        -Method Post `
        -Uri $Webhook `
        -ContentType "application/json; charset=utf-8" `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($Payload))
    if ($Response.code -ne 0) {
        if ($Response.code -eq 19021) {
            throw (
                "Feishu signature verification failed (19021). Configure " +
                "the bot signing secret or disable signature verification."
            )
        }
        throw "Feishu webhook rejected the message: code=$($Response.code)"
    }

    $State[$EventKey] = $MessageHash
    $Temporary = Join-Path (
        $StateDirectory
    ) "feishu_progress_state.$PID.$([guid]::NewGuid().ToString('N')).tmp"
    try {
        $State | ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath $Temporary -Encoding utf8
        Move-Item -LiteralPath $Temporary -Destination $StatePath -Force
    } finally {
        if (Test-Path -LiteralPath $Temporary -PathType Leaf) {
            Remove-Item -LiteralPath $Temporary -Force
        }
    }
    Write-Output "Feishu progress event delivered: $EventKey"
} finally {
    if ($LockAcquired) {
        [void]$Mutex.ReleaseMutex()
    }
    $Mutex.Dispose()
}

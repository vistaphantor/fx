param(
    [string]$ProjectRoot = "C:\Users\Admin\Desktop\fx",
    [string]$TerminalPath = "C:\Program Files (x86)\HFM Metatrader 4\terminal.exe",
    [string]$DataPath = "C:\Users\Admin\AppData\Roaming\MetaQuotes\Terminal\91DD9F8C3ED3720A3F71FC024EC6B483",
    [string]$Symbol = "XAUUSD",
    [string]$Period = "M1",
    [switch]$NoRestartExisting
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TerminalPath)) {
    throw "MT4 terminal not found: $TerminalPath"
}

$envPath = Join-Path $ProjectRoot ".env"
if (-not (Test-Path -LiteralPath $envPath)) {
    throw ".env not found: $envPath"
}

function Read-DotEnvValue {
    param([string]$Path, [string]$Key)
    $line = Get-Content -LiteralPath $Path | Where-Object { $_ -match "^\s*$([regex]::Escape($Key))=" } | Select-Object -First 1
    if (-not $line) {
        return ""
    }
    return ($line -split "=", 2)[1].Trim()
}

$login = Read-DotEnvValue -Path $envPath -Key "MT4_LOGIN"
if (-not $login) {
    $login = Read-DotEnvValue -Path $envPath -Key "HFM_LOGIN"
}
$password = Read-DotEnvValue -Path $envPath -Key "MT4_PASSWORD"
if (-not $password) {
    $password = Read-DotEnvValue -Path $envPath -Key "HFM_PASSWORD"
}
$server = Read-DotEnvValue -Path $envPath -Key "MT4_SERVER"
if (-not $server) {
    $server = Read-DotEnvValue -Path $envPath -Key "HFM_SERVER"
}

if (-not $login -or -not $password -or -not $server) {
    throw "Missing MT4 login, password, or server in .env"
}

$expertSource = Join-Path $ProjectRoot "tools\mt4\FxPythonBridge.mq4"
$compiledExpertSource = Join-Path $ProjectRoot "tools\mt4\FxPythonBridge.ex4"
$expertDestDir = Join-Path $DataPath "MQL4\Experts"
$expertDest = Join-Path $expertDestDir "FxPythonBridge.mq4"
$compiledExpertDest = Join-Path $expertDestDir "FxPythonBridge.ex4"
New-Item -ItemType Directory -Force -Path $expertDestDir | Out-Null
Copy-Item -LiteralPath $expertSource -Destination $expertDest -Force

$metaEditor = Join-Path (Split-Path -Parent $TerminalPath) "metaeditor.exe"
if (Test-Path -LiteralPath $metaEditor) {
    $compileLog = Join-Path $ProjectRoot ".runtime\mt4-bridge-compile.log"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $compileLog) | Out-Null
    Start-Process -FilePath $metaEditor -ArgumentList @("/compile:$expertDest", "/log:$compileLog") -Wait -WindowStyle Hidden
}
if (Test-Path -LiteralPath $compiledExpertSource) {
    Copy-Item -LiteralPath $compiledExpertSource -Destination $compiledExpertDest -Force
}

$runtimeDir = Join-Path $ProjectRoot ".runtime"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$startupConfig = Join-Path $runtimeDir "mt4-autotrade-start.ini"

@"
Login=$login
Password=$password
Server=$server
ExpertsEnable=true
ExpertsTrades=true
ExpertsDllImport=false
ExpertsExpImport=true

Symbol=$Symbol
Period=$Period
Expert=FxPythonBridge
"@ | Set-Content -LiteralPath $startupConfig -Encoding ASCII

$commonFiles = Join-Path (Join-Path $env:APPDATA "MetaQuotes\Terminal\Common") "Files"
New-Item -ItemType Directory -Force -Path $commonFiles | Out-Null
$heartbeat = Join-Path $commonFiles "fx_bridge_heartbeat.csv"

if (-not $NoRestartExisting) {
    Get-Process terminal -ErrorAction SilentlyContinue | ForEach-Object {
        $processPath = ""
        try {
            $processPath = $_.MainModule.FileName
        } catch {
            $processPath = ""
        }

        if ($processPath -eq $TerminalPath) {
            Write-Host "Closing existing MT4 so startup Expert=FxPythonBridge is applied..."
            $_.CloseMainWindow() | Out-Null
            if (-not $_.WaitForExit(7000)) {
                Stop-Process -Id $_.Id -Force
            }
        }
    }
}

Remove-Item -LiteralPath $heartbeat -Force -ErrorAction SilentlyContinue

Start-Process -FilePath $TerminalPath -ArgumentList @($startupConfig)
Write-Host "MT4 started with FxPythonBridge auto-attach requested."
Write-Host "Startup config: $startupConfig"
Write-Host "Bridge files folder: $commonFiles"

$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $heartbeat) {
        $heartbeatText = Get-Content -LiteralPath $heartbeat -ErrorAction SilentlyContinue | Select-Object -First 1
        Write-Host "FxPythonBridge heartbeat detected: $heartbeatText"
        exit 0
    }
    Start-Sleep -Seconds 1
}

Write-Warning "FxPythonBridge heartbeat was not detected. In MT4, check Navigator > Expert Advisors and the Experts tab for startup errors."

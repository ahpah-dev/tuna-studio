[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [switch]$OpenBrowser
)

$ErrorActionPreference = 'Stop'
$appDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$serverFile = Join-Path $appDirectory 'server.py'
$dataDirectory = [System.IO.Path]::GetFullPath((Join-Path $appDirectory 'data'))
$baseUrl = 'http://127.0.0.1:{0}' -f $Port

function Test-LocalPort {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $attempt = $client.ConnectAsync('127.0.0.1', $Port)
        if (-not $attempt.Wait(1000)) { return $false }
        return $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Get-VerifiedTunaStatus {
    try {
        $response = Invoke-WebRequest -Uri ($baseUrl + '/api/status') -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ne 200 -or [string]$response.Headers['Server'] -notmatch '^TunaStudio/') {
            return $null
        }
        $status = $response.Content | ConvertFrom-Json
        if (-not $status.data_directory -or $status.offline -isnot [bool]) { return $null }
        $reportedDirectory = [System.IO.Path]::GetFullPath([string]$status.data_directory)
        if (-not [string]::Equals($reportedDirectory.TrimEnd('\', '/'), $dataDirectory.TrimEnd('\', '/'), [System.StringComparison]::OrdinalIgnoreCase)) {
            return $null
        }
        return $status
    } catch {
        return $null
    }
}

function Show-TunaUrl {
    Write-Host ('Tuna Studio: ' + $baseUrl)
    if ($OpenBrowser) { Start-Process -FilePath $baseUrl }
}

if (-not (Test-Path -LiteralPath $serverFile -PathType Leaf)) {
    throw 'server.py was not found beside this launcher. Keep the app folder together.'
}

if (Test-LocalPort) {
    if ($null -ne (Get-VerifiedTunaStatus)) {
        Write-Host 'Tuna Studio is already running for this app folder.'
        Show-TunaUrl
        return
    }
    throw ('Port {0} is occupied by another service or another Tuna data directory. Nothing was stopped. Use -Port with a free port, or stop that service yourself.' -f $Port)
}

$venvPython = Join-Path $appDirectory '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $pythonFile = $venvPython
} else {
    $pythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $pythonCommand) {
        throw 'Python was not found. Install Python 3.10 or later, or create .venv in this app folder, then try again.'
    }
    $pythonFile = $pythonCommand.Source
}

$logDirectory = Join-Path $dataDirectory 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stdoutLog = Join-Path $logDirectory ('server-' + $logStamp + '.stdout.log')
$stderrLog = Join-Path $logDirectory ('server-' + $logStamp + '.stderr.log')
$serverArguments = @('-u', ('"{0}"' -f $serverFile), '--host', '127.0.0.1', '--port', [string]$Port, '--data-dir', ('"{0}"' -f $dataDirectory))
$serverProcess = Start-Process -FilePath $pythonFile -ArgumentList $serverArguments -WorkingDirectory $appDirectory -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
$ready = $false
for ($attemptNumber = 0; $attemptNumber -lt 40; $attemptNumber++) {
    $serverProcess.Refresh()
    if ($serverProcess.HasExited) { break }
    if ($null -ne (Get-VerifiedTunaStatus)) { $ready = $true; break }
    Start-Sleep -Milliseconds 250
}

if (-not $ready) {
    $serverProcess.Refresh()
    if (-not $serverProcess.HasExited) {
        # This handle belongs only to the child created above; never stop a port owner.
        $serverProcess.Kill()
        $serverProcess.WaitForExit(3000) | Out-Null
    }
    throw ('Tuna Studio did not expose its verified local endpoint. Read {0} and {1}. No unrelated process was stopped.' -f $stdoutLog, $stderrLog)
}

@{
    process_id = $serverProcess.Id
    started_at = $serverProcess.StartTime.ToUniversalTime().ToString('o')
    python = $pythonFile
    server = $serverFile
    port = $Port
    data_directory = $dataDirectory
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dataDirectory ('server-process-' + $Port + '.json')) -Encoding UTF8

Write-Host ('Started local server, process ID {0}.' -f $serverProcess.Id)
Write-Host ('Logs: ' + $logDirectory)
Show-TunaUrl

<#
.SYNOPSIS
    Start TradeSense locally in the background. It keeps running after this window closes.

.DESCRIPTION
    Binds to 127.0.0.1 by default (this machine only). Output goes to logs\tradesense.log.
    Stop it with scripts\stop-local.cmd (or stop-local.ps1).

.EXAMPLE
    scripts\start-local.cmd                    # http://127.0.0.1:8000
    scripts\start-local.cmd -Port 8080
    scripts\start-local.cmd -BindHost 0.0.0.0  # reachable from your network: NO authentication!
#>
param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "logs"
$log = Join-Path $logDir "tradesense.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    $owner = Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue
    Write-Host "Port $Port is already in use by $($owner.ProcessName) (PID $($busy[0].OwningProcess))."
    Write-Host "If that is TradeSense, it is already running: http://127.0.0.1:$Port"
    exit 1
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    throw "uv was not found on PATH. Install it from https://docs.astral.sh/uv/"
}

Add-Content -Path $log -Value "`r`n===== $(Get-Date -Format s) starting TradeSense on ${BindHost}:$Port =====" -Encoding UTF8

# Win32_Process.Create starts the server outside this console's process tree, so it keeps
# running when the window (or the tool that launched it) closes.
$commandLine = "cmd.exe /c `"set `"PYTHONUTF8=1`" && `"$($uv.Source)`" run tradesense " +
               "--host $BindHost --port $Port >> `"$log`" 2>&1`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $commandLine; CurrentDirectory = $root }
if ($result.ReturnValue -ne 0) {
    throw "Could not start TradeSense (Win32_Process.Create returned $($result.ReturnValue))."
}

$probeHost = if ($BindHost -eq "0.0.0.0") { "127.0.0.1" } else { $BindHost }
$url = "http://${probeHost}:$Port"
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 2
        if ($health.status -eq "ok") {
            Write-Host "TradeSense is running at $url"
            Write-Host "MCP endpoint:  $url/mcp"
            Write-Host "Logs:          $log"
            Write-Host "Stop with:     scripts\stop-local.cmd"
            exit 0
        }
    } catch {
        # not up yet
    }
}

Write-Host "TradeSense did not report healthy within 45 seconds. Last log lines:"
Get-Content -Path $log -Tail 40
exit 1

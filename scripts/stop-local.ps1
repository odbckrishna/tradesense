<#
.SYNOPSIS
    Stop the local TradeSense server started by start-local (default port 8000).

.EXAMPLE
    scripts\stop-local.cmd
    scripts\stop-local.cmd -Port 8080
#>
param(
    [int]$Port = 8000
)

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listeners) {
    Write-Host "Nothing is listening on port $Port."
    exit 0
}

foreach ($processId in ($listeners.OwningProcess | Sort-Object -Unique)) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
    if ($process.CommandLine -notmatch "tradesense") {
        Write-Host "Port $Port belongs to '$($process.Name)' (PID $processId), not TradeSense. Leaving it alone."
        continue
    }
    # The server runs as cmd.exe -> uv.exe -> python.exe; stop the whole chain.
    $chain = @($processId)
    $parent = $process.ParentProcessId
    while ($parent) {
        $parentProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $parent"
        if (-not $parentProcess -or $parentProcess.CommandLine -notmatch "tradesense") { break }
        $chain += $parent
        $parent = $parentProcess.ParentProcessId
    }
    foreach ($id in $chain) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Stopped TradeSense (PIDs $($chain -join ', '))."
}

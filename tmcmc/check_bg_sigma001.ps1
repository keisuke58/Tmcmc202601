# Check background execution status for sigma=0.001
# Usage: .\check_bg_sigma001.ps1

$PidFile = "run_bg_sigma001.pid"

Write-Host "================================================================================"
Write-Host "Background Execution Status (sigma=0.001)"
Write-Host "================================================================================"
Write-Host ""

if (Test-Path $PidFile) {
    $ProcessId = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($ProcessId) {
        $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($Process) {
            Write-Host "Status: RUNNING" -ForegroundColor Green
            Write-Host "PID: $($Process.Id)"
            Write-Host "Process Name: $($Process.ProcessName)"
            Write-Host "CPU Time: $($Process.CPU)"
            Write-Host "Memory: $([math]::Round($Process.WorkingSet / 1MB, 2)) MB"
            Write-Host "Start Time: $($Process.StartTime)"
            Write-Host ""
            
            # Check for latest run directory
            $RunDirs = Get-ChildItem "_runs" -Directory | Where-Object { $_.Name -match "^\d{8}_\d{6}_" } | Sort-Object LastWriteTime -Descending
            if ($RunDirs.Count -gt 0) {
                $LatestRun = $RunDirs[0]
                Write-Host "Latest Run Directory: $($LatestRun.Name)"
                
                # Check for log file
                $LogFile = Join-Path $LatestRun.FullName "run.log"
                if (Test-Path $LogFile) {
                    Write-Host "Log File: $LogFile"
                    Write-Host ""
                    Write-Host "Last 10 lines of log:"
                    Write-Host "----------------------------------------"
                    Get-Content $LogFile -Tail 10
                } else {
                    Write-Host "Log file not found yet (may still be initializing)"
                }
            }
        } else {
            Write-Host "Status: COMPLETED or TERMINATED" -ForegroundColor Yellow
            Write-Host "PID: $Pid (process not found)"
            Write-Host ""
            Write-Host "Checking for output directories..."
            $RunDirs = Get-ChildItem "_runs" -Directory | Where-Object { $_.Name -match "^\d{8}_\d{6}_" } | Sort-Object LastWriteTime -Descending
            if ($RunDirs.Count -gt 0) {
                $LatestRun = $RunDirs[0]
                Write-Host "Latest Run Directory: $($LatestRun.Name)"
                Write-Host "Created: $($LatestRun.CreationTime)"
                Write-Host "Modified: $($LatestRun.LastWriteTime)"
            }
        }
    } else {
        Write-Host "Status: PID file is empty" -ForegroundColor Red
    }
} else {
    Write-Host "Status: PID file not found" -ForegroundColor Red
    Write-Host "Background process may not have started"
}

Write-Host ""
Write-Host "================================================================================"

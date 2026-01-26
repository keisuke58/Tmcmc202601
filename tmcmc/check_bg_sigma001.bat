@echo off
REM Check background execution status for sigma=0.001

echo ================================================================================
echo Background Execution Status (sigma=0.001)
echo ================================================================================
echo.

if exist run_bg_sigma001.pid (
    echo PID file found: run_bg_sigma001.pid
    type run_bg_sigma001.pid
    echo.
    echo Checking process status...
    powershell -Command "Get-Process -Id (Get-Content 'run_bg_sigma001.pid' -ErrorAction SilentlyContinue) -ErrorAction SilentlyContinue | Format-List Id, ProcessName, StartTime, CPU, WorkingSet"
) else (
    echo PID file not found: run_bg_sigma001.pid
    echo Background process may not have started
)

echo.
echo Checking for latest run directories...
powershell -Command "Get-ChildItem '_runs' -Directory | Where-Object { $_.Name -match '^\d{8}_\d{6}_' } | Sort-Object LastWriteTime -Descending | Select-Object -First 3 | Format-Table Name, CreationTime, LastWriteTime -AutoSize"

echo.
echo ================================================================================
pause

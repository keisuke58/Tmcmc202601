# case2_main.py background runner for Windows
# Note: Process will terminate if PC shuts down

param(
    [string]$Mode = "debug",
    [double]$SigmaObs = 0.0001,
    [int]$NParticles = 5000,
    [string]$Models = "M1,M2,M3",
    [int]$Seed = 42
)

Write-Host "================================================================================"
Write-Host "Starting case2_main.py in background"
Write-Host "================================================================================"
Write-Host ""

# Change to script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Setup log directory
$LogDir = "tmcmc\_runs\bg_logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# Create log file with timestamp
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "case2_${Mode}_sigma${SigmaObs}_np${NParticles}_${Timestamp}.log"
$PidFile = Join-Path $LogDir "case2_bg.pid"

Write-Host "Log file: $LogFile"
Write-Host "PID file: $PidFile"
Write-Host ""

# Build command
$PythonCmd = "python -m main.case2_main --mode $Mode --sigma-obs $SigmaObs --n-particles $NParticles --models $Models --seed $Seed"

Write-Host "Command: $PythonCmd"
Write-Host ""

# Start background process
Write-Host "Starting background process..."
Write-Host ""

try {
    # Start process in background with log redirection
    $Process = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "$PythonCmd > `"$LogFile`" 2>&1" `
        -WindowStyle Hidden `
        -PassThru
    
    # Save PID
    $Process.Id | Out-File -FilePath $PidFile -Encoding ASCII
    
    Write-Host ""
    Write-Host "================================================================================"
    Write-Host "Background process started"
    Write-Host "================================================================================"
    Write-Host ""
    Write-Host "Process ID: $($Process.Id)"
    Write-Host "Log file: $LogFile"
    Write-Host "PID file: $PidFile"
    Write-Host ""
    Write-Host "To monitor progress:"
    Write-Host "  Get-Content $LogFile -Tail 50 -Wait"
    Write-Host ""
    Write-Host "To check if running:"
    Write-Host "  Get-Process -Id $($Process.Id)"
    Write-Host ""
    Write-Host "WARNING: Process will terminate if PC shuts down."
    Write-Host "         Keep PC running until execution completes."
    Write-Host ""
    Write-Host "================================================================================"
    
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to start background process" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

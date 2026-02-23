# バックグラウンド実行中の run_pipeline.py を停止

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $ScriptDir "run_pipeline_bg.pid"

Write-Host "================================================================================"
Write-Host "run_pipeline.py バックグラウンドプロセスの停止"
Write-Host "================================================================================"
Write-Host ""

if (-not (Test-Path $PidFile)) {
    Write-Host "エラー: PIDファイルが見つかりません" -ForegroundColor Red
    Write-Host "実行中のプロセスはありません"
    exit 1
}

$Pid = Get-Content $PidFile -ErrorAction SilentlyContinue

if (-not $Pid) {
    Write-Host "エラー: PIDファイルが空です" -ForegroundColor Red
    Remove-Item $PidFile -Force
    exit 1
}

# プロセスの存在確認
$Process = Get-Process -Id $Pid -ErrorAction SilentlyContinue

if (-not $Process) {
    Write-Host "警告: プロセスは既に終了しています (PID: $Pid)" -ForegroundColor Yellow
    Remove-Item $PidFile -Force
    exit 0
}

Write-Host "プロセスを停止します (PID: $Pid)"
Write-Host "プロセス名: $($Process.ProcessName)"
Write-Host ""
Write-Host "続行しますか? (Y/N): " -NoNewline
$Response = Read-Host

if ($Response -ne "Y" -and $Response -ne "y") {
    Write-Host "キャンセルしました"
    exit 0
}

try {
    Stop-Process -Id $Pid -Force
    Write-Host ""
    Write-Host "プロセスを停止しました" -ForegroundColor Green
    
    # PIDファイルを削除
    Remove-Item $PidFile -Force
    $StatusFile = Join-Path $ScriptDir "run_pipeline_bg_status.txt"
    if (Test-Path $StatusFile) {
        Remove-Item $StatusFile -Force
    }
    
    Write-Host "PIDファイルを削除しました"
    
} catch {
    Write-Host ""
    Write-Host "エラー: プロセスの停止に失敗しました" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

Write-Host ""
Write-Host "================================================================================"

# バックグラウンド実行中の run_pipeline.py の状態を確認

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $ScriptDir "run_pipeline_bg.pid"
$StatusFile = Join-Path $ScriptDir "run_pipeline_bg_status.txt"

Write-Host "================================================================================"
Write-Host "run_pipeline.py バックグラウンド実行状態確認"
Write-Host "================================================================================"
Write-Host ""

if (-not (Test-Path $PidFile)) {
    Write-Host "状態: 実行中のプロセスが見つかりません" -ForegroundColor Yellow
    Write-Host "PIDファイルが存在しません: $PidFile"
    exit 0
}

$Pid = Get-Content $PidFile -ErrorAction SilentlyContinue

if (-not $Pid) {
    Write-Host "状態: PIDファイルが空です" -ForegroundColor Yellow
    exit 0
}

# プロセスの存在確認
$Process = Get-Process -Id $Pid -ErrorAction SilentlyContinue

if (-not $Process) {
    Write-Host "状態: プロセスは実行されていません (PID: $Pid)" -ForegroundColor Red
    Write-Host ""
    Write-Host "PIDファイルを削除しますか? (Y/N): " -NoNewline
    $Response = Read-Host
    if ($Response -eq "Y" -or $Response -eq "y") {
        Remove-Item $PidFile -Force
        if (Test-Path $StatusFile) {
            Remove-Item $StatusFile -Force
        }
        Write-Host "PIDファイルを削除しました"
    }
    exit 0
}

Write-Host "状態: 実行中" -ForegroundColor Green
Write-Host ""

# ステータス情報の表示
if (Test-Path $StatusFile) {
    Write-Host "実行情報:"
    Get-Content $StatusFile
    Write-Host ""
}

# プロセス情報
Write-Host "プロセス情報:"
Write-Host "  PID: $($Process.Id)"
Write-Host "  プロセス名: $($Process.ProcessName)"
Write-Host "  CPU時間: $($Process.CPU)"
Write-Host "  メモリ使用量: $([math]::Round($Process.WorkingSet64 / 1MB, 2)) MB"
Write-Host "  開始時刻: $($Process.StartTime)"
Write-Host ""

# 実行ディレクトリから最新のrun_idを探す
$RootDir = Split-Path -Parent $ScriptDir
$RunsRoot = Join-Path $RootDir "tmcmc\_runs"

if (Test-Path $RunsRoot) {
    $LatestRun = Get-ChildItem -Path $RunsRoot -Directory | 
        Sort-Object LastWriteTime -Descending | 
        Select-Object -First 1
    
    if ($LatestRun) {
        Write-Host "最新の実行ディレクトリ:"
        Write-Host "  $($LatestRun.FullName)"
        Write-Host "  最終更新: $($LatestRun.LastWriteTime)"
        Write-Host ""
        
        # ログファイルの確認
        $LogFile = Join-Path $LatestRun.FullName "run.log"
        if (Test-Path $LogFile) {
            $LogSize = (Get-Item $LogFile).Length
            Write-Host "ログファイル: $LogFile"
            Write-Host "  サイズ: $([math]::Round($LogSize / 1KB, 2)) KB"
            Write-Host ""
            
            # ログの最後の数行を表示
            Write-Host "ログの最後の10行:"
            Write-Host "  " + ("-" * 70)
            Get-Content $LogFile -Tail 10 | ForEach-Object {
                Write-Host "  $_"
            }
            Write-Host "  " + ("-" * 70)
        }
        
        # REPORT.mdの確認
        $ReportFile = Join-Path $LatestRun.FullName "REPORT.md"
        if (Test-Path $ReportFile) {
            Write-Host ""
            Write-Host "完了: REPORT.md が生成されました!" -ForegroundColor Green
            Write-Host "  パス: $ReportFile"
        }
    }
}

Write-Host ""
Write-Host "================================================================================"

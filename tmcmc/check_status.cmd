@echo off
REM バックグラウンド実行中のプロセスの状態を確認（cmd用）

cd /d "%~dp0\.."

echo ================================================================================
echo バックグラウンド実行状態確認
echo ================================================================================
echo.

REM PIDファイルの確認
if exist "tmcmc\run_pipeline_bg.pid" (
    echo PIDファイルから読み込み中...
    powershell -NoProfile -Command "$pid = (Get-Content 'tmcmc\run_pipeline_bg.pid' -Raw).Trim(); if ($pid) { Write-Host 'PID:' $pid; $proc = Get-Process -Id ([int]$pid) -ErrorAction SilentlyContinue; if ($proc) { Write-Host '状態: 実行中' -ForegroundColor Green; Write-Host 'プロセス名:' $proc.ProcessName; Write-Host 'CPU時間:' $proc.CPU; Write-Host 'メモリ:' ([math]::Round($proc.WorkingSet64/1MB,2)) 'MB'; Write-Host '開始時刻:' $proc.StartTime } else { Write-Host '状態: プロセスが見つかりません' -ForegroundColor Red } } else { Write-Host 'PIDファイルが空です' }"
) else (
    echo PIDファイルが見つかりません
    echo 実行中のプロセスはありません
)

echo.
echo ================================================================================
echo 最新の実行ディレクトリ
echo ================================================================================
echo.

REM 最新の実行ディレクトリを探す
for /f "delims=" %%i in ('dir /b /ad /o-d "tmcmc\_runs" 2^>nul') do (
    set LATEST_RUN=%%i
    goto :found
)
:found
if defined LATEST_RUN (
    echo 実行ID: %LATEST_RUN%
    echo.
    
    if exist "tmcmc\_runs\%LATEST_RUN%\run.log" (
        echo ログファイルの最後の20行:
        echo --------------------------------------------------------------------------------
        powershell -Command "Get-Content 'tmcmc\_runs\%LATEST_RUN%\run.log' -Tail 20"
        echo --------------------------------------------------------------------------------
    ) else (
        echo ログファイルが見つかりません
    )
    
    if exist "tmcmc\_runs\%LATEST_RUN%\REPORT.md" (
        echo.
        echo [完了] REPORT.md が生成されました!
    )
) else (
    echo 実行ディレクトリが見つかりません
)

echo.
pause

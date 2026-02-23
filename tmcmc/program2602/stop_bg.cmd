@echo off
REM バックグラウンド実行中のプロセスを停止（cmd用）

cd /d "%~dp0\.."

if not exist "tmcmc\run_pipeline_bg.pid" (
    echo エラー: PIDファイルが見つかりません
    echo 実行中のプロセスはありません
    pause
    exit /b 1
)

set /p PID=<tmcmc\run_pipeline_bg.pid

echo ================================================================================
echo バックグラウンドプロセスの停止
echo ================================================================================
echo.
echo PID: %PID%
echo.

powershell -Command "$proc = Get-Process -Id %PID% -ErrorAction SilentlyContinue; if ($proc) { Write-Host 'プロセス名: ' $proc.ProcessName; Write-Host ''; Write-Host 'プロセスを停止しますか? (Y/N): ' -NoNewline; $response = Read-Host; if ($response -eq 'Y' -or $response -eq 'y') { Stop-Process -Id %PID% -Force; Write-Host ''; Write-Host 'プロセスを停止しました' -ForegroundColor Green; Remove-Item 'tmcmc\run_pipeline_bg.pid' -Force -ErrorAction SilentlyContinue; Write-Host 'PIDファイルを削除しました' } else { Write-Host 'キャンセルしました' } } else { Write-Host '警告: プロセスは既に終了しています' -ForegroundColor Yellow; Remove-Item 'tmcmc\run_pipeline_bg.pid' -Force -ErrorAction SilentlyContinue }"

echo.
pause

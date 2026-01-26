@echo off
REM ログをリアルタイムで監視（cmd用）

cd /d "%~dp0\.."

REM 最新の実行ディレクトリを探す
for /f "delims=" %%i in ('dir /b /ad /o-d "tmcmc\_runs" 2^>nul') do (
    set LATEST_RUN=%%i
    goto :found
)
:found

if not defined LATEST_RUN (
    echo エラー: 実行ディレクトリが見つかりません
    pause
    exit /b 1
)

if not exist "tmcmc\_runs\%LATEST_RUN%\run.log" (
    echo エラー: ログファイルが見つかりません
    pause
    exit /b 1
)

echo ================================================================================
echo ログ監視: tmcmc\_runs\%LATEST_RUN%\run.log
echo ================================================================================
echo Ctrl+C で終了
echo.

powershell -Command "Get-Content 'tmcmc\_runs\%LATEST_RUN%\run.log' -Tail 50 -Wait"

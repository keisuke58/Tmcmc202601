@echo off
REM sweep_m1.sh の実行状況を確認するバッチファイル

setlocal enabledelayedexpansion

REM プロジェクトルートに移動
cd /d "%~dp0"
cd ..

set LOG_DIR=tmcmc\_runs\sweep_logs
set PID_FILE=%LOG_DIR%\sweep_m1.pid

echo ================================================================================
echo sweep_m1.sh 実行状況確認
echo ================================================================================
echo.

REM PIDファイルを確認
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    echo PIDファイル: %PID_FILE%
    echo プロセスID: %PID%
    echo.
    
    REM プロセスが実行中か確認
    tasklist /FI "PID eq %PID%" 2>nul | findstr /I "%PID%" >nul
    if %ERRORLEVEL% EQU 0 (
        echo ステータス: 実行中
    ) else (
        echo ステータス: 終了済み（または見つかりません）
    )
) else (
    echo PIDファイルが見つかりません: %PID_FILE%
    echo.
    echo 実行中のbashプロセスを確認:
    tasklist | findstr bash
)

echo.
echo ================================================================================
echo 最新のログファイル:
echo ================================================================================
if exist "%LOG_DIR%" (
    for /f "delims=" %%f in ('dir /b /o-d "%LOG_DIR%\sweep_m1_*.log" 2^>nul') do (
        echo %%f
        echo   サイズ: 
        for %%s in ("%LOG_DIR%\%%f") do echo %%~zs bytes
        echo   最終更新:
        for %%t in ("%LOG_DIR%\%%f") do echo %%~zt
        goto :found_log
    )
    echo ログファイルが見つかりません
) else (
    echo ログディレクトリが見つかりません: %LOG_DIR%
)

:found_log
echo.
echo ================================================================================
echo スイープ結果ディレクトリ:
echo ================================================================================
if exist "tmcmc\_runs" (
    dir /b /ad tmcmc\_runs\sweep_m1_* 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo スイープ結果ディレクトリが見つかりません
    )
) else (
    echo tmcmc\_runs ディレクトリが見つかりません
)

echo.
exit /b 0

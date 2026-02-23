@echo off
REM sweep_m1.py をバックグラウンドで実行するバッチファイル（Python版）
REM 電源が切れても続行できるように、ログをファイルに出力

setlocal enabledelayedexpansion

echo ================================================================================
echo sweep_m1.py バックグラウンド実行開始
echo ================================================================================
echo.

REM プロジェクトルートに移動
cd /d "%~dp0"
cd ..

REM ログディレクトリとファイル名を設定
set LOG_DIR=tmcmc\_runs\sweep_logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM タイムスタンプ付きログファイル名（PowerShellで生成）
set PID_FILE=%LOG_DIR%\sweep_m1.pid

echo PIDファイル: %PID_FILE%
echo.

REM バックグラウンドで実行（PowerShellを使用）
echo バックグラウンド実行を開始します...
echo.

REM PowerShellスクリプトを実行
powershell -ExecutionPolicy Bypass -File "%~dp0run_sweep_m1_bg.ps1"

if %ERRORLEVEL% EQU 0 (
    REM PIDを取得
    if exist "%PID_FILE%" (
        set /p PID=<"%PID_FILE%"
    )
    
    echo.
    echo ================================================================================
    echo バックグラウンド実行を開始しました
    echo ================================================================================
    echo.
    echo ログファイル: %LOG_FILE%
    echo PIDファイル: %PID_FILE%
    echo.
    echo 進捗確認方法:
    echo   1. ログファイルを確認:
    echo      type %LOG_FILE%
    echo.
    echo   2. 実行中のプロセスを確認:
    echo      tasklist | findstr python
    echo.
    echo   3. スイープ結果ディレクトリを確認:
    echo      dir tmcmc\_runs\sweep_m1_* /b
    echo.
    echo   4. ステータス確認:
    echo      tmcmc\check_sweep_status.bat
    echo.
    echo ================================================================================
) else (
    echo.
    echo エラー: バックグラウンド実行の開始に失敗しました
    exit /b 1
)

exit /b 0

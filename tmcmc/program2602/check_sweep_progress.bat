@echo off
REM スイープ実行の進捗を確認するバッチファイル（CMD用）

setlocal enabledelayedexpansion

REM プロジェクトルートに移動
cd /d "%~dp0"
cd ..

set LOG_DIR=tmcmc\_runs\sweep_logs
set PID_FILE=%LOG_DIR%\sweep_m1.pid

echo ================================================================================
echo スイープ実行進捗確認
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
    echo 実行中のPythonプロセスを確認:
    tasklist | findstr python
)

echo.
echo ================================================================================
echo 最新のログファイル（最後の30行）:
echo ================================================================================
if exist "%LOG_DIR%" (
    for /f "delims=" %%f in ('dir /b /o-d "%LOG_DIR%\sweep_m1_*.log" 2^>nul') do (
        echo ログファイル: %%f
        echo.
        powershell -Command "Get-Content '%LOG_DIR%\%%f' -Tail 30"
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
    echo 実行中のスイープ:
    dir /b /ad tmcmc\_runs\sweep_m1_* 2>nul | findstr /V "_failed"
    if %ERRORLEVEL% NEQ 0 (
        echo スイープ結果ディレクトリが見つかりません
    )
    echo.
    echo 各スイープの完了状況:
    for /d %%d in (tmcmc\_runs\sweep_m1_*) do (
        if exist "%%d\sweep_summary.csv" (
            echo   [完了] %%d
        ) else (
            echo   [実行中] %%d
            REM 各ジョブの状況を確認
            for /d %%j in ("%%d\sig*") do (
                if exist "%%j\REPORT.md" (
                    echo     [OK] %%~nxj
                ) else (
                    echo     [実行中] %%~nxj
                )
            )
        )
    )
) else (
    echo tmcmc\_runs ディレクトリが見つかりません
)

echo.
echo ================================================================================
echo 進捗確認コマンド:
echo ================================================================================
echo   1. ログファイルを確認:
echo      type %LOG_DIR%\sweep_m1_*.log
echo.
echo   2. 実行中のプロセスを確認:
echo      tasklist | findstr python
echo.
echo   3. スイープ結果を確認:
echo      dir tmcmc\_runs\sweep_m1_* /s /b
echo.
exit /b 0

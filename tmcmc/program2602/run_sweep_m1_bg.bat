@echo off
REM sweep_m1.sh をバックグラウンドで実行するバッチファイル
REM 電源が切れても続行できるように、ログをファイルに出力

setlocal enabledelayedexpansion

echo ================================================================================
echo sweep_m1.sh バックグラウンド実行開始
echo ================================================================================
echo.

REM プロジェクトルートに移動
cd /d "%~dp0"
cd ..

REM ログディレクトリとファイル名を設定
set LOG_DIR=tmcmc\_runs\sweep_logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM タイムスタンプ付きログファイル名
for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set "dt=%%a"
set LOG_FILE=%LOG_DIR%\sweep_m1_%dt:~0,8%_%dt:~8,6%.log
set PID_FILE=%LOG_DIR%\sweep_m1.pid

echo ログファイル: %LOG_FILE%
echo PIDファイル: %PID_FILE%
echo.

REM Git Bashのパスを確認（一般的なインストール場所）
set GIT_BASH=
if exist "C:\Program Files\Git\bin\bash.exe" (
    set GIT_BASH=C:\Program Files\Git\bin\bash.exe
) else if exist "C:\Program Files (x86)\Git\bin\bash.exe" (
    set GIT_BASH=C:\Program Files (x86)\Git\bin\bash.exe
) else (
    REM 環境変数PATHから探す
    where bash.exe >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set GIT_BASH=bash.exe
    )
)

if "%GIT_BASH%"=="" (
    echo エラー: Git Bash が見つかりません。
    echo Git Bash をインストールするか、WSL を使用してください。
    echo.
    echo WSLを使用する場合、以下のコマンドを手動で実行してください:
    echo   wsl bash tmcmc\sweep_m1.sh ^> %LOG_FILE% 2^>^&1
    exit /b 1
)

echo Git Bash が見つかりました: %GIT_BASH%
echo.

REM バックグラウンドで実行（PowerShellを使用）
echo バックグラウンド実行を開始します...
echo.

REM PowerShellでStart-Processを使用してバックグラウンド実行
powershell -Command "Start-Process -FilePath '%GIT_BASH%' -ArgumentList '-c', 'cd /c/Users/nishioka/Neuer\ Ordner/tmcmc_docs && bash tmcmc/sweep_m1.sh > tmcmc/_runs/sweep_logs/sweep_m1_$(date +%%Y%%m%%d_%%H%%M%%S).log 2>&1' -WindowStyle Hidden -PassThru | Select-Object -ExpandProperty Id | Out-File -FilePath '%PID_FILE%' -Encoding ASCII"

if %ERRORLEVEL% EQU 0 (
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
    echo      tasklist | findstr bash
    echo.
    echo   3. スイープ結果ディレクトリを確認:
    echo      dir tmcmc\_runs\sweep_m1_* /b
    echo.
    echo ================================================================================
) else (
    echo.
    echo エラー: バックグラウンド実行の開始に失敗しました
    exit /b 1
)

exit /b 0

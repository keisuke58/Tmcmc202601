@echo off
REM case2_main.py をバックグラウンドで実行するバッチファイル
REM PCをシャットダウンするとプロセスは終了します

echo ================================================================================
echo case2_main.py バックグラウンド実行開始
echo ================================================================================
echo.

REM プロジェクトルートに移動
cd /d "%~dp0"

REM PowerShellスクリプトを実行
powershell -ExecutionPolicy Bypass -File "%~dp0run_case2_bg.ps1" -Mode debug -SigmaObs 0.0001 -NParticles 5000

if %ERRORLEVEL% EQU 0 (
    echo.
    echo バックグラウンド実行を開始しました
    echo.
    echo 注意: PCをシャットダウンするとプロセスは終了します。
    echo       長時間実行が必要な場合は、実行が完了するまでPCを起動したままにしてください。
) else (
    echo.
    echo エラー: バックグラウンド実行の開始に失敗しました
)

pause

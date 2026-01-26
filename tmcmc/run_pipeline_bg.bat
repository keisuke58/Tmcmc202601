@echo off
REM run_pipeline.py をバックグラウンドで実行するバッチファイル
REM 使用例:
REM   run_pipeline_bg.bat
REM   run_pipeline_bg.bat debug 42 M1 5000 30 5

setlocal

REM デフォルト値
set MODE=debug
set SEED=42
set MODELS=
set NPARTICLES=
set NSTAGES=
set NMUTATIONSTEPS=
set NCHAINS=

REM 引数の解析
if not "%~1"=="" set MODE=%~1
if not "%~2"=="" set SEED=%~2
if not "%~3"=="" set MODELS=%~3
if not "%~4"=="" set NPARTICLES=%~4
if not "%~5"=="" set NSTAGES=%~5
if not "%~6"=="" set NMUTATIONSTEPS=%~6
if not "%~7"=="" set NCHAINS=%~7

echo ================================================================================
echo run_pipeline.py バックグラウンド実行
echo ================================================================================
echo.

REM スクリプトディレクトリに移動
cd /d "%~dp0"

REM PowerShellスクリプトを実行
set PS_ARGS=-Mode %MODE% -Seed %SEED%

if not "%MODELS%"=="" set PS_ARGS=%PS_ARGS% -Models %MODELS%
if not "%NPARTICLES%"=="" set PS_ARGS=%PS_ARGS% -NParticles %NPARTICLES%
if not "%NSTAGES%"=="" set PS_ARGS=%PS_ARGS% -NStages %NSTAGES%
if not "%NMUTATIONSTEPS%"=="" set PS_ARGS=%PS_ARGS% -NMutationSteps %NMUTATIONSTEPS%
if not "%NCHAINS%"=="" set PS_ARGS=%PS_ARGS% -NChains %NCHAINS%

powershell -ExecutionPolicy Bypass -File "%~dp0run_pipeline_bg.ps1" %PS_ARGS%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo バックグラウンド実行を開始しました
    echo.
    echo 実行状況の確認:
    echo   check_pipeline_bg.bat
    echo.
    echo ログの監視:
    echo   watch_pipeline_bg.bat
    echo.
    echo 注意: PCをシャットダウンするとプロセスは終了します。
    echo       実行が完了するまでPCを起動したままにしてください。
) else (
    echo.
    echo エラー: バックグラウンド実行の開始に失敗しました
)

pause

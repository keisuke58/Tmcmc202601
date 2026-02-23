@echo off
REM M1のみで5000パーティクル実行用バッチファイル

setlocal enabledelayedexpansion

echo ================================================================================
echo M1計算実行 (5000パーティクル)
echo ================================================================================
echo.

REM 実行IDを生成
for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set "dt=%%a"
set RUN_ID=m1_5000_%dt:~0,8%_%dt:~8,6%

echo Run ID: %RUN_ID%
echo Settings:
echo   n_particles: 5000
echo   n_stages: 30
echo   n_mutation_steps: 5
echo   n_chains: 1
echo   models: M1 only
echo ================================================================================
echo.

REM プロジェクトルートに移動
cd /d "%~dp0"
cd ..

REM 実行
python tmcmc\run_pipeline.py ^
    --mode debug ^
    --run-id %RUN_ID% ^
    --models M1 ^
    --n-particles 5000 ^
    --n-stages 30 ^
    --n-mutation-steps 5 ^
    --n-chains 1 ^
    --sigma-obs 0.01 ^
    --cov-rel 0.005 ^
    --debug-level MINIMAL ^
    --seed 42

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo 実行完了
    echo 結果: tmcmc\_runs\%RUN_ID%\
    echo レポート: tmcmc\_runs\%RUN_ID%\REPORT.md
    echo ================================================================================
) else (
    echo.
    echo ================================================================================
    echo 実行失敗 (終了コード: %ERRORLEVEL%)
    echo ログ: tmcmc\_runs\%RUN_ID%\
    echo ================================================================================
)

exit /b %ERRORLEVEL%



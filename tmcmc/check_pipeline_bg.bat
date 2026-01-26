@echo off
REM バックグラウンド実行中の run_pipeline.py の状態を確認

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0check_pipeline_bg.ps1"
pause

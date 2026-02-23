@echo off
REM バックグラウンド実行中の run_pipeline.py を停止

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0stop_pipeline_bg.ps1"
pause

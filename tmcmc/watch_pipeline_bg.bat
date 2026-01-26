@echo off
REM バックグラウンド実行中の run_pipeline.py のログをリアルタイムで監視

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0watch_pipeline_bg.ps1"
pause

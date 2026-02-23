# バックグラウンド実行中の run_pipeline.py のログをリアルタイムで監視

param(
    [int]$Tail = 50,
    [switch]$Follow = $true
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$RunsRoot = Join-Path $RootDir "tmcmc\_runs"

if (-not (Test-Path $RunsRoot)) {
    Write-Host "エラー: 実行ディレクトリが見つかりません: $RunsRoot" -ForegroundColor Red
    exit 1
}

# 最新の実行ディレクトリを探す
$LatestRun = Get-ChildItem -Path $RunsRoot -Directory | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1

if (-not $LatestRun) {
    Write-Host "エラー: 実行ディレクトリが見つかりません" -ForegroundColor Red
    exit 1
}

$LogFile = Join-Path $LatestRun.FullName "run.log"

if (-not (Test-Path $LogFile)) {
    Write-Host "エラー: ログファイルが見つかりません: $LogFile" -ForegroundColor Red
    exit 1
}

Write-Host "================================================================================"
Write-Host "ログ監視: $LogFile"
Write-Host "================================================================================"
Write-Host ""

if ($Follow) {
    Write-Host "リアルタイム監視モード (Ctrl+C で終了)"
    Write-Host ""
    # Get-Content -Tail -Wait でリアルタイム監視
    Get-Content $LogFile -Tail $Tail -Wait
} else {
    # 最後のN行のみ表示
    Get-Content $LogFile -Tail $Tail
}

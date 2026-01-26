# sweep_m1.py をバックグラウンドで実行するPowerShellスクリプト
# 電源が切れても続行できるように、ログをファイルに出力

Write-Host "================================================================================"
Write-Host "sweep_m1.py バックグラウンド実行開始"
Write-Host "================================================================================"
Write-Host ""

# プロジェクトルートに移動
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Set-Location ..

# ログディレクトリとファイル名を設定
$LogDir = "tmcmc\_runs\sweep_logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# タイムスタンプ付きログファイル名
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "sweep_m1_$Timestamp.log"
$PidFile = Join-Path $LogDir "sweep_m1.pid"

Write-Host "ログファイル: $LogFile"
Write-Host "PIDファイル: $PidFile"
Write-Host ""

# バックグラウンドで実行
Write-Host "バックグラウンド実行を開始します..."
Write-Host ""

try {
    # Start-Processでバックグラウンド実行（ログファイルにリダイレクト）
    # PowerShellでは2>&1を使ってstdoutとstderrを同じファイルにリダイレクト
    # cmd.exe経由で実行することで2>&1が使える
    # 堅牢化されたsweep_m1.pyを使用（環境チェック、リトライ機能付き）
    $Process = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "python tmcmc\sweep_m1.py > `"$LogFile`" 2>&1" `
        -WindowStyle Hidden `
        -PassThru
    
    # PIDを保存
    $Process.Id | Out-File -FilePath $PidFile -Encoding ASCII
    
    Write-Host ""
    Write-Host "================================================================================"
    Write-Host "バックグラウンド実行を開始しました"
    Write-Host "================================================================================"
    Write-Host ""
    Write-Host "プロセスID: $($Process.Id)"
    Write-Host "ログファイル: $LogFile"
    Write-Host "PIDファイル: $PidFile"
    Write-Host ""
    Write-Host "進捗確認方法:"
    Write-Host "  1. ログファイルを確認:"
    Write-Host "     Get-Content $LogFile -Tail 50 -Wait"
    Write-Host ""
    Write-Host "  2. 実行中のプロセスを確認:"
    Write-Host "     Get-Process -Id $($Process.Id)"
    Write-Host ""
    Write-Host "  3. スイープ結果ディレクトリを確認:"
    Write-Host "     Get-ChildItem tmcmc\_runs\sweep_m1_*"
    Write-Host ""
    Write-Host "  4. ステータス確認:"
    Write-Host "     .\check_sweep_status.bat"
    Write-Host ""
    Write-Host "================================================================================"
    
} catch {
    Write-Host ""
    Write-Host "エラー: バックグラウンド実行の開始に失敗しました" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

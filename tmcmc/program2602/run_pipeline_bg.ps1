# run_pipeline.py バックグラウンド実行スクリプト (Windows用)
# ターミナルを閉じても実行が継続します
# 注意: PCをシャットダウンするとプロセスは終了します

param(
    [string]$Mode = "debug",
    [int]$Seed = 42,
    [string]$RunId = $null,
    [string]$Models = $null,
    [int]$NParticles = $null,
    [int]$NStages = $null,
    [int]$NMutationSteps = $null,
    [int]$NChains = $null,
    [double]$SigmaObs = $null,
    [double]$CovRel = $null,
    [switch]$ForceBetaOne = $false,
    [switch]$LockPaperConditions = $false
)

Write-Host "================================================================================"
Write-Host "run_pipeline.py バックグラウンド実行開始"
Write-Host "================================================================================"
Write-Host ""

# スクリプトディレクトリに移動
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

# PIDファイルとログディレクトリの設定
$PidFile = Join-Path $ScriptDir "run_pipeline_bg.pid"
$StatusFile = Join-Path $ScriptDir "run_pipeline_bg_status.txt"

# 既に実行中かチェック
if (Test-Path $PidFile) {
    $OldPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($OldPid) {
        $Process = Get-Process -Id $OldPid -ErrorAction SilentlyContinue
        if ($Process) {
            Write-Host "警告: 既に実行中のプロセスがあります (PID: $OldPid)" -ForegroundColor Yellow
            Write-Host "新しいプロセスを開始しますか? (Y/N): " -NoNewline
            $Response = Read-Host
            if ($Response -ne "Y" -and $Response -ne "y") {
                Write-Host "キャンセルしました"
                exit 0
            }
        }
    }
}

# コマンドライン引数の構築
$ArgsList = @(
    "tmcmc\run_pipeline.py"
    "--mode", $Mode
    "--seed", $Seed
)

if ($RunId) {
    $ArgsList += "--run-id", $RunId
}
if ($Models) {
    $ArgsList += "--models", $Models
}
if ($NParticles) {
    $ArgsList += "--n-particles", $NParticles
}
if ($NStages) {
    $ArgsList += "--n-stages", $NStages
}
if ($NMutationSteps) {
    $ArgsList += "--n-mutation-steps", $NMutationSteps
}
if ($NChains) {
    $ArgsList += "--n-chains", $NChains
}
if ($SigmaObs) {
    $ArgsList += "--sigma-obs", $SigmaObs
}
if ($CovRel) {
    $ArgsList += "--cov-rel", $CovRel
}
if ($ForceBetaOne) {
    $ArgsList += "--force-beta-one"
}
if ($LockPaperConditions) {
    $ArgsList += "--lock-paper-conditions"
}

# コマンドを文字列に変換（表示用）
$CmdString = "python " + ($ArgsList -join " ")

Write-Host "実行コマンド: $CmdString"
Write-Host ""

# バックグラウンドプロセスを開始
Write-Host "バックグラウンドプロセスを開始しています..."
Write-Host ""

try {
    # Start-Process を使用して完全に独立したプロセスとして起動
    # -WindowStyle Hidden: ウィンドウを表示しない
    # -PassThru: プロセスオブジェクトを返す
    # これにより、ターミナルを閉じてもプロセスは継続します
    
    $Process = Start-Process -FilePath "python" `
        -ArgumentList $ArgsList `
        -WorkingDirectory $RootDir `
        -WindowStyle Hidden `
        -PassThru
    
    # PIDを保存
    $Process.Id | Out-File -FilePath $PidFile -Encoding ASCII -Force
    
    # ステータス情報を保存
    $StatusInfo = @"
開始時刻: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
PID: $($Process.Id)
コマンド: $CmdString
実行ディレクトリ: $RootDir
"@
    $StatusInfo | Out-File -FilePath $StatusFile -Encoding UTF8 -Force
    
    Write-Host ""
    Write-Host "================================================================================"
    Write-Host "バックグラウンドプロセスを開始しました"
    Write-Host "================================================================================"
    Write-Host ""
    Write-Host "プロセスID (PID): $($Process.Id)"
    Write-Host "PIDファイル: $PidFile"
    Write-Host "ステータスファイル: $StatusFile"
    Write-Host ""
    Write-Host "実行状況の確認:"
    Write-Host "  .\tmcmc\check_pipeline_bg.ps1"
    Write-Host ""
    Write-Host "ログの確認:"
    Write-Host "  .\tmcmc\watch_pipeline_bg.ps1"
    Write-Host ""
    Write-Host "プロセスの停止:"
    Write-Host "  .\tmcmc\stop_pipeline_bg.ps1"
    Write-Host ""
    Write-Host "注意:"
    Write-Host "  - ターミナルを閉じても実行は継続します"
    Write-Host "  - PCをシャットダウンするとプロセスは終了します"
    Write-Host "  - 実行が完了するまでPCを起動したままにしてください"
    Write-Host ""
    Write-Host "================================================================================"
    
} catch {
    Write-Host ""
    Write-Host "エラー: バックグラウンドプロセスの開始に失敗しました" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# Slack通知の設定方法

## 環境変数の設定場所

Slack通知を有効にするには、以下の環境変数を設定する必要があります。

### 設定方法

#### 方法1: `.bashrc` または `.bash_profile` に設定（推奨）

```bash
# ホームディレクトリの .bashrc に追加
echo 'export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"' >> ~/.bashrc
source ~/.bashrc
```

#### 方法2: 実行時に環境変数を設定

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
./tmcmc/run_m1.sh
```

#### 方法3: `.env` ファイルを作成（プロジェクトルート）

```bash
# /home/nishioka/IKM_Hiwi/.env を作成
cat > /home/nishioka/IKM_Hiwi/.env << EOF
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
EOF

# 実行前に読み込む
source .env
./tmcmc/run_m1.sh
```

### 必要な環境変数

コード内の説明（`tmcmc/debug/logger.py` 36-39行目）:

```python
# Enabled only when credentials are provided via environment variables.
# - Webhook: SLACK_WEBHOOK_URL
# - Bot: SLACK_BOT_TOKEN (+ SLACK_CHANNEL, depending on stranger/d.py)
SLACK_ENABLED = bool(os.getenv("SLACK_WEBHOOK_URL") or os.getenv("SLACK_BOT_TOKEN"))
```

#### Webhook方式（推奨）
- `SLACK_WEBHOOK_URL`: Slack Incoming Webhook URL

#### Bot Token方式
- `SLACK_BOT_TOKEN`: Slack Bot Token
- `SLACK_CHANNEL`: 送信先チャンネル（例: `#general`）

### Slack Webhook URLの取得方法

1. Slackワークスペースにログイン
2. https://api.slack.com/apps にアクセス
3. "Create New App" → "From scratch"
4. App名とワークスペースを選択
5. "Incoming Webhooks" を有効化
6. "Add New Webhook to Workspace" をクリック
7. チャンネルを選択してWebhook URLをコピー

### 動作確認

```bash
# 環境変数が設定されているか確認
echo $SLACK_WEBHOOK_URL

# または
env | grep SLACK
```

### 注意事項

- 環境変数が設定されていない場合、通知は送信されませんがエラーにはなりません
- Webhook URLは機密情報のため、Gitにコミットしないでください
- `.env` ファイルは `.gitignore` に追加することを推奨

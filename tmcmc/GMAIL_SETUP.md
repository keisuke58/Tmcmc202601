# Gmail通知の設定方法

## 概要

TMCMC計算完了時にGmailで通知を送信する機能です。Slack URLがない場合でも、メールで通知を受け取ることができます。

## 設定手順

### 1. Gmailアプリパスワードの取得

Gmailで2段階認証を有効にし、アプリパスワードを取得する必要があります。

1. https://myaccount.google.com/ にアクセス
2. 「セキュリティ」→「2段階認証プロセス」を有効化
3. 「アプリパスワード」を選択
4. 「アプリを選択」→「メール」
5. 「デバイスを選択」→「その他（カスタム名）」→「TMCMC通知」など
6. 「生成」をクリックして16文字のパスワードをコピー

### 2. .envファイルの設定

`.env`ファイルを編集して、Gmail認証情報を設定します：

```bash
# 送信先メールアドレス（通知を受け取るアドレス）
EMAIL_TO=your-email@gmail.com

# 送信元メールアドレス（Gmailアカウント）
EMAIL_FROM=your-gmail@gmail.com

# Gmail SMTP認証情報
EMAIL_USER=your-gmail@gmail.com
EMAIL_PASSWORD=your-16-char-app-password
```

**重要**: `EMAIL_PASSWORD`には通常のGmailパスワードではなく、**アプリパスワード**を設定してください。

### 3. 動作確認

設定後、テスト通知を送信できます：

```bash
cd /home/nishioka/IKM_Hiwi
source .env
python3 tmcmc/utils/email_notifier.py test
```

## 通知内容

計算完了時に以下の情報がメールで送信されます：

- ✅ 実行完了通知
- Run ID
- 実行時間
- ステータス（PASS/WARN/FAIL）
- 結果ディレクトリのパス
- レポートのパス

## 注意事項

- Gmailアプリパスワードは16文字の英数字です
- 通常のGmailパスワードでは動作しません
- 2段階認証が有効になっている必要があります
- `.env`ファイルは`.gitignore`に含まれているため、Gitにはコミットされません

## トラブルシューティング

### メールが送信されない場合

1. 環境変数が正しく設定されているか確認：
   ```bash
   source .env
   echo $EMAIL_TO
   echo $EMAIL_FROM
   echo $EMAIL_USER
   ```

2. アプリパスワードが正しいか確認（16文字の英数字）

3. 2段階認証が有効か確認

4. ファイアウォールでSMTPポート（587）がブロックされていないか確認

### エラーメッセージ

- `Authentication failed`: アプリパスワードが間違っている可能性
- `Connection refused`: SMTPサーバーに接続できない（ネットワーク問題）

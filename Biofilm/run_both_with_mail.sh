#!/bin/bash

MAIL_TO="kei128608@gmail.com"

run_job() {
    SCRIPT_NAME=$1
    LOG_FILE=$2

    echo "開始: $SCRIPT_NAME"
    START=$(date)

    nohup python3 $SCRIPT_NAME > $LOG_FILE 2>&1
    STATUS=$?
    END=$(date)

    if [ $STATUS -eq 0 ]; then
        SUBJECT="【完了通知】$SCRIPT_NAME 終了"
        BODY="計算が正常に終了しました。\n\n開始: $START\n終了: $END\nログ: $LOG_FILE"
    else
        SUBJECT="【⚠️エラー】$SCRIPT_NAME 停止"
        BODY="計算がエラーで停止しました。\n\n開始: $START\n終了: $END\nログ: $LOG_FILE"
    fi

    echo -e "$BODY" | mail -s "$SUBJECT" "$MAIL_TO"
}

# ---------- ここから並列実行 -------------

# re3
run_job biofilm1130_re3_best.py log3best.txt &

# re4
run_job biofilm1130_re4_accurecy.py log4best.txt &

wait    # ← どちらも終了を待つ

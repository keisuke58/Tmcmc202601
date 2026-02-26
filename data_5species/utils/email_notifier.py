"""
Email notification utility for TMCMC execution completion.

Supports Gmail SMTP for sending completion notifications.
"""

from __future__ import annotations

import os
import smtplib
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional


def send_email_notification(
    subject: str,
    body: str,
    to_email: Optional[str] = None,
    from_email: Optional[str] = None,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
) -> bool:
    """
    Send email notification via SMTP (Gmail supported).

    Parameters
    ----------
    subject : str
        Email subject
    body : str
        Email body (plain text)
    to_email : str, optional
        Recipient email address (default: from EMAIL_TO env var)
    from_email : str, optional
        Sender email address (default: from EMAIL_FROM env var)
    smtp_server : str
        SMTP server (default: smtp.gmail.com)
    smtp_port : int
        SMTP port (default: 587 for TLS)
    smtp_user : str, optional
        SMTP username (default: from EMAIL_USER env var)
    smtp_password : str, optional
        SMTP password/app password (default: from EMAIL_PASSWORD env var)

    Returns
    -------
    bool
        True if email sent successfully, False otherwise
    """
    # Get credentials from environment variables if not provided
    to_email = to_email or os.getenv("EMAIL_TO")
    from_email = from_email or os.getenv("EMAIL_FROM")
    smtp_user = smtp_user or os.getenv("EMAIL_USER")
    smtp_password = smtp_password or os.getenv("EMAIL_PASSWORD")

    # Check if email is enabled
    if not all([to_email, from_email, smtp_user, smtp_password]):
        return False

    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject

        # Add body
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        return True
    except Exception as e:
        # Silent fail - don't break execution if email fails
        print(f"Email notification failed: {e}", file=sys.stderr)
        return False


def notify_tmcmc_completion(
    run_id: str,
    status: str,
    elapsed_time: Optional[str] = None,
    run_dir: Optional[str] = None,
    report_path: Optional[str] = None,
    error_message: Optional[str] = None,
) -> bool:
    """
    Send email notification for TMCMC completion.

    Parameters
    ----------
    run_id : str
        Run identifier
    status : str
        Status: "SUCCESS", "FAIL", "PASS", "WARN"
    elapsed_time : str, optional
        Execution time (e.g., "30分15秒")
    run_dir : str, optional
        Run directory path
    report_path : str, optional
        Report file path
    error_message : str, optional
        Error message if failed

    Returns
    -------
    bool
        True if email sent successfully
    """
    # Check if email is enabled
    if not os.getenv("EMAIL_TO"):
        return False

    # Build subject
    if status in ["SUCCESS", "PASS"]:
        subject = f"✅ TMCMC計算完了: {run_id}"
        emoji = "✅"
    elif status == "WARN":
        subject = f"⚠️ TMCMC計算完了（警告）: {run_id}"
        emoji = "⚠️"
    else:
        subject = f"❌ TMCMC計算失敗: {run_id}"
        emoji = "❌"

    # Build body
    body_lines = [
        f"{emoji} TMCMC計算が完了しました",
        "",
        f"Run ID: {run_id}",
        f"ステータス: {status}",
    ]

    if elapsed_time:
        body_lines.append(f"実行時間: {elapsed_time}")

    if run_dir:
        body_lines.append(f"結果ディレクトリ: {run_dir}")

    if report_path:
        body_lines.append(f"レポート: {report_path}")

    if error_message:
        body_lines.extend(
            [
                "",
                "エラー詳細:",
                error_message,
            ]
        )

    body = "\n".join(body_lines)

    return send_email_notification(subject, body)


if __name__ == "__main__":
    # Test email notification
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        result = notify_tmcmc_completion(
            run_id="test_run",
            status="SUCCESS",
            elapsed_time="5分30秒",
            run_dir="/path/to/results",
            report_path="/path/to/REPORT.md",
        )
        print(f"Email sent: {result}")

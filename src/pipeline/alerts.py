"""7B. Alert System — Telegram + Email notifications for daily predictions.

Both channels are optional; configured via config.yaml → alerts section.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def send_alerts(config: dict, alert_data: dict) -> dict:
    """Send prediction alerts via configured channels.

    Args:
        config: Full config dict (reads alerts section)
        alert_data: Dict with date, direction, confidence, report

    Returns:
        Dict with send status per channel
    """
    cfg = config.get("alerts", {})
    results = {"telegram": False, "email": False}

    # Telegram
    token = cfg.get("telegram_token", "")
    chat_id = cfg.get("telegram_chat_id", "")
    if token and chat_id:
        results["telegram"] = _send_telegram(token, chat_id, alert_data)

    # Email
    smtp_server = cfg.get("email_smtp", "")
    email_from = cfg.get("email_from", "")
    email_to = cfg.get("email_to", "")
    if smtp_server and email_from and email_to:
        results["email"] = _send_email(smtp_server, email_from, email_to, alert_data)

    sent_any = any(results.values())
    if not sent_any:
        logger.info("No alert channels configured — skipping")
    return {"sent": sent_any, **results}


def _format_message(data: dict) -> str:
    """Format prediction data into a readable message."""
    direction = data.get("direction", "N/A")
    confidence = data.get("confidence", 0)
    date = data.get("date", "")

    # Emoji mapping
    emoji = "🟢" if "BULLISH" in direction else "🔴" if "BEARISH" in direction else "⚪"

    msg = (
        f"{emoji} SPY Prediction — {date}\n"
        f"Direction: {direction}\n"
        f"Confidence: {confidence:.0f}%\n"
    )

    report = data.get("report", "")
    if report:
        # Truncate for Telegram (4096 char limit)
        if len(report) > 3500:
            report = report[:3500] + "..."
        msg += f"\n{report}"

    return msg


def _send_telegram(token: str, chat_id: str, data: dict) -> bool:
    """Send alert via Telegram Bot API."""
    msg = _format_message(data)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "HTML",
        }, timeout=15)
        if resp.status_code == 200:
            logger.info("Telegram alert sent")
            return True
        else:
            logger.warning(f"Telegram API error: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
        return False


def _send_email(smtp_server: str, from_addr: str, to_addr: str,
                data: dict) -> bool:
    """Send alert via SMTP email."""
    direction = data.get("direction", "N/A")
    confidence = data.get("confidence", 0)
    date = data.get("date", "")
    report = data.get("report", "")

    subject = f"SPY Prediction {date}: {direction} ({confidence:.0f}%)"

    html = f"""
    <html><body>
    <h2>SPY/SPX Daily Prediction — {date}</h2>
    <table style="border-collapse:collapse;">
      <tr><td style="padding:4px 12px;font-weight:bold;">Direction</td>
          <td style="padding:4px 12px;">{direction}</td></tr>
      <tr><td style="padding:4px 12px;font-weight:bold;">Confidence</td>
          <td style="padding:4px 12px;">{confidence:.0f}%</td></tr>
    </table>
    <hr>
    <h3>Daily Report</h3>
    <p>{report.replace(chr(10), '<br>')}</p>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(_format_message(data), "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(smtp_server, 587, timeout=30) as server:
            server.starttls()
            server.send_message(msg)
        logger.info("Email alert sent")
        return True
    except Exception as e:
        logger.warning(f"Email send failed: {e}")
        return False

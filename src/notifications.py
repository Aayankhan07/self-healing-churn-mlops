"""
Enterprise Notification & Webhook Dispatcher Module.
Sends instant notifications to Slack / Email Webhooks when data drift or retraining events occur.
"""

import os
import json
import logging
import urllib.request

logger = logging.getLogger(__name__)


def send_slack_alert(
    event_type: str, title: str, details: str, domain_id: str = "telecom"
) -> bool:
    """
    Send Webhook notification to Slack / External Webhook URL if configured via SLACK_WEBHOOK_URL.
    """
    # Read directly rather than via api.config: src/ is the ML pipeline and
    # must not depend on the API package. Same variable, same default.
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

    payload = {
        "text": f"[ChurnGuard Alert | Domain: {domain_id}] {title}",
        "attachments": [
            {
                "color": "#e74c3c" if event_type in ("drift", "failure") else "#2ecc71",
                "fields": [
                    {"title": "Event Type", "value": event_type, "short": True},
                    {"title": "Domain", "value": domain_id, "short": True},
                    {"title": "Details", "value": details, "short": False},
                ],
            }
        ],
    }

    if not webhook_url:
        logger.info(f"[Mock Notification Payload Dispatch] {json.dumps(payload)}")
        return True

    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception as e:
        logger.warning(f"Failed to send Slack webhook alert: {e}")
        return False

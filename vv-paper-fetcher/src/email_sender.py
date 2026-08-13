"""Sends the HTML digest via Resend's REST API (plain requests, no SDK).

Setup caveat (see README/.env.example): Resend's sandbox sender
(onboarding@resend.dev) only delivers to the account owner's own verified
address; sending to any other recipient requires a verified custom domain.
"""
from __future__ import annotations

import logging
from typing import List

import requests

logger = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"
TIMEOUT_S = 30


def send_digest_email(
    api_key: str,
    sender: str,
    recipients: str,
    subject: str,
    html_body: str,
) -> bool:
    """Best-effort send — failures are logged and return False, never raised,
    so an email problem never prevents the report file/state from being written.
    """
    to_list: List[str] = [addr.strip() for addr in recipients.split(",") if addr.strip()]

    try:
        resp = requests.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"from": sender, "to": to_list, "subject": subject, "html": html_body},
            timeout=TIMEOUT_S,
        )
        if resp.status_code >= 300:
            logger.warning("Resend send failed: %s %s", resp.status_code, resp.text)
            return False
        return True
    except requests.RequestException as exc:
        logger.warning("Resend send failed: %s", exc)
        return False

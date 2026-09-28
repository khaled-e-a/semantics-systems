"""Sends the HTML digest via Resend's REST API (plain requests, no SDK).

Setup caveat (see README/.env.example): Resend's sandbox sender
(onboarding@resend.dev) only delivers to the account owner's own verified
address; sending to any other recipient requires a verified custom domain.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"
TIMEOUT_S = 30


def attachment_from_path(path: Path, filename: Optional[str] = None) -> Dict[str, str]:
    """Build one Resend attachment ({filename, content: base64}) from a local file."""
    path = Path(path)
    return {
        "filename": filename or path.name,
        "content": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def send_digest_email(
    api_key: str,
    sender: str,
    recipients: str,
    subject: str,
    html_body: str,
    attachments: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """Best-effort send — failures are logged and return False, never raised,
    so an email problem never prevents the report file/state from being written.

    ``attachments`` is passed through as Resend's ``attachments`` field:
    ``[{"filename": ..., "content": <base64 str>}]`` (see attachment_from_path).
    """
    to_list: List[str] = [addr.strip() for addr in recipients.split(",") if addr.strip()]

    payload: Dict[str, object] = {"from": sender, "to": to_list, "subject": subject, "html": html_body}
    if attachments:
        payload["attachments"] = attachments

    try:
        resp = requests.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT_S,
        )
        if resp.status_code >= 300:
            logger.warning("Resend send failed: %s %s", resp.status_code, resp.text)
            return False
        return True
    except requests.RequestException as exc:
        logger.warning("Resend send failed: %s", exc)
        return False

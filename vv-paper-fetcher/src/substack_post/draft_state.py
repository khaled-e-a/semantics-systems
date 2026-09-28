"""Idempotency state for Substack drafts: which report dates already have one.

``state/substack_drafts.json`` maps the report date to the draft it produced::

    {"2026-09-28": {"draft_id": 123, "edit_url": "https://…/publish/post/123",
                    "title": "…", "updated_at": "2026-09-28T12:00:00+00:00"}}

write_post.py skips a date that is already here unless --force is given, in
which case it updates the same draft instead of creating a duplicate.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .models import DraftResult

DRAFTS_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "state" / "substack_drafts.json"


def load_drafts(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    p = path or DRAFTS_STATE_PATH
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_drafts(drafts: Dict[str, Dict[str, Any]], path: Path | None = None) -> None:
    p = path or DRAFTS_STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(drafts, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def get_draft(drafts: Dict[str, Dict[str, Any]], report_date: str) -> Optional[Dict[str, Any]]:
    """The recorded draft for ``report_date``, or None when there is no usable record."""
    record = drafts.get(report_date)
    if not record or record.get("draft_id") is None:
        return None
    return record


def record_draft(
    drafts: Dict[str, Dict[str, Any]],
    report_date: str,
    result: DraftResult,
    title: str,
    now: Optional[datetime] = None,
) -> None:
    """Record (or overwrite) the draft created/updated for ``report_date``."""
    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    drafts[report_date] = {
        "draft_id": result.draft_id,
        "edit_url": result.edit_url,
        "title": title,
        "updated_at": ts.isoformat(timespec="seconds"),
    }

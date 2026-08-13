"""OpenReview collector (NeurIPS/ICML/ICLR).

Conferences aren't weekly, so this returning nothing most weeks is a normal
outcome, not an error. OpenReview is mid-migration between API v1 and v2
with different client classes and response shapes; each configured venue
declares which version it's on. Exact per-venue mapping and v1/v2 field
names should be re-verified against openreview-py's docs at implementation
time, since venues migrate over time.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List

from ..models import Paper

logger = logging.getLogger(__name__)


def _v1_title(note) -> str:
    return note.content.get("title", "")


def _v2_title(note) -> str:
    val = note.content.get("title", {})
    return val.get("value", "") if isinstance(val, dict) else str(val)


def _v1_abstract(note) -> str:
    return note.content.get("abstract", "")


def _v2_abstract(note) -> str:
    val = note.content.get("abstract", {})
    return val.get("value", "") if isinstance(val, dict) else str(val)


def _v1_authors(note) -> List[str]:
    return list(note.content.get("authors", []))


def _v2_authors(note) -> List[str]:
    val = note.content.get("authors", {})
    return list(val.get("value", [])) if isinstance(val, dict) else list(val)


ADAPTERS = {
    1: {"title": _v1_title, "abstract": _v1_abstract, "authors": _v1_authors},
    2: {"title": _v2_title, "abstract": _v2_abstract, "authors": _v2_authors},
}


def _fetch_venue_notes(venue_id: str, api_version: int) -> List[Any]:
    if api_version == 2:
        from openreview.api import OpenReviewClient

        client = OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=os.environ.get("OPENREVIEW_USERNAME"),
            password=os.environ.get("OPENREVIEW_PASSWORD"),
        )
        return list(client.get_all_notes(content={"venueid": venue_id}))
    else:
        import openreview

        client = openreview.Client(
            baseurl="https://api.openreview.net",
            username=os.environ.get("OPENREVIEW_USERNAME"),
            password=os.environ.get("OPENREVIEW_PASSWORD"),
        )
        return list(client.get_all_notes(invitation=f"{venue_id}/-/Blind_Submission"))


def fetch(src_cfg: Dict[str, Any], window_start: date, window_end: date, state: Dict[str, Any]) -> List[Paper]:
    papers: List[Paper] = []

    for venue in src_cfg.get("venues", []):
        venue_id = venue["id"]
        api_version = venue.get("api_version", 2)
        label = venue.get("label", venue_id)
        adapter = ADAPTERS[api_version]

        try:
            notes = _fetch_venue_notes(venue_id, api_version)
        except Exception as exc:  # noqa: BLE001 - one venue failing shouldn't drop the others
            logger.warning("OpenReview venue %s failed: %s", venue_id, exc)
            continue

        for note in notes:
            ts_ms = getattr(note, "cdate", None) or getattr(note, "tmdate", None)
            if not ts_ms:
                continue
            note_date = datetime.fromtimestamp(ts_ms / 1000).date()
            if not (window_start <= note_date <= window_end):
                continue

            try:
                title = adapter["title"](note)
                abstract = adapter["abstract"](note)
                authors = adapter["authors"](note)
            except Exception as exc:  # noqa: BLE001 - malformed note shouldn't drop the batch
                logger.warning("OpenReview note %s parse failed: %s", getattr(note, "id", "?"), exc)
                continue

            papers.append(
                Paper(
                    title=" ".join(title.split()),
                    authors=authors,
                    abstract=" ".join(abstract.split()),
                    url=f"https://openreview.net/forum?id={note.id}",
                    venue=label,
                    source="openreview",
                    published_date=note_date,
                    extra={"openreview_id": note.id},
                )
            )

    return papers

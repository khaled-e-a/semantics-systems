"""DBLP collector (CAV, TACAS, FM, POPL, CADE, LICS, VMCAI, ...).

DBLP has no queryable per-paper submission date, so this source can't be
windowed like the others (see design plan). Instead we track each venue's
known proceedings-*volume* set in `state["dblp_seen_keys"]` and process only
newly appeared volumes each run — a diff, not a date filter. This naturally
matches how these venues actually update: in bursts, as a proceedings volume
gets indexed, not on a weekly cadence.

Two-stage, both verified live against the real DBLP endpoints (the
originally-planned `search/publ/api` approach was tried first and found to
return empty/irrelevant results for venue filtering — this replaces it):

1. `https://dblp.org/streams/conf/<key>.rss` — one entry per proceedings
   volume ever indexed for that venue (e.g. "CAV (1) 2026"), with a stable
   link like `https://dblp.org/db/conf/cav/cav2026-1.html`.
2. For each volume not already in state, fetch that same link with `.html`
   swapped to `.xml` — a well-formed XML document listing every individual
   `<inproceedings>` paper in that volume (title/authors/year/DOI).

No abstracts are available from DBLP, so `abstract=""` and
`date_precision="year"` (DBLP only gives publication year, not day).

Two behaviors added after live-testing against the real API (confirmed
during implementation, not just theorized):

- DBLP rate-limits/connection-resets fast under back-to-back requests (no
  documented per-second budget found) — a fixed delay is inserted between
  each per-volume XML fetch, and 429s get one retry with backoff, matching
  the pattern used by the other collectors.
- On a venue's *first* run (no prior state), a stream's RSS lists every
  volume ever indexed — for CAV that's 60+ volumes back to 1989. Reporting
  all of that as "new" in week one would both hammer DBLP unnecessarily and
  violate the actual intent (newly *appearing* proceedings, not a full
  historical backfill). So a first run silently seeds state with every
  currently-known volume id and returns no papers; only volumes that appear
  in *subsequent* runs are treated as new and parsed.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any, Dict, List

import feedparser
import requests

from ..models import Paper

TIMEOUT_S = 30
VOLUME_FETCH_DELAY_S = 2
logger = logging.getLogger(__name__)


def _get_with_retry(url: str) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=TIMEOUT_S)
            if resp.status_code == 429 and attempt == 0:
                time.sleep(5)
                continue
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            time.sleep(2)
        except requests.HTTPError as exc:
            if resp.status_code >= 500 and attempt == 0:
                last_exc = exc
                time.sleep(2)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _parse_volume(xml_url: str, label: str) -> List[Paper]:
    resp = _get_with_retry(xml_url)
    root = ET.fromstring(resp.text)

    papers: List[Paper] = []
    for entry in root.findall(".//inproceedings"):
        key = entry.get("key", "")
        title_el = entry.find("title")
        title = (title_el.text or "").rstrip(".") if title_el is not None else ""
        authors = [a.text for a in entry.findall("author") if a.text]

        year_el = entry.find("year")
        published_dt = None
        if year_el is not None and year_el.text:
            try:
                published_dt = date(int(year_el.text), 1, 1)
            except ValueError:
                pass

        ee_el = entry.find("ee")
        doi = None
        if ee_el is not None and ee_el.text and ee_el.text.startswith("https://doi.org/"):
            doi = ee_el.text[len("https://doi.org/"):]

        url_el = entry.find("url")
        url = f"https://dblp.org/{url_el.text}" if url_el is not None and url_el.text else xml_url

        papers.append(
            Paper(
                title=" ".join(title.split()),
                authors=authors,
                abstract="",
                url=url,
                venue=label,
                source="dblp",
                published_date=published_dt,
                date_precision="year",
                doi=doi,
                extra={"dblp_key": key},
            )
        )

    return papers


def fetch(src_cfg: Dict[str, Any], window_start: date, window_end: date, state: Dict[str, Any]) -> List[Paper]:
    papers: List[Paper] = []
    dblp_state = state.setdefault("dblp_seen_keys", {})

    for i, venue in enumerate(src_cfg.get("venues", [])):
        if i > 0:
            time.sleep(VOLUME_FETCH_DELAY_S)

        venue_key = venue["key"]
        label = venue.get("label", venue_key)
        previously_seen = set(dblp_state.get(venue_key, []))

        try:
            resp = _get_with_retry(f"https://dblp.org/streams/{venue_key}.rss")
            feed = feedparser.parse(resp.content)
        except Exception as exc:  # noqa: BLE001 - one venue failing shouldn't drop the others
            logger.warning("DBLP venue %s stream fetch failed: %s", venue_key, exc)
            continue

        current_volume_ids = {
            (entry.get("id") or entry.get("link", "")) for entry in feed.entries
        }
        current_volume_ids.discard("")

        is_bootstrap = len(previously_seen) == 0
        if is_bootstrap:
            logger.info(
                "DBLP venue %s: first run, seeding %d known volumes without parsing (avoids a "
                "one-time historical backfill and unnecessary load on DBLP)",
                venue_key,
                len(current_volume_ids),
            )
        else:
            new_volume_ids = current_volume_ids - previously_seen
            for volume_entry in feed.entries:
                volume_id = volume_entry.get("id") or volume_entry.get("link", "")
                if volume_id not in new_volume_ids:
                    continue

                html_link = volume_entry.get("link", volume_id)
                xml_url = html_link.replace(".html", ".xml")
                try:
                    papers.extend(_parse_volume(xml_url, label))
                except Exception as exc:  # noqa: BLE001 - one bad volume shouldn't drop the others
                    logger.warning("DBLP volume %s parse failed: %s", xml_url, exc)
                time.sleep(VOLUME_FETCH_DELAY_S)

        dblp_state[venue_key] = sorted(current_volume_ids | previously_seen)

    return papers

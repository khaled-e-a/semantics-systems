"""Hugging Face Papers collector.

Community-curated daily papers list with an upvote signal. The endpoint's
exact date-filtering support is unconfirmed from official docs, so we fetch
the default payload and filter client-side on `publishedAt` — verify at
implementation/runtime whether a `?date=` param narrows the server-side
payload; either way client-side filtering is correct.

Each list item wraps the actual paper record under an `item["paper"]` key
(id/authors/upvotes/githubStars all live there, not at the top level, even
though title/summary/publishedAt happen to be duplicated at both levels) —
confirmed against the live endpoint.
"""
from __future__ import annotations

import time
from datetime import date, datetime
from typing import Any, Dict, List

import requests

from ..models import Paper

URL = "https://huggingface.co/api/daily_papers"
TIMEOUT_S = 30


def _get_with_retry() -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.get(URL, timeout=TIMEOUT_S)
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


def fetch(src_cfg: Dict[str, Any], window_start: date, window_end: date, state: Dict[str, Any]) -> List[Paper]:
    resp = _get_with_retry()
    items = resp.json()

    papers: List[Paper] = []
    for item in items:
        paper_obj = item.get("paper", {})

        published_raw = paper_obj.get("publishedAt") or item.get("publishedAt", "")
        try:
            published_dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).date()
        except ValueError:
            continue
        if not (window_start <= published_dt <= window_end):
            continue

        arxiv_id = paper_obj.get("id")
        authors = [a.get("name", "") for a in paper_obj.get("authors", []) if a.get("name")]
        title = paper_obj.get("title") or item.get("title", "")
        summary = paper_obj.get("summary") or item.get("summary", "")

        papers.append(
            Paper(
                title=" ".join(title.split()),
                authors=authors,
                abstract=" ".join(summary.split()),
                url=f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else "",
                venue="HF Daily Papers",
                source="hf_papers",
                published_date=published_dt,
                arxiv_id=arxiv_id,
                extra={
                    "hf_upvotes": paper_obj.get("upvotes", 0),
                    "hf_github_stars": paper_obj.get("githubStars"),
                },
            )
        )

    return papers

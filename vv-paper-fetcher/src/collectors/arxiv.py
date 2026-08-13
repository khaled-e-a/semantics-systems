"""arXiv collector — primary volume source.

Queries export.arxiv.org/api/query filtered by configured categories and a
submittedDate range, paginating until a short page or a safety cap.
"""
from __future__ import annotations

import time
from datetime import date, datetime
from typing import Any, Dict, List

import feedparser
import requests

from ..models import Paper

BASE_URL = "https://export.arxiv.org/api/query"
TIMEOUT_S = 30
PAGE_DELAY_S = 3  # arXiv's documented guidance between successive calls
NS_ARXIV = "{http://arxiv.org/schemas/atom}"


def _get_with_retry(params: Dict[str, Any]) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=TIMEOUT_S)
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
    categories = src_cfg.get("categories", ["cs.AI"])
    max_results_per_call = src_cfg.get("max_results_per_call", 200)
    max_total = src_cfg.get("max_total_results", 1000)

    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    date_from = window_start.strftime("%Y%m%d") + "0000"
    date_to = window_end.strftime("%Y%m%d") + "2359"
    search_query = f"({cat_query}) AND submittedDate:[{date_from} TO {date_to}]"

    papers: List[Paper] = []
    start = 0
    while start < max_total:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results_per_call,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = _get_with_retry(params)
        feed = feedparser.parse(resp.content)
        entries = feed.entries
        if not entries:
            break

        for entry in entries:
            raw_id = entry.get("id", "")
            arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
            arxiv_id = arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id.split("/")[-1] else arxiv_id

            pdf_url = None
            for link in entry.get("links", []):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break

            published_dt = None
            if entry.get("published"):
                try:
                    published_dt = datetime.strptime(entry["published"][:10], "%Y-%m-%d").date()
                except ValueError:
                    pass

            primary_category = entry.get("arxiv_primary_category", {}).get("term", "arXiv")

            papers.append(
                Paper(
                    title=" ".join(entry.get("title", "").split()),
                    authors=[a.get("name", "") for a in entry.get("authors", [])],
                    abstract=" ".join(entry.get("summary", "").split()),
                    url=raw_id,
                    venue=primary_category,
                    source="arxiv",
                    published_date=published_dt,
                    arxiv_id=arxiv_id,
                    pdf_url=pdf_url,
                )
            )

        if len(entries) < max_results_per_call:
            break
        start += max_results_per_call
        time.sleep(PAGE_DELAY_S)

    return papers

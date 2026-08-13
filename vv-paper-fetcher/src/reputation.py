"""Semantic Scholar reputation lookups: paper citation counts and, more
importantly, each author's historical h-index/citation count.

A paper published this week has ~0 citations of its own — reputation
scoring therefore leans on the AUTHORS' track record (h-index) rather than
the new paper's citation count (see design plan). Only called on the
post-triage relevant set to keep call volume small.
"""
from __future__ import annotations

import difflib
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from .models import Paper, RunStats, TriagedPaper

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
TIMEOUT_S = 30
BATCH_LIMIT = 500
TITLE_MATCH_THRESHOLD = 0.85


class ReputationLookup:
    def __init__(self, api_key: Optional[str] = None):
        self.headers = {"x-api-key": api_key} if api_key else {}

    def _post_with_retry(self, url: str, params: Dict[str, Any], json_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for attempt in range(2):
            try:
                resp = requests.post(url, params=params, json=json_body, headers=self.headers, timeout=TIMEOUT_S)
                if resp.status_code == 429 and attempt == 0:
                    time.sleep(3)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("Semantic Scholar request failed (attempt %d): %s", attempt + 1, exc)
                if attempt == 0:
                    time.sleep(2)
                    continue
        return None

    def _get_with_retry(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for attempt in range(2):
            try:
                resp = requests.get(url, params=params, headers=self.headers, timeout=TIMEOUT_S)
                if resp.status_code == 429 and attempt == 0:
                    time.sleep(3)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("Semantic Scholar request failed (attempt %d): %s", attempt + 1, exc)
                if attempt == 0:
                    time.sleep(2)
                    continue
        return None

    def _match_by_title(self, paper: Paper) -> Optional[Dict[str, Any]]:
        data = self._get_with_retry(
            f"{BASE_URL}/paper/search/match",
            {"query": paper.title, "fields": "title,citationCount,authors.authorId,authors.name"},
        )
        if not data or "data" not in data or not data["data"]:
            return None
        candidate = data["data"][0]
        ratio = difflib.SequenceMatcher(None, candidate.get("title", "").lower(), paper.title.lower()).ratio()
        if ratio < TITLE_MATCH_THRESHOLD:
            return None
        if paper.authors:
            candidate_authors = {a.get("name", "").lower() for a in candidate.get("authors", [])}
            first_author_last = paper.authors[0].lower().split()[-1] if paper.authors[0] else ""
            if first_author_last and not any(first_author_last in name for name in candidate_authors):
                return None
        return candidate

    def lookup_papers(self, papers: List[Paper], stats: RunStats) -> Dict[str, Dict[str, Any]]:
        """Returns dedup_key -> {citationCount, authors: [{authorId, name}]}."""
        results: Dict[str, Dict[str, Any]] = {}

        with_arxiv = [p for p in papers if p.arxiv_id]
        without_arxiv = [p for p in papers if not p.arxiv_id]

        for start in range(0, len(with_arxiv), BATCH_LIMIT):
            batch = with_arxiv[start : start + BATCH_LIMIT]
            ids = [f"arXiv:{p.arxiv_id}" for p in batch]
            data = self._post_with_retry(
                f"{BASE_URL}/paper/batch",
                {"fields": "citationCount,authors.authorId,authors.name"},
                {"ids": ids},
            )
            if data is None:
                stats.reputation_lookup_errors += len(batch)
                continue
            for paper, entry in zip(batch, data):
                if entry:
                    results[paper.dedup_key()] = entry

        for paper in without_arxiv:
            match = self._match_by_title(paper)
            if match:
                results[paper.dedup_key()] = match

        return results

    def lookup_authors(self, author_ids: List[str], stats: RunStats) -> Dict[str, Dict[str, Any]]:
        """Returns authorId -> {hIndex, citationCount, paperCount}."""
        results: Dict[str, Dict[str, Any]] = {}
        unique_ids = list(dict.fromkeys(a for a in author_ids if a))

        for start in range(0, len(unique_ids), BATCH_LIMIT):
            batch = unique_ids[start : start + BATCH_LIMIT]
            data = self._post_with_retry(
                f"{BASE_URL}/author/batch",
                {"fields": "hIndex,citationCount,paperCount"},
                {"ids": batch},
            )
            if data is None:
                stats.reputation_lookup_errors += len(batch)
                continue
            for author_id, entry in zip(batch, data):
                if entry:
                    results[author_id] = entry

        return results


def enrich_with_reputation(
    triaged: List[TriagedPaper], api_key: Optional[str], stats: RunStats
) -> Dict[str, Dict[str, Any]]:
    """Returns dedup_key -> {citationCount, max_h_index, avg_h_index, unmatched}."""
    lookup = ReputationLookup(api_key)
    relevant_papers = [t.paper for t in triaged if t.relevant]

    paper_data = lookup.lookup_papers(relevant_papers, stats)

    all_author_ids: List[str] = []
    for entry in paper_data.values():
        for author in entry.get("authors", []) or []:
            author_id = author.get("authorId")
            if author_id:
                all_author_ids.append(author_id)

    author_data = lookup.lookup_authors(all_author_ids, stats)

    enriched: Dict[str, Dict[str, Any]] = {}
    for paper in relevant_papers:
        key = paper.dedup_key()
        entry = paper_data.get(key)
        if entry is None:
            enriched[key] = {"citationCount": 0, "max_h_index": 0, "avg_h_index": 0.0, "unmatched": True}
            continue

        h_indices = []
        for author in entry.get("authors", []) or []:
            author_entry = author_data.get(author.get("authorId", ""))
            if author_entry and author_entry.get("hIndex") is not None:
                h_indices.append(author_entry["hIndex"])

        enriched[key] = {
            "citationCount": entry.get("citationCount") or 0,
            "max_h_index": max(h_indices) if h_indices else 0,
            "avg_h_index": (sum(h_indices) / len(h_indices)) if h_indices else 0.0,
            "unmatched": False,
        }

    return enriched

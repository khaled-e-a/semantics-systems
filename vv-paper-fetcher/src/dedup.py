"""In-run merge dedup: the same paper surfaced by multiple sources is merged,
not arbitrarily dropped down to one record — signals like hf_upvotes from an
HF Papers duplicate get unioned onto the arXiv record for the same paper.
"""
from __future__ import annotations

from typing import Dict, List

from .models import Paper


def merge_duplicates(papers: List[Paper]) -> List[Paper]:
    merged: Dict[str, Paper] = {}

    for paper in papers:
        key = paper.dedup_key()
        if key not in merged:
            paper.sources = [paper.source]
            merged[key] = paper
            continue

        existing = merged[key]
        existing.sources.append(paper.source)

        if len(paper.abstract) > len(existing.abstract):
            existing.abstract = paper.abstract
        if not existing.arxiv_id and paper.arxiv_id:
            existing.arxiv_id = paper.arxiv_id
        if not existing.doi and paper.doi:
            existing.doi = paper.doi
        if not existing.pdf_url and paper.pdf_url:
            existing.pdf_url = paper.pdf_url
        if paper.published_date and (
            not existing.published_date or paper.published_date < existing.published_date
        ):
            existing.published_date = paper.published_date
        if len(paper.authors) > len(existing.authors):
            existing.authors = paper.authors
        existing.extra.update({k: v for k, v in paper.extra.items() if k not in existing.extra})

    return list(merged.values())

"""Core data records passed through the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class Paper:
    """A paper normalized from any collector source."""

    title: str
    authors: List[str]
    abstract: str
    url: str
    venue: str
    source: str  # "arxiv" | "hf_papers" | "openreview" | "dblp"
    published_date: Optional[date]
    date_precision: str = "day"  # "day" | "year" (DBLP has no day-level date)
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    sources: List[str] = field(default_factory=list)  # populated on merge
    extra: Dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> str:
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        norm_title = "".join(ch for ch in self.title.lower() if ch.isalnum())
        first_author = ""
        if self.authors:
            first_author = "".join(ch for ch in self.authors[0].lower() if ch.isalnum())
        return f"title:{norm_title}:{first_author}"


@dataclass
class TriagedPaper:
    """A Paper after LLM relevance triage."""

    paper: Paper
    relevant: bool
    relevance_score: int  # 0-5
    summary: str
    tags: List[str]


@dataclass
class ScoredPaper:
    """A TriagedPaper after reputation lookup and composite scoring."""

    triaged: TriagedPaper
    max_author_h_index: int = 0
    avg_author_h_index: float = 0.0
    paper_citation_count: int = 0
    composite_score: float = 0.0
    reputation_unmatched: bool = False


@dataclass
class RunStats:
    """Accumulated counters/errors for the final run summary."""

    collected_per_source: Dict[str, int] = field(default_factory=dict)
    source_errors: Dict[str, str] = field(default_factory=dict)
    after_in_run_dedup: int = 0
    after_keyword_filter: int = 0
    after_cross_run_dedup: int = 0
    llm_batches_total: int = 0
    llm_batches_failed: int = 0
    after_llm_triage: int = 0
    reputation_lookup_errors: int = 0
    final_top_n: int = 0
